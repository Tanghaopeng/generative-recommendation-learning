"""Continue full-parameter SFT from verified SFT weights, adding historical ICA."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
from transformers import TrainingArguments, TrainerCallback, EarlyStoppingCallback, set_seed

from qwen_sft import (read_json, write_json, sha256, load_prepared, select_subset,
                      SFTDataset, ValidationDataset, CompletionCollator, RankingTrainer)
from qwen_eval_fast import evaluate_retrieval_fast
from qwen_hpr import verify_sft_identity, checkpoint_metadata
from qwen_ica import ICA_PROTOCOL, QwenICAForCausalLM, load_ica_tokenizer, tokenizer_binding

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True, help='Existing trained ordinary SID SFT; never original Qwen/Q0')
    p.add_argument('--sft-record', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT / 'experiments/qwen_ica_full.json')
    p.add_argument('--resume', type=Path)
    p.add_argument('--control', action='store_true', help='Same continued SFT with ICA disabled; isolates additional training')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--smoke-steps', type=int, default=0, help='Diagnostic only; does not count as a formal experiment')
    args = p.parse_args()
    config = read_json(args.config)
    if args.smoke_steps < 0:
        raise ValueError('Nonnegative smoke steps required')
    if args.smoke_steps:
        config.update(max_steps=args.smoke_steps, eval_steps=args.smoke_steps, validation_users=64, train_limit=1024)
    if config.get('max_steps', -1) > 0 and not args.smoke_steps:
        raise ValueError('Step-limited runs must be explicitly diagnostic')
    config['ica']['enabled'] = not args.control
    if args.output.exists() and not args.resume:
        raise ValueError('Preserve experiments: use a new output or --resume')
    manifest, rows, catalog, examples = load_prepared(args.data)
    weight_hash = sha256(args.checkpoint / 'model.safetensors')
    data_hash = sha256(args.data / 'manifest.json')
    verify_sft_identity(read_json(args.sft_record), weight_hash, data_hash)
    if config['max_history'] != manifest['max_history']:
        raise ValueError('History limit differs from prepared training positions')
    tokenizer = load_ica_tokenizer(args.checkpoint)
    binding = tokenizer_binding(tokenizer, manifest['tokens_per_namespace'])
    for item in catalog.values():
        ids = tokenizer.encode(''.join(f'<sid_{layer}_{v}>' for layer, v in enumerate(item['sid'])), add_special_tokens=False)
        if ids != [binding['sid_token_ids'][i][v] for i, v in enumerate(item['sid'])]:
            raise ValueError('SID tokenizer/catalog mismatch')
    contract = dict(protocol=ICA_PROTOCOL, config=config, control=args.control,
        diagnostic_only=bool(args.smoke_steps), source_sft_weight_sha256=weight_hash,
        source_sft_metadata_sha256=checkpoint_metadata(args.checkpoint),
        prepared_manifest_sha256=data_hash,
        code_sha256={name: sha256(Path(__file__).with_name(name)) for name in
                     ('qwen_ica.py', 'train_qwen_ica.py', 'qwen_sft.py', 'qwen_eval_fast.py')})
    if args.resume:
        if read_json(args.output / 'contract.json') != contract:
            raise ValueError('Resume config/code/source/data mismatch')
        if not (args.resume / 'optimizer.pt').is_file():
            raise ValueError('Resume requires optimizer, scheduler and RNG states')
    train_examples = select_subset(examples, config['train_limit'], config['seed'])
    valid_rows = select_subset(rows, config['validation_users'], config['seed'] + 1)
    train = SFTDataset(rows, catalog, train_examples, tokenizer, 'q1', config['max_length'], config['max_history'])
    valid = ValidationDataset(valid_rows, catalog, tokenizer, config['max_length'], config['max_history'])
    collator = CompletionCollator(tokenizer.pad_token_id)
    if args.dry_run:
        args.output.mkdir(parents=True)
        write_json(args.output / 'dry_run.json', dict(contract=contract, train_rows=len(train),
            example_shape=list(collator([train[0], valid[0]])['input_ids'].shape), actual_gpu_training=False))
        print('ICA tokenizer, SID/source identity and samples verified. No model or optimizer loaded.', flush=True)
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Full ICA SFT requires one BF16 CUDA GPU')
    if torch.cuda.device_count() != 1:
        raise ValueError('Expose exactly one GPU')
    set_seed(config['seed'])
    if args.resume:
        model = QwenICAForCausalLM.from_pretrained(args.resume, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='sdpa')
    else:
        model = QwenICAForCausalLM.from_sft(args.checkpoint, tokenizer, manifest['tokens_per_namespace'],
            config['ica'], dtype=torch.bfloat16, attn_implementation='sdpa')
    model.requires_grad_(True)
    if args.control:
        model.model.ica.requires_grad_(False)
    model.config.use_cache = False
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'contract.json', contract)
    run = dict(status='training', contract=contract, train_rows=len(train),
        total_parameters=sum(p.numel() for p in model.parameters()),
        ica_parameters=sum(p.numel() for p in model.model.ica.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        source_qwen_parameters_updated=True, rqvae_updated=False, sid_mapping_updated=False)
    write_json(args.output / 'run.json', run)

    class GradientAudit(TrainerCallback):
        def __init__(self):
            self.values = []

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            if state.global_step < 3:
                def norm(parameters):
                    values = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
                    return float(torch.stack(values).sum().sqrt()) if values else 0.0
                record = dict(step=state.global_step + 1,
                    ica_grad_norm=norm(model.model.ica.parameters()),
                    qwen_grad_norm=norm(model.model.layers[0].parameters()),
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
                if not __import__('math').isfinite(record['ica_grad_norm'] + record['qwen_grad_norm']):
                    raise RuntimeError('Non-finite ICA/Qwen gradient')
                if record['qwen_grad_norm'] <= 0 or (config['ica']['enabled'] and record['ica_grad_norm'] <= 0):
                    raise RuntimeError('Missing expected ICA/Qwen gradients')
                self.values.append(record)
                write_json(Path(args.output_dir) / 'gradient_audit.json', self.values)
                print('ICA_GRADIENT_AUDIT', record, flush=True)

    def rank(model):
        return evaluate_retrieval_fast(model, tokenizer, catalog, valid_rows, 'valid', config['beam'],
            config['max_history'], config['max_length'], config['evaluation_batch_size'])

    training_args = TrainingArguments(output_dir=str(args.output),
        per_device_train_batch_size=config['micro_batch_size'], per_device_eval_batch_size=config['evaluation_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'], num_train_epochs=config['epochs'], max_steps=config['max_steps'],
        warmup_ratio=0.03, max_grad_norm=1.0, bf16=True, tf32=True, optim='adamw_torch',
        lr_scheduler_type='cosine', gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, eval_strategy='steps',
        eval_steps=config['eval_steps'], save_strategy='steps', save_steps=config['eval_steps'],
        save_total_limit=2, save_only_model=False, load_best_model_at_end=True,
        metric_for_best_model='ndcg10', greater_is_better=True, logging_steps=5,
        prediction_loss_only=True, report_to=[], seed=config['seed'], data_seed=config['seed'],
        dataloader_num_workers=0, remove_unused_columns=False)
    trainer = RankingTrainer(model=model, args=training_args, train_dataset=train, eval_dataset=valid,
        data_collator=collator, processing_class=tokenizer, ranking_evaluator=rank,
        callbacks=[GradientAudit(), EarlyStoppingCallback(early_stopping_patience=3)])
    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    final = trainer.evaluate()
    trainer.save_model(str(args.output / 'selected'))
    tokenizer.save_pretrained(args.output / 'selected')
    trainer.save_state()
    artifact = dict(checkpoint_weight_sha256=sha256(args.output / 'selected/model.safetensors'),
        prepared_manifest_sha256=data_hash, protocol=ICA_PROTOCOL, diagnostic_only=bool(args.smoke_steps),
        source_sft_weight_sha256=weight_hash, ica_enabled=not args.control, config=config)
    write_json(args.output / 'selected/model_artifact.json', artifact)
    run.update(status='completed', seconds=time.perf_counter() - started,
        best_checkpoint=trainer.state.best_model_checkpoint, final_validation=final,
        training_steps=trainer.state.global_step, test_evaluated=False)
    write_json(args.output / 'run.json', run)
    print('ICA SFT completed; selected/ contains Qwen + ICA + tokenizer, all required for inference.', flush=True)


if __name__ == '__main__':
    main()
