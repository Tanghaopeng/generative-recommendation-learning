"""From the original trained SFT, jointly update Qwen + ICA + HPR using CE + HPR."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import torch
from transformers import TrainingArguments, TrainerCallback, EarlyStoppingCallback, set_seed

from qwen_sft import (read_json, write_json, sha256, load_prepared, select_subset,
                      SFTDataset, ValidationDataset, CompletionCollator, RankingTrainer)
from qwen_ica import load_ica_tokenizer, tokenizer_binding
from qwen_hpr import verify_sft_identity, checkpoint_metadata
from qwen_hpr_decode import predict_hpr
from evaluate_qwen_hpr import retrieval_metrics
from qwen_ica_hpr_joint import JOINT_PROTOCOL, QwenICAHPRForCausalLM

ROOT = Path(__file__).resolve().parents[1]


class JointTrainer(RankingTrainer):
    def __init__(self, *args, hpr_learning_rate, **kwargs):
        super().__init__(*args, **kwargs)
        self.hpr_learning_rate = hpr_learning_rate
        self.last_losses = None

    def create_optimizer(self):
        if self.optimizer is None:
            decay = set(self.get_decay_parameter_names(self.model))
            groups = []
            for is_head in (False, True):
                for use_decay in (False, True):
                    parameters = [p for name, p in self.model.named_parameters() if p.requires_grad
                                  and name.startswith('hpr.') == is_head and (name in decay) == use_decay]
                    if parameters:
                        groups.append(dict(params=parameters, lr=self.hpr_learning_rate if is_head else self.args.learning_rate,
                                           weight_decay=self.args.weight_decay if use_decay else 0.0))
            self.optimizer = torch.optim.AdamW(groups)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, output = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        if model.training:
            self.last_losses = (output.ce_loss.detach(), output.hpr_loss.detach())
        return (loss, output) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if 'loss' in logs and self.last_losses is not None:
            logs.update(last_microbatch_ce=float(self.last_losses[0]), last_microbatch_hpr=float(self.last_losses[1]))
        return super().log(logs, *args, **kwargs)


class JointGradientAudit(TrainerCallback):
    def __init__(self):
        self.records = []

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if state.global_step < 3:
            def norm(module):
                values = [p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
                return float(torch.stack(values).sum().sqrt()) if values else 0.0
            record = dict(step=state.global_step + 1, ica_grad_norm=norm(model.model.ica),
                qwen_grad_norm=norm(model.model.layers[0]), hpr_grad_norm=norm(model.hpr),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
            for key in ('ica_grad_norm', 'qwen_grad_norm', 'hpr_grad_norm'):
                if not math.isfinite(record[key]) or record[key] <= 0:
                    raise RuntimeError('Joint gradient audit failed: ' + key)
            self.records.append(record)
            write_json(Path(args.output_dir) / 'gradient_audit.json', self.records)
            print('JOINT_GRADIENT_AUDIT', record, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True, help='Original ordinary trained SFT selected/, not ICA or Q0')
    p.add_argument('--sft-record', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT / 'experiments/qwen_ica_hpr_joint.json')
    p.add_argument('--resume', type=Path)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--smoke-steps', type=int, default=0)
    args = p.parse_args()
    config = read_json(args.config)
    if args.smoke_steps < 0:
        raise ValueError('Nonnegative smoke steps required')
    if args.smoke_steps:
        config.update(max_steps=args.smoke_steps, eval_steps=args.smoke_steps, validation_users=64, train_limit=1024)
    if config['max_steps'] > 0 and not args.smoke_steps:
        raise ValueError('Limited training must be labelled smoke-only')
    if args.output.exists() and not args.resume:
        raise ValueError('Use a new output or explicitly --resume')
    if not config['ica']['enabled'] or config['joint']['loss_weight'] <= 0:
        raise ValueError('This combination requires both ICA and a positive HPR objective')
    manifest, rows, catalog, examples = load_prepared(args.data)
    weight_hash, data_hash = sha256(args.checkpoint / 'model.safetensors'), sha256(args.data / 'manifest.json')
    verify_sft_identity(read_json(args.sft_record), weight_hash, data_hash)
    if config['max_history'] != manifest['max_history']:
        raise ValueError('Prepared history limit mismatch')
    tokenizer = load_ica_tokenizer(args.checkpoint)
    tokenizer_binding(tokenizer, manifest['tokens_per_namespace'])
    contract = dict(protocol=JOINT_PROTOCOL, config=config, diagnostic_only=bool(args.smoke_steps),
        source_sft_weight_sha256=weight_hash, source_sft_metadata_sha256=checkpoint_metadata(args.checkpoint),
        prepared_manifest_sha256=data_hash, code_sha256={name: sha256(Path(__file__).with_name(name)) for name in
            ('qwen_ica_hpr_joint.py', 'train_qwen_ica_hpr_joint.py', 'qwen_ica.py', 'qwen_hpr.py',
             'qwen_sft.py', 'qwen_eval_fast.py', 'qwen_hpr_decode.py')})
    if args.resume and (read_json(args.output / 'contract.json') != contract or not (args.resume / 'optimizer.pt').exists()):
        raise ValueError('Resume requires matching source/config/code/data and optimizer checkpoint')
    train_examples = select_subset(examples, config['train_limit'], config['seed'])
    valid_rows = select_subset(rows, config['validation_users'], config['seed'] + 1)
    train = SFTDataset(rows, catalog, train_examples, tokenizer, 'q1', config['max_length'], config['max_history'])
    valid = ValidationDataset(valid_rows, catalog, tokenizer, config['max_length'], config['max_history'])
    collator = CompletionCollator(tokenizer.pad_token_id)
    if args.dry_run:
        args.output.mkdir(parents=True)
        write_json(args.output / 'dry_run.json', dict(contract=contract, train_rows=len(train),
            example_shape=list(collator([train[0], valid[0]])['input_ids'].shape), actual_gpu_training=False))
        print('Joint tokenizer/source/SID samples verified; no model or optimizer loaded.', flush=True)
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose one BF16 CUDA GPU; use --dry-run for CPU sample checks')
    set_seed(config['seed'])
    if args.resume:
        model = QwenICAHPRForCausalLM.from_pretrained(args.resume, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='sdpa')
    else:
        model = QwenICAHPRForCausalLM.from_sft(args.checkpoint, tokenizer, manifest['tokens_per_namespace'],
            config['ica'], config['joint'], dtype=torch.bfloat16, attn_implementation='sdpa')
    model.set_catalog(catalog)
    model.requires_grad_(True)
    model.config.use_cache = False
    if any(p.dtype != torch.float32 for p in model.hpr.parameters()):
        raise RuntimeError('HPR parameters must stay FP32')
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'contract.json', contract)
    run = dict(status='training', contract=contract, all_components_jointly_updated=True, frozen_context_cache=False,
        total_parameters=sum(p.numel() for p in model.parameters()),
        ica_parameters=sum(p.numel() for p in model.model.ica.parameters()),
        hpr_parameters=sum(p.numel() for p in model.hpr.parameters()), train_rows=len(train))
    write_json(args.output / 'run.json', run)

    def rank(model):
        predictions = predict_hpr(model, tokenizer, catalog, [r['train'] for r in valid_rows], model.hpr,
            config['evaluation_batch_size'], config['beam'], config['pool_size'], config['validation_strength'],
            config['max_history'], config['max_length'])
        return retrieval_metrics([r['valid'][0] for r in valid_rows], predictions)

    training_args = TrainingArguments(output_dir=str(args.output),
        per_device_train_batch_size=config['micro_batch_size'], per_device_eval_batch_size=config['evaluation_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'], num_train_epochs=config['epochs'],
        max_steps=config['max_steps'], learning_rate=config['learning_rate'], weight_decay=0.01,
        warmup_ratio=0.03, max_grad_norm=1.0, bf16=True, tf32=True, lr_scheduler_type='cosine',
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='steps', eval_steps=config['eval_steps'], save_strategy='steps', save_steps=config['eval_steps'],
        save_total_limit=2, save_only_model=False, load_best_model_at_end=True, metric_for_best_model='ndcg10',
        greater_is_better=True, logging_steps=5, prediction_loss_only=True, report_to=[],
        seed=config['seed'], data_seed=config['seed'], dataloader_num_workers=0, remove_unused_columns=False)
    trainer = JointTrainer(model=model, args=training_args, train_dataset=train, eval_dataset=valid,
        data_collator=collator, processing_class=tokenizer, ranking_evaluator=rank,
        hpr_learning_rate=config['hpr_learning_rate'],
        callbacks=[JointGradientAudit(), EarlyStoppingCallback(early_stopping_patience=3)])
    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    metrics = trainer.evaluate()
    trainer.save_model(str(args.output / 'selected'))
    tokenizer.save_pretrained(args.output / 'selected')
    trainer.save_state()
    write_json(args.output / 'selected/model_artifact.json', dict(protocol=JOINT_PROTOCOL,
        checkpoint_weight_sha256=sha256(args.output / 'selected/model.safetensors'),
        prepared_manifest_sha256=data_hash, diagnostic_only=bool(args.smoke_steps), config=config,
        source_sft_weight_sha256=weight_hash, source='original_sft_jointly_continued'))
    run.update(status='completed', seconds=time.perf_counter() - started, final_validation=metrics,
        best_checkpoint=trainer.state.best_model_checkpoint, training_steps=trainer.state.global_step, test_evaluated=False)
    write_json(args.output / 'run.json', run)
    print('Joint ICA + Qwen + HPR training completed. Validate decoding strengths before test.', flush=True)


if __name__ == '__main__':
    main()
