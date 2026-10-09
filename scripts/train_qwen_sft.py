"""Full-parameter Qwen SID SFT, with tokenizer-only dry run and NDCG selection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from qwen_sft import (ROOT, MODEL_REVISION, MODEL_SHA256, read_json, write_json, sha256,
                      load_prepared, extend_tokenizer, select_subset, SFTDataset,
                      ValidationDataset, CompletionCollator, evaluate_retrieval, RankingTrainer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true', help='Tokenizer and sample checks only; NO 1.5B model loaded')
    parser.add_argument('--resume', type=Path, help='A Trainer checkpoint, including optimizer/scheduler/RNG')
    args = parser.parse_args()
    config = read_json(args.config)
    manifest, rows, catalog, examples = load_prepared(args.data)
    if config['max_history'] != manifest['max_history']:
        raise ValueError('Config history length differs from prepared training positions')
    if args.output.exists() and not args.resume:
        raise ValueError('Use a new output directory, or --resume an existing checkpoint')
    if config.get('max_steps', -1) > 0 and not config['pilot']:
        raise ValueError('Step-limited training must be labelled pilot')
    set_seed(config['seed'])
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    vocab = extend_tokenizer(tokenizer, manifest['tokens_per_namespace'])
    train_examples = select_subset(examples, config['train_limit'], config['seed'])
    valid_rows = select_subset(rows, config['validation_users'], config['seed'] + 1)
    train = SFTDataset(rows, catalog, train_examples, tokenizer, config['recipe'], config['max_length'], config['max_history'])
    valid = ValidationDataset(valid_rows, catalog, tokenizer, config['max_length'], config['max_history'])
    collator = CompletionCollator(tokenizer.pad_token_id)
    # Validate every catalog mapping, not just the tokens used by a pilot.
    from qwen_sft import sid_text
    for item in catalog.values():
        if len(tokenizer.encode(sid_text(item['sid']), add_special_tokens=False)) != 4:
            raise ValueError('Complete SID must encode to exactly four tokens')
    checks = [train[0], train[len(train) - 1], valid[0]]
    if any(sum(label != -100 for label in row['labels']) < 2 for row in checks):
        raise ValueError('Empty completion supervision')
    batch = collator(checks)
    run = dict(status='tokenizer_and_sample_checks_passed', actual_gpu_training=False,
               model='Qwen/Qwen2.5-1.5B-Instruct', model_revision=MODEL_REVISION,
               model_metadata_sha256={name: sha256(args.model / name) for name in
                                      ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt')},
               config=config, prepared_manifest_sha256=sha256(args.data / 'manifest.json'),
               vocabulary=vocab, train_rows=len(train), main_next_item_rows=len(train_examples),
               validation_users=len(valid_rows), example_batch_shape=list(batch['input_ids'].shape),
               supervised_tokens_per_example=[sum(x != -100 for x in row['labels']) for row in checks],
               code_sha256={name: sha256(ROOT / 'scripts' / name) for name in
                            ('qwen_sft.py', 'prepare_qwen_sft.py', 'train_qwen_sft.py')},
               timestamp_utc=datetime.now(timezone.utc).isoformat())
    if args.dry_run:
        args.output.mkdir(parents=True)
        tokenizer.save_pretrained(args.output / 'tokenizer')
        write_json(args.output / 'dry_run.json', run)
        print({key: run[key] for key in ('status', 'vocabulary', 'train_rows', 'validation_users', 'example_batch_shape')})
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Training requires a CUDA GPU with BF16; CPU 1.5B training is disabled. Use --dry-run.')
    if torch.cuda.device_count() != 1 or int(__import__('os').environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This recipe is validated for one GPU only; expose one using CUDA_VISIBLE_DEVICES')
    if sha256(args.model / 'model.safetensors') != MODEL_SHA256:
        raise ValueError('Base model weight differs from the pinned, verified Qwen checkpoint')
    from transformers import TrainingArguments, EarlyStoppingCallback

    # Resume guards prevent mixing SIDs, subsets, recipes, or code with old optimizer states.
    contract = {key: run[key] for key in ('config', 'prepared_manifest_sha256', 'code_sha256', 'model_revision', 'model_metadata_sha256')}
    if args.resume:
        if args.dry_run or not args.resume.is_dir():
            raise ValueError('Invalid resume checkpoint')
        if read_json(args.output / 'contract.json') != contract:
            raise ValueError('Resume contract mismatch: configuration, data or code changed')
        if not (args.resume / 'optimizer.pt').is_file():
            raise ValueError('Resume requires optimizer state, not just model weights')
        saved_tokenizer = AutoTokenizer.from_pretrained(args.resume, local_files_only=True)
        if saved_tokenizer.get_vocab() != tokenizer.get_vocab():
            raise ValueError('Resume SID tokenizer differs')
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                                               trust_remote_code=False, dtype=torch.bfloat16,
                                               attn_implementation='sdpa')
    # Explicit normal initialization avoids a costly covariance computation on the large Qwen vocabulary.
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.requires_grad_(True)
    model.config.use_cache = False
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError('Full-parameter SFT must have no frozen Qwen parameters')
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'contract.json', contract)
    run.update(status='training', actual_gpu_training=True,
               parameters=sum(p.numel() for p in model.parameters()), gpu=torch.cuda.get_device_name(0),
               versions={package: importlib.metadata.version(package) for package in
                         ('torch', 'transformers', 'accelerate', 'numpy')},
               base_weight_sha256=MODEL_SHA256,
               effective_batch_size=config['micro_batch_size'] * config['gradient_accumulation_steps'])
    write_json(args.output / 'run.json', run)
    training_args = TrainingArguments(
        output_dir=str(args.output), per_device_train_batch_size=config['micro_batch_size'],
        per_device_eval_batch_size=config['micro_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        num_train_epochs=config['epochs'], max_steps=config['max_steps'],
        learning_rate=config['learning_rate'], warmup_ratio=0.03, max_grad_norm=1.0,
        bf16=True, tf32=True, optim='adamw_torch', lr_scheduler_type='cosine',
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='steps', eval_steps=config['eval_steps'], save_strategy='steps',
        save_steps=config['eval_steps'], save_total_limit=2, save_only_model=False,
        load_best_model_at_end=True, metric_for_best_model='ndcg10', greater_is_better=True,
        logging_steps=5, prediction_loss_only=True, report_to=[],
        seed=config['seed'], data_seed=config['seed'], dataloader_num_workers=0,
        remove_unused_columns=False)
    trainer = RankingTrainer(model=model, args=training_args, train_dataset=train,
                                    eval_dataset=valid, data_collator=collator, processing_class=tokenizer,
                                    ranking_evaluator=lambda model: evaluate_retrieval(
                                        model, tokenizer, catalog, valid_rows, 'valid', config['beam'],
                                        config['max_history'], config['max_length']),
                                    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    # Export selected weights; periodic checkpoint-* directories retain optimizer/scheduler/RNG.
    final_metrics = trainer.evaluate()
    trainer.save_model(str(args.output / 'selected'))
    tokenizer.save_pretrained(args.output / 'selected')
    trainer.save_state()
    run.update(status='sft_completed_requires_full_validation_before_test',
               best_checkpoint=trainer.state.best_model_checkpoint, final_validation=final_metrics,
               training_steps=trainer.state.global_step, log_history=trainer.state.log_history,
               test_evaluated=False)
    write_json(args.output / 'run.json', run)
    print('SFT completed. Run full validation on finalists before a frozen single test evaluation.')


if __name__ == '__main__':
    main()
