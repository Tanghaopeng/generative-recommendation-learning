"""Validate/test frozen SFT + HPR. Test requires an independently frozen choice."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM

from qwen_sft import load_prepared, load_qwen_tokenizer, select_subset, sha256, read_json, write_json
from qwen_hpr import HPR_PROTOCOL, load_scorer, checkpoint_metadata
from qwen_hpr_decode import predict_hpr


def retrieval_metrics(targets, predictions):
    if not targets or len(targets) != len(predictions):
        raise ValueError('Nonempty matched target/prediction rows required')
    totals = {f'{name}@{k}': 0.0 for k in (10, 20) for name in ('Recall', 'NDCG')}
    for target, predicted in zip(targets, predictions):
        if len(set(predicted)) != len(predicted):
            raise ValueError('Duplicate item in a recommendation list')
        if target in predicted:
            rank = predicted.index(target) + 1
            for k in (10, 20):
                if rank <= k:
                    totals[f'Recall@{k}'] += 1
                    totals[f'NDCG@{k}'] += 1 / math.log2(rank + 1)
    return dict({k: v / len(targets) for k, v in totals.items()}, users=len(targets),
                shortfall_users=sum(len(p) < 20 for p in predictions))


def evaluation_contract(args, head_config):
    provenance = head_config['provenance']
    contract = dict(protocol=HPR_PROTOCOL,
        checkpoint_weight_sha256=sha256(args.checkpoint / 'model.safetensors'),
        checkpoint_metadata_sha256=checkpoint_metadata(args.checkpoint),
        hpr_weight_sha256=sha256(args.hpr / 'model.safetensors'),
        hpr_config_sha256=sha256(args.hpr / 'config.json'),
        prepared_manifest_sha256=sha256(args.data / 'manifest.json'),
        strength=args.strength, pool_size=args.pool_size, beam=args.beam,
        max_history=provenance['max_history'], max_length=provenance['max_length'],
        evaluation_execution=dict(backend='qwen_hpr_kv_beam_v1', batch_size=args.batch_size,
                                  qwen_precision='bf16', hpr_precision='fp32'),
        code_sha256={name: sha256(Path(__file__).with_name(name)) for name in
                     ('qwen_sft.py', 'qwen_eval_fast.py', 'qwen_hpr.py', 'qwen_hpr_decode.py', 'evaluate_qwen_hpr.py')})
    for key in ('checkpoint_weight_sha256', 'checkpoint_metadata_sha256', 'prepared_manifest_sha256'):
        if provenance[key] != contract[key]:
            raise ValueError('HPR was trained for different SFT weights or SID data: ' + key)
    if provenance['source_sft_code_sha256'] != contract['code_sha256']['qwen_sft.py']:
        raise ValueError('Prompt/SID implementation differs from cached HPR training inputs')
    if provenance['prompt_preparation_code_sha256'] != contract['code_sha256']['qwen_eval_fast.py']:
        raise ValueError('Prompt preparation differs from cached HPR training inputs')
    return contract


def check_test_selection(selection, contract, diagnostic_head=False):
    if diagnostic_head or not selection.get('retrieval_validation_completed'):
        raise ValueError('Formal test requires a non-pilot head and completed retrieval validation')
    if selection.get('contract') != contract:
        raise ValueError('Frozen HPR choice differs: checkpoint, SID data, scorer, decode settings or code')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--hpr', type=Path, required=True, help='HPR run selected/ directory')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--split', choices=['valid', 'test'], default='valid')
    p.add_argument('--users', type=int, default=0, help='0=all; a positive count is a fixed diagnostic subset')
    p.add_argument('--seed', type=int, default=2027)
    p.add_argument('--strength', type=float, default=0.25)
    p.add_argument('--pool-size', type=int, default=400)
    p.add_argument('--beam', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--selection-record', type=Path)
    p.add_argument('--export-candidates', type=Path, help='Optional private JSON of first 33 evaluated users')
    args = p.parse_args()
    if args.output.exists() or (args.export_candidates and args.export_candidates.exists()):
        raise ValueError('Preserve prior reports; choose a new output path')
    if (args.users < 0 or args.batch_size < 1 or args.beam < 20 or args.pool_size < args.beam
            or not math.isfinite(args.strength) or args.strength < 0):
        raise ValueError('Invalid HPR retrieval settings')
    head_config = read_json(args.hpr / 'config.json')
    contract = evaluation_contract(args, head_config)
    if args.split == 'test':
        if not args.selection_record:
            raise ValueError('Freeze a validation report before testing')
        check_test_selection(read_json(args.selection_record), contract, head_config['provenance']['diagnostic_only'])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Qwen 1.5B retrieval requires a BF16 CUDA GPU')
    manifest, rows, catalog, _ = load_prepared(args.data)
    rows = select_subset(rows, args.users, args.seed)
    histories = [row['train'] + (row['valid'] if args.split == 'test' else []) for row in rows]
    targets = [row[args.split][0] for row in rows]
    if any(target in h for target, h in zip(targets, histories)):
        raise ValueError('Held-out target overlaps observed history')
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    scorer, _ = load_scorer(args.hpr, 'cuda')
    tokenizer = load_qwen_tokenizer(args.checkpoint)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    candidate_rows = {}

    def collect(user, entries):
        if user < 33:
            candidate_rows[user] = entries

    predictions = predict_hpr(model, tokenizer, catalog, histories, scorer,
        args.batch_size, args.beam, args.pool_size, args.strength, manifest['max_history'], contract['max_length'],
        detail_sink=collect if args.export_candidates else None)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    report = dict(retrieval_metrics(targets, predictions), split=args.split,
        diagnostic_only=args.users > 0 or head_config['provenance']['diagnostic_only'],
        diagnostic_head=head_config['provenance']['diagnostic_only'], subset_seed=args.seed,
        contract=contract, seconds=seconds, users_per_second=len(rows) / seconds,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        hpr_head_selection='contrastive validation loss; retrieval validation chooses decoding strength',
        qwen_updated=False)
    if args.export_candidates:
        # Export from the actual evaluation batches, never regenerate under a different BF16 batch shape.
        details = [candidate_rows[i] for i in sorted(candidate_rows)]
        for entries in details:
            for rank, entry in enumerate(entries, 1):
                item = catalog[str(entry['item_id'])]
                entry.update(rank=rank, sid=item['sid'], title=item.get('title', ''))
        write_json(args.export_candidates, dict(contract=contract, users=[dict(
            evaluation_row=i, target_item=targets[i], candidates=entries) for i, entries in enumerate(details)]))
    write_json(args.output, report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
