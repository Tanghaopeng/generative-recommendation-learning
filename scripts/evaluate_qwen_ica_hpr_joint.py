"""Validate/test the jointly trained ICA + Qwen + integrated HPR checkpoint."""
import argparse
from pathlib import Path
import time

import torch

from qwen_sft import read_json, write_json, sha256, load_prepared, select_subset
from qwen_ica import load_ica_tokenizer
from qwen_hpr import checkpoint_metadata
from qwen_hpr_decode import predict_hpr
from qwen_ica_hpr_joint import JOINT_PROTOCOL, QwenICAHPRForCausalLM
from evaluate_qwen_hpr import retrieval_metrics, check_test_selection


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['valid', 'test'], default='valid')
    p.add_argument('--users', type=int, default=0)
    p.add_argument('--seed', type=int, default=2027)
    p.add_argument('--strength', type=float, default=0.25)
    p.add_argument('--selection-record', type=Path)
    p.add_argument('--export-candidates', type=Path)
    args = p.parse_args()
    if args.output.exists() or (args.export_candidates and args.export_candidates.exists()):
        raise ValueError('Preserve existing results; choose new outputs')
    if args.users < 0 or not __import__('math').isfinite(args.strength) or args.strength < 0:
        raise ValueError('Nonnegative users and finite nonnegative strength required')
    artifact = read_json(args.checkpoint / 'model_artifact.json')
    if artifact['protocol'] != JOINT_PROTOCOL:
        raise ValueError('This loader requires a jointly trained ICA/HPR checkpoint')
    config = artifact['config']
    contract = dict(protocol=JOINT_PROTOCOL,
        checkpoint_weight_sha256=sha256(args.checkpoint / 'model.safetensors'),
        checkpoint_metadata_sha256=checkpoint_metadata(args.checkpoint),
        prepared_manifest_sha256=sha256(args.data / 'manifest.json'),
        strength=args.strength, pool_size=config['pool_size'], beam=config['beam'],
        max_history=config['max_history'], max_length=config['max_length'],
        evaluation_execution=dict(backend='qwen_hpr_kv_beam_v1', batch_size=config['evaluation_batch_size'],
                                  qwen_precision='bf16', hpr_precision='fp32'),
        code_sha256={name: sha256(Path(__file__).with_name(name)) for name in
            ('qwen_ica_hpr_joint.py', 'evaluate_qwen_ica_hpr_joint.py', 'qwen_ica.py', 'qwen_hpr.py',
             'qwen_sft.py', 'qwen_eval_fast.py', 'qwen_hpr_decode.py')})
    for key in ('checkpoint_weight_sha256', 'prepared_manifest_sha256'):
        if artifact[key] != contract[key]:
            raise ValueError('Joint checkpoint/SID artifact mismatch: ' + key)
    if args.split == 'test':
        if not args.selection_record:
            raise ValueError('Freeze validation decoding choice before testing')
        check_test_selection(read_json(args.selection_record), contract, artifact['diagnostic_only'])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Requires BF16 CUDA')
    _, rows, catalog, _ = load_prepared(args.data)
    rows = select_subset(rows, args.users, args.seed)
    histories = [r['train'] + (r['valid'] if args.split == 'test' else []) for r in rows]
    targets = [r[args.split][0] for r in rows]
    if any(t in h for t, h in zip(targets, histories)):
        raise ValueError('Held-out item is visible in history')
    model = QwenICAHPRForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
    if any(p.dtype != torch.float32 for p in model.hpr.parameters()):
        raise RuntimeError('Integrated HPR failed to restore FP32 parameters')
    tokenizer = load_ica_tokenizer(args.checkpoint)
    details = {}

    def collect(user, entries):
        if user < 33:
            details[user] = entries

    started = time.perf_counter()
    predictions = predict_hpr(model, tokenizer, catalog, histories, model.hpr,
        config['evaluation_batch_size'], config['beam'], config['pool_size'], args.strength,
        config['max_history'], config['max_length'], detail_sink=collect if args.export_candidates else None)
    torch.cuda.synchronize()
    report = dict(retrieval_metrics(targets, predictions), contract=contract, split=args.split,
        subset_seed=args.seed, diagnostic_only=args.users > 0 or artifact['diagnostic_only'],
        diagnostic_head=artifact['diagnostic_only'], joint_training=True, seconds=time.perf_counter() - started)
    if args.export_candidates:
        users = []
        for user in sorted(details):
            for rank, entry in enumerate(details[user], 1):
                item = catalog[str(entry['item_id'])]
                entry.update(rank=rank, sid=item['sid'], title=item.get('title', ''))
            users.append(dict(evaluation_row=user, target_item=targets[user], candidates=details[user]))
        write_json(args.export_candidates, dict(contract=contract, users=users))
    write_json(args.output, report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
