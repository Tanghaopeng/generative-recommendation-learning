"""Evaluate saved Qwen+ICA with the existing SID beam/trie retrieval protocol."""
import argparse
from pathlib import Path
import time

import torch

from qwen_ica import QwenICAForCausalLM, load_ica_tokenizer, ICA_PROTOCOL
from qwen_sft import read_json, write_json, sha256, load_prepared, select_subset
from qwen_eval_fast import predict_histories
from evaluate_qwen_hpr import retrieval_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['valid', 'test'], default='valid')
    p.add_argument('--users', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--selection-record', type=Path, help='A completed validation report from this exact ICA model/code')
    p.add_argument('--export-candidates', type=Path)
    args = p.parse_args()
    if args.output.exists() or (args.export_candidates and args.export_candidates.exists()):
        raise ValueError('Use new output paths; preserve existing results')
    if args.users < 0 or args.batch_size < 1:
        raise ValueError('Invalid evaluation subset/batch')
    artifact = read_json(args.checkpoint / 'model_artifact.json')
    contract = dict(protocol=ICA_PROTOCOL, checkpoint_weight_sha256=sha256(args.checkpoint / 'model.safetensors'),
        prepared_manifest_sha256=sha256(args.data / 'manifest.json'),
        checkpoint_config_sha256=sha256(args.checkpoint / 'config.json'),
        tokenizer_sha256=sha256(args.checkpoint / 'tokenizer.json'),
        batch_size=args.batch_size, beam=artifact['config']['beam'], precision='bf16',
        max_length=artifact['config']['max_length'], max_history=artifact['config']['max_history'],
        code_sha256={name: sha256(Path(__file__).with_name(name)) for name in
                     ('qwen_ica.py', 'evaluate_qwen_ica.py', 'qwen_sft.py', 'qwen_eval_fast.py')})
    for key in ('checkpoint_weight_sha256', 'prepared_manifest_sha256'):
        if contract[key] != artifact[key]:
            raise ValueError('ICA model/data artifact mismatch: ' + key)
    if args.split == 'test':
        if not args.selection_record:
            raise ValueError('Choose the checkpoint on validation before testing')
        selection = read_json(args.selection_record)
        if selection['split'] != 'valid' or selection['contract'] != contract:
            raise ValueError('Test differs from the validation-selected model/protocol')
        if selection.get('diagnostic_model'):
            raise ValueError('Do not label a smoke model as a formal test experiment')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Requires BF16 CUDA')
    _, rows, catalog, _ = load_prepared(args.data)
    rows = select_subset(rows, args.users, 2027)
    histories = [row['train'] + (row['valid'] if args.split == 'test' else []) for row in rows]
    targets = [row[args.split][0] for row in rows]
    if any(target in h for target, h in zip(targets, histories)):
        raise ValueError('Held-out target in observed history')
    tokenizer = load_ica_tokenizer(args.checkpoint)
    model = QwenICAForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    last = 0

    def progress(done, total):
        nonlocal last
        if done - last >= 160 or done == total:
            print(f'ICA retrieval: {done}/{total}', flush=True)
            last = done

    predictions = predict_histories(model, tokenizer, catalog, histories, args.batch_size,
        contract['beam'], contract['max_history'], contract['max_length'], on_batch=progress)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    report = dict(retrieval_metrics(targets, predictions), split=args.split, contract=contract,
        diagnostic_only=args.users > 0 or artifact['diagnostic_only'], diagnostic_model=artifact['diagnostic_only'],
        seconds=seconds, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        ica_enabled=artifact['ica_enabled'])
    if args.export_candidates:
        write_json(args.export_candidates, dict(contract=contract, users=[dict(evaluation_row=i, target_item=targets[i],
            candidates=[dict(rank=rank, item_id=item, sid=catalog[str(item)]['sid'],
                             title=catalog[str(item)].get('title', '')) for rank, item in enumerate(items, 1)])
            for i, items in enumerate(predictions[:33])]))
    write_json(args.output, report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
