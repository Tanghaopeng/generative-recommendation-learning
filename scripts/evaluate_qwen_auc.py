"""Teacher-forced four-SID log-probability AUC, separate from beam Top-K metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_sft import load_prepared, prompt_ids, read_json, select_subset, sha256, sid_text, write_json
from recommendation_auc import candidate_ids, grouped_auc


@torch.inference_mode()
def score_candidates(model, prompt, codes, batch_size):
    scores = []
    for start in range(0, len(codes), batch_size):
        selected = codes[start:start + batch_size]
        ids = torch.tensor([prompt + code for code in selected], device=model.device)
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                       use_cache=False, logits_to_keep=5).logits[:, -5:-1, :].float()
        targets = torch.tensor(selected, device=model.device)
        chosen = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        log_probs = chosen - torch.logsumexp(logits, dim=-1)
        scores.extend(log_probs.sum(-1).cpu().tolist())
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', choices=['valid', 'test'], default='valid')
    parser.add_argument('--users', type=int, default=1000, help='Fixed subset; 0=all users')
    parser.add_argument('--negatives', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--candidate-batch-size', type=int, default=4)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--output', type=Path, required=True, help='A new output directory')
    parser.add_argument('--selection-record', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory; preserve existing metrics')
    if args.users < 0 or args.negatives < 1 or args.candidate_batch_size < 1:
        raise ValueError('Invalid users/negative count/batch size')
    manifest, rows, catalog, _ = load_prepared(args.data)
    weight_hash = sha256(args.checkpoint / 'model.safetensors')
    data_hash = sha256(args.data / 'manifest.json')
    if args.split == 'test':
        if not args.selection_record:
            raise ValueError('Test requires the validation-frozen selection record')
        selected = read_json(args.selection_record)
        if (not selected.get('full_validation_completed') or
                selected['checkpoint_weight_sha256'] != weight_hash or
                selected['prepared_manifest_sha256'] != data_hash or
                selected['max_length'] != args.max_length):
            raise ValueError('Selection record does not match checkpoint/data/prompt length')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This scoring entry requires a BF16 CUDA GPU; no training is started')
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
    codes = {int(item): tokenizer.encode(sid_text(value['sid']), add_special_tokens=False)
             for item, value in catalog.items()}
    if any(len(code) != 4 for code in codes.values()):
        raise ValueError('Load the saved SID-extended tokenizer, not the original Qwen tokenizer')
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
                                               trust_remote_code=False, dtype=torch.bfloat16,
                                               attn_implementation='sdpa').to('cuda').eval()
    rows = select_subset(rows, args.users, args.seed)
    groups, candidate_digest = [], hashlib.sha256()
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (args.output / 'user_scores.jsonl').open('w', encoding='utf-8') as handle:
        for index, row in enumerate(rows):
            history = row['train'] + (row['valid'] if args.split == 'test' else [])
            visible = history[-manifest['max_history']:]
            prompt = prompt_ids(tokenizer, 'next_sid', visible, 0, catalog, manifest['max_history'])
            while len(prompt) + 5 > args.max_length and visible:
                visible = visible[1:]
                prompt = prompt_ids(tokenizer, 'next_sid', visible, 0, catalog, manifest['max_history'])
            if len(prompt) + 5 > args.max_length:
                raise ValueError('Prompt exceeds max length')
            items = candidate_ids(row, len(catalog), args.split, args.negatives, args.seed)
            labels = [1] + [0] * args.negatives
            values = score_candidates(model, prompt, [codes[item] for item in items], args.candidate_batch_size)
            groups.append({'labels': labels, 'scores': values})
            candidate_digest.update((json.dumps([row['user_id'], items], separators=(',', ':')) + '\n').encode())
            handle.write(json.dumps({'user_id': row['user_id'], 'item_ids': items,
                                     'labels': labels, 'scores': values}) + '\n')
            if (index + 1) % 100 == 0:
                print(f'AUC users: {index+1}/{len(rows)}', flush=True)
    metrics = grouped_auc(groups)
    metrics.update(status='completed', split=args.split, users=len(rows),
                   diagnostic_only=len(rows) < manifest['users'], sampled_negatives=args.negatives,
                   protocol=f'uniform_without_replacement_S{args.negatives}', seed=args.seed,
                   score='sum of four SID token log-probabilities under the unmasked full vocabulary; no EOS',
                   negative_interpretation='unobserved proxy negatives, not exposed dislikes',
                   pooled_auc_warning='cross-user raw scores; not calibrated CTR probabilities',
                   candidate_set_sha256=candidate_digest.hexdigest(),
                   checkpoint_weight_sha256=weight_hash, prepared_manifest_sha256=data_hash,
                   seconds=time.perf_counter()-started, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                   max_length=args.max_length, code_sha256={name: sha256(Path(__file__).with_name(name))
                                                           for name in ('evaluate_qwen_auc.py', 'recommendation_auc.py', 'qwen_sft.py')})
    write_json(args.output / 'metrics.json', metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
