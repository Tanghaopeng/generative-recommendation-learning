"""Post-hoc candidate inspection using the unchanged retrieval implementation.

Exports final beam candidates, SID components and transition scores; does not
retrain, select checkpoints, or replace the frozen formal test result.
"""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM
from qwen_sft import (CatalogConstraint, SID_DECODING_PROTOCOL, load_prepared,
                      load_qwen_tokenizer, retrieve, select_subset, sha256, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--users', type=int, default=32, help='0 exports all users; sample is diagnostic only')
    parser.add_argument('--include-user', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--beam', type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest, rows, catalog, _ = load_prepared(args.data)
    chosen = select_subset(rows, args.users, args.seed)
    if args.include_user not in {r['user_id'] for r in chosen}:
        chosen = [next(r for r in rows if r['user_id'] == args.include_user)] + chosen
    tokenizer = load_qwen_tokenizer(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
              dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    original_generate = model.generate
    captured = {}

    def capture_generate(*a, **kw):
        result = original_generate(*a, **kw)
        captured['result'] = result
        captured['prompt_length'] = kw['input_ids'].shape[1]
        captured['prompt_ids'] = kw['input_ids'][0].tolist()
        return result

    model.generate = capture_generate
    constraint = CatalogConstraint(tokenizer, catalog, 0, [])
    records = []
    started = time.perf_counter()
    try:
        with (args.output / 'users.jsonl').open('w', encoding='utf-8') as out:
            for row in chosen:
                history = row['train'] + row['valid']
                target = row['test'][0]
                predicted = retrieve(model, tokenizer, catalog, history, args.beam,
                                     manifest['max_history'], 512, constraint)
                result = captured['result']
                with torch.inference_mode():
                    transition = model.compute_transition_scores(result.sequences, result.scores,
                                    result.beam_indices, normalize_logits=False).float().cpu().tolist()
                by_item = {}
                for sequence, score, steps in zip(result.sequences, result.sequences_scores, transition):
                    code = tuple(sequence[captured['prompt_length']:].tolist())
                    item = constraint.leaves.get(code)
                    score = float(score)
                    if not item or item in history or not math.isfinite(score):
                        continue
                    if len(steps) != 4 or not math.isclose(sum(steps), score, abs_tol=0.02):
                        raise ValueError('Transition scores do not reconstruct the formal beam score')
                    value = {'item_id': item, 'title': catalog[str(item)]['title'],
                             'sid': catalog[str(item)]['sid'], 'token_ids': list(code),
                             'sequence_log_probability': score,
                             'step_log_probabilities': steps,
                             'step_probabilities': [math.exp(x) for x in steps],
                             'is_test_target': item == target}
                    if item not in by_item or score > by_item[item]['sequence_log_probability']:
                        by_item[item] = value
                candidates = [dict(rank=i+1, **by_item[item]) for i, item in enumerate(predicted)]
                prefix_counts = Counter(c['sid'][0] for c in candidates)
                record = {'user_id': row['user_id'], 'split': 'test', 'post_hoc_diagnostic': True,
                          'history_item_ids': history, 'visible_history_item_ids': history[-manifest['max_history']:],
                          'prompt_token_ids': captured['prompt_ids'],
                          'prompt_text': tokenizer.decode(captured['prompt_ids'], skip_special_tokens=False),
                          'target_item_id': target, 'target_title': catalog[str(target)]['title'],
                          'target_sid': catalog[str(target)]['sid'],
                          'target_rank': predicted.index(target)+1 if target in predicted else None,
                          'candidate_count': len(candidates),
                          'unique_prefixes_by_depth': {str(d): len({tuple(c['sid'][:d]) for c in candidates}) for d in (1,2,3,4)},
                          'first_sid_counts': dict(sorted(prefix_counts.items())), 'candidates': candidates}
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
                out.flush()
                records.append(record)
                print(f"EXPORTED user={row['user_id']} candidates={len(candidates)} first_sid_count={len(prefix_counts)}", flush=True)
                captured.clear()
    finally:
        model.generate = original_generate
    with (args.output / 'candidates.csv').open('w', encoding='utf-8-sig', newline='') as f:
        fields = ['user_id','rank','item_id','title','sid_0','sid_1','sid_2','sid_3',
                  'sequence_log_probability','step_log_probabilities','is_test_target']
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for r in records:
            for c in r['candidates']:
                writer.writerow({'user_id': r['user_id'], **{k:c[k] for k in ('rank','item_id','title','sequence_log_probability','is_test_target')},
                                 **{f'sid_{d}': c['sid'][d] for d in range(4)},
                                 'step_log_probabilities': json.dumps(c['step_log_probabilities'])})
    distribution = Counter(r['unique_prefixes_by_depth']['1'] for r in records)
    write_json(args.output / 'summary.json', {
        'status': 'post_hoc_candidate_export_completed', 'users': len(records),
        'sampling': f'seed={args.seed}, requested={args.users}, additionally include user={args.include_user}',
        'not_a_new_formal_test_or_model_selection': True, 'beam': args.beam,
        'first_sid_unique_count_distribution': dict(sorted(distribution.items())),
        'mean_unique_first_sid': sum(k*v for k,v in distribution.items()) / len(records),
        'decoding_protocol': SID_DECODING_PROTOCOL, 'seconds': time.perf_counter()-started,
        'checkpoint_weight_sha256': sha256(args.checkpoint / 'model.safetensors'),
        'prepared_manifest_sha256': sha256(args.data / 'manifest.json'),
        'retrieve_code_sha256': sha256(Path(__file__).with_name('qwen_sft.py')),
        'export_code_sha256': sha256(__file__),
        'scores_note': 'Full-vocabulary log probabilities before illegal candidates are masked; not renormalized over legal children or top20.',
        'coverage_note': 'Final surviving candidates only; first-stage beam trajectory is not recorded.'})
    print('CANDIDATE_EXPORT_COMPLETED', flush=True)


if __name__ == '__main__':
    main()
