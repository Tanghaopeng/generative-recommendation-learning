"""Audit ordered SID candidates and benchmark isolated evaluation prototypes."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM

from qwen_sft import (CatalogConstraint, load_prepared, load_qwen_tokenizer,
                      retrieve, select_subset, sha256, write_json)
from qwen_eval_fast import predict_histories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--users', type=int, default=128)
    parser.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.users < 32:
        raise ValueError('Preserve existing output; audit at least 32 users')
    manifest, rows, catalog, _ = load_prepared(args.data)
    selected = select_subset(rows, args.users, 2027)
    histories = [row['train'] for row in selected]
    tokenizer = load_qwen_tokenizer(args.checkpoint)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, local_files_only=True,
        dtype=torch.bfloat16 if args.precision == 'bf16' else torch.float32,
        attn_implementation='sdpa').to('cuda').eval()
    index = CatalogConstraint(tokenizer, catalog, 0, [])
    reference = None
    report = dict(status='evaluation_prototype_benchmark', diagnostic_only=True,
                  checkpoint=str(args.checkpoint), users=len(histories), split='valid',
                  subset_seed=2027, beam=20, precision=args.precision, tf32=False,
                  prepared_manifest_sha256=sha256(args.data / 'manifest.json'),
                  checkpoint_weight_sha256=sha256(args.checkpoint / 'model.safetensors'),
                  code_sha256={name: sha256(Path(__file__).with_name(name)) for name in
                               ('qwen_sft.py', 'qwen_eval_fast.py', 'benchmark_qwen_eval.py')},
                  model_training_or_weights_modified=False, modes={})
    for name, batch in [('reference', 1), ('bulk_io_b1', 1), ('bucket_b4', 4), ('bucket_b8', 8)]:
        def predict(values):
            if name == 'reference':
                return [retrieve(model, tokenizer, catalog, history, constraint=index)
                        for history in values]
            return predict_histories(model, tokenizer, catalog, values, batch,
                                     max_history=manifest['max_history'], index=index)
        predict(histories[:8])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        predictions = predict(histories)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        if reference is None:
            reference = predictions
        mismatches = [i for i, (a, b) in enumerate(zip(reference, predictions)) if a != b]
        item_set_mismatches = sum(set(a) != set(b) for a, b in zip(reference, predictions))
        record = dict(batch_size=batch, seconds=seconds, users_per_second=len(histories) / seconds,
                      peak_cuda_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_cuda_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                      exact_order_matches=len(histories) - len(mismatches),
                      item_set_mismatches=item_set_mismatches,
                      all_ordered_candidates_identical=not mismatches,
                      shortfall_users=sum(len(items) < 20 for items in predictions),
                      prediction_sha256=hashlib.sha256(json.dumps(predictions, separators=(',', ':')).encode()).hexdigest(),
                      mismatch_examples=[dict(user=i, reference=reference[i], candidate=predictions[i])
                                         for i in mismatches[:3]])
        report['modes'][name] = record
        write_json(args.output, report)
        print('EVAL_BENCHMARK_MODE', name, json.dumps(record), flush=True)
    report['limitations'] = ('Short validation-only prototype benchmark, not formal model evaluation. '
                            'Batched BF16 computation can change near-tied rankings. '
                            'A matching audit is evidence for this checkpoint/subset only; '
                            're-audit a selected trained checkpoint before deployment.')
    write_json(args.output, report)


if __name__ == '__main__':
    main()
