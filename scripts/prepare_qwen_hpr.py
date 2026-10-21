"""Cache frozen SFT Qwen prompt states for HPR head training. No SFT/RL occurs."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from qwen_sft import (ROOT, read_json, load_prepared, load_qwen_tokenizer, select_subset,
                      sha256, write_json, CatalogConstraint)
from qwen_eval_fast import prepare_request
from qwen_hpr import HPR_PROTOCOL, encode_context_batch, verify_sft_identity, checkpoint_metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True, help='Existing trained SFT selected/, NOT original Qwen or Q0')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--sft-record', type=Path,
                   default=ROOT / 'experiments/qwen_sft_office2018_20261013/model_artifact.json',
                   help='Binds trained SFT weight SHA256 to its original prepared SID manifest')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--validation-users', type=int, default=5000)
    p.add_argument('--validation-seed', type=int, default=2027)
    p.add_argument('--train-limit', type=int, default=0, help='Positive values label the cache diagnostic-only')
    p.add_argument('--max-length', type=int, default=512)
    args = p.parse_args()
    if args.output.exists() or args.batch_size < 1 or args.train_limit < 0 or args.validation_users < 1:
        raise ValueError('Use a new output directory and positive batch/validation sizes')
    if (args.checkpoint / 'q0_initialization.json').exists():
        raise ValueError('HPR must start from a trained SFT checkpoint, not Q0')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Feature extraction requires a BF16 CUDA GPU; do not load 1.5B on the local CPU')
    manifest, rows, catalog, examples = load_prepared(args.data)
    if args.train_limit:
        examples = examples[:args.train_limit]
    valid = select_subset(rows, args.validation_users, args.validation_seed)
    tokenizer = load_qwen_tokenizer(args.checkpoint)
    CatalogConstraint(tokenizer, catalog, 0, [])  # Audit all FOUR atomic tokens and unique item mapping.
    # Fail rather than silently introducing random SID embeddings.
    token_ids = []
    for layer in range(3):
        tokens = [tokenizer.encode(f'<sid_{layer}_{code}>', add_special_tokens=False)
                  for code in range(manifest['tokens_per_namespace'])]
        if any(len(value) != 1 for value in tokens):
            raise ValueError('Existing SFT tokenizer must contain atomic SID tokens')
        token_ids.append([value[0] for value in tokens])
    if len(set(sum(token_ids, []))) != 3 * manifest['tokens_per_namespace']:
        raise ValueError('SID namespaces must have distinct tokenizer IDs')
    weight_hash = sha256(args.checkpoint / 'model.safetensors')
    verify_sft_identity(read_json(args.sft_record), weight_hash, sha256(args.data / 'manifest.json'))
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    hidden_size = model.config.hidden_size
    estimated = (len(examples) + len(valid)) * hidden_size * 4 + 32 * 1024 * 1024
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output.parent).free < estimated * 1.1:
        raise RuntimeError(f'Feature cache needs approximately {estimated / 2**30:.2f} GiB plus margin')
    args.output.mkdir()
    record = dict(status='preparing', protocol=HPR_PROTOCOL, frozen_qwen=True,
        source_checkpoint=str(args.checkpoint.resolve()), checkpoint_weight_sha256=weight_hash,
        checkpoint_metadata_sha256=checkpoint_metadata(args.checkpoint),
        source_sft_record_sha256=sha256(args.sft_record),
        prepared_manifest_sha256=sha256(args.data / 'manifest.json'),
        source_sft_code_sha256=sha256(Path(__file__).with_name('qwen_sft.py')),
        hidden_size=hidden_size, train_rows=len(examples), valid_rows=len(valid),
        validation_seed=args.validation_seed, max_history=manifest['max_history'], max_length=args.max_length,
        diagnostic_only=args.train_limit > 0, context_dtype='float32',
        context_source='last prompt state before answer; target never enters Qwen input',
        code_sha256={name: sha256(Path(__file__).with_name(name)) for name in
                     ('prepare_qwen_hpr.py', 'qwen_hpr.py', 'qwen_eval_fast.py', 'qwen_sft.py')})
    write_json(args.output / 'manifest.json', record)
    embeddings = model.get_input_embeddings().weight[torch.tensor(token_ids, device=model.device)].float().cpu().numpy()
    np.save(args.output / 'sid_embeddings.npy', embeddings, allow_pickle=False)
    catalog_sids = np.array([catalog[str(i)]['sid'][:3] for i in range(1, len(catalog) + 1)], dtype=np.int64)
    np.save(args.output / 'catalog_sids.npy', catalog_sids, allow_pickle=False)
    np.save(args.output / 'train_positions.npy', examples, allow_pickle=False)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    def train_requests():
        for user, start, end in examples:
            history = rows[user]['train'][start:end]
            yield history, rows[user]['train'][end]

    def valid_requests():
        for row in valid:
            yield row['train'], row['valid'][0]

    for split, size, iterator in [('train', len(examples), train_requests()), ('valid', len(valid), valid_requests())]:
        contexts = np.lib.format.open_memmap(args.output / f'{split}_contexts.npy', mode='w+',
                                             dtype=np.float32, shape=(size, hidden_size))
        targets = np.lib.format.open_memmap(args.output / f'{split}_targets.npy', mode='w+',
                                            dtype=np.int64, shape=(size, 3))
        buffer, buffer_targets, done = [], [], 0

        def flush():
            nonlocal done
            if not buffer:
                return
            value = encode_context_batch(model, tokenizer, buffer).cpu().numpy()
            if not np.isfinite(value).all():
                raise RuntimeError('Non-finite frozen context features')
            contexts[done:done + len(buffer)] = value
            targets[done:done + len(buffer)] = buffer_targets
            done += len(buffer)
            buffer.clear()
            buffer_targets.clear()

        for history, target in iterator:
            if target in history:
                raise ValueError('Target leaked into historical interactions')
            buffer.append(prepare_request(tokenizer, catalog, history, manifest['max_history'], args.max_length))
            buffer_targets.append(catalog[str(target)]['sid'][:3])
            if len(buffer) == args.batch_size:
                flush()
                if done % (args.batch_size * 50) == 0:
                    print(dict(split=split, completed=done, total=size,
                               elapsed_seconds=round(time.perf_counter() - started, 1)), flush=True)
        flush()
        if done != size:
            raise RuntimeError('Feature extraction row count changed')
        contexts.flush()
        targets.flush()
        del contexts, targets
    record.update(status='completed', seconds=time.perf_counter() - started,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                  files={f.name: sha256(f) for f in sorted(args.output.glob('*.npy'))})
    write_json(args.output / 'manifest.json', record)
    print('HPR_FEATURE_CACHE_COMPLETED', {k: record[k] for k in
          ('train_rows', 'valid_rows', 'hidden_size', 'diagnostic_only', 'seconds')}, flush=True)


if __name__ == '__main__':
    main()
