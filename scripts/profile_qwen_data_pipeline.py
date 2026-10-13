"""CPU-only audit of Q1 sample preparation, tokenization, collation and Parquet.

Does not load model weights or change training artifacts. The sampled throughput
is an estimate for a full pass, not the historical cloud preprocessing duration.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import time

STARTED = time.perf_counter()
import numpy as np
from qwen_sft import (SFTDataset, CompletionCollator, load_prepared,
                      load_qwen_tokenizer, extend_tokenizer, select_subset,
                      positions, write_json, sha256)
IMPORT_SECONDS = time.perf_counter() - STARTED


def timed(call):
    start = time.perf_counter()
    value = call()
    return value, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=4096)
    args = parser.parse_args()
    if args.output.exists() or args.samples < 128:
        raise ValueError('Use a new output directory and at least 128 samples')
    args.output.mkdir(parents=True)
    print('PROFILE imports completed', round(IMPORT_SECONDS, 3), flush=True)
    (manifest, rows, catalog, examples), load_seconds = timed(lambda: load_prepared(args.data))
    tokenizer, tokenizer_seconds = timed(lambda: load_qwen_tokenizer(args.tokenizer))
    vocab = extend_tokenizer(tokenizer, manifest['tokens_per_namespace'])
    rebuilt, positions_seconds = timed(lambda: positions(rows, manifest['max_history']))
    if not np.array_equal(rebuilt, examples):
        raise ValueError('Recomputed training positions differ')
    subset = select_subset(examples, args.samples, 2026)
    dataset, init_seconds = timed(lambda: SFTDataset(rows, catalog, subset, tokenizer,
                                                    'q1', 512, manifest['max_history']))
    # Warm tokenizer/template caches; do not count cold Jinja compilation as every-row work.
    for i in range(min(32, len(dataset))):
        dataset[i]
    samples, encode_seconds = timed(lambda: [dataset[i] for i in range(len(dataset))])
    print('PROFILE sample encoding', len(samples), round(encode_seconds, 3), flush=True)
    collator = CompletionCollator(tokenizer.pad_token_id)
    _, collate_seconds = timed(lambda: [collator(samples[i:i + 4])
                                        for i in range(0, len(samples), 4)])
    lengths = np.asarray([len(row['input_ids']) for row in samples])
    result = {
        'status': 'cpu_profile_completed', 'historical_cloud_timing': False,
        'model_weights_loaded': False, 'recipe': 'q1', 'seed': 2026,
        'platform': platform.platform(), 'processor': platform.processor(),
        'python': platform.python_version(),
        'versions': {name: importlib.metadata.version(name)
                     for name in ('numpy', 'torch', 'transformers', 'tokenizers')},
        'prepared_manifest_sha256': sha256(args.data / 'manifest.json'),
        'qwen_sft_sha256': sha256(Path(__file__).with_name('qwen_sft.py')),
        'vocabulary': vocab, 'users': len(rows), 'full_train_examples': len(examples),
        'sampled_examples': len(samples), 'import_seconds': IMPORT_SECONDS,
        'load_prepared_including_hashes_and_split_validation_seconds': load_seconds,
        'tokenizer_load_seconds': tokenizer_seconds,
        'rebuild_positions_only_seconds': positions_seconds,
        'sampled_dataset_init_seconds': init_seconds,
        'encode_sampled_examples_seconds': encode_seconds,
        'encoding_examples_per_second': len(samples) / encode_seconds,
        'encoding_ms_per_example': 1000 * encode_seconds / len(samples),
        'estimated_full_pass_encoding_seconds': encode_seconds * len(examples) / len(samples),
        'sampled_batch4_collation_seconds': collate_seconds,
        'estimated_full_pass_collation_seconds': collate_seconds * len(examples) / len(samples),
        'length_min': int(lengths.min()), 'length_max': int(lengths.max()),
        'length_mean': float(lengths.mean()),
        'supervised_token_counts': sorted({sum(label != -100 for label in row['labels']) for row in samples}),
        'limitations': 'Local CPU sampled measurement, not original server timing; excludes embeddings/RQ-VAE, GPU compute, evaluation, checkpoints and network I/O. Parquet test uses warm local filesystem and this subset only.'
    }
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        # Dynamic-length rows; padding remains a batch operation. Never persist a
        # fixed padding width of 512, which would inflate storage and GPU work.
        schema = pa.schema([(name, pa.list_(pa.int32())) for name in
                            ('input_ids', 'attention_mask', 'labels')])
        table, arrow_seconds = timed(lambda: pa.Table.from_pylist(samples, schema=schema))
        path = args.output / 'tokenized_sample.parquet'
        _, write_seconds = timed(lambda: pq.write_table(table, path, compression='zstd'))
        loaded, read_seconds = timed(lambda: pq.read_table(path))
        cached, convert_seconds = timed(loaded.to_pylist)
        if samples != cached:
            raise ValueError('Parquet round trip changed token IDs, masks or labels')
        _, cached_collate_seconds = timed(lambda: [collator(cached[i:i + 4])
                                                  for i in range(0, len(cached), 4)])
        result['parquet'] = dict(version=pa.__version__, round_trip_exact=True,
                                 arrow_table_seconds=arrow_seconds,
                                 write_seconds=write_seconds, read_seconds=read_seconds,
                                 convert_to_python_seconds=convert_seconds,
                                 cached_batch4_collation_seconds=cached_collate_seconds,
                                 sampled_bytes=path.stat().st_size,
                                 estimated_full_train_bytes=path.stat().st_size * len(examples) / len(samples),
                                 estimated_full_read_and_convert_seconds=(read_seconds + convert_seconds) * len(examples) / len(samples))
    except ImportError:
        result['parquet'] = {'status': 'pyarrow_not_installed'}
    write_json(args.output / 'profile.json', result)
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
