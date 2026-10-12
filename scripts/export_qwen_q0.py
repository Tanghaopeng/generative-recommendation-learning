"""Export original Qwen plus deterministic untrained SID rows; no optimizer exists."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from qwen_initialization import initialize_sid_rows
from qwen_sft import (MODEL_REVISION, MODEL_SHA256, load_prepared, sha256,
                      sid_text, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new Q0 directory; preserve the existing initialization')
    if args.seed < 0:
        raise ValueError('Seed must be nonnegative')
    manifest, _, catalog, _ = load_prepared(args.data)
    if sha256(args.model / 'model.safetensors') != MODEL_SHA256:
        raise ValueError('Original model weight differs from the pinned official Qwen')
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    # Export is a CPU operation. No GPU allocation, gradient, or optimizer step.
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                                               trust_remote_code=False, dtype=torch.bfloat16,
                                               attn_implementation='sdpa')
    info = initialize_sid_rows(model, tokenizer, manifest['tokens_per_namespace'], args.seed)
    for item in catalog.values():
        if len(tokenizer.encode(sid_text(item['sid']), add_special_tokens=False)) != 4:
            raise ValueError('Catalog SID does not encode to exactly four tokens')
    model.eval()
    model.requires_grad_(False)
    args.output.mkdir(parents=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size='5GB')
    tokenizer.save_pretrained(args.output)
    if (args.model / 'LICENSE').is_file():
        shutil.copyfile(args.model / 'LICENSE', args.output / 'LICENSE')
    record = dict(**info, status='q0_export_completed', optimizer_steps=0,
                  model='Qwen/Qwen2.5-1.5B-Instruct', model_revision=MODEL_REVISION,
                  source_model_sha256=MODEL_SHA256,
                  prepared_manifest_sha256=sha256(args.data / 'manifest.json'),
                  checkpoint_weight_sha256=sha256(args.output / 'model.safetensors'),
                  source_model_metadata_sha256={name: sha256(args.model / name) for name in
                                               ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt')},
                  files_sha256={p.name: sha256(p) for p in args.output.iterdir() if p.is_file()},
                  versions={name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'tokenizers')},
                  code_sha256={name: sha256(Path(__file__).with_name(name)) for name in
                               ('qwen_initialization.py', 'qwen_sft.py', 'export_qwen_q0.py')},
                  timestamp_utc=datetime.now(timezone.utc).isoformat())
    # Written last; its presence marks a completed, checksummed export.
    write_json(args.output / 'q0_initialization.json', record)
    print({'status': record['status'], 'output': str(args.output), 'optimizer_steps': 0,
           'vocabulary': info['vocabulary'], 'sid_rows_initialized': info['sid_rows_initialized'],
           'reused_padding_rows': info['reused_padding_rows'],
           'checkpoint_weight_sha256': record['checkpoint_weight_sha256']}, flush=True)


if __name__ == '__main__':
    main()
