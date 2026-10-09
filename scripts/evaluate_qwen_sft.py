"""Explicit frozen-checkpoint item retrieval evaluation, separate from SFT training."""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_sft import load_prepared, evaluate_retrieval, select_subset, write_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', choices=['valid', 'test'], default='valid')
    parser.add_argument('--users', type=int, default=0, help='0 = all users; nonzero is diagnostic only')
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--beam', type=int, default=20)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection-record', type=Path, help='Required for test: frozen selection JSON with checkpoint and data hashes')
    args = parser.parse_args()
    manifest, rows, catalog, _ = load_prepared(args.data)
    if args.output.exists():
        raise ValueError('Evaluation output already exists; preserve the prior record')
    if args.split == 'test':
        from qwen_sft import read_json
        if not args.selection_record:
            raise ValueError('Freeze the validation-selected checkpoint before testing')
        selected = read_json(args.selection_record)
        if selected['checkpoint_weight_sha256'] != sha256(args.checkpoint / 'model.safetensors') or selected['prepared_manifest_sha256'] != sha256(args.data / 'manifest.json'):
            raise ValueError('Selection record does not match checkpoint/data')
        if selected.get('beam') != args.beam or selected.get('max_length') != args.max_length or not selected.get('full_validation_completed'):
            raise ValueError('Selection must include full validation and frozen beam')
    if not torch.cuda.is_available():
        raise RuntimeError('Qwen 1.5B evaluation requires the GPU environment')
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
                                               dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    result = evaluate_retrieval(model, tokenizer, catalog, select_subset(rows, args.users, args.seed),
                                args.split, args.beam, manifest['max_history'], args.max_length)
    result.update(split=args.split, diagnostic_only=args.users > 0,
                  max_length=args.max_length,
                  checkpoint_weight_sha256=sha256(args.checkpoint / 'model.safetensors'),
                  prepared_manifest_sha256=sha256(args.data / 'manifest.json'))
    write_json(args.output, result)
    print(result)


if __name__ == '__main__':
    main()
