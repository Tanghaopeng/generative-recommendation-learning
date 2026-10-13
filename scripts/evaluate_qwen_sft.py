"""Explicit frozen-checkpoint item retrieval evaluation, separate from SFT training."""
import argparse
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_sft import (SID_DECODING_PROTOCOL, load_prepared, load_qwen_tokenizer,
                      evaluate_retrieval, select_subset, write_json, sha256)


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
    q0 = None
    if (args.checkpoint / 'q0_initialization.json').is_file():
        from qwen_initialization import check_initialization
        q0 = check_initialization(args.checkpoint, args.data)
    if args.output.exists():
        raise ValueError('Evaluation output already exists; preserve the prior record')
    if args.split == 'test':
        from qwen_sft import read_json
        if not args.selection_record:
            raise ValueError('Freeze the validation-selected checkpoint before testing')
        selected = read_json(args.selection_record)
        if selected['checkpoint_weight_sha256'] != sha256(args.checkpoint / 'model.safetensors') or selected['prepared_manifest_sha256'] != sha256(args.data / 'manifest.json'):
            raise ValueError('Selection record does not match checkpoint/data')
        validation_completed = selected.get('full_validation_completed') or (
            selected.get('validation_completed') and
            selected.get('validation_scope') == 'fixed_training_validation_subset' and
            selected.get('validation_users', 0) > 0 and
            selected.get('selection_evidence_sha256'))
        if selected.get('beam') != args.beam or selected.get('max_length') != args.max_length or not validation_completed:
            raise ValueError('Selection requires completed validation evidence and frozen beam')
        if selected.get('decoding_protocol') != SID_DECODING_PROTOCOL:
            raise ValueError('Decoding changed; repeat validation and freeze the current protocol before testing')
    if not torch.cuda.is_available():
        raise RuntimeError('Qwen 1.5B evaluation requires the GPU environment')
    tokenizer = load_qwen_tokenizer(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
                                               dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    torch.cuda.synchronize()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    result = evaluate_retrieval(model, tokenizer, catalog, select_subset(rows, args.users, args.seed),
                                args.split, args.beam, manifest['max_history'], args.max_length)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    result.update(split=args.split, diagnostic_only=args.users > 0,
                  model_stage='q0_untrained_sid' if q0 else 'trained_checkpoint',
                  initialization_seed=q0['seed'] if q0 else None,
                  seconds=seconds, users_per_second=result['users'] / seconds,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                  code_sha256={name: sha256(Path(__file__).with_name(name)) for name in
                               ('qwen_sft.py', 'evaluate_qwen_sft.py')},
                  max_length=args.max_length,
                  checkpoint_weight_sha256=sha256(args.checkpoint / 'model.safetensors'),
                  prepared_manifest_sha256=sha256(args.data / 'manifest.json'))
    write_json(args.output, result)
    print(result)


if __name__ == '__main__':
    main()
