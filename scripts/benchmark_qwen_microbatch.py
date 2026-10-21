"""Disposable full-parameter SFT capacity benchmark; never saves model weights.

Compare micro-batches 2/4/8/16 at effective batch 128 on identical examples.
This is not a GRPO memory benchmark, which also needs rollout/reference buffers.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, set_seed
from qwen_sft import CompletionCollator, SFTDataset, load_prepared, load_qwen_tokenizer, select_subset, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--batch', type=int, choices=[2,4,8,16], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing benchmark record')
    set_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = True
    manifest, rows, catalog, examples = load_prepared(args.data)
    tokenizer = load_qwen_tokenizer(args.checkpoint)
    subset = select_subset(examples, 128, 2026)
    dataset = SFTDataset(rows, catalog, subset, tokenizer, 'q1', 512, manifest['max_history'])
    samples = [dataset[i] for i in range(128)]
    collator = CompletionCollator(tokenizer.pad_token_id)
    batches = [collator(samples[i:i+args.batch]) for i in range(0,128,args.batch)]
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
                     dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    model.requires_grad_(True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(.9,.999), eps=1e-8, weight_decay=0.0)
    accum = 128 // args.batch
    steps = []
    try:
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            losses = []
            for cpu_batch in batches:
                batch = {k:v.to('cuda') for k,v in cpu_batch.items()}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss')
                (loss / accum).backward()
                losses.append(float(loss.detach()))
                del loss, batch
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite gradient')
            optimizer.step()
            torch.cuda.synchronize()
            free,total = torch.cuda.mem_get_info()
            record = {'step':step+1, 'warmup':step==0,
                      'seconds':time.perf_counter()-start,
                      'loss':sum(losses)/len(losses), 'grad_norm':float(norm),
                      'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                      'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
                      'driver_free_gib_after_step':free/2**30}
            steps.append(record)
            print(json.dumps(record),flush=True)
        # Stress the longest sampled sequence after optimizer states exist.
        optimizer.zero_grad(set_to_none=True)
        longest = max(samples,key=lambda s:len(s['input_ids']))
        stress = {k:v.to('cuda') for k,v in collator([longest]*args.batch).items()}
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            stress_loss = model(**stress).loss
        stress_loss.backward()
        torch.cuda.synchronize()
        stress_peak = torch.cuda.max_memory_allocated()/2**30
        result = {'status':'passed', 'benchmark':'disposable_sft_not_grpo', 'batch':args.batch,
                  'gradient_accumulation_steps':accum,'effective_batch_size':128,
                  'precision':'bf16','optimizer':'adamw_torch','gradient_checkpointing':True,
                  'samples':128,'sample_seed':2026,'sequence_min':min(len(s['input_ids']) for s in samples),
                  'sequence_max':len(longest['input_ids']),
                  'padded_batch_width_min':min(b['input_ids'].shape[1] for b in batches),
                  'padded_batch_width_max':max(b['input_ids'].shape[1] for b in batches),
                  'warmup_optimizer_steps':1,'timed_optimizer_steps':2,'steps':steps,
                  'seconds_per_effective_step':sum(s['seconds'] for s in steps[1:])/2,
                  'samples_per_second':256/sum(s['seconds'] for s in steps[1:]),
                  'longest_sample_backward_peak_gib':stress_peak,
                  'weights_saved':False,'formal_checkpoint_modified':False,
                  'limitations':'Short SFT benchmark on actual sampled lengths, not 512-token stress or GRPO; timing is approximate.'}
    except torch.OutOfMemoryError as exc:
        result={'status':'cuda_out_of_memory','batch':args.batch,'steps':steps,'error':str(exc)}
    write_json(args.output,result)
    print('BENCHMARK_RESULT '+json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
