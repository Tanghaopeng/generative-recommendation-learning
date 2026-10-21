"""Evaluation prototypes: bulk constraint I/O and exact-length request batches.

The original qwen_sft implementation remains the reference. Full-vocabulary
log-softmax and SID beam-search settings are unchanged. Batched BF16 GEMMs may
change near-tied rankings; audit complete candidate lists before adoption.
"""
from collections import defaultdict
import math

import torch
from transformers import GenerationConfig, LogitsProcessor

from qwen_sft import CatalogConstraint, prompt_ids, SID_DECODING_PROTOCOL


class BulkCatalogConstraint(LogitsProcessor):
    def __init__(self, index, prompt_width, histories, beam):
        self.index, self.prompt_width, self.beam = index, prompt_width, beam
        self.excluded = [set(history) for history in histories]

    def __call__(self, input_ids, scores):
        if len(input_ids) != len(self.excluded) * self.beam:
            raise ValueError('Beam-to-user mapping mismatch')
        # One device-to-host transfer for all prefixes, rather than one per beam.
        prefixes = input_ids[:, self.prompt_width:].tolist()
        row_ids, token_ids = [], []
        for row, prefix in enumerate(prefixes):
            prefix = tuple(prefix)
            allowed = self.index.children.get(prefix, ())
            if len(prefix) == 3:
                excluded = self.excluded[row // self.beam]
                allowed = [token for token in allowed
                           if self.index.leaves[prefix + (token,)] not in excluded]
            for token in allowed:
                row_ids.append(row)
                token_ids.append(token)
        masked = torch.full_like(scores, -float('inf'))
        if row_ids:
            rows = torch.tensor(row_ids, dtype=torch.long, device=scores.device)
            tokens = torch.tensor(token_ids, dtype=torch.long, device=scores.device)
            masked[rows, tokens] = scores[rows, tokens]
        return masked


def prepare_request(tokenizer, catalog, history, max_history=20, max_length=512):
    visible = list(history[-max_history:])
    prompt = prompt_ids(tokenizer, 'next_sid', visible, 0, catalog, max_history)
    while len(prompt) + 5 > max_length and visible:
        visible = visible[1:]
        prompt = prompt_ids(tokenizer, 'next_sid', visible, 0, catalog, max_history)
    if len(prompt) + 5 > max_length:
        raise ValueError('Generation prompt too long')
    return dict(prompt=prompt, history=list(history))


@torch.inference_mode()
def retrieve_group(model, tokenizer, requests, index, beam=20):
    if not requests or beam < 20:
        raise ValueError('Nonempty requests and beam >= 20 required')
    widths = {len(request['prompt']) for request in requests}
    if len(widths) != 1:
        raise ValueError('Batch only equal-length prompts; no padding or position change')
    width = widths.pop()
    constraint = BulkCatalogConstraint(index, width, [r['history'] for r in requests], beam)
    ids = torch.tensor([r['prompt'] for r in requests], device=model.device)
    result = model.generate(
        input_ids=ids, attention_mask=torch.ones_like(ids),
        generation_config=GenerationConfig(), use_model_defaults=False,
        max_new_tokens=4, min_new_tokens=0, num_beams=beam,
        num_return_sequences=beam, do_sample=False, length_penalty=0.0,
        eos_token_id=None, forced_eos_token_id=None, forced_bos_token_id=None,
        pad_token_id=tokenizer.pad_token_id, use_cache=True,
        renormalize_logits=False, logits_processor=[constraint],
        return_dict_in_generate=True, output_scores=True)
    # Two bulk transfers instead of reading each sequence and scalar separately.
    codes = result.sequences[:, width:].tolist()
    scores = result.sequences_scores.tolist()
    predictions = []
    for user in range(len(requests)):
        candidates = {}
        for row in range(user * beam, (user + 1) * beam):
            item = index.leaves.get(tuple(codes[row]))
            score = scores[row]
            if item and item not in constraint.excluded[user] and math.isfinite(score):
                candidates[item] = max(candidates.get(item, -math.inf), score)
        predictions.append([item for item, _ in sorted(
            candidates.items(), key=lambda pair: (-pair[1], pair[0]))[:20]])
    return predictions


def predict_histories(model, tokenizer, catalog, histories, batch_size=1,
                      beam=20, max_history=20, max_length=512, index=None, on_batch=None):
    if batch_size < 1:
        raise ValueError('Positive evaluation batch required')
    index = index if index is not None else CatalogConstraint(tokenizer, catalog, 0, [])
    requests = [prepare_request(tokenizer, catalog, history, max_history, max_length)
                for history in histories]
    if batch_size == 1:
        predictions = []
        for request in requests:
            predictions.append(retrieve_group(model, tokenizer, [request], index, beam)[0])
            if on_batch:
                on_batch(len(predictions), len(requests))
        return predictions
    buckets = defaultdict(list)
    for user, request in enumerate(requests):
        buckets[len(request['prompt'])].append((user, request))
    predictions = [None] * len(requests)
    completed = 0
    for entries in buckets.values():
        for start in range(0, len(entries), batch_size):
            chunk = entries[start:start + batch_size]
            values = retrieve_group(model, tokenizer, [r for _, r in chunk], index, beam)
            for (user, _), value in zip(chunk, values):
                predictions[user] = value
            completed += len(chunk)
            if on_batch:
                on_batch(completed, len(requests))
    return predictions


def evaluate_retrieval_fast(model, tokenizer, catalog, rows, split='valid', beam=20,
                            max_history=20, max_length=512, batch_size=8):
    if not rows or split not in ('valid', 'test'):
        raise ValueError('Nonempty valid/test rows required')
    histories = [row['train'] + (row['valid'] if split == 'test' else []) for row in rows]
    targets = [row[split][0] for row in rows]
    if any(target in history for target, history in zip(targets, histories)):
        raise ValueError('Held-out target overlaps observed history')
    was_training, use_cache = model.training, model.config.use_cache
    model.eval()
    last_report = 0

    def progress(done, total):
        nonlocal last_report
        if done - last_report >= 100 or done == total:
            print(f'{split} batched retrieval: {done}/{total} (batch={batch_size})', flush=True)
            last_report = done

    try:
        predictions = predict_histories(model, tokenizer, catalog, histories, batch_size,
                                        beam, max_history, max_length, on_batch=progress)
    finally:
        model.config.use_cache = use_cache
        model.train(was_training)
    totals = {f'{name}@{k}': 0.0 for k in (10, 20) for name in ('Recall', 'NDCG')}
    for target, predicted in zip(targets, predictions):
        if target in predicted:
            rank = predicted.index(target) + 1
            for k in (10, 20):
                if rank <= k:
                    totals[f'Recall@{k}'] += 1
                    totals[f'NDCG@{k}'] += 1 / math.log2(rank + 1)
    return {**{key: value / len(rows) for key, value in totals.items()},
            'users': len(rows), 'beam': beam,
            'shortfall_users': sum(len(items) < 20 for items in predictions),
            'decoding_protocol': SID_DECODING_PROTOCOL,
            'search': 'full catalog, finite beam approximation; complete history excluded',
            'evaluation_execution': dict(backend='equal_length_bucket_v1', batch_size=batch_size,
                precision='bf16' if next(model.parameters()).dtype == torch.bfloat16 else 'fp32')}
