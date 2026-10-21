"""HPR beam search: score expanded partial paths BEFORE pruning, with a KV cache."""
from __future__ import annotations

from collections import defaultdict
import math

import torch

from qwen_eval_fast import prepare_request, retrieve_group
from qwen_sft import CatalogConstraint


def select_expansion(lm_scores, path_scores, beam, pool_size, strength):
    """Keep top-N by cumulative LM score, then top-beam by LM + alpha*logsoftmax(HPR).

    LM and fused scores stay separate: previous HPR bonuses are NOT repeatedly
    added to future LM sums. A pool-excluded path cannot be rescued by HPR.
    """
    if beam < 1 or pool_size < beam or not math.isfinite(strength) or strength < 0:
        raise ValueError('Requires pool >= beam >= 1 and finite nonnegative HPR strength')
    if lm_scores.shape != path_scores.shape or lm_scores.ndim != 1:
        raise ValueError('One score pair per expanded path required')
    valid = torch.nonzero(torch.isfinite(lm_scores), as_tuple=False).flatten()
    if not len(valid):
        return valid, lm_scores[valid]
    pool = valid[torch.argsort(lm_scores[valid], descending=True, stable=True)[:pool_size]]
    if not torch.isfinite(path_scores[pool]).all():
        raise ValueError('Non-finite HPR path score')
    fused = lm_scores[pool] + strength * torch.log_softmax(path_scores[pool].float(), dim=0)
    chosen = torch.argsort(fused, descending=True, stable=True)[:beam]
    return pool[chosen], fused[chosen]


@torch.inference_mode()
def retrieve_hpr_group(model, tokenizer, requests, index, scorer, beam=20, pool_size=400, strength=0.25,
                       return_details=False):
    if not requests or beam < 20 or pool_size < beam or not math.isfinite(strength) or strength < 0:
        raise ValueError('Nonempty requests, beam>=20, pool>=beam and nonnegative strength required')
    # A true off-switch: reuse the existing SFT retrieval implementation bit-for-bit.
    if strength == 0:
        baseline = retrieve_group(model, tokenizer, requests, index, beam)
        if not return_details:
            return baseline
        item_codes = {item: list(code) for code, item in index.leaves.items()}
        return [[dict(item_id=item, sid_token_ids=item_codes[item], lm_log_probability=None,
                      fused_score=None, score_note='zero-strength delegates unchanged SFT generation')
                 for item in items] for items in baseline]
    widths = {len(r['prompt']) for r in requests}
    if len(widths) != 1:
        raise ValueError('Batch only equal-length prompts')
    if scorer.config.hidden_size != model.config.hidden_size:
        raise ValueError('HPR and Qwen hidden sizes differ')
    device, users = model.device, len(requests)
    ids = torch.tensor([r['prompt'] for r in requests], device=device, dtype=torch.long)
    width = ids.shape[1]
    initial = model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, return_dict=True)
    contexts = initial.last_hidden_state[:, -1].float()
    hidden, cache = initial.last_hidden_state[:, -1], initial.past_key_values
    excluded = [set(r['history']) for r in requests]
    prefix = torch.empty((users, 0), dtype=torch.long, device=device)
    path_sums = torch.zeros((users, scorer.config.hidden_size), device=device)
    lm = torch.zeros(users, device=device)
    fused = lm.clone()
    current_beam = 1
    for depth in range(1, 5):
        # Full-vocabulary log probabilities retain the original SFT scoring definition.
        log_probs = model.get_output_embeddings()(hidden).float().log_softmax(-1)
        prefixes = prefix.tolist()
        active = torch.isfinite(lm).tolist()
        selected_parents, selected_tokens, selected_lm, selected_paths, selected_fused = [], [], [], [], []
        for user in range(users):
            parents, tokens = [], []
            for parent in range(user * current_beam, (user + 1) * current_beam):
                if not active[parent]:
                    continue
                base = tuple(prefixes[parent])
                for token in sorted(index.children.get(base, ())):
                    if depth == 4 and index.leaves[base + (token,)] in excluded[user]:
                        continue
                    parents.append(parent)
                    tokens.append(token)
            if parents:
                parent_ids = torch.tensor(parents, device=device, dtype=torch.long)
                token_ids = torch.tensor(tokens, device=device, dtype=torch.long)
                candidates_lm = lm[parent_ids] + log_probs[parent_ids, token_ids]
                # Only evaluate HPR for the top-N LM pool, not every catalog expansion.
                pool = torch.argsort(candidates_lm, descending=True, stable=True)[:pool_size]
                parent_ids, token_ids, candidates_lm = parent_ids[pool], token_ids[pool], candidates_lm[pool]
                candidates_paths = path_sums[parent_ids]
                if depth <= 3:
                    candidates_paths = candidates_paths + model.get_input_embeddings()(token_ids).float()
                layer = min(depth, 3) - 1
                path_scores = scorer.paired_scores(contexts[user:user + 1], candidates_paths, layer)
                chosen, chosen_fused = select_expansion(candidates_lm, path_scores, beam, pool_size, strength)
                selected_parents.extend(parent_ids[chosen].tolist())
                selected_tokens.extend(token_ids[chosen].tolist())
                selected_lm.extend(candidates_lm[chosen].unbind())
                selected_paths.extend(candidates_paths[chosen].unbind())
                selected_fused.extend(chosen_fused.unbind())
                count = len(chosen)
            else:
                count = 0
            # Fixed beam slots support users with fewer legal paths or no unseen items.
            for _ in range(beam - count):
                selected_parents.append(user * current_beam)
                selected_tokens.append(tokenizer.pad_token_id)
                selected_lm.append(torch.tensor(-torch.inf, device=device))
                selected_paths.append(torch.zeros_like(contexts[user]))
                selected_fused.append(torch.tensor(-torch.inf, device=device))
        parent_ids = torch.tensor(selected_parents, device=device, dtype=torch.long)
        token_ids = torch.tensor(selected_tokens, device=device, dtype=torch.long)
        prefix = torch.cat([prefix[parent_ids], token_ids[:, None]], dim=1)
        path_sums, lm, fused = torch.stack(selected_paths), torch.stack(selected_lm), torch.stack(selected_fused)
        current_beam = beam
        if depth < 4:
            cache.reorder_cache(parent_ids)
            step = model.model(input_ids=token_ids[:, None],
                attention_mask=torch.ones((users * beam, width + depth), device=device, dtype=torch.long),
                past_key_values=cache, use_cache=True, return_dict=True)
            hidden, cache = step.last_hidden_state[:, -1], step.past_key_values
    codes, raw_scores, final_scores = prefix.tolist(), lm.tolist(), fused.tolist()
    results = []
    for user in range(users):
        candidates = {}
        for row in range(user * beam, (user + 1) * beam):
            code = tuple(codes[row])
            item = index.leaves.get(code)
            if item is not None and item not in excluded[user] and math.isfinite(final_scores[row]):
                entry = dict(item_id=item, sid_token_ids=list(code), lm_log_probability=raw_scores[row],
                             fused_score=final_scores[row])
                if item not in candidates or entry['fused_score'] > candidates[item]['fused_score']:
                    candidates[item] = entry
        ranked = sorted(candidates.values(), key=lambda x: (-x['fused_score'], x['item_id']))[:20]
        results.append(ranked if return_details else [x['item_id'] for x in ranked])
    return results


def predict_hpr(model, tokenizer, catalog, histories, scorer, batch_size=8, beam=20,
                pool_size=400, strength=0.25, max_history=20, max_length=512, return_details=False,
                detail_sink=None):
    if batch_size < 1:
        raise ValueError('Positive evaluation batch required')
    index = CatalogConstraint(tokenizer, catalog, 0, [])
    buckets = defaultdict(list)
    for user, history in enumerate(histories):
        request = prepare_request(tokenizer, catalog, history, max_history, max_length)
        buckets[len(request['prompt'])].append((user, request))
    result, done = [None] * len(histories), 0
    model_was_training, head_was_training = model.training, scorer.training
    model.eval()
    scorer.eval()
    try:
        for entries in buckets.values():
            for start in range(0, len(entries), batch_size):
                chunk = entries[start:start + batch_size]
                values = retrieve_hpr_group(model, tokenizer, [r for _, r in chunk], index, scorer,
                                             beam, pool_size, strength, return_details or detail_sink is not None)
                for (user, _), value in zip(chunk, values):
                    if detail_sink is not None:
                        detail_sink(user, value)
                    result[user] = value if return_details or detail_sink is None else [x['item_id'] for x in value]
                done += len(chunk)
                if done % (batch_size * 20) == 0 or done == len(histories):
                    print(f'HPR retrieval: {done}/{len(histories)}', flush=True)
    finally:
        model.train(model_was_training)
        scorer.train(head_was_training)
    return result
