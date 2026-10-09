"""Office Products four-token SID SFT adapter.

Training recipe reference: wbn11/minionerec-single-gpu @ bccd7ef.
This file is a local implementation; preserved upstream sources are reference only.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import LogitsProcessor, Trainer

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
MODEL_SHA256 = 'dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def positions(rows, max_history=20):
    """Match existing SASRec/TIGER positive positions, including the tail start."""
    result = []
    for user, row in enumerate(rows):
        start = max(0, len(row['train']) - 1 - max_history)
        result.extend((user, start, target) for target in range(start + 1, len(row['train'])))
    return np.asarray(result, dtype=np.int64).reshape(-1, 3)


def validate_rows(rows, item_count):
    for row in rows:
        train, valid, test = row['train'], row['valid'], row['test']
        if not train or len(valid) != 1 or len(test) != 1:
            raise ValueError('Expected nonempty train and one valid/test target')
        full = train + valid + test
        if row.get('items', full) != full or len(set(full)) != len(full):
            raise ValueError('Invalid split or repeated item in deduplicated sequence')
        if min(full) < 1 or max(full) > item_count:
            raise ValueError('Item outside the frozen catalog')


def sid_text(code):
    return ''.join(f'<sid_{level}_{int(value)}>' for level, value in enumerate(code))


def sid_tokens(width, depth=4):
    return [f'<sid_{level}_{value}>' for level in range(depth) for value in range(width)]


def extend_tokenizer(tokenizer, width):
    original = len(tokenizer)
    tokens = sid_tokens(width)
    added = tokenizer.add_tokens(tokens, special_tokens=False)
    ids = [tokenizer.encode(token, add_special_tokens=False) for token in tokens]
    if any(len(value) != 1 for value in ids) or len({value[0] for value in ids}) != len(tokens):
        raise ValueError('SID tokens must be unique atomic ordinary tokens')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    return {'original_tokenizer_size': original, 'added_tokens': added,
            'tokenizer_size': len(tokenizer), 'sid_token_count': len(tokens)}


def clean_title(value):
    # Prevent metadata strings from injecting chat/SID control tokens.
    return ' '.join(str(value or '').replace('<', '＜').replace('>', '＞').split())[:1000]


def prompt_ids(tokenizer, task, history, target, catalog, max_history=20):
    if task in ('next_sid', 'next_title'):
        history = history[-max_history:]
        body = '\n'.join(sid_text(catalog[str(item)]['sid']) for item in history)
        instruction = ('Predict the next product. Return only its four SID tokens.' if task == 'next_sid'
                       else 'Predict the title of the next product. Return only its title.')
        content = instruction + '\nHistory (oldest to newest):\n' + body
    elif task == 'title_to_sid':
        title_ids = tokenizer.encode(catalog[str(target)]['title'], add_special_tokens=False)[:128]
        title = tokenizer.decode(title_ids, skip_special_tokens=False)
        content = 'Identify this product. Return only its four SID tokens.\nTitle: ' + title
    elif task == 'sid_to_title':
        content = 'Return only the title of this product.\nSID: ' + sid_text(catalog[str(target)]['sid'])
    else:
        raise ValueError(task)
    messages = [{'role': 'system', 'content': 'You are a product recommendation assistant.'},
                {'role': 'user', 'content': content}]
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)


def encode_example(tokenizer, catalog, history, target, task='next_sid', max_length=512, max_history=20):
    history = list(history[-max_history:])
    answer = (sid_text(catalog[str(target)]['sid']) if task in ('next_sid', 'title_to_sid')
              else catalog[str(target)]['title'])
    completion = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
    # Auxiliary titles may be long. Keep EOS and reserve at least 128 prompt tokens.
    if task in ('next_title', 'sid_to_title'):
        completion = completion[:min(64, max_length - 128) - 1] + [tokenizer.eos_token_id]
    prompt = prompt_ids(tokenizer, task, history, target, catalog, max_history)
    while len(prompt) + len(completion) > max_length and history:
        history.pop(0)  # Never truncate part of a product's SID.
        prompt = prompt_ids(tokenizer, task, history, target, catalog, max_history)
    if len(prompt) + len(completion) > max_length:
        raise ValueError('Prompt exceeds max_length; shorten catalog titles or increase max_length')
    ids = prompt + completion
    return {'input_ids': ids, 'attention_mask': [1] * len(ids),
            'labels': [-100] * len(prompt) + completion}


class SFTDataset(Dataset):
    def __init__(self, rows, catalog, examples, tokenizer, recipe='q1', max_length=512, max_history=20):
        if recipe not in ('q1', 'q2'):
            raise ValueError('recipe must be q1 or q2')
        self.rows, self.catalog, self.examples, self.tokenizer = rows, catalog, examples, tokenizer
        self.recipe, self.max_length, self.max_history = recipe, max_length, max_history
        # Full-catalog textual alignment uses metadata only, never held-out interactions.
        self.title_items = [int(item) for item, value in catalog.items() if value['title']]
        self.fusion = np.asarray([i for i, (u, _, t) in enumerate(examples)
                                  if catalog[str(rows[u]['train'][t])]['title']], dtype=np.int64)

    def __len__(self):
        return len(self.examples) + (2 * len(self.title_items) + len(self.fusion) if self.recipe == 'q2' else 0)

    def __getitem__(self, index):
        n = len(self.examples)
        if index < n:
            u, start, target = self.examples[index]
            history, target = self.rows[u]['train'][start:target], self.rows[u]['train'][target]
            task = 'next_sid'
        elif index < n + 2 * len(self.title_items):
            offset = index - n
            target = self.title_items[offset // 2]
            history, task = [], ('title_to_sid' if offset % 2 == 0 else 'sid_to_title')
        else:
            u, start, target = self.examples[self.fusion[index - n - 2 * len(self.title_items)]]
            history, target = self.rows[u]['train'][start:target], self.rows[u]['train'][target]
            task = 'next_title'
        return encode_example(self.tokenizer, self.catalog, history, target, task, self.max_length, self.max_history)


class ValidationDataset(Dataset):
    def __init__(self, rows, catalog, tokenizer, max_length=512, max_history=20):
        self.rows, self.catalog, self.tokenizer = rows, catalog, tokenizer
        self.max_length, self.max_history = max_length, max_history

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return encode_example(self.tokenizer, self.catalog, row['train'], row['valid'][0],
                              max_length=self.max_length, max_history=self.max_history)


class CompletionCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, examples):
        width = (max(len(row['input_ids']) for row in examples) + 7) // 8 * 8
        batch = {}
        for key, pad in [('input_ids', self.pad_id), ('attention_mask', 0), ('labels', -100)]:
            batch[key] = torch.tensor([row[key] + [pad] * (width - len(row[key])) for row in examples])
        return batch


class RankingTrainer(Trainer):
    """Attach item metrics before Trainer logs, saves, or invokes early stopping."""
    def __init__(self, *args, ranking_evaluator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ranking_evaluator = ranking_evaluator

    def evaluation_loop(self, *args, metric_key_prefix='eval', **kwargs):
        output = super().evaluation_loop(*args, metric_key_prefix=metric_key_prefix, **kwargs)
        if self.ranking_evaluator is not None:
            ranking = self.ranking_evaluator(self.model)
            output.metrics[metric_key_prefix + '_ndcg10'] = ranking['NDCG@10']
            output.metrics[metric_key_prefix + '_recall10'] = ranking['Recall@10']
            output.metrics[metric_key_prefix + '_shortfalls'] = ranking['shortfall_users']
        return output


def load_prepared(directory):
    directory = Path(directory)
    manifest = read_json(directory / 'manifest.json')
    if manifest['status'] != 'completed':
        raise ValueError('Incomplete prepared data')
    for filename, fingerprint in manifest['files'].items():
        if sha256(directory / filename) != fingerprint:
            raise ValueError(f'Prepared artifact hash mismatch: {filename}')
    rows, catalog = read_rows(directory / 'histories.jsonl'), read_json(directory / 'catalog.json')
    validate_rows(rows, len(catalog))
    examples = np.load(directory / 'train_examples.npy', allow_pickle=False)
    return manifest, rows, catalog, examples


def select_subset(values, count, seed):
    if count <= 0 or count >= len(values):
        return values
    selected = sorted(random.Random(seed).sample(range(len(values)), count))
    return [values[i] for i in selected] if isinstance(values, list) else values[selected]


class CatalogConstraint(LogitsProcessor):
    """Mask after beam-search log_softmax; keep full-vocabulary probabilities."""
    def __init__(self, tokenizer, catalog, prompt_length, excluded):
        self.prompt_length = prompt_length
        self.children, self.leaves = {}, {}
        self.excluded = set(excluded)
        for item, value in catalog.items():
            code = tuple(tokenizer.encode(sid_text(value['sid']), add_special_tokens=False))
            if len(code) != 4 or code in self.leaves:
                raise ValueError('Non-atomic or colliding full SID')
            self.leaves[code] = int(item)
            for depth in range(4):
                self.children.setdefault(code[:depth], set()).add(code[depth])

    def __call__(self, input_ids, scores):
        masked = torch.full_like(scores, -float('inf'))
        for row, ids in enumerate(input_ids):
            prefix = tuple(ids[self.prompt_length:].tolist())
            allowed = self.children.get(prefix, set())
            if len(prefix) == 3:
                allowed = [token for token in allowed if self.leaves[prefix + (token,)] not in self.excluded]
            if allowed:
                allowed = list(allowed)
                masked[row, allowed] = scores[row, allowed]
        return masked


@torch.inference_mode()
def retrieve(model, tokenizer, catalog, history, beam=20, max_history=20, max_length=512, constraint=None):
    if beam < 20:
        raise ValueError('beam must be >= 20 for Recall@20')
    visible = history[-max_history:]
    prompt = prompt_ids(tokenizer, 'next_sid', visible, 0, catalog, max_history)
    while len(prompt) + 5 > max_length and visible:
        visible = visible[1:]
        prompt = prompt_ids(tokenizer, 'next_sid', visible, 0, catalog, max_history)
    if len(prompt) + 5 > max_length:
        raise ValueError('Generation prompt too long')
    if constraint is None:
        constraint = CatalogConstraint(tokenizer, catalog, len(prompt), history)
    constraint.prompt_length, constraint.excluded = len(prompt), set(history)
    ids = torch.tensor([prompt], device=model.device)
    result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                            max_new_tokens=4, min_new_tokens=0, num_beams=beam,
                            num_return_sequences=beam, do_sample=False, length_penalty=0.0,
                            eos_token_id=None, forced_eos_token_id=None, forced_bos_token_id=None,
                            pad_token_id=tokenizer.pad_token_id, use_cache=True,
                            renormalize_logits=False, logits_processor=[constraint],
                            return_dict_in_generate=True, output_scores=True)
    candidates = {}
    for sequence, score in zip(result.sequences, result.sequences_scores):
        code = tuple(sequence[len(prompt):].tolist())
        item = constraint.leaves.get(code)
        if item and item not in constraint.excluded and math.isfinite(float(score)):
            candidates[item] = max(candidates.get(item, -math.inf), float(score))
    return [item for item, _ in sorted(candidates.items(), key=lambda pair: (-pair[1], pair[0]))[:20]]


def evaluate_retrieval(model, tokenizer, catalog, rows, split='valid', beam=20, max_history=20, max_length=512):
    if not rows or split not in ('valid', 'test'):
        raise ValueError('Nonempty valid/test rows required')
    was_training, use_cache = model.training, model.config.use_cache
    model.eval()
    totals = {f'{name}@{k}': 0.0 for k in (10, 20) for name in ('Recall', 'NDCG')}
    shortfalls = 0
    constraint = CatalogConstraint(tokenizer, catalog, 0, [])
    try:
        for index, row in enumerate(rows):
            history = row['train'] + (row['valid'] if split == 'test' else [])
            target = row[split][0]
            if target in history:
                raise ValueError('Held-out target overlaps observed history')
            predicted = retrieve(model, tokenizer, catalog, history, beam, max_history, max_length, constraint)
            shortfalls += len(predicted) < 20
            if target in predicted:
                rank = predicted.index(target) + 1
                for k in (10, 20):
                    if rank <= k:
                        totals[f'Recall@{k}'] += 1
                        totals[f'NDCG@{k}'] += 1 / math.log2(rank + 1)
            if (index + 1) % 100 == 0:
                print(f'{split} retrieval: {index+1}/{len(rows)}', flush=True)
    finally:
        model.config.use_cache = use_cache
        model.train(was_training)
    return {**{key: value / len(rows) for key, value in totals.items()},
            'users': len(rows), 'beam': beam, 'shortfall_users': shortfalls,
            'search': 'full catalog, finite beam approximation; complete history excluded'}
