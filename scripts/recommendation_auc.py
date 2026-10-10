"""Offline sampled AUC/GAUC; unobserved items are proxy negatives, not dislikes."""
from __future__ import annotations

import hashlib
import random

import numpy as np


def binary_auc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError('Expected equally sized finite one-dimensional labels/scores')
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('AUC requires binary labels')
    positive, negative = scores[labels == 1], np.sort(scores[labels == 0])
    if not len(positive) or not len(negative):
        raise ValueError('AUC requires both positive and negative examples')
    left = np.searchsorted(negative, positive, side='left')
    right = np.searchsorted(negative, positive, side='right')
    return float((left + 0.5 * (right - left)).sum() / (len(positive) * len(negative)))


def grouped_auc(groups):
    """Pooled AUC and GAUC weighted by each valid group's candidate count."""
    labels, scores, aucs, weights = [], [], [], []
    skipped = 0
    for group in groups:
        y, s = group['labels'], group['scores']
        # Reject malformed scores before considering single-class groups.
        if len(y) != len(s) or not len(y) or not np.isfinite(s).all() or not np.isin(y, [0, 1]).all():
            raise ValueError('Malformed AUC group')
        labels.extend(y)
        scores.extend(s)
        if len(set(y)) < 2:
            skipped += 1
            continue
        aucs.append(binary_auc(y, s))
        weights.append(len(y))
    if not aucs:
        raise ValueError('No user group with both classes')
    return {'AUC': binary_auc(labels, scores),
            'GAUC': float(np.average(aucs, weights=weights)),
            'macro_user_AUC': float(np.mean(aucs)),
            'valid_groups': len(aucs), 'skipped_single_class_groups': skipped,
            'gauc_weight': 'number of candidate examples per user'}


def candidate_ids(row, item_count, split='valid', negatives=100, seed=2026):
    if split not in ('valid', 'test') or negatives < 1:
        raise ValueError('Expected valid/test split and at least one negative')
    history = row['train'] + (row['valid'] if split == 'test' else [])
    target = int(row[split][0])
    if target in history or not 1 <= target <= item_count:
        raise ValueError('Invalid or observed held-out target')
    blocked = set(history) | {target}
    if negatives > item_count - len(blocked):
        raise ValueError('Not enough distinct eligible negatives')
    material = f'{seed}:{split}:{row["user_id"]}'.encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], 'big'))
    sampled = []
    used = set(blocked)
    while len(sampled) < negatives:
        item = rng.randint(1, item_count)
        if item not in used:
            sampled.append(item)
            used.add(item)
    return [target] + sampled
