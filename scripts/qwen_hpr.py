"""Frozen-Qwen HPR heads, duplicate-aware contrastive loss and prefix negatives.

This is a local BARGE-inspired adaptation, not the authors' implementation.
Only the first three SID namespaces are semantic; the fourth disambiguates items.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from qwen_sft import read_json, write_json, sha256

HPR_PROTOCOL = 'qwen_frozen_hpr_v1: semantic prefixes 1-3; fourth uses third-prefix score'


def checkpoint_metadata(directory):
    directory = Path(directory)
    return {name: sha256(directory / name) for name in
            ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')}


def verify_sft_identity(record, weight_hash, data_hash):
    """An extended tokenizer alone cannot prove which SID mapping trained the SFT."""
    expected_weight = record.get('checkpoint_weight_sha256', record.get('weight_sha256'))
    expected_data = record.get('prepared_manifest_sha256', record.get('data_manifest_sha256'))
    if expected_weight != weight_hash or expected_data != data_hash:
        raise ValueError('SFT source record does not match checkpoint weights and prepared SID data')


@dataclass
class HPRConfig:
    hidden_size: int
    projection_dim: int = 64
    semantic_depth: int = 3
    initial_temperature: float = 0.07


class HPRScorer(nn.Module):
    """Three independent dual towers. Inputs are frozen SFT representations."""
    def __init__(self, config: HPRConfig):
        super().__init__()
        if (config.hidden_size < 1 or config.projection_dim < 1
                or config.semantic_depth != 3 or config.initial_temperature <= 0):
            raise ValueError('Requires positive dimensions/temperature and three semantic layers')
        self.config = config
        self.context_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.projection_dim, bias=False) for _ in range(3)])
        self.path_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.projection_dim, bias=False) for _ in range(3)])
        self.log_scale = nn.Parameter(torch.full((3,), math.log(1 / config.initial_temperature)))

    def context(self, value, layer):
        return F.normalize(self.context_heads[layer](value.float()), dim=-1)

    def path(self, value, layer):
        return F.normalize(self.path_heads[layer](value.float()), dim=-1)

    def scale(self, layer):
        return self.log_scale[layer].exp().clamp(max=100)

    def matrix_scores(self, contexts, paths, layer):
        return self.context(contexts, layer) @ self.path(paths, layer).T * self.scale(layer)

    def paired_scores(self, contexts, paths, layer):
        return (self.context(contexts, layer) * self.path(paths, layer)).sum(-1) * self.scale(layer)


def cumulative_paths(sids, sid_embeddings):
    """[N,3] codes -> [N,3,D] sums of Qwen input-token embeddings, NOT VQ vectors."""
    if sids.ndim != 2 or sids.shape[1] != 3:
        raise ValueError('Expected three semantic SID columns')
    if (sids < 0).any() or (sids >= sid_embeddings.shape[1]).any():
        raise ValueError('Semantic SID outside token table')
    return torch.stack([sid_embeddings[layer, sids[:, layer]] for layer in range(3)], dim=1).cumsum(1)


class PrefixNegativeSampler:
    """Valid catalog prefixes: same-parent alternatives first, global fallback.

    These are structural negatives, not Qwen-mined high-probability negatives and
    not unclicked exposures. Full-prefix equality is always excluded.
    """
    def __init__(self, catalog_sids):
        self.prefixes, self.groups = [], []
        for depth in range(1, 4):
            prefixes = np.unique(np.asarray(catalog_sids)[:, :depth], axis=0)
            groups = {}
            for i, prefix in enumerate(prefixes):
                groups.setdefault(tuple(prefix[:-1]), []).append(i)
            self.prefixes.append(prefixes)
            self.groups.append(groups)

    def sample(self, targets, layer, count, rng):
        prefixes = self.prefixes[layer]
        values = np.zeros((len(targets), count, layer + 1), dtype=np.int64)
        mask = np.zeros((len(targets), count), dtype=bool)
        for row, target in enumerate(np.asarray(targets)[:, :layer + 1]):
            pool = np.asarray(self.groups[layer].get(tuple(target[:-1]), []), dtype=np.int64)
            pool = pool[np.any(prefixes[pool] != target, axis=1)]
            chosen = rng.choice(pool, min(count, len(pool)), replace=False).tolist()
            # Rejection sampling avoids an O(catalog-size) scan for every training row.
            attempts = 0
            while len(chosen) < count and attempts < max(32, count * 8):
                candidate = int(rng.integers(len(prefixes)))
                if candidate not in chosen and np.any(prefixes[candidate] != target):
                    chosen.append(candidate)
                attempts += 1
            if chosen:
                values[row, :len(chosen)] = prefixes[chosen]
                mask[row, :len(chosen)] = True
        return values, mask


def contrastive_loss(scorer, contexts, targets, sid_embeddings, sampler, negatives, rng):
    """Symmetric multi-positive InfoNCE; row-specific catalog negatives in c2p.

    Samples sharing the same true prefix are ALL positives, never false negatives.
    The optional extra catalog negatives have no positive context anchor, so they
    only enter context-to-path denominators, not the reverse path-to-context loss.
    """
    paths = cumulative_paths(targets, sid_embeddings)
    losses, logs = [], {}
    target_array = targets.detach().cpu().numpy()
    for layer in range(3):
        scores = scorer.matrix_scores(contexts, paths[:, layer], layer)
        positive = (targets[:, None, :layer + 1] == targets[None, :, :layer + 1]).all(-1)
        numerator = scores.masked_fill(~positive, -torch.inf).logsumexp(1)
        denominator_scores = scores
        if negatives:
            codes, mask = sampler.sample(target_array, layer, negatives, rng)
            codes = torch.as_tensor(codes, device=contexts.device)
            mask = torch.as_tensor(mask, device=contexts.device)
            negative_paths = sum(sid_embeddings[j, codes[:, :, j]] for j in range(layer + 1))
            extra = scorer.paired_scores(contexts[:, None, :], negative_paths, layer)
            denominator_scores = torch.cat([scores, extra.masked_fill(~mask, -torch.inf)], dim=1)
        c2p = denominator_scores.logsumexp(1) - numerator
        p2c = scores.logsumexp(0) - scores.masked_fill(~positive, -torch.inf).logsumexp(0)
        losses.append((c2p.mean() + p2c.mean()) / 2)
        with torch.no_grad():
            logs[f'layer{layer + 1}_loss'] = float(losses[-1])
            logs[f'layer{layer + 1}_positive_mass'] = float((numerator - denominator_scores.logsumexp(1)).exp().mean())
            logs[f'layer{layer + 1}_distinct_prefixes'] = int(torch.unique(targets[:, :layer + 1], dim=0).shape[0])
    return torch.stack(losses).mean(), logs


@torch.inference_mode()
def encode_context_batch(model, tokenizer, requests):
    """h0 is the last ASSISTANT-PROMPT state. No target tokens enter this forward."""
    if not requests:
        raise ValueError('Nonempty prompts required')
    lengths = torch.tensor([len(r['prompt']) for r in requests], device=model.device)
    width = int(lengths.max())
    ids = torch.full((len(requests), width), tokenizer.pad_token_id, device=model.device, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, request in enumerate(requests):
        ids[i, :len(request['prompt'])] = torch.tensor(request['prompt'], device=model.device)
        mask[i, :len(request['prompt'])] = 1
    # Call the Qwen backbone directly: avoid materializing vocabulary logits or all-layer hidden states.
    output = model.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
    return output.last_hidden_state[torch.arange(len(requests), device=model.device), lengths - 1].float()


def load_feature_cache(directory):
    directory = Path(directory)
    manifest = read_json(directory / 'manifest.json')
    if manifest.get('status') != 'completed' or manifest['protocol'] != HPR_PROTOCOL:
        raise ValueError('Incomplete or incompatible HPR feature cache')
    for name, checksum in manifest['files'].items():
        if Path(name).name != name or sha256(directory / name) != checksum:
            raise ValueError('Feature cache checksum mismatch: ' + name)
    arrays = {name: np.load(directory / (name + '.npy'), mmap_mode='r', allow_pickle=False)
              for name in ('train_contexts', 'train_targets', 'valid_contexts', 'valid_targets',
                           'sid_embeddings', 'catalog_sids')}
    for split in ('train', 'valid'):
        if (arrays[split + '_contexts'].shape != (manifest[split + '_rows'], manifest['hidden_size'])
                or arrays[split + '_targets'].shape != (manifest[split + '_rows'], 3)):
            raise ValueError('Feature cache shape mismatch')
    return manifest, arrays


def save_scorer(directory, scorer, provenance):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in scorer.state_dict().items()},
              str(directory / 'model.safetensors'))
    write_json(directory / 'config.json', dict(protocol=HPR_PROTOCOL, architecture=asdict(scorer.config),
                                             provenance=provenance))


def load_scorer(directory, device='cpu'):
    directory = Path(directory)
    config = read_json(directory / 'config.json')
    if config['protocol'] != HPR_PROTOCOL:
        raise ValueError('Unknown HPR checkpoint protocol')
    model = HPRScorer(HPRConfig(**config['architecture']))
    model.load_state_dict(load_file(str(directory / 'model.safetensors')))
    return model.to(device).eval(), config
