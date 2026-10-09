"""Freeze unique four-token SIDs from the trained upstream residual quantizer."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from tiger_common import PROCESSED, configure, sha256, utc_now, write_json
from train_rqvae import make_model


def disambiguate(raw):
    counts = Counter()
    result = np.empty((len(raw), raw.shape[1] + 1), dtype=np.int64)
    for index, code in enumerate(raw):
        key = tuple(int(v) for v in code)
        result[index, :-1] = code
        result[index, -1] = counts[key]
        counts[key] += 1
    return result, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--embeddings', default=str(PROCESSED / 'tiger_embeddings'))
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', default=str(PROCESSED / 'tiger_sids'))
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    configure(threads=args.threads)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty SID output directory to freeze a new version')
    out.mkdir(parents=True, exist_ok=True)
    ep = Path(args.embeddings)
    manifest = json.loads((ep / 'manifest.json').read_text(encoding='utf-8'))
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if manifest['status'] != 'completed' or manifest['contract']['smoke_only']:
        raise ValueError('Requires completed full catalog embeddings')
    if sha256(ep / 'embeddings.npy') != saved['embedding_sha256']:
        raise ValueError('RQ-VAE and embeddings do not match')
    if sha256(PROCESSED / 'items.jsonl') != manifest['contract']['items_source_sha256']:
        raise ValueError('Catalog changed since encoding')
    model = make_model(saved['config'])
    model.load_state_dict(saved['state_dict'])
    model.eval()
    x = torch.from_numpy(np.load(ep / 'embeddings.npy')[1:])
    with torch.inference_mode():
        raw = torch.cat([model.get_semantic_ids(b).sem_ids for b in x.split(512)]).numpy()
    full, counts = disambiguate(raw)
    sid = np.full((len(x) + 1, 4), -1, dtype=np.int64)
    sid[1:] = full
    if len(np.unique(full, axis=0)) != len(x):
        raise ValueError('Failed unique SID verification')
    reverse = {tuple(code): item for item, code in enumerate(full, 1)}
    if any(reverse[tuple(code)] != item for item, code in enumerate(full, 1)):
        raise ValueError('SID roundtrip failed')
    np.save(out / 'semantic_ids.npy', sid)
    duplicate_items = len(x) - len(counts)
    info = dict(status='completed', completed_at_utc=utc_now(), items=len(x), sid_length=4,
                codebook_size=saved['config']['codebook_size'],
                tokens_per_hierarchy=max(saved['config']['codebook_size'], int(full[:, -1].max()) + 1),
                raw_unique_ids=len(counts), raw_duplicate_items=duplicate_items,
                raw_duplicate_fraction=duplicate_items/len(x),
                max_collision_group=max(counts.values()),
                codebook_usage=[int(np.unique(raw[:, d]).size) for d in range(3)],
                final_unique_ids=len(x), roundtrip_verified=True, padding_row=-1,
                collision_order='ascending item_id, fourth token=within-group counter',
                items_source_sha256=manifest['contract']['items_source_sha256'],
                embedding_sha256=saved['embedding_sha256'], checkpoint_sha256=sha256(args.checkpoint),
                rqvae_best_epoch=saved['epoch'], sid_sha256=sha256(out/'semantic_ids.npy'))
    write_json(out / 'manifest.json', info)
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
