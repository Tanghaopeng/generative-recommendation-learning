"""Fixed-feature SC/PD diagnostics, distinct from differentiable training proxies.

SC uses unordered item pairs and macro-averages non-singleton clusters.
PD uses distinct nonzero empirical cluster centroids; smaller is better.
The fourth collision counter is never used to define a semantic cluster.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tiger_common import PROCESSED, sha256, utc_now, write_json


def cluster_quality(features, labels, *, temperature=2., max_pairs=200000, seed=2026):
    x = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if x.ndim != 2 or len(x) < 1 or len(labels) != len(x):
        raise ValueError('Expected nonempty aligned feature and cluster arrays')
    if not np.isfinite(x).all() or temperature <= 0 or max_pairs < 1:
        raise ValueError('Invalid features or PD configuration')
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError('Zero feature vectors have undefined cosine similarity')
    x = x / norms[:, None]
    _, inverse, counts = np.unique(labels, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(counts), x.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, x)
    eligible = counts > 1
    # ||sum unit(x_i)||^2 - n equals sum_{i != j} cosine(x_i, x_j).
    per_cluster = ((sums[eligible] ** 2).sum(1) - counts[eligible]) / (
        counts[eligible] * (counts[eligible] - 1))
    sc = float(np.clip(per_cluster, -1, 1).mean()) if eligible.any() else None
    centers = sums / counts[:, None]
    center_norms = np.linalg.norm(centers, axis=1)
    centers = centers[center_norms > 1e-12] / center_norms[center_norms > 1e-12, None]
    m = len(centers)
    total_pairs = m * (m - 1) // 2
    pd = None
    evaluated_pairs = min(total_pairs, max_pairs)
    if total_pairs:
        if total_pairs <= max_pairs:
            a, b = np.triu_indices(m, k=1)
        else:
            # Sample unordered pairs without constructing an M x M matrix.
            ranks = np.random.default_rng(seed).choice(total_pairs, max_pairs, replace=False)
            starts = np.arange(m, dtype=np.int64)
            starts = starts * (2 * m - starts - 1) // 2
            a = np.searchsorted(starts, ranks, side='right') - 1
            b = a + 1 + ranks - starts[a]
        similarity = np.empty(evaluated_pairs, dtype=np.float64)
        for start in range(0, evaluated_pairs, 4096):
            section = slice(start, start + 4096)
            similarity[section] = np.einsum('ij,ij->i', centers[a[section]], centers[b[section]])
        similarity = np.clip(similarity, -1, 1)
        values = -temperature * (1 - similarity)
        top = float(values.max())
        pd = float(top + np.log(np.exp(values - top).mean()))
    return dict(SC=sc, PD=pd, items=len(x), active_clusters=len(counts),
                sc_eligible_clusters=int(eligible.sum()),
                singleton_clusters=int((counts == 1).sum()),
                sc_item_coverage=float(counts[eligible].sum() / len(x)),
                pd_valid_centroids=m, pd_undefined_centroids=int(len(counts) - m),
                pd_total_pairs=total_pairs, pd_evaluated_pairs=evaluated_pairs,
                pd_exact=total_pairs <= max_pairs, max_cluster_items=int(counts.max()))


def semantic_quality(features, semantic_ids, *, codebook_size=256, max_pairs=200000, seed=2026):
    codes = np.asarray(semantic_ids)
    if codes.shape != (len(features), 3) or not np.issubdtype(codes.dtype, np.integer):
        raise ValueError('Expected exactly three integer semantic segments, no collision counter')
    if np.any(codes < 0) or np.any(codes >= codebook_size):
        raise ValueError('Code outside configured codebook')
    rows = []
    for depth in range(3):
        for grouping in ('layer', 'prefix'):
            labels = codes[:, depth] if grouping == 'layer' else codes[:, :depth + 1]
            quality = cluster_quality(features, labels, max_pairs=max_pairs, seed=seed + depth)
            rows.append(dict(grouping=grouping, depth=depth + 1, **quality))
    layers = [row for row in rows if row['grouping'] == 'layer']
    values = lambda key: [row[key] for row in layers if row[key] is not None]
    return dict(protocol='fixed_minilm_cosine_macro_sc_centroid_pd_v1',
                feature_space='same frozen input vectors across codecs', pd_temperature=2.,
                sc_pair_convention='unordered i<j average; SC range [-1,1]',
                sc_singleton_policy='exclude undefined singleton clusters; report coverage',
                pd_centroids='empirical means of unit input item vectors; zero centroids excluded',
                pd_sampling_seed=seed, pd_max_pairs=max_pairs,
                collision_counter_excluded=True,
                SC_layer_macro=float(np.mean(values('SC'))) if values('SC') else None,
                PD_layer_macro=float(np.mean(values('PD'))) if values('PD') else None,
                rows=rows,
                codebook_utilization=[len(np.unique(codes[:, d])) / codebook_size for d in range(3)],
                dead_codes=[codebook_size - len(np.unique(codes[:, d])) for d in range(3)],
                raw_collision_excess_fraction=1 - len(np.unique(codes, axis=0)) / len(codes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embeddings', type=Path, default=PROCESSED / 'tiger_embeddings')
    parser.add_argument('--sids', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-pairs', type=int, default=200000)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve old reports; use a new output file')
    em = json.loads((args.embeddings / 'manifest.json').read_text(encoding='utf-8'))
    sm = json.loads((args.sids / 'manifest.json').read_text(encoding='utf-8'))
    eh, sh = sha256(args.embeddings / 'embeddings.npy'), sha256(args.sids / 'semantic_ids.npy')
    if (em['status'] != 'completed' or em['contract']['smoke_only'] or sm['status'] != 'completed'
        or eh != em['embedding_sha256'] or eh != sm['embedding_sha256'] or sh != sm['sid_sha256']
        or em['contract']['items_source_sha256'] != sm['items_source_sha256']):
        raise ValueError('Embedding/SID artifact contracts differ')
    x, codes = np.load(args.embeddings / 'embeddings.npy'), np.load(args.sids / 'semantic_ids.npy')
    if codes.shape != (len(x), 4) or not (codes[0] == -1).all() or len(x) != sm['items'] + 1:
        raise ValueError('Padding/alignment contract differs')
    result = semantic_quality(x[1:], codes[1:, :3], codebook_size=sm['codebook_size'],
                              max_pairs=args.max_pairs, seed=args.seed)
    result.update(recorded_at_utc=utc_now(), embedding_sha256=eh, sid_sha256=sh,
                  quality_code_sha256=sha256(Path(__file__)),
                  interpretation='geometric proxies, not direct interaction preference metrics or downstream guarantees')
    write_json(args.output, result)
    print(json.dumps({key:result[key] for key in ('SC_layer_macro','PD_layer_macro','dead_codes')}))


if __name__ == '__main__':
    main()
