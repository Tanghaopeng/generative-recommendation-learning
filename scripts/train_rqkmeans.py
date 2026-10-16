"""Fit residual MiniBatchKMeans on fixed features; export versioned four-token SIDs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import sklearn
import torch
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

from tiger_common import ROOT, PROCESSED, configure, sha256, utc_now, write_json
from build_semantic_ids import disambiguate
from audit_codebooks import occupancy
from sid_quality import semantic_quality


def assign(features, centers, batch_size=1024):
    """Bound memory and use float64 distances to avoid cancellation near ties."""
    x, c = np.asarray(features), np.asarray(centers, dtype=np.float64)
    if x.ndim != 2 or c.ndim != 2 or x.shape[1] != c.shape[1]:
        raise ValueError('Features and centers must have matching dimensions')
    ids = []
    for start in range(0, len(x), batch_size):
        b = np.asarray(x[start:start + batch_size], dtype=np.float64)
        distance = (b*b).sum(1)[:, None] + (c*c).sum(1)[None, :] - 2*b@c.T
        ids.append(distance.argmin(1))
    return np.concatenate(ids).astype(np.int64)


def encode(features, codebooks):
    residual = np.asarray(features, dtype=np.float32).copy()
    codes = []
    errors = [float(np.mean(np.sum(residual.astype(np.float64)**2, axis=1)))]
    for centers in codebooks:
        labels = assign(residual, centers)
        codes.append(labels)
        residual -= centers[labels]
        errors.append(float(np.mean(np.sum(residual.astype(np.float64)**2, axis=1))))
    return np.stack(codes, axis=1), residual, errors


def fit_residual(features, *, layers=3, size=256, seed=2026, threads=2,
                 batch_size=1024, n_init=3, max_iter=100):
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or len(x) < size or not np.isfinite(x).all():
        raise ValueError('Need at least codebook_size finite training features')
    residual, codebooks, history = x.copy(), [], []
    for depth in range(layers):
        started = time.perf_counter()
        km = MiniBatchKMeans(n_clusters=size, init='k-means++', n_init=n_init,
            max_iter=max_iter, batch_size=batch_size, random_state=seed+depth,
            reassignment_ratio=.01, max_no_improvement=20)
        with threadpool_limits(limits=threads):
            km.fit(residual)
            centers = km.cluster_centers_.astype(np.float32)
            labels = assign(residual, centers, batch_size)
        residual -= centers[labels]
        row = dict(layer=depth+1, seconds=time.perf_counter()-started,
            n_iter=int(km.n_iter_), n_steps=int(km.n_steps_),
            train_residual_squared_l2=float(np.mean(np.sum(residual.astype(np.float64)**2, axis=1))),
            occupancy=occupancy(np.bincount(labels, minlength=size)))
        codebooks.append(centers)
        history.append(row)
        print(json.dumps({k:row[k] for k in ('layer','seconds','n_iter','train_residual_squared_l2')}), flush=True)
    return np.stack(codebooks), history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--embeddings', type=Path, default=PROCESSED/'tiger_embeddings')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--encoder-checkpoint', type=Path)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--n-init', type=int, default=3)
    p.add_argument('--max-iter', type=int, default=100)
    args = p.parse_args()
    configure(args.seed, args.threads)
    if args.output.exists():
        raise ValueError('Use a new output directory')
    em = json.loads((args.embeddings/'manifest.json').read_text(encoding='utf-8'))
    eh = sha256(args.embeddings/'embeddings.npy')
    if (em['status'] != 'completed' or em['contract']['smoke_only'] or
        eh != em['embedding_sha256'] or
        sha256(PROCESSED/'items.jsonl') != em['contract']['items_source_sha256']):
        raise ValueError('Incomplete embeddings or catalog fingerprint mismatch')
    padded = np.load(args.embeddings/'embeddings.npy')
    if padded.shape != (em['contract']['items']+1, em['contract']['dimension']):
        raise ValueError('Embedding dimensions differ from manifest')
    x = padded[1:]
    features, encoder_info = x.copy(), None
    started = time.perf_counter()
    if args.encoder_checkpoint:
        from train_rqvae import make_model
        saved = torch.load(args.encoder_checkpoint, map_location='cpu', weights_only=False)
        if saved['embedding_sha256'] != eh:
            raise ValueError('Encoder checkpoint feature fingerprint differs')
        model = make_model(saved['config'])
        model.load_state_dict(saved['state_dict'])
        model.eval()
        with torch.inference_mode():
            features = torch.cat([model.encode(b) for b in torch.from_numpy(x).split(512)]).numpy()
        encoder_info = dict(checkpoint_sha256=sha256(args.encoder_checkpoint),
            checkpoint_config=saved['config'], frozen=True,
            interpretation='conditional ablation: reuse trained RQ-VAE encoder; not independent content baseline')
    order = np.random.default_rng(args.seed).permutation(len(x))
    nvalid = max(1, round(len(x)*.05))
    valid_ids, train_ids = order[:nvalid], order[nvalid:]
    args.output.mkdir(parents=True)
    fit_started = time.perf_counter()
    books, history = fit_residual(features[train_ids], seed=args.seed, threads=args.threads,
        batch_size=args.batch_size, n_init=args.n_init, max_iter=args.max_iter)
    fit_seconds = time.perf_counter()-fit_started
    with threadpool_limits(limits=args.threads):
        raw, residual, errors = encode(features, books)
    full, groups = disambiguate(raw)
    sid = np.full((len(x)+1, 4), -1, dtype=np.int64)
    sid[1:] = full
    if len(np.unique(full, axis=0)) != len(x):
        raise ValueError('SID uniqueness failed')
    reverse = {tuple(code):item for item,code in enumerate(full,1)}
    assert all(reverse[tuple(code)]==item for item,code in enumerate(full,1))
    sd = args.output/'sids'
    sd.mkdir()
    np.save(sd/'semantic_ids.npy', sid)
    np.savez(args.output/'codebooks.npz', codebooks=books)
    info = dict(status='completed', codec='residual_minibatch_kmeans', items=len(x),
        sid_length=4, semantic_layers=3, codebook_size=256,
        tokens_per_hierarchy=max(256,int(full[:,-1].max())+1), padding_row=-1,
        raw_unique_ids=len(groups), raw_duplicate_fraction=1-len(groups)/len(x),
        max_collision_group=max(groups.values()),
        codebook_usage=[len(np.unique(raw[:,d])) for d in range(3)],
        final_unique_ids=len(x), roundtrip_verified=True,
        collision_order='ascending item_id, fourth token=within-group counter',
        embedding_sha256=eh, items_source_sha256=em['contract']['items_source_sha256'],
        checkpoint_sha256=sha256(args.output/'codebooks.npz'), sid_sha256=sha256(sd/'semantic_ids.npy'))
    write_json(sd/'manifest.json', info)
    with threadpool_limits(limits=args.threads):
        quality = semantic_quality(x, raw)
    quality.update(embedding_sha256=eh, sid_sha256=info['sid_sha256'],
        quality_code_sha256=sha256(ROOT/'scripts/sid_quality.py'))
    write_json(args.output/'sid_quality.json', quality)
    split_stats = {}
    for name,indices in [('train',train_ids),('validation',valid_ids),('full_catalog',np.arange(len(x)))]:
        norms = np.linalg.norm((features-residual)[indices], axis=1)
        original_norms = np.linalg.norm(features[indices], axis=1)
        denominator = norms*original_norms
        cosine = np.sum(features[indices]*(features-residual)[indices],axis=1)/np.maximum(denominator,1e-12)
        split_stats[name] = dict(items=len(indices),
            residual_squared_l2=float(np.mean(np.sum(residual[indices].astype(np.float64)**2,axis=1))),
            cosine_reconstruction_loss=float(np.mean(1-cosine)),
            cosine_undefined_items=int((denominator<=1e-12).sum()),
            layers=[occupancy(np.bincount(raw[indices,d],minlength=256)) for d in range(3)])
    _, counts = np.unique(raw,axis=0,return_counts=True)
    report = dict(status='completed', recorded_at_utc=utc_now(),
        feature_space='frozen_rqvae_encoder32' if encoder_info else 'frozen_minilm384',
        dimension=features.shape[1], codebook_size=256, layers=3,
        codebook_scalar_count=int(books.size), train_items=len(train_ids), validation_items=len(valid_ids),
        split_seed=args.seed, validation_fraction=.05, fit_uses_training_items_only=True,
        encoder=encoder_info, configuration=dict(seed=args.seed,batch_size=args.batch_size,n_init=args.n_init,
            max_iter=args.max_iter,max_no_improvement=20,reassignment_ratio=.01,threads=args.threads,
            metric='squared Euclidean', residual_renormalization=False),
        history=history, fit_seconds=fit_seconds, total_seconds=time.perf_counter()-started,
        residual_squared_l2_by_depth=errors, splits=split_stats,
        collision_groups=int((counts>1).sum()), items_in_collision_groups=int(counts[counts>1].sum()),
        collision_item_fraction=float(counts[counts>1].sum()/len(x)),
        manifest=info, quality=quality,
        versions=dict(numpy=np.__version__,sklearn=sklearn.__version__,torch=torch.__version__),
        source_sha256={name:sha256(ROOT/'scripts'/name) for name in
            ('train_rqkmeans.py','sid_quality.py','build_semantic_ids.py','audit_codebooks.py','tiger_common.py')},
        downstream_recommendation_trained=False)
    write_json(args.output/'run.json', report)
    print(json.dumps(dict(status='completed',dimension=features.shape[1],fit_seconds=fit_seconds,
        SC=quality['SC_layer_macro'],PD=quality['PD_layer_macro'],dead=quality['dead_codes'],
        collision_fraction=info['raw_duplicate_fraction'],max_group=info['max_collision_group'])),flush=True)


if __name__ == '__main__':
    main()
