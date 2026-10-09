"""Train the pinned upstream RQ-VAE on frozen catalog vectors, without compile."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

from tiger_common import (PROCESSED, configure, sha256, utc_now, write_json,
                          atomic_checkpoint, random_state, restore_random, source_hashes)
from modules.rqvae import RqVae
from modules.quantize import QuantizeForwardMode


def make_model(config):
    return RqVae(**config, codebook_mode=QuantizeForwardMode.STE)


def codec_loss(model, x):
    quantized = model.get_semantic_ids(x)
    reconstructed = torch.nn.functional.normalize(model.decode(quantized.embeddings.sum(-1)), dim=-1)
    reconstruction = ((reconstructed - x) ** 2).sum(-1)
    return (reconstruction + quantized.quantize_loss).mean(), reconstruction.mean()


@torch.inference_mode()
def initialize_codebooks(model, vectors, seed, batch_size=512):
    model.eval()
    residual = model.encode(vectors).numpy()
    for depth, layer in enumerate(model.layers):
        km = MiniBatchKMeans(n_clusters=model.codebook_size, n_init=1, max_iter=100,
                            batch_size=batch_size, random_state=seed + depth,
                            reassignment_ratio=.01)
        with threadpool_limits(limits=2):
            labels = km.fit_predict(residual)
        layer.embedding.weight.copy_(torch.from_numpy(km.cluster_centers_))
        layer.kmeans_initted = True
        residual = residual - km.cluster_centers_[labels]


@torch.inference_mode()
def evaluate_codec(model, x, batch_size):
    model.eval()
    recon, total, n = 0., 0., 0
    for batch in x.split(batch_size):
        loss, rec = codec_loss(model, batch)
        recon += rec.item() * len(batch)
        total += loss.item() * len(batch)
        n += len(batch)
    return {'reconstruction_loss': recon / n, 'total_loss': total / n, 'items': n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--embeddings', default=str(PROCESSED / 'tiger_embeddings'))
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--resume')
    args = p.parse_args()
    configure(args.seed, args.threads)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    emb_path = Path(args.embeddings)
    em = json.loads((emb_path / 'manifest.json').read_text(encoding='utf-8'))
    if em['status'] != 'completed' or em['contract']['smoke_only']:
        raise ValueError('Formal RQ-VAE needs complete full-catalog embeddings')
    fingerprint = sha256(emb_path / 'embeddings.npy')
    if fingerprint != em['embedding_sha256']:
        raise ValueError('Embedding fingerprint mismatch')
    x = torch.from_numpy(np.load(emb_path / 'embeddings.npy')[1:])
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(x))
    nvalid = max(1, round(len(x) * .05))
    valid_x, train_x = x[order[:nvalid]], x[order[nvalid:]]
    config = dict(input_dim=x.shape[1], embed_dim=32, hidden_dims=[256, 128],
                  codebook_size=256, codebook_kmeans_init=False,
                  n_layers=3, n_cat_features=0, commitment_weight=.25)
    model = make_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch, best, history, stale = 1, float('inf'), [], 0
    if args.resume:
        previous = torch.load(args.resume, map_location='cpu', weights_only=False)
        if previous['config'] != config or previous['embedding_sha256'] != fingerprint:
            raise ValueError('Resume artifact contract differs')
        if previous['source_sha256'] != source_hashes():
            raise ValueError('Codec source differs from the checkpoint')
        model.load_state_dict(previous['state_dict'])
        optimizer.load_state_dict(previous['optimizer'])
        restore_random(previous['random_state'], rng)
        start_epoch = previous['epoch'] + 1
        parent = json.loads((Path(args.resume).parent / 'run.json').read_text(encoding='utf-8'))
        for key in ('batch_size', 'seed', 'lr', 'patience'):
            if vars(args)[key] != parent['arguments'][key]:
                raise ValueError(f'Resume configuration differs: {key}')
        if args.epochs < start_epoch or parent['history'][-1]['epoch'] != previous['epoch']:
            raise ValueError('Resume the latest checkpoint into a greater total epoch count')
        history, best = parent['history'], previous['best_metric']
        stale = previous['stale']
        (out / 'best.pth').write_bytes((Path(args.resume).parent / 'best.pth').read_bytes())
        best_epoch = previous['best_epoch']
    else:
        initialize_codebooks(model, train_x[rng.choice(len(train_x), min(4096, len(train_x)), replace=False)], args.seed)
        best_epoch = 0
    run = dict(status='running', started_at_utc=utc_now(), arguments=vars(args), config=config,
               embedding_sha256=fingerprint, embedding_manifest=em, history=history,
               codec_train_items=len(train_x), codec_validation_items=len(valid_x),
               selection='catalog-vector validation reconstruction, not recommendation test',
               upstream_forward_used=False, upstream_quantizer_used=True,
               initialization='bounded sklearn MiniBatchKMeans, separate residual codebooks',
               reconstruction='L2-normalized decoder output; n_cat_features=0',
               source_sha256=source_hashes())
    write_json(out / 'run.json', run)
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        losses, count = 0., 0
        for start in range(0, len(train_x), args.batch_size):
            if start == 0:
                indices = rng.permutation(len(train_x))
            batch = train_x[indices[start:start + args.batch_size]]
            loss, _ = codec_loss(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite codec loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses += loss.item() * len(batch)
            count += len(batch)
        valid = evaluate_codec(model, valid_x, args.batch_size)
        history.append(dict(epoch=epoch, loss=losses/count, validation=valid,
                            seconds=time.perf_counter()-epoch_start))
        improved = valid['reconstruction_loss'] < best
        if improved:
            best, best_epoch, stale = valid['reconstruction_loss'], epoch, 0
        else:
            stale += 1
        state = dict(state_dict=model.state_dict(), config=config, epoch=epoch,
                     embedding_sha256=fingerprint, optimizer=optimizer.state_dict(),
                     random_state=random_state(rng), source_sha256=run['source_sha256'],
                     best_metric=best,best_epoch=best_epoch,stale=stale)
        atomic_checkpoint(out / 'last.pth', state)
        if improved:
            atomic_checkpoint(out / 'best.pth', state)
        run.update(history=history, best_epoch=best_epoch, best_validation_reconstruction=best)
        write_json(out / 'run.json', run)
        print(f'rqvae epoch={epoch} loss={losses/count:.6f} '
              f'validation_reconstruction={valid["reconstruction_loss"]:.6f} '
              f'seconds={time.perf_counter()-epoch_start:.1f}', flush=True)
        if stale >= args.patience:
            run['stop_reason'] = 'codec_validation_plateau'
            break
    run.update(status='completed', completed_at_utc=utc_now(), elapsed_seconds=time.perf_counter()-started,
               parameters=sum(p.numel() for p in model.parameters()))
    run.setdefault('stop_reason', 'requested_epochs_completed')
    write_json(out / 'run.json', run)


if __name__ == '__main__':
    main()
