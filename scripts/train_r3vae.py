"""Train/export an isolated MiniLM R3-VAE adaptation; never rewrites old SIDs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from tiger_common import (ROOT, PROCESSED, configure, sha256, utc_now, write_json,
                          atomic_checkpoint, random_state, restore_random)
from build_semantic_ids import disambiguate
from r3vae import R3VAE, REFERENCE
from sid_quality import semantic_quality


def source_hashes():
    return {name:sha256(ROOT / 'scripts' / name) for name in
            ('r3vae.py','train_r3vae.py','sid_quality.py','build_semantic_ids.py','tiger_common.py')}


@torch.inference_mode()
def evaluate(model, x, batch_size):
    model.eval()
    sums = dict(reconstruction_loss=0.,sc_proxy_loss=0.,pd_proxy_loss=0.)
    for batch in x.split(batch_size):
        output = model(batch)
        for key in sums:
            sums[key] += float(output[key]) * len(batch)
    return {key:value / len(x) for key,value in sums.items()}


@torch.inference_mode()
def export_sids(model, x, output, metadata, checkpoint):
    model.eval()
    raw = torch.cat([model(batch)['ids'].cpu() for batch in x.split(512)]).numpy()
    full, counts = disambiguate(raw)
    if len(np.unique(full, axis=0)) != len(x):
        raise ValueError('SID disambiguation failed')
    sid = np.full((len(x) + 1, 4), -1, dtype=np.int64)
    sid[1:] = full
    output.mkdir(exist_ok=False)
    np.save(output / 'semantic_ids.npy', sid)
    info = dict(status='completed', codec='r3vae_local_adaptation', items=len(x),
                sid_length=4, semantic_layers=3, codebook_size=model.codebook_size,
                tokens_per_hierarchy=max(model.codebook_size, int(full[:,-1].max()) + 1),
                padding_row=-1, raw_unique_ids=len(counts),
                raw_duplicate_fraction=1-len(counts)/len(x),max_collision_group=max(counts.values()),
                codebook_usage=[int(len(np.unique(raw[:,depth]))) for depth in range(3)],
                final_unique_ids=len(x),roundtrip_verified=True,
                collision_order='ascending item_id, fourth token=within-group counter',
                embedding_sha256=metadata['embedding_sha256'],
                items_source_sha256=metadata['embedding_manifest']['contract']['items_source_sha256'],
                checkpoint_sha256=sha256(checkpoint),sid_sha256=sha256(output / 'semantic_ids.npy'))
    write_json(output / 'manifest.json', info)
    quality = semantic_quality(x.cpu().numpy(), raw, codebook_size=model.codebook_size)
    quality.update(embedding_sha256=info['embedding_sha256'],sid_sha256=info['sid_sha256'],
                   quality_code_sha256=sha256(ROOT / 'scripts/sid_quality.py'))
    write_json(output.parent / 'sid_quality.json', quality)
    return info, quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embeddings', type=Path, default=PROCESSED / 'tiger_embeddings')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if (min(config['epochs'],config['batch_size'],config['patience'],config['init_items']) < 1
        or config['lr'] <= 0 or min(config['sc_weight'],config['pd_weight']) < 0
        or config['initialization'] not in ('residual_kmeans','uniform')):
        raise ValueError('Invalid training configuration')
    configure(config['seed'], args.threads)
    if args.output.exists():
        raise ValueError('Use a new output directory, also when resuming')
    em = json.loads((args.embeddings / 'manifest.json').read_text(encoding='utf-8'))
    eh = sha256(args.embeddings / 'embeddings.npy')
    if (em['status'] != 'completed' or em['contract']['smoke_only'] or eh != em['embedding_sha256']
        or em['contract']['items_source_sha256'] != sha256(PROCESSED / 'items.jsonl')):
        raise ValueError('Complete verified catalog embeddings required')
    x = torch.from_numpy(np.load(args.embeddings / 'embeddings.npy')[1:]).float()
    if x.shape != (em['contract']['items'],config['model']['input_dim']) or not torch.isfinite(x).all():
        raise ValueError('Embedding shape/data invalid')
    rng = np.random.default_rng(config['seed'])
    order = rng.permutation(len(x))
    nvalid = max(1,round(len(x) * .05))
    valid, train = x[order[:nvalid]], x[order[nvalid:]]
    model = R3VAE(**config['model'])
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    start_epoch, best, stale, history, best_epoch = 1, float('inf'), 0, [], 0
    snapshot = source_hashes()
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    if args.resume:
        saved = torch.load(args.resume,map_location='cpu',weights_only=False)
        expected = dict(saved['config']); expected['epochs'] = config['epochs']
        if (expected != config or saved['embedding_sha256'] != eh or saved['source_sha256'] != snapshot
            or config['epochs'] <= saved['epoch']):
            raise ValueError('Resume contract differs; only larger epoch budget is allowed')
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        restore_random(saved['random_state'],rng)
        history, best, stale, best_epoch = saved['history'],saved['best_metric'],saved['stale'],saved['best_epoch']
        start_epoch = saved['epoch'] + 1
        (args.output / 'best.pth').write_bytes((args.resume.parent / 'best.pth').read_bytes())
    elif config['initialization'] == 'residual_kmeans':
        chosen = rng.choice(len(train), min(config['init_items'], len(train)), replace=False)
        model.initialize_kmeans(train[chosen],config['seed'])
    initialization_seconds = time.perf_counter() - started
    run = dict(status='training', started_at_utc=utc_now(),config=config,reference=REFERENCE,
               source_sha256=snapshot,embedding_sha256=eh,embedding_manifest=em,
               initialization_seconds=initialization_seconds,
               codec_train_items=len(train),codec_validation_items=len(valid),
               catalog_validation_fraction=.05,split_seed=config['seed'],
               selection='held-out catalog-vector cosine reconstruction only; no downstream test',
               precision='FP32 CPU',parameters=sum(p.numel() for p in model.parameters()),history=history)
    write_json(args.output / 'run.json',run)
    for epoch in range(start_epoch,config['epochs'] + 1):
        timer = time.perf_counter()
        model.train()
        indices = rng.permutation(len(train))
        sums = dict(loss=0., reconstruction_loss=0.,sc_proxy_loss=0.,pd_proxy_loss=0.)
        for start in range(0,len(train),config['batch_size']):
            batch = train[indices[start:start + config['batch_size']]]
            optimizer.zero_grad(set_to_none=True)
            result = model(batch,config['sc_weight'],config['pd_weight'])
            if not torch.isfinite(result['loss']):
                raise FloatingPointError('Nonfinite R3-VAE loss')
            result['loss'].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite R3-VAE gradient')
            optimizer.step()
            for key in sums:
                sums[key] += float(result[key].detach()) * len(batch)
        validation = evaluate(model,valid,config['batch_size'])
        row = dict(epoch=epoch,seconds=time.perf_counter()-timer,validation=validation,
                   **{key:value/len(train) for key,value in sums.items()})
        history.append(row)
        if validation['reconstruction_loss'] < best:
            best,best_epoch,stale = validation['reconstruction_loss'],epoch,0
        else:
            stale += 1
        checkpoint = dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
                          random_state=random_state(rng),embedding_sha256=eh,config=config,
                          source_sha256=snapshot,epoch=epoch,best_metric=best,best_epoch=best_epoch,
                          stale=stale,history=history)
        atomic_checkpoint(args.output / 'last.pth',checkpoint)
        if best_epoch == epoch:
            atomic_checkpoint(args.output / 'best.pth',checkpoint)
        run.update(history=history,best_epoch=best_epoch)
        write_json(args.output / 'run.json',run)
        print(json.dumps(row),flush=True)
        if stale >= config['patience']:
            break
    best_state = torch.load(args.output / 'best.pth',map_location='cpu',weights_only=False)
    model.load_state_dict(best_state['state_dict'])
    info, quality = export_sids(model,x,args.output / 'sids',run,args.output / 'best.pth')
    run.update(status='completed',completed_at_utc=utc_now(),elapsed_seconds=time.perf_counter()-started,
               best_epoch=best_epoch,best_validation_reconstruction=best,sid_manifest=info,
               stop_reason='catalog_validation_plateau' if stale >= config['patience'] else 'requested_epochs_completed',
               SC_layer_macro=quality['SC_layer_macro'],PD_layer_macro=quality['PD_layer_macro'],
               downstream_recommendation_trained=False)
    write_json(args.output / 'run.json',run)
    print('R3VAE_COMPLETED',json.dumps({key:run[key] for key in ('best_epoch','SC_layer_macro','PD_layer_macro','elapsed_seconds')}),flush=True)


if __name__ == '__main__':
    main()
