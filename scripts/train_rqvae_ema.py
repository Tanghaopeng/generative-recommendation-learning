"""Matched RQ-VAE ablation: EMA codebooks, commitment + reconstruction loss."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from tiger_common import (ROOT,PROCESSED,configure,sha256,utc_now,write_json,
                          atomic_checkpoint,random_state,restore_random,source_hashes)
from train_rqvae import make_model,initialize_codebooks,codec_loss,evaluate_codec
from rqvae_ema import attach_ema,ema_step,canonical_state
from audit_codebooks import audit_split


def sources():
    result = source_hashes()
    for name in ('rqvae_ema.py','train_rqvae_ema.py','audit_codebooks.py'):
        result[f'scripts/{name}'] = sha256(ROOT/'scripts'/name)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--embeddings',default=str(PROCESSED/'tiger_embeddings'))
    p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--decay',type=float,default=.99)
    p.add_argument('--epsilon',type=float,default=1e-5)
    p.add_argument('--initial-mass',type=float,default=1.)
    p.add_argument('--resume')
    args = p.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.patience<1:
        raise ValueError('Positive epochs, batch size and patience required')
    configure(args.seed,args.threads)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    ep = Path(args.embeddings)
    em = json.loads((ep/'manifest.json').read_text(encoding='utf-8'))
    fingerprint = sha256(ep/'embeddings.npy')
    if em['status']!='completed' or em['contract']['smoke_only'] or fingerprint!=em['embedding_sha256']:
        raise ValueError('Requires verified full-catalog embeddings')
    x = torch.from_numpy(np.load(ep/'embeddings.npy')[1:])
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(x))
    nvalid = max(1,round(len(x)*.05))
    valid_x,train_x = x[order[:nvalid]],x[order[nvalid:]]
    config = dict(input_dim=x.shape[1],embed_dim=32,hidden_dims=[256,128],
                  codebook_size=256,codebook_kmeans_init=False,
                  n_layers=3,n_cat_features=0,commitment_weight=.25)
    model = make_model(config)
    original_keys = set(model.state_dict())
    ema_config = dict(decay=args.decay,epsilon=args.epsilon,initial_mass=args.initial_mass)
    if not args.resume:
        chosen = rng.choice(len(train_x),min(4096,len(train_x)),replace=False)
        initialize_codebooks(model,train_x[chosen],args.seed)
    attach_ema(model,**ema_config)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=args.lr)
    history,best,best_epoch,stale,start_epoch = [],float('inf'),0,0,1
    snapshot = sources()
    if args.resume:
        saved = torch.load(args.resume,map_location='cpu',weights_only=False)
        if (saved['config']!=config or saved['ema_config']!=ema_config or
                saved['embedding_sha256']!=fingerprint or saved['source_sha256']!=snapshot):
            raise ValueError('EMA resume contract differs')
        parent = json.loads((Path(args.resume).parent/'run.json').read_text(encoding='utf-8'))
        for key in ('batch_size','seed','lr','patience'):
            if vars(args)[key] != parent['arguments'][key]:
                raise ValueError(f'Resume configuration differs: {key}')
        if parent['history'][-1]['epoch']!=saved['epoch'] or args.epochs<=saved['epoch']:
            raise ValueError('Resume latest state into a larger total epoch count')
        model.load_state_dict(saved['ema_state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        restore_random(saved['random_state'],rng)
        best,best_epoch,stale = (saved[k] for k in ('best_metric','best_epoch','stale'))
        history,start_epoch = parent['history'],saved['epoch']+1
        (out/'best.pth').write_bytes((Path(args.resume).parent/'best.pth').read_bytes())
    run = dict(status='running',started_at_utc=utc_now(),arguments=vars(args),config=config,
               codebook_update='EMA; no codebook gradient, no dead-code reset',ema_config=ema_config,
               embedding_sha256=fingerprint,source_sha256=snapshot,history=history,
               codec_train_items=len(train_x),codec_validation_items=len(valid_x),
               selection='same 5% catalog-vector reconstruction validation as Adam baseline',
               loss='normalized reconstruction + 0.25 * summed per-level commitment; no codebook loss',
               optimizer='Adam encoder/decoder only; EMA applied after optimizer step using pre-step residuals',
               residual_graph='same upstream per-level STE as Adam; no residual-gradient changes',
               initialization='same bounded residual MiniBatchKMeans and 4096 training vectors; EMA pseudocount=initial_mass',
               dead_code_reset=False)
    write_json(out/'run.json',run)
    started = time.perf_counter()
    for epoch in range(start_epoch,args.epochs+1):
        model.train()
        indices = rng.permutation(len(train_x))
        total,count,epoch_start = 0.,0,time.perf_counter()
        for start in range(0,len(train_x),args.batch_size):
            batch = train_x[indices[start:start+args.batch_size]]
            loss,_ = codec_loss(model,batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite EMA loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            ema_step(model)
            total+=loss.item()*len(batch)
            count+=len(batch)
        valid = evaluate_codec(model,valid_x,args.batch_size)
        # Fixed train-corpus snapshot every epoch, distinct from ever-assigned
        # or minibatch usage. Frozen eval never applies EMA updates.
        audit = audit_split(model,train_x,args.batch_size)
        layers = [{k:row[k] for k in ('layer','active_codes','dead_codes','utilization','effective_codes','top1_fraction','top10_fraction')}
                  for row in audit['layers']]
        history.append(dict(epoch=epoch,loss=total/count,validation=valid,
                            train_occupancy_snapshot=layers,seconds=time.perf_counter()-epoch_start))
        improved = valid['reconstruction_loss']<best
        if improved:
            best,best_epoch,stale = valid['reconstruction_loss'],epoch,0
        else:
            stale+=1
        state = dict(state_dict=canonical_state(model,original_keys),ema_state_dict=model.state_dict(),
                     config=config,ema_config=ema_config,epoch=epoch,embedding_sha256=fingerprint,
                     optimizer=optimizer.state_dict(),random_state=random_state(rng),source_sha256=snapshot,
                     best_metric=best,best_epoch=best_epoch,stale=stale,codebook_update='EMA')
        atomic_checkpoint(out/'last.pth',state)
        if improved:
            atomic_checkpoint(out/'best.pth',state)
        run.update(history=history,best_epoch=best_epoch,best_validation_reconstruction=best)
        write_json(out/'run.json',run)
        print(f'ema epoch={epoch} loss={total/count:.6f} validation_reconstruction={valid["reconstruction_loss"]:.6f} '
              f'active={[r["active_codes"] for r in layers]} seconds={time.perf_counter()-epoch_start:.1f}',flush=True)
        if stale>=args.patience:
            run['stop_reason']='codec_validation_plateau'
            break
    run.update(status='completed',completed_at_utc=utc_now(),elapsed_seconds=time.perf_counter()-started,
               parameters=sum(p.numel() for p in model.parameters()),
               optimizer_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
               best_checkpoint_sha256=sha256(out/'best.pth'))
    run.setdefault('stop_reason','requested_epochs_completed')
    write_json(out/'run.json',run)


if __name__=='__main__':
    main()
