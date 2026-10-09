"""Quantizer occupancy, imbalance, residual error, and actual gradient routing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tiger_common import PROCESSED, configure, sha256, utc_now, write_json
from train_rqvae import make_model


def occupancy(counts):
    counts = np.asarray(counts,dtype=np.int64)
    total, active = int(counts.sum()), int((counts>0).sum())
    p = counts[counts>0]/total
    entropy = float(-(p*np.log(p)).sum())
    effective = float(np.exp(entropy))
    ranked = np.sort(counts)[::-1]
    return dict(items=total,codebook_size=len(counts),active_codes=active,
                dead_codes=int((counts==0).sum()),utilization=active/len(counts),
                effective_codes=effective,normalized_effective_codes=effective/len(counts),
                entropy_nats=entropy,top1_fraction=float(ranked[0]/total),
                top10_fraction=float(ranked[:10].sum()/total),
                rare_codes_1_to_5=int(((counts>=1)&(counts<=5)).sum()),
                dead_code_ids=np.flatnonzero(counts==0).tolist(),counts=counts.tolist())


@torch.inference_mode()
def audit_split(model,x,batch_size=512):
    model.eval()
    size,depth = model.codebook_size,model.n_layers
    counts = np.zeros((depth,size),dtype=np.int64)
    error, energy = np.zeros(depth),np.zeros(depth)
    reconstruction = 0.
    raw = []
    for batch in x.split(batch_size):
        q = model.get_semantic_ids(batch)
        ids = q.sem_ids.numpy()
        raw.append(ids)
        actual = torch.stack([layer.embedding(q.sem_ids[:,d]) for d,layer in enumerate(model.layers)],-1)
        for d in range(depth):
            counts[d] += np.bincount(ids[:,d],minlength=size)
            energy[d] += q.residuals[:,:,d].square().sum().item()
            error[d] += (q.residuals[:,:,d]-actual[:,:,d]).square().sum().item()
        restored = torch.nn.functional.normalize(model.decode(actual.sum(-1)),dim=-1)
        reconstruction += (restored-batch).square().sum().item()
    ids = np.concatenate(raw)
    _,groups = np.unique(ids,axis=0,return_counts=True)
    layers = [dict(layer=d+1,**occupancy(counts[d]),residual_energy=energy[d]/len(x),
                   quantization_error=error[d]/len(x),
                   relative_residual_error=error[d]/max(energy[d],1e-12)) for d in range(depth)]
    return dict(items=len(x),reconstruction_loss=reconstruction/len(x),layers=layers,
                raw_unique_ids=len(groups),extra_duplicate_fraction=(len(x)-len(groups))/len(x),
                items_in_collision_groups=int(groups[groups>1].sum()),max_collision_group=int(groups.max()))


def gradient_norm(loss,parameters):
    parameters = [p for p in parameters if p.requires_grad]
    if not parameters or not loss.requires_grad:
        return 0.
    grads = torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    return float(sum(g.square().sum().item() for g in grads if g is not None)**.5)


def audit_gradients(model,x):
    model.train()
    q = model.get_semantic_ids(x[:128])
    rows = []
    for d,layer in enumerate(model.layers):
        r = q.residuals[:,:,d]
        e = layer.embedding(q.sem_ids[:,d])
        codebook = (r.detach()-e).square().sum(-1).mean()
        commitment = (r-e.detach()).square().sum(-1).mean()
        rows.append(dict(layer=d+1,codebook_loss=codebook.item(),
                         commitment_loss_unweighted=commitment.item(),
                         codebook_loss_encoder_gradient=gradient_norm(codebook,model.encoder.parameters()),
                         codebook_loss_codebook_gradient=gradient_norm(codebook,layer.parameters()),
                         commitment_loss_encoder_gradient=gradient_norm(commitment,model.encoder.parameters()),
                         commitment_loss_codebook_gradient=gradient_norm(commitment,layer.parameters())))
    model.eval()
    return dict(scope='128 training vectors, isolated per-layer terms, no optimizer step',layers=rows,
                residual_graph='upstream: r_next=r-(r+stop_gradient(e-r)); later commitment encoder gradient cancels')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--embeddings',default=str(PROCESSED/'tiger_embeddings'))
    p.add_argument('--output',required=True)
    p.add_argument('--threads',type=int,default=2)
    args = p.parse_args()
    configure(threads=args.threads)
    saved = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    ep = Path(args.embeddings)
    if sha256(ep/'embeddings.npy') != saved['embedding_sha256']:
        raise ValueError('Embedding version differs from checkpoint')
    model = make_model(saved['config'])
    if 'ema_state_dict' in saved:
        from rqvae_ema import attach_ema
        attach_ema(model,**saved['ema_config'])
        model.load_state_dict(saved['ema_state_dict'])
    else:
        model.load_state_dict(saved['state_dict'])
    x = torch.from_numpy(np.load(ep/'embeddings.npy')[1:])
    parent = json.loads((Path(args.checkpoint).parent/'run.json').read_text(encoding='utf-8'))
    rng = np.random.default_rng(parent['arguments']['seed'])
    order = rng.permutation(len(x))
    nvalid = max(1,round(len(x)*.05))
    valid,train = x[order[:nvalid]],x[order[nvalid:]]
    report = dict(recorded_at_utc=utc_now(),checkpoint_epoch=saved['epoch'],
                  checkpoint_sha256=sha256(args.checkpoint),embedding_sha256=saved['embedding_sha256'],
                  update=parent.get('codebook_update','Adam + codebook loss'),
                  dead_definition='zero hard assignments in this frozen-checkpoint corpus; not never updated during training',
                  full_catalog=audit_split(model,x),train=audit_split(model,train),validation=audit_split(model,valid),
                  gradients=audit_gradients(model,train))
    write_json(args.output,report)
    for row in report['full_catalog']['layers']:
        print(f'layer={row["layer"]} used={row["active_codes"]}/{row["codebook_size"]} dead={row["dead_codes"]} '
              f'effective={row["effective_codes"]:.2f} top10={row["top10_fraction"]:.2%}',flush=True)
    print('Gradient routing:',report['gradients'],flush=True)


if __name__ == '__main__':
    main()
