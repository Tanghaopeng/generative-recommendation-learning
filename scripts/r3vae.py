"""R3-VAE-inspired codec; independently written adapter with pinned provenance.

Default rating follows wwqq/R3-VAE@87f1c0c: normalized residual/codebook,
top1 cosine-weighted codeword, no STE. The optional paper_softmax mode follows
paper equations 3-7 and is explicitly a separate quantization ablation.
Training SC/PD are the public code's proxies, NOT hard-cluster offline metrics.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits


REFERENCE = dict(repository='https://github.com/wwqq/R3-VAE',
                 commit='87f1c0c21cecd16a1c44870dabe29d970475b5e1',
                 paper='https://arxiv.org/html/2604.11440v3',
                 implementation='independently written local adaptation, not verbatim upstream files')


def pd_codebook_loss(codebook, temperature=2.):
    if len(codebook) < 2:
        raise ValueError('PD requires at least two codewords')
    unit = F.normalize(codebook, dim=-1)
    indices = torch.triu_indices(len(unit), len(unit), offset=1, device=unit.device)
    similarities = (unit[indices[0]] * unit[indices[1]]).sum(-1).clamp(-1, 1)
    return torch.logsumexp(-temperature * (1 - similarities), dim=0) - np.log(len(similarities))


def mlp(dimensions):
    layers = []
    for depth, (left, right) in enumerate(zip(dimensions, dimensions[1:])):
        layers.append(nn.Linear(left, right, bias=False))
        if depth < len(dimensions) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class R3VAE(nn.Module):
    def __init__(self, input_dim=384, latent_dim=32, hidden_dims=(256,128),
                 codebook_size=256, layers=3, rating_mode='official_top1'):
        super().__init__()
        if min(input_dim, latent_dim, codebook_size) < 2 or layers != 3:
            raise ValueError('This SID adapter requires three layers and nontrivial dimensions')
        if rating_mode not in ('official_top1','paper_softmax'):
            raise ValueError('Unknown rating mode')
        self.rating_mode, self.codebook_size, self.layers = rating_mode, codebook_size, layers
        self.encoder = mlp([input_dim, *hidden_dims, latent_dim])
        self.decoder = mlp([latent_dim, *reversed(hidden_dims), input_dim])
        self.reference = nn.Parameter(torch.empty(1, latent_dim).uniform_(-1,1))
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.empty(codebook_size, latent_dim).uniform_(-1/(d+1),1/(d+1)))
            for d in range(layers)])
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def initial_residual(self, encoded):
        encoded = F.normalize(encoded, dim=-1)
        reference = F.normalize(self.reference, dim=-1)
        anchor = (encoded @ reference.T) * reference
        return encoded - anchor, anchor

    def rate(self, residual, codebook):
        unit_codes = F.normalize(codebook, dim=-1)
        scores = F.normalize(residual, dim=-1) @ unit_codes.T
        selected = scores.argmax(-1)
        if self.rating_mode == 'official_top1':
            residual = F.normalize(residual, dim=-1)
            values = scores.gather(1, selected[:, None])
            output = values * unit_codes[selected]
        else:
            output = scores.softmax(-1) @ codebook
        return residual - output, output, selected

    def quantize_encoded(self, encoded):
        residual, total = self.initial_residual(encoded)
        initial = residual
        ids, sc_terms, pd_terms = [], [], []
        for codebook in self.codebooks:
            residual, output, selected = self.rate(residual, codebook)
            total = total + output
            ids.append(selected)
            # The public implementation compares the initial residual to the
            # accumulated representation including its reference contribution.
            sc_terms.append(1 - F.cosine_similarity(initial, total, dim=-1).mean())
            pd_terms.append(pd_codebook_loss(codebook))
        return dict(quantized=total, ids=torch.stack(ids, dim=1),
                    sc_proxy_loss=torch.stack(sc_terms).mean(),
                    pd_proxy_loss=torch.stack(pd_terms).mean())

    def forward(self, x, sc_weight=0., pd_weight=0.):
        encoded = self.encoder(F.normalize(x, dim=-1))
        output = self.quantize_encoded(encoded)
        decoded = self.decoder(output['quantized'])
        reconstruction = 1 - F.cosine_similarity(decoded, x, dim=-1).mean()
        output.update(reconstructed=decoded, reconstruction_loss=reconstruction,
                      loss=reconstruction + sc_weight * output['sc_proxy_loss'] + pd_weight * output['pd_proxy_loss'])
        return output

    @torch.inference_mode()
    def initialize_kmeans(self, x, seed=2026):
        if len(x) < self.codebook_size:
            raise ValueError('Need at least as many initialization items as codewords')
        self.eval()
        residual, _ = self.initial_residual(self.encoder(F.normalize(x, dim=-1)))
        for depth, codebook in enumerate(self.codebooks):
            training_vectors = F.normalize(residual, dim=-1) if self.rating_mode == 'official_top1' else residual
            km = MiniBatchKMeans(n_clusters=self.codebook_size, n_init=1, max_iter=100,
                                batch_size=min(512, len(x)), random_state=seed + depth,
                                reassignment_ratio=.01)
            with threadpool_limits(limits=2):
                km.fit(training_vectors.cpu().numpy())
            codebook.copy_(torch.from_numpy(km.cluster_centers_).to(codebook))
            residual, _, _ = self.rate(residual, codebook)
