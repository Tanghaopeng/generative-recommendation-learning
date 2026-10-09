"""EMA-only codebook updates; retains the pinned upstream residual STE graph."""
from __future__ import annotations

import torch
from torch import nn

import tiger_common
from modules.quantize import QuantizeOutput


class EmaQuantize(nn.Module):
    def __init__(self, initial_weight, commitment_weight=.25, decay=.99, epsilon=1e-5, initial_mass=1.):
        super().__init__()
        if not 0 < decay < 1 or epsilon <= 0 or initial_mass <= 0:
            raise ValueError('Invalid EMA hyperparameters')
        self.embedding = nn.Embedding.from_pretrained(initial_weight.detach().clone(), freeze=True)
        self.decay, self.epsilon = decay, epsilon
        self.commitment_weight = commitment_weight
        self.register_buffer('ema_count', torch.full((len(initial_weight),), initial_mass))
        self.register_buffer('ema_sum', initial_weight.detach().clone()*initial_mass)
        self.register_buffer('update_steps', torch.zeros((),dtype=torch.long))
        self.pending = None

    @property
    def weight(self):
        return self.embedding.weight

    def forward(self, x, temperature=None):
        weight = self.embedding.weight
        distance = x.square().sum(-1,keepdim=True) + weight.square().sum(-1) - 2*x@weight.T
        ids = distance.detach().argmin(-1)
        chosen = self.embedding(ids)
        # Exactly the upstream per-level STE. Residual-gradient changes are a
        # separate ablation, not silently bundled with EMA.
        output = x+(chosen-x).detach() if self.training else chosen
        commitment = self.commitment_weight*(x-chosen.detach()).square().sum(-1)
        if self.training:
            count = torch.bincount(ids,minlength=len(weight)).to(weight.dtype)
            sums = torch.zeros_like(weight).index_add_(0,ids,x.detach())
            self.pending = count, sums
        return QuantizeOutput(embeddings=output,ids=ids,loss=commitment)

    @torch.no_grad()
    def ema_step(self):
        if not self.training or self.pending is None:
            raise ValueError('EMA update requires pending training assignments')
        count, sums = self.pending
        self.ema_count.mul_(self.decay).add_(count,alpha=1-self.decay)
        self.ema_sum.mul_(self.decay).add_(sums,alpha=1-self.decay)
        # Preserve numerically depleted/unassigned centers; do not shrink them
        # to zero or confuse EMA with a dead-code replacement policy.
        valid = self.ema_count > self.epsilon
        self.embedding.weight[valid] = self.ema_sum[valid]/self.ema_count[valid,None]
        self.update_steps.add_(1)
        self.pending = None


def attach_ema(model, decay=.99, epsilon=1e-5, initial_mass=1.):
    model.layers = nn.ModuleList([EmaQuantize(layer.embedding.weight,
        commitment_weight=model.commitment_weight,decay=decay,epsilon=epsilon,
        initial_mass=initial_mass) for layer in model.layers])
    return model


def ema_step(model):
    for layer in model.layers:
        layer.ema_step()


def canonical_state(model, original_keys):
    """Eval-compatible weights; full EMA buffers are saved separately for resume."""
    return {key:value for key,value in model.state_dict().items() if key in original_keys}
