"""Local full-SID adapter around the unchanged third-party T5 implementation."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

import tiger_common  # Adds the pinned vendor package to sys.path.
from modules.model import EncoderDecoderRetrievalModel


class PrefixIndex:
    def __init__(self, sid, vocabulary):
        children, leaves, smallest = [{}], [0], [len(sid)]
        for item, code in enumerate(sid[1:], 1):
            node = 0
            for token in code:
                smallest[node] = min(smallest[node], item)
                token = int(token)
                if token not in children[node]:
                    children[node][token] = len(children)
                    children.append({})
                    leaves.append(0)
                    smallest.append(len(sid))
                node = children[node][token]
            if leaves[node]:
                raise ValueError('Duplicate full SID')
            leaves[node], smallest[node] = item, item
        edges = np.full((len(children), vocabulary), -1, dtype=np.int64)
        for node, mapping in enumerate(children):
            for token, child in mapping.items():
                edges[node, token] = child
        self.edges = torch.from_numpy(edges)
        self.leaves = torch.tensor(leaves)
        self.smallest = torch.tensor(smallest)


class TigerModel(EncoderDecoderRetrievalModel):
    def __init__(self, sid, config):
        super().__init__(codebooks=torch.as_tensor(sid[1:], dtype=torch.long),
                         num_hierarchies=4, num_embeddings_per_hierarchy=config['vocabulary'],
                         t5_d_model=config['hidden'], t5_num_heads=config['heads'],
                         t5_d_ff=config['ff'], t5_num_layers=config['layers'],
                         should_add_sep_token=False, num_user_bins=None)
        self.index = PrefixIndex(sid, config['vocabulary'])
        self.config_local = config

    def logits(self, tokens, mask, targets):
        enc, enc_mask = self.encoder_forward_pass(mask.long(), tokens)
        # Input BOS,a,b,c predicts a,b,c,d. No target d is fed to this pass.
        dec = self.decoder_forward_pass(future_ids=targets[:, :-1], encoder_output=enc,
                                        attention_mask_for_encoder=enc_mask, use_cache=False)
        return torch.stack([self.decoder_mlp[h](dec[:, h]) for h in range(4)], dim=1)

    def forward(self, tokens, mask, targets):
        logits = self.logits(tokens, mask, targets)
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())

    @torch.inference_mode()
    def retrieve(self, tokens, mask, excluded, beam=20, topk=20):
        """Deterministic trie-constrained beam; raw conditional log probabilities.

        Masking happens AFTER log_softmax: illegal branches are not renormalized.
        Complete observed histories are excluded at the leaf, never held-out targets.
        A finite beam is approximate search over the full catalog, not exact scoring.
        """
        if self.training:
            raise ValueError('Call eval() before retrieval')
        if beam < topk:
            raise ValueError('Beam must cover requested top-k')
        batch = len(tokens)
        enc, enc_mask = self.encoder_forward_pass(mask.long(), tokens)
        nodes = torch.zeros((batch, 1), dtype=torch.long)
        scores = torch.zeros((batch, 1))
        prefixes = torch.empty((batch, 1, 0), dtype=torch.long)
        cache = EncoderDecoderCache(DynamicCache(), DynamicCache())
        for depth in range(4):
            width = nodes.shape[1]
            expanded_enc = enc.repeat_interleave(width, 0)
            expanded_mask = enc_mask.repeat_interleave(width, 0)
            dec, cache = self.decoder_forward_pass(
                future_ids=prefixes.flatten(0, 1) if depth else None,
                encoder_output=expanded_enc, attention_mask_for_encoder=expanded_mask,
                use_cache=True, past_key_values=cache)
            logp = F.log_softmax(self.decoder_mlp[depth](dec[:, -1]), dim=-1)
            if not torch.isfinite(logp).all():
                raise FloatingPointError('Non-finite generation probabilities')
            logp = logp.reshape(batch, width, -1)
            next_nodes = self.index.edges[nodes.clamp_min(0)]
            allowed = (next_nodes >= 0) & (nodes.unsqueeze(-1) >= 0)
            if depth == 3:
                candidates = self.index.leaves[next_nodes.clamp_min(0)]
                for row, history in enumerate(excluded):
                    if history:
                        allowed[row] &= ~torch.isin(candidates[row], torch.tensor(sorted(history)))
            cumulative = (scores.unsqueeze(-1) + logp).masked_fill(~allowed, -torch.inf)
            flat_scores = cumulative.flatten(1)
            flat_nodes = next_nodes.flatten(1)
            # Tie rule: lowest reachable item ID, then stable prefix order.
            ids = self.index.smallest[flat_nodes.clamp_min(0)]
            tie_order = torch.argsort(ids, dim=1, stable=True)
            order = torch.argsort(flat_scores.gather(1, tie_order), dim=1,
                                  descending=True, stable=True)
            chosen = tie_order.gather(1, order[:, :min(beam, flat_scores.shape[1])])
            new_scores = flat_scores.gather(1, chosen)
            parent = chosen // self.num_embeddings_per_hierarchy
            token = chosen % self.num_embeddings_per_hierarchy
            prefixes = torch.cat([prefixes.gather(1, parent.unsqueeze(-1).expand(-1, -1, depth)),
                                  token.unsqueeze(-1)], dim=-1)
            nodes = flat_nodes.gather(1, chosen).masked_fill(~torch.isfinite(new_scores), -1)
            scores = new_scores
            parent_global = (parent + torch.arange(batch).unsqueeze(1) * width).flatten()
            cache.reorder_cache(parent_global)
        result = self.index.leaves[nodes.clamp_min(0)].masked_fill(nodes < 0, 0)
        return result[:, :topk], scores[:, :topk]


def pack_histories(histories, sid, maxlen=20):
    cropped = [list(history)[-maxlen:] for history in histories]
    if not cropped or any(not h for h in cropped):
        raise ValueError('At least one valid history item is required')
    length = max(map(len, cropped)) * 4
    tokens = np.zeros((len(cropped), length), dtype=np.int64)
    mask = np.zeros_like(tokens, dtype=bool)
    for row, history in enumerate(cropped):
        code = sid[history].reshape(-1)
        tokens[row, :len(code)] = code
        mask[row, :len(code)] = True
    return torch.from_numpy(tokens), torch.from_numpy(mask)


def training_examples(rows, maxlen=20):
    """The same positive positions and visible tail window as SASRec training."""
    examples = []
    for row_index, row in enumerate(rows):
        train = row['train']
        start = max(0, len(train) - 1 - maxlen)
        examples.extend((row_index, start, target) for target in range(start + 1, len(train)))
    return np.asarray(examples, dtype=np.int64)


def training_batch(examples, rows, sid, maxlen=20):
    histories = [rows[r]['train'][start:target] for r, start, target in examples]
    targets = torch.from_numpy(np.stack([sid[rows[r]['train'][target]] for r, _, target in examples]))
    tokens, mask = pack_histories(histories, sid, maxlen)
    return tokens, mask, targets
