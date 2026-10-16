"""Hard-assignment codebook perplexity: exp(Shannon entropy of code frequencies).

Definition used in DeepMind Sonnet VectorQuantizer / VectorQuantizerEMA:
https://github.com/google-deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
This implementation counts unique catalog products, not batches or interactions.
"""
import numpy as np


def codebook_usage(counts):
    counts = np.asarray(counts)
    if (counts.ndim != 1 or not len(counts) or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts < 0) or counts.sum() <= 0):
        raise ValueError('Expected nonempty nonnegative integer counts with positive sum')
    total = int(counts.sum())
    p = counts[counts > 0].astype(np.float64) / total
    entropy = float(-(p * np.log(p)).sum())
    effective = float(np.exp(entropy))
    ranked = np.sort(counts)[::-1]
    active = int((counts > 0).sum())
    return dict(items=total, codebook_size=len(counts), active_codes=active,
        dead_codes=len(counts)-active, utilization=active/len(counts),
        entropy_nats=entropy, codebook_perplexity=effective, effective_codes=effective,
        normalized_effective_codes=effective/len(counts),
        top1_fraction=float(ranked[0]/total), top10_fraction=float(ranked[:10].sum()/total),
        counts=counts.tolist())


def semantic_usage(codes, codebook_size=256):
    codes = np.asarray(codes)
    if (codes.ndim != 2 or codes.shape[1] != 3 or not len(codes)
        or not np.issubdtype(codes.dtype, np.integer)
        or np.any(codes < 0) or np.any(codes >= codebook_size)):
        raise ValueError('Expected three semantic segments without padding or collision counter')
    return [dict(layer=d+1, **codebook_usage(np.bincount(codes[:,d],minlength=codebook_size)))
            for d in range(3)]
