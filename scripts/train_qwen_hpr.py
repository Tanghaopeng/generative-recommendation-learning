"""Train small FP32 HPR heads from cached frozen SFT features; Qwen is not loaded."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch

from qwen_sft import ROOT, read_json, write_json, sha256
from qwen_hpr import (HPRConfig, HPRScorer, PrefixNegativeSampler, contrastive_loss,
                      load_feature_cache, save_scorer)


def fit_heads(cache, output, config, device='cpu', resume=False):
    """Normal optimizer loop. No policy gradients and no update to SFT weights."""
    cache, output = Path(cache), Path(output)
    manifest, arrays = load_feature_cache(cache)
    if (config['epochs'] < 1 or config['batch_size'] < 2 or config['catalog_negatives'] < 0
            or config['learning_rate'] <= 0 or config['early_stopping_patience'] < 1):
        raise ValueError('Invalid HPR training configuration')
    if output.exists() != resume:
        raise ValueError('Use a new output directory, or --resume an existing epoch checkpoint')
    if not manifest['train_rows'] or not manifest['valid_rows']:
        raise ValueError('Training and validation features must be nonempty')
    torch.manual_seed(config['seed'])
    architecture = HPRConfig(manifest['hidden_size'], config['projection_dim'], 3, config['initial_temperature'])
    scorer = HPRScorer(architecture).to(device).train()
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    table = torch.tensor(np.asarray(arrays['sid_embeddings']), device=device, dtype=torch.float32)
    sampler = PrefixNegativeSampler(arrays['catalog_sids'])
    provenance = dict(checkpoint_weight_sha256=manifest['checkpoint_weight_sha256'],
        checkpoint_metadata_sha256=manifest['checkpoint_metadata_sha256'],
        prepared_manifest_sha256=manifest['prepared_manifest_sha256'],
        feature_manifest_sha256=sha256(cache / 'manifest.json'),
        source_sft_code_sha256=manifest['source_sft_code_sha256'],
        prompt_preparation_code_sha256=manifest['code_sha256']['qwen_eval_fast.py'],
        max_history=manifest['max_history'], max_length=manifest['max_length'],
        diagnostic_only=manifest['diagnostic_only'], qwen_frozen=True,
        negative_policy='multi-positive in-batch + same-parent catalog alternatives/global fallback')
    contract = dict(config=config, provenance=provenance, architecture=asdict(architecture),
                    code_sha256={name: sha256(Path(__file__).with_name(name)) for name in
                                 ('qwen_hpr.py', 'train_qwen_hpr.py')})
    start_epoch, best, bad_epochs, history = 0, float('inf'), 0, []
    if resume:
        if read_json(output / 'contract.json') != contract:
            raise ValueError('Resume contract differs: data, SFT features, code or config changed')
        state = torch.load(output / 'last.pt', map_location=device, weights_only=True)
        scorer.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch, best, bad_epochs, history = state['epoch'], state['best_loss'], state['bad_epochs'], state['history']
    else:
        output.mkdir(parents=True)
        write_json(output / 'contract.json', contract)
    started = time.perf_counter()
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()

    def batches(split, order):
        for start in range(0, len(order), config['batch_size']):
            selected = order[start:start + config['batch_size']]
            contexts = torch.tensor(arrays[split + '_contexts'][selected], device=device, dtype=torch.float32)
            targets = torch.tensor(arrays[split + '_targets'][selected], device=device, dtype=torch.long)
            yield contexts, targets

    for epoch in range(start_epoch, config['epochs']):
        if bad_epochs >= config['early_stopping_patience']:
            break
        scorer.train()
        rng = np.random.default_rng(config['seed'] + epoch)
        order = np.random.default_rng(config['seed'] + epoch).permutation(manifest['train_rows'])
        total, count, norms, first_log = 0.0, 0, [], None
        for contexts, targets in batches('train', order):
            optimizer.zero_grad(set_to_none=True)
            loss, log = contrastive_loss(scorer, contexts, targets, table, sampler, config['catalog_negatives'], rng)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite HPR loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(scorer.parameters(), config['max_grad_norm'])
            if not torch.isfinite(norm):
                raise RuntimeError('Non-finite HPR gradient')
            optimizer.step()
            total += float(loss.detach()) * len(contexts)
            count += len(contexts)
            norms.append(float(norm))
            first_log = log if first_log is None else first_log
        scorer.eval()
        val_total, val_count = 0.0, 0
        # Fixed ordering/negatives make validation contrastive loss comparable across epochs.
        rng = np.random.default_rng(config['seed'] + 1000000)
        with torch.no_grad():
            for contexts, targets in batches('valid', np.arange(manifest['valid_rows'])):
                loss, _ = contrastive_loss(scorer, contexts, targets, table, sampler, config['catalog_negatives'], rng)
                val_total += float(loss) * len(contexts)
                val_count += len(contexts)
        val_loss = val_total / val_count
        if not np.isfinite(val_loss):
            raise RuntimeError('Non-finite HPR validation loss')
        log = dict(epoch=epoch + 1, train_loss=total / count, valid_contrastive_loss=val_loss,
                   mean_gradient_norm=float(np.mean(norms)), max_gradient_norm=max(norms),
                   first_batch_diagnostics=first_log)
        history.append(log)
        print(log, flush=True)
        if val_loss < best:
            best, bad_epochs = val_loss, 0
            save_scorer(output / 'selected', scorer, dict(provenance, head_selection='validation contrastive loss',
                                                        epoch=epoch + 1, valid_contrastive_loss=val_loss))
        else:
            bad_epochs += 1
        # Exact restart at completed epoch boundaries; interrupted partial epochs are replayed.
        state = dict(epoch=epoch + 1, model=scorer.state_dict(), optimizer=optimizer.state_dict(),
                     best_loss=best, bad_epochs=bad_epochs, history=history)
        torch.save(state, output / 'last.pt.partial')
        (output / 'last.pt.partial').replace(output / 'last.pt')
        write_json(output / 'history.json', history)
    result = dict(status='hpr_heads_trained_requires_retrieval_validation',
                  qwen_loaded=False, qwen_updated=False, parameters=sum(p.numel() for p in scorer.parameters()),
                  precision='fp32_heads; frozen cached features from BF16 Qwen',
                  best_valid_contrastive_loss=best, completed_epochs=len(history), seconds=time.perf_counter() - started,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,
                  diagnostic_only=manifest['diagnostic_only'], test_evaluated=False)
    write_json(output / 'run.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT / 'experiments/qwen_hpr_v1.json')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    print(fit_heads(args.cache, args.output, read_json(args.config), args.device, args.resume), flush=True)


if __name__ == '__main__':
    main()
