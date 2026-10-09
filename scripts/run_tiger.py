"""CPU runner: shared SASRec split, upstream T5, full unique semantic IDs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import psutil
import torch

from tiger_common import (PROCESSED, configure, sequences, load_sid, sha256, utc_now,
                          write_json, atomic_checkpoint, random_state, restore_random, source_hashes)
from tiger_model import TigerModel, training_examples, training_batch, pack_histories


@torch.inference_mode()
def evaluate(model, rows, sid, split, batch_size=32, beam=20, maxlen=20, limit=0):
    model.eval()
    evaluated = rows[:limit] if limit else rows
    hits = {10: 0, 20: 0}
    ndcg = {10: 0., 20: 0.}
    shortfalls, generated, times = 0, 0, []
    started = time.perf_counter()
    for start in range(0, len(evaluated), batch_size):
        batch = evaluated[start:start + batch_size]
        histories = [r['train'] + (r['valid'] if split == 'test' else []) for r in batch]
        targets = [r[split][0] for r in batch]
        if any(target in history for target, history in zip(targets, histories)):
            raise ValueError('Held-out target overlaps observed history')
        tokens, mask = pack_histories(histories, sid, maxlen)
        before = time.perf_counter()
        result, _ = model.retrieve(tokens, mask, [set(h) for h in histories], beam=beam, topk=20)
        times.append(time.perf_counter() - before)
        for predicted, target, history in zip(result.tolist(), targets, histories):
            nonzero = [i for i in predicted if i]
            if len(set(nonzero)) != len(nonzero) or any(i in history for i in nonzero):
                raise ValueError('Duplicate or observed item in retrieval output')
            shortfalls += len(nonzero) < 20
            generated += len(nonzero)
            rank = predicted.index(target) + 1 if target in predicted else len(sid)
            for k in (10, 20):
                if rank <= k:
                    hits[k] += 1
                    ndcg[k] += 1 / np.log2(rank + 1)
        if start == 0 or (start // batch_size) % 100 == 0:
            print(f'  {split}: {min(start+batch_size,len(evaluated))}/{len(evaluated)} users', flush=True)
    n = len(evaluated)
    result = {f'{name}@{k}': float((hits if name == 'Recall' else ndcg)[k] / n)
              for k in (10, 20) for name in ('Recall', 'NDCG')}
    result.update(users=n, seconds=time.perf_counter()-started, beam=beam,
                  shortfall_users=shortfalls, returned_items=generated,
                  legal_item_fraction=1.0 if generated else 0.,
                  warm_batch_seconds_p50=float(np.median(times[1:] or times)),
                  warm_batch_seconds_p95=float(np.quantile(times[1:] or times, .95)),
                  timing_batch_size=batch_size, smoke_only=bool(limit))
    return result


def batch_order(examples, rng, batch_size):
    # Length buckets reduce padding; every positive position is visited once.
    order = rng.permutation(len(examples))
    lengths = examples[order, 2] - examples[order, 1]
    order = order[np.argsort(lengths, kind='stable')]
    batches = [order[i:i+batch_size] for i in range(0, len(order), batch_size)]
    return [batches[i] for i in rng.permutation(len(batches))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sids', default=str(PROCESSED/'tiger_sids'))
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--eval-batch-size', type=int, default=32)
    p.add_argument('--maxlen', type=int, default=20)
    p.add_argument('--beam', type=int, default=20)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--ff', type=int, default=256)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--patience', type=int, default=2)
    p.add_argument('--resume')
    p.add_argument('--smoke-steps', type=int, default=0,
                   help='Separate benchmark artifact, not a formal experiment')
    p.add_argument('--smoke-eval-users', type=int, default=64)
    args = p.parse_args()
    configure(args.seed, args.threads)
    if args.beam < 20 or args.epochs < 1 or args.maxlen < 1:
        raise ValueError('Positive epochs/history length and beam >=20 are required')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    sid, sid_info = load_sid(args.sids)
    rows, stats = sequences()
    if stats['items'] != len(sid)-1:
        raise ValueError('Catalog sizes differ')
    examples = training_examples(rows, args.maxlen)
    config = dict(vocabulary=sid_info['tokens_per_hierarchy'], hidden=args.hidden,
                  heads=args.heads, layers=args.layers, ff=args.ff)
    model = TigerModel(sid, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    best, best_epoch, best_loss, streak, start_epoch, history = -1., 0, None, 0, 1, []
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if (saved['config'] != config or saved['sid_sha256'] != sid_info['sid_sha256']
                or saved['maxlen'] != args.maxlen or saved['smoke_only']):
            raise ValueError('Resume contract differs or is a smoke artifact')
        parent = json.loads((Path(args.resume).parent/'run.json').read_text(encoding='utf-8'))
        for key in ('batch_size', 'seed', 'beam', 'lr', 'patience'):
            if parent['arguments'][key] != vars(args)[key]:
                raise ValueError(f'Resume configuration changed: {key}')
        if saved['source_sha256'] != source_hashes():
            raise ValueError('Source changed since checkpoint; use an explicit new experiment')
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        restore_random(saved['random_state'], rng)
        best, best_epoch, best_loss, streak = (saved[k] for k in ('best_metric','best_epoch','best_loss','streak'))
        start_epoch = saved['epoch']+1
        history = [r for r in parent['history'] if r['epoch'] <= saved['epoch']]
        if args.epochs < start_epoch:
            raise ValueError('Requested total epochs already reached')
        parent_best = torch.load(Path(args.resume).parent/'best.pth',map_location='cpu',weights_only=False)
        if parent_best['epoch'] != best_epoch:
            raise ValueError('The matching best checkpoint is unavailable for this resume point')
        shutil.copy2(Path(args.resume).parent/'best.pth', out/'best.pth')
    fingerprints = source_hashes()
    run = dict(status='running', started_at_utc=utc_now(), arguments=vars(args), config=config,
               process_id=os.getpid(),parent_process_id=os.getppid(),
               parameters=sum(p.numel() for p in model.parameters()), training_examples=len(examples),
               data_sha256=stats['sasrec_input_sha256'], sequence_sha256=sha256(PROCESSED/'sequences.jsonl'),
               sid_manifest=sid_info, source_sha256=fingerprints, history=history,
               smoke_only=bool(args.smoke_steps), evaluation='full catalog trie, deterministic finite beam; history excluded',
               checkpoint_selection='validation NDCG@10 only',
               training='all effective SASRec positive positions once/epoch, length buckets, four-token CE mean',
               negative_sampling='none; per-hierarchy vocabulary softmax', stop_rule='two consecutive lower validation NDCG@10 while loss below validation-best epoch')
    write_json(out/'run.json', run)
    started, train_steps = time.perf_counter(), 0
    for epoch in range(start_epoch, args.epochs+1):
        model.train()
        epoch_start, loss_sum, n = time.perf_counter(), 0., 0
        batches = batch_order(examples, rng, args.batch_size)
        for step, indices in enumerate(batches):
            if args.smoke_steps and train_steps >= args.smoke_steps:
                break
            tokens, mask, targets = training_batch(examples[indices], rows, sid, args.maxlen)
            loss = model(tokens, mask, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            loss_sum += loss.item()*len(indices)
            n += len(indices)
            train_steps += 1
            if step == 0 or step % 100 == 0:
                run.update(current_epoch=epoch, current_batch=step+1, epoch_batches=len(batches),
                           current_loss=loss_sum/n, peak_observed_rss_bytes=psutil.Process().memory_info().rss)
                write_json(out/'run.json', run)
                print(f'tiger epoch={epoch} batch={step+1}/{len(batches)} loss={loss_sum/n:.5f} '
                      f'seconds={time.perf_counter()-epoch_start:.1f}', flush=True)
        training_seconds = time.perf_counter()-epoch_start
        valid = evaluate(model, rows, sid, 'valid', args.eval_batch_size, args.beam, args.maxlen,
                         args.smoke_eval_users if args.smoke_steps else 0)
        average_loss = loss_sum/n
        metric = valid['NDCG@10']
        if metric > best:
            best, best_epoch, best_loss, streak = metric, epoch, average_loss, 0
            improved = True
        else:
            streak = streak+1 if metric < best and average_loss < best_loss else 0
            improved = False
        history.append(dict(epoch=epoch, training_loss=average_loss, training_examples=n,
                            training_seconds=training_seconds, validation=valid,
                            consecutive_overfit_warnings=streak))
        state = dict(state_dict=model.state_dict(), config=config, optimizer=optimizer.state_dict(),
                     random_state=random_state(rng), epoch=epoch, maxlen=args.maxlen,
                     sid_sha256=sid_info['sid_sha256'], source_sha256=fingerprints,
                     smoke_only=bool(args.smoke_steps), best_metric=best, best_epoch=best_epoch,
                     best_loss=best_loss, streak=streak)
        atomic_checkpoint(out/'last.pth', state)
        if improved:
            atomic_checkpoint(out/'best.pth', state)
        run.update(history=history, best_epoch=best_epoch, best_validation_ndcg10=best)
        write_json(out/'run.json', run)
        print(f'tiger epoch={epoch} validation={valid} best_epoch={best_epoch}', flush=True)
        if args.smoke_steps or streak >= args.patience:
            run['stop_reason'] = 'smoke_benchmark' if args.smoke_steps else 'overfit_warning'
            break
    run['stop_reason'] = run.get('stop_reason', 'requested_epochs_completed')
    if not args.smoke_steps:
        saved = torch.load(out/'best.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(saved['state_dict'])
        run['test'] = evaluate(model, rows, sid, 'test', args.eval_batch_size, args.beam, args.maxlen)
    run.update(status='completed', completed_at_utc=utc_now(), elapsed_seconds=time.perf_counter()-started,
               checkpoint_sha256=sha256(out/'best.pth'))
    write_json(out/'run.json', run)
    print('Completed TIGER run. Smoke artifacts are not formal recommendation results.', flush=True)


if __name__ == '__main__':
    main()
