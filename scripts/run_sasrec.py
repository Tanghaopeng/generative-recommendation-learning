"""Reproducible SASRec runner using the unchanged pmixer model.

Training follows the upstream shifted targets, uniform negative sampling and
BCE loss. This entry point uses one deterministic user pass per epoch, avoids
Windows multiprocessing, and evaluates the full catalog using validation only
for checkpoint selection. It is not an exact reproduction of upstream results.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "baselines" / "sasrec"
sys.path.insert(0, str(MODEL_DIR))
from model import SASRec  # noqa: E402
from utils import data_partition  # noqa: E402


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_data():
    stats = json.loads((ROOT / "experiments/data_stats.json").read_text(encoding="utf-8"))
    source = MODEL_DIR / "data/office2018.txt"
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    if checksum != stats["sasrec_input_sha256"]:
        raise ValueError("SASRec input checksum differs from verified data")
    old_cwd = Path.cwd()
    try:
        os.chdir(MODEL_DIR)
        train, valid, test, users, items = data_partition("office2018")
    finally:
        os.chdir(old_cwd)
    assert users == stats["users"] and items == stats["items"]
    assert all(len(train[u]) >= 2 and len(valid[u]) == len(test[u]) == 1 for u in train)
    return train, valid, test, users, items, checksum


def training_arrays(train, maxlen):
    users = np.array(sorted(train), dtype=np.int64)
    seq = np.zeros((len(users), maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    for row, uid in enumerate(users):
        history = train[int(uid)]
        inputs, targets = history[:-1][-maxlen:], history[1:][-maxlen:]
        seq[row, -len(inputs):] = inputs
        pos[row, -len(targets):] = targets
    return users, seq, pos


def training_keys(train, itemnum):
    # Membership key: user * (catalog size + 1) + item. No held-out data is used.
    return np.sort(np.fromiter((u * (itemnum + 1) + i for u, ids in train.items()
                               for i in ids), dtype=np.int64))


def sample_negatives(users, pos, itemnum, known_keys, rng):
    neg = np.zeros_like(pos)
    rows, cols = np.nonzero(pos)
    base = users[rows] * (itemnum + 1)
    choices = rng.integers(1, itemnum + 1, size=len(rows))
    while True:
        keys = base + choices
        indices = np.searchsorted(known_keys, keys)
        rejected = (indices < len(known_keys))
        rejected[rejected] = known_keys[indices[rejected]] == keys[rejected]
        if not rejected.any():
            break
        choices[rejected] = rng.integers(1, itemnum + 1, size=int(rejected.sum()))
    neg[rows, cols] = choices
    return neg


def padded_histories(histories, maxlen):
    seq = np.zeros((len(histories), maxlen), dtype=np.int64)
    for row, history in enumerate(histories):
        tail = history[-maxlen:]
        if tail:
            seq[row, -len(tail):] = tail
    return seq


def target_ranks(scores, targets, histories):
    """Exact 1-based ranks, excluding known history, ties by smaller item ID."""
    targets = torch.as_tensor(targets, dtype=torch.long, device=scores.device)
    rows = torch.arange(len(targets), device=scores.device)
    target_scores = scores[rows, targets].clone()
    if not torch.isfinite(target_scores).all():
        raise ValueError("Non-finite target scores")
    scores[:, 0] = -torch.inf
    mask_rows, mask_items = [], []
    for row, history in enumerate(histories):
        if int(targets[row]) in history:
            raise ValueError("Held-out target overlaps known history")
        mask_rows.extend([row] * len(history))
        mask_items.extend(history)
    if mask_items:
        scores[mask_rows, mask_items] = -torch.inf
    item_ids = torch.arange(scores.shape[1], device=scores.device)
    better = (scores > target_scores[:, None]).sum(dim=1)
    tied_before = ((scores == target_scores[:, None]) & (item_ids < targets[:, None])).sum(dim=1)
    return (1 + better + tied_before).cpu().numpy()


def metrics_from_ranks(ranks):
    return {f"{metric}@{k}": float(np.mean((ranks <= k) if metric == "Recall" else
             np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0)))
            for k in (10, 20) for metric in ("Recall", "NDCG")}


class OverfitMonitor:
    """Pause on sustained worse validation while fitting training more closely.

    This is a practical warning rule, not a proof of overfitting. Comparing to
    the training loss at the validation-best epoch avoids treating one noisy
    batch/epoch loss increase as a recovery of generalization.
    """
    def __init__(self, best_metric=-math.inf, best_loss=None, patience=2):
        self.best_metric = best_metric
        self.best_loss = best_loss
        self.patience = patience
        self.streak = 0

    def observe(self, metric, loss):
        if metric > self.best_metric:
            self.best_metric, self.best_loss, self.streak = metric, loss, 0
        validation_worse = metric < self.best_metric
        training_better = self.best_loss is not None and loss < self.best_loss
        if validation_worse and training_better:
            self.streak += 1
        else:
            self.streak = 0
        return {"validation_below_best": validation_worse,
                "training_loss_below_validation_best_epoch": training_better,
                "consecutive_warnings": self.streak,
                "patience": self.patience,
                "best_validation_ndcg10": self.best_metric,
                "training_loss_at_validation_best_epoch": self.best_loss,
                "pause": self.streak >= self.patience}


@torch.inference_mode()
def evaluate(model, train, valid, test, split, users, maxlen, batch_size, popularity=None):
    started = time.perf_counter()
    ranks = []
    device = next(model.parameters()).device
    model.eval()
    for start in range(0, len(users), batch_size):
        batch_users = users[start:start + batch_size]
        histories = [train[int(u)] + (valid[int(u)] if split == "test" else []) for u in batch_users]
        targets = [(test if split == "test" else valid)[int(u)][0] for u in batch_users]
        if popularity is None:
            feats = model.log2feats(padded_histories(histories, maxlen))[:, -1, :]
            scores = feats @ model.item_emb.weight.T
        else:
            scores = popularity.to(device).expand(len(batch_users), -1).clone()
        ranks.append(target_ranks(scores, targets, histories))
        if start == 0 or (start // batch_size) % 100 == 0:
            print(f"  {split}: {min(start + batch_size, len(users))}/{len(users)} users", flush=True)
    result = metrics_from_ranks(np.concatenate(ranks))
    result.update(users=len(users), seconds=round(time.perf_counter() - started, 3))
    return result


def initialize_model(usernum, itemnum, config):
    model = SASRec(usernum, itemnum, SimpleNamespace(**config)).to(config["device"])
    for name, param in model.named_parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_normal_(param)
    with torch.no_grad():
        model.pos_emb.weight[0].zero_()
        model.item_emb.weight[0].zero_()
    return model


def checkpoint(path, model, config, users, items, epoch, validation, checksum, optimizer=None, rng=None):
    temporary = path.with_suffix(".tmp")
    saved = {"state_dict": model.state_dict(), "config": config,
                "usernum": users, "itemnum": items, "epoch": epoch,
                "validation": validation, "data_sha256": checksum}
    if optimizer is not None and rng is not None:
        saved["training_state"] = {"optimizer": optimizer.state_dict(),
                                   "numpy_rng": rng.bit_generator.state,
                                   "torch_rng": torch.get_rng_state(),
                                   "python_rng": random.getstate(),
                                   "cuda_rng": torch.cuda.get_rng_state_all() if config["device"] == "cuda" else []}
    torch.save(saved, temporary)
    temporary.replace(path)


def restore_training_state(saved, optimizer, rng):
    state = saved.get("training_state")
    if state is None:
        return False
    optimizer.load_state_dict(state["optimizer"])
    rng.bit_generator.state = state["numpy_rng"]
    torch.set_rng_state(state["torch_rng"].cpu())
    random.setstate(state["python_rng"])
    if state["cuda_rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    return True


def restore_model(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=True)
    config = {**saved["config"], "device": device}
    model = SASRec(saved["usernum"], saved["itemnum"], SimpleNamespace(**config)).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved


@torch.inference_mode()
def recommend(model, history, maxlen, k=10):
    if not history:
        raise ValueError("SASRec requires history; use popularity for cold-start users")
    if any(i < 1 or i > model.item_num for i in history):
        raise ValueError("Unknown item ID")
    feats = model.log2feats(padded_histories([history], maxlen))[:, -1, :]
    scores = (feats @ model.item_emb.weight.T)[0]
    scores[[0] + history] = -torch.inf
    # Stable descending order keeps item ID ascending for equal scores.
    ids = torch.argsort(scores, descending=True, stable=True)[:k]
    return [(int(i), float(scores[i])) for i in ids if torch.isfinite(scores[i])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Resume an epoch checkpoint in a new output directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--maxlen", type=int, default=50)
    parser.add_argument("--hidden-units", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--patience", type=int, default=3, help="Validation checks without improvement")
    parser.add_argument("--stop-on-overfit", action="store_true",
                        help="Pause if validation is below best and training loss below best-epoch loss for patience checks")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=20)
    args = parser.parse_args()
    for field in ("epochs", "batch_size", "eval_batch_size", "maxlen", "hidden_units",
                  "num_blocks", "num_heads", "threads", "eval_every", "patience", "smoke_steps"):
        if getattr(args, field) <= 0:
            parser.error(f"{field} must be positive")
    if args.hidden_units % args.num_heads:
        parser.error("hidden_units must be divisible by num_heads")
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Output directory is not empty; choose a new run directory")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(args.seed)
    config = {"device": args.device, "maxlen": args.maxlen, "hidden_units": args.hidden_units,
              "num_blocks": args.num_blocks, "num_heads": args.num_heads,
              "dropout_rate": args.dropout_rate, "norm_first": False}
    train, valid, test, usernum, itemnum, checksum = load_data()
    users, seq, pos = training_arrays(train, args.maxlen)
    known_keys = training_keys(train, itemnum)
    model = initialize_model(usernum, itemnum, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    criterion = torch.nn.BCEWithLogitsLoss()
    run = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
           "training_executed": True, "config": config,
           "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
           "environment": {"python": platform.python_version(), "torch": torch.__version__,
                           "numpy": np.__version__, "platform": platform.platform(),
                           "threads": torch.get_num_threads()},
           "users": usernum, "items": itemnum, "data_sha256": checksum,
           "parameters": sum(p.numel() for p in model.parameters()),
           "supervised_positions_per_epoch": int(np.count_nonzero(pos)),
           "protocol": {"split": "per-user leave-two-out", "candidates": "all processed items",
                        "history_filter": "all known history, including validation for test",
                        "ties": "score descending, item ID ascending",
                        "selection": "validation NDCG@10 only", "validation_users": "all",
                        "test_users": "all", "negative_sampling": "uniform excluding training history only",
                        "epoch": "one shuffled pass over all training users",
                        "global_time_split": False}, "history": []}
    write_json(out / "run.json", run)
    print(f"Environment: torch={torch.__version__}, CPU threads={args.threads}", flush=True)
    print(f"Data: {usernum} users, {itemnum} items; parameters={run['parameters']:,}; "
          f"supervised positions={run['supervised_positions_per_epoch']:,}", flush=True)
    best_metric, best_epoch, stale = -math.inf, 0, 0
    start_epoch = 1
    if args.resume:
        resume_path = args.resume.resolve()
        saved = torch.load(resume_path, map_location=args.device, weights_only=True)
        if saved["data_sha256"] != checksum or saved["usernum"] != usernum or saved["itemnum"] != itemnum:
            parser.error("Resume checkpoint and current data differ")
        for key in config:
            if key != "device" and config[key] != saved["config"][key]:
                parser.error(f"Resume model configuration differs: {key}")
        start_epoch = saved["epoch"] + 1
        if not args.smoke_only and start_epoch > args.epochs:
            parser.error("epochs must exceed the resume checkpoint epoch")
        model.load_state_dict(saved["state_dict"])
        state_restored = restore_training_state(saved, optimizer, rng)
        parent_path = resume_path.parent / "run.json"
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        run["history"] = [e for e in parent["history"] if e["epoch"] < start_epoch]
        validations = [e for e in run["history"] if "validation" in e]
        if validations:
            best = max(validations, key=lambda e: e["validation"]["NDCG@10"])
            best_epoch, best_metric = best["epoch"], best["validation"]["NDCG@10"]
            parent_best = resume_path.parent / "best.pth"
            best_saved = torch.load(parent_best, map_location="cpu", weights_only=True)
            if best_saved["epoch"] != best_epoch or best_saved["data_sha256"] != checksum:
                parser.error("Parent best checkpoint and validation history disagree")
            shutil.copy2(parent_best, out / "best.pth")
            stale = len([e for e in validations if e["epoch"] > best_epoch])
        run["resume"] = {"checkpoint": str(args.resume), "checkpoint_epoch": saved["epoch"],
                         "checkpoint_sha256": hashlib.sha256(resume_path.read_bytes()).hexdigest(),
                         "optimizer_and_rng_restored": state_restored,
                         "mode": "full training state" if state_restored else "weights only; Adam and RNG reinitialized",
                         "parent_arguments": parent["arguments"],
                         "parent_environment": parent["environment"],
                         "parent_started_at_utc": parent["started_at_utc"],
                         "parent_resume": parent.get("resume"),
                         "parent_stop_reason": parent.get("stop_reason")}
        print(f"Resume epoch {saved['epoch']}: {run['resume']['mode']}", flush=True)
        write_json(out / "run.json", run)
    last_epoch = start_epoch if args.smoke_only else args.epochs
    best_loss = next((e["loss"] for e in run["history"] if e["epoch"] == best_epoch), None)
    monitor = OverfitMonitor(best_metric, best_loss, args.patience)
    run.update(training_start_epoch=start_epoch, requested_last_epoch=last_epoch,
               requested_additional_epochs=last_epoch - start_epoch + 1,
               training_paused=False, stop_reason="maximum_epochs_reached")
    run["protocol"]["overfitting_pause_rule"] = (
        f"validation NDCG@10 below best AND training loss below best-epoch loss "
        f"for {args.patience} consecutive validation checks" if args.stop_on_overfit else None)
    write_json(out / "run.json", run)
    for epoch in range(start_epoch, last_epoch + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = rng.permutation(len(users))
        loss_sum, position_count, steps = 0.0, 0, 0
        for start in range(0, len(users), args.batch_size):
            index = order[start:start + args.batch_size]
            batch_users, batch_seq, batch_pos = users[index], seq[index], pos[index]
            neg = sample_negatives(batch_users, batch_pos, itemnum, known_keys, rng)
            pos_logits, neg_logits = model(batch_users, batch_seq, batch_pos, neg)
            mask = torch.as_tensor(batch_pos != 0, device=args.device)
            loss = criterion(pos_logits[mask], torch.ones_like(pos_logits[mask]))
            loss += criterion(neg_logits[mask], torch.zeros_like(neg_logits[mask]))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = int(mask.sum())
            loss_sum += loss.item() * count
            position_count += count
            steps += 1
            if steps == 1 or steps % 100 == 0:
                print(f"epoch {epoch:02d} step {steps:03d}/{math.ceil(len(users)/args.batch_size)} "
                      f"loss={loss.item():.5f} elapsed={time.perf_counter()-epoch_started:.1f}s", flush=True)
            if args.smoke_only and steps >= args.smoke_steps:
                break
        record = {"epoch": epoch, "loss": loss_sum / position_count, "steps": steps,
                  "training_seconds": round(time.perf_counter() - epoch_started, 3)}
        if args.smoke_only:
            checkpoint(out / "last.pth", model, config, usernum, itemnum, epoch, None, checksum, optimizer, rng)
            original = model.eval().log2feats(seq[:2]).detach().clone()
            restored, _ = restore_model(out / "last.pth", args.device)
            with torch.inference_mode():
                torch.testing.assert_close(restored.log2feats(seq[:2]), original, rtol=0, atol=0)
            run.update(status="smoke_passed", checkpoint_reload_verified=True,
                       elapsed_seconds=round(time.perf_counter() - started, 3), history=[record])
            write_json(out / "run.json", run)
            print("Smoke passed: finite loss, backward/update, save/reload identical.", flush=True)
            return
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            validation = evaluate(model, train, valid, test, "valid", users, args.maxlen, args.eval_batch_size)
            record["validation"] = validation
            print(f"epoch {epoch:02d} validation={json.dumps(validation)}", flush=True)
            if args.stop_on_overfit:
                record["overfit_monitor"] = monitor.observe(validation["NDCG@10"], record["loss"])
                print(f"Overfit monitor: {json.dumps(record['overfit_monitor'])}", flush=True)
            if validation["NDCG@10"] > best_metric:
                best_metric, best_epoch, stale = validation["NDCG@10"], epoch, 0
                checkpoint(out / "best.pth", model, config, usernum, itemnum, epoch, validation, checksum, optimizer, rng)
            else:
                stale += 1
        run["history"].append(record)
        checkpoint(out / "last.pth", model, config, usernum, itemnum, epoch,
                   record.get("validation"), checksum, optimizer, rng)
        run.update(best_epoch=best_epoch, best_validation_ndcg10=best_metric if math.isfinite(best_metric) else None)
        write_json(out / "run.json", run)
        if args.stop_on_overfit and record.get("overfit_monitor", {}).get("pause"):
            run.update(training_paused=True, stop_reason="overfitting_signal",
                       training_stopped_at_epoch=epoch, overfit_evidence=record["overfit_monitor"])
            write_json(out / "run.json", run)
            print("Training paused: sustained validation degradation despite lower training loss.", flush=True)
            break
        if not args.stop_on_overfit and stale >= args.patience:
            run.update(stop_reason="validation_no_improvement", training_stopped_at_epoch=epoch)
            write_json(out / "run.json", run)
            print("Early stopping based only on validation NDCG@10.", flush=True)
            break
    model, saved = restore_model(out / "best.pth", args.device)
    run["test"] = evaluate(model, train, valid, test, "test", users, args.maxlen, args.eval_batch_size)
    counts = Counter(i for history in train.values() for i in history)
    popularity = torch.tensor([counts[i] for i in range(itemnum + 1)], dtype=torch.float32)
    run["popularity_test"] = evaluate(model, train, valid, test, "test", users, args.maxlen,
                                      args.eval_batch_size, popularity)
    example_history = train[1] + valid[1]
    for _ in range(5):
        recommend(model, example_history, args.maxlen)
    durations = []
    for _ in range(30):
        begin = time.perf_counter()
        recommendation = recommend(model, example_history, args.maxlen)
        if args.device == "cuda":
            torch.cuda.synchronize()
        durations.append(1000 * (time.perf_counter() - begin))
    run["inference"] = {"single_user_full_catalog_top10_p50_ms": float(np.median(durations)),
                        "single_user_full_catalog_top10_p95_ms": float(np.percentile(durations, 95)),
                        "repeats": 30, "excludes": "disk/network IO; warm model, one user's history"}
    write_json(out / "demo.local.json", {"user_id": 1, "history": example_history,
               "test_target": test[1][0], "recommendations": recommendation})
    run.update(status="completed", best_epoch=saved["epoch"],
               elapsed_seconds=round(time.perf_counter() - started, 3),
               completed_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(out / "run.json", run)
    print(f"Completed. Best epoch={saved['epoch']}; test={json.dumps(run['test'])}", flush=True)
    print(f"Popularity test={json.dumps(run['popularity_test'])}", flush=True)
    print(f"Artifacts: {out}", flush=True)


if __name__ == "__main__":
    main()
