"""Load a trained SASRec checkpoint and recommend from numeric item IDs."""
import argparse
import json
from pathlib import Path

import torch

from run_sasrec import ROOT, load_data, recommend, restore_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--history", type=int, nargs="+", help="Processed item IDs, oldest first")
    selection.add_argument("--user-id", type=int, help="Local numeric user ID; history includes validation")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.k <= 0 or args.threads <= 0:
        parser.error("k and threads must be positive")
    torch.set_num_threads(args.threads)
    model, saved = restore_model(args.checkpoint)
    history = args.history
    if args.user_id is not None:
        train, valid, _, _, _, checksum = load_data()
        if checksum != saved["data_sha256"]:
            parser.error("Checkpoint and data differ")
        if args.user_id not in train:
            parser.error("Unknown user ID")
        history = train[args.user_id] + valid[args.user_id]
    metadata = {}
    with (ROOT / "data/processed/office2018/items.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            metadata[row["item_id"]] = row
    results = []
    for item, score in recommend(model, history, saved["config"]["maxlen"], args.k):
        row = metadata[item]
        results.append({"item_id": item, "asin": row["asin"], "title": row["title"], "score": score})
    print(json.dumps({"history_length": len(history), "checkpoint_epoch": saved["epoch"],
                      "recommendations": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
