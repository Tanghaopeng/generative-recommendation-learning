"""Prepare Amazon 2018 Office Products using only the Python standard library.

Original data and cloned upstream sources are preserved. Review text is not used
as an item feature, to avoid bringing held-out reviews into item representations.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import html
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "office2018"


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_gz(path):
    # Reading to EOF also verifies gzip CRC/truncation.
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                yield line_no, json.loads(line)


def interactions():
    source = ROOT / "data" / "raw" / "Office_Products_5.json.gz"
    pairs = {}
    rows = 0
    for line_no, row in read_gz(source):
        rows += 1
        key = (row["reviewerID"], row["asin"])
        event = (int(row["unixReviewTime"]), line_no, float(row["overall"]))
        previous = pairs.get(key)
        if previous is None or event[:2] < previous[:2]:
            pairs[key] = event
        if rows % 200000 == 0:
            print(f"reviews parsed: {rows:,}", flush=True)
    edges = [(u, i, *event) for (u, i), event in pairs.items()]
    unique_before_core = len(edges)
    del pairs
    iterations = 0
    while True:
        users = Counter(edge[0] for edge in edges)
        items = Counter(edge[1] for edge in edges)
        keep = [e for e in edges if users[e[0]] >= 5 and items[e[1]] >= 5]
        if len(keep) == len(edges):
            break
        edges = keep
        iterations += 1
    if not edges:
        raise ValueError("No interactions remain after deduplication and 5-core.")
    user_map = {u: n for n, u in enumerate(sorted(users), 1)}
    item_map = {i: n for n, i in enumerate(sorted(items), 1)}
    histories = defaultdict(list)
    for user, item, timestamp, source_line, rating in edges:
        histories[user_map[user]].append((timestamp, source_line, item_map[item], rating))
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(OUT / "user_mapping.json", user_map)
    save_json(OUT / "item_mapping.json", item_map)
    train_count = 0
    tie_users = 0
    examples = []
    with (OUT / "interactions.csv").open("w", encoding="utf-8", newline="") as table, \
         (OUT / "sequences.jsonl").open("w", encoding="utf-8") as sequences, \
         (OUT / "office2018.txt").open("w", encoding="utf-8") as sasrec:
        writer = csv.writer(table)
        writer.writerow(["user_id", "item_id", "timestamp", "rating"])
        for uid in sorted(histories):
            events = sorted(histories[uid])
            ids = [event[2] for event in events]
            timestamps = [event[0] for event in events]
            assert len(ids) >= 5 and len(ids) == len(set(ids))
            tie_users += int(len(timestamps) != len(set(timestamps)))
            for timestamp, _, iid, rating in events:
                writer.writerow([uid, iid, timestamp, rating])
                sasrec.write(f"{uid} {iid}\n")
            sample = {"user_id": uid, "items": ids, "timestamps": timestamps,
                      "train": ids[:-2], "valid": ids[-2:-1], "test": ids[-1:]}
            sequences.write(json.dumps(sample) + "\n")
            train_count += len(sample["train"])
            if len(examples) < 3:
                examples.append(sample)
    repo_data = ROOT / "baselines" / "sasrec" / "data"
    repo_data.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT / "office2018.txt", repo_data / "office2018.txt")
    timestamp_min = min(e[2] for e in edges)
    timestamp_max = max(e[2] for e in edges)
    stats = {
        "dataset": "Amazon Reviews 2018 / Office Products / official 5-core",
        "source_review_rows": rows,
        "unique_user_item_pairs_before_recore": unique_before_core,
        "duplicate_review_rows_removed": rows - unique_before_core,
        "recore_iterations": iterations,
        "users": len(user_map), "items": len(item_map), "interactions": len(edges),
        "train_interactions": train_count, "validation_targets": len(user_map),
        "test_targets": len(user_map), "min_user_interactions": min(users.values()),
        "min_item_interactions": min(items.values()),
        "users_with_equal_timestamps": tie_users,
        "date_min_utc": datetime.fromtimestamp(timestamp_min, timezone.utc).isoformat(),
        "date_max_utc": datetime.fromtimestamp(timestamp_max, timezone.utc).isoformat(),
        "protocol": {
            "feedback": "all rating levels treated as implicit interactions",
            "deduplication": "keep earliest review per reviewerID/asin pair",
            "filter": "iterative user-item 5-core after deduplication",
            "ordering": "unixReviewTime, then original source line as stable tie-break",
            "split": "per-user last item=test, penultimate=valid, earlier=train",
            "catalog": "fixed full processed catalog; not a strict global-time benchmark",
            "id_zero": "padding only",
            "metadata": "catalog text only; no held-out review text",
        },
        "review_archive_sha256": digest(source),
        "sasrec_input_sha256": digest(OUT / "office2018.txt"),
    }
    save_json(OUT / "stats.json", stats)
    save_json(OUT / "sample_sequences.json", examples)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


def clean_text(value):
    if isinstance(value, list):
        return " ".join(filter(None, (clean_text(v) for v in value)))
    if value is None:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(str(value)))).strip()


def metadata():
    source = ROOT / "data" / "raw" / "meta_Office_Products.json.gz"
    item_map = json.loads((OUT / "item_mapping.json").read_text(encoding="utf-8"))
    selected = {}
    rows = 0
    matched_rows = 0
    for _, row in read_gz(source):
        rows += 1
        asin = row.get("asin")
        if asin in item_map:
            matched_rows += 1
            obj = {"item_id": item_map[asin], "asin": asin, "metadata_found": True}
            for field in ("title", "brand", "description", "feature", "category"):
                obj[field] = clean_text(row.get(field))
            obj["price"] = row.get("price", "")
            # Image links and also-buy graphs are intentionally not used as features.
            obj["text"] = ". ".join(f"{f}: {obj[f]}" for f in
                                      ("title", "brand", "category", "description", "feature")
                                      if obj[f])
            previous = selected.get(asin)
            if previous is None or len(obj["text"]) > len(previous["text"]):
                selected[asin] = obj
        if rows % 100000 == 0:
            print(f"metadata rows parsed: {rows:,}", flush=True)
    missing = []
    with (OUT / "items.jsonl").open("w", encoding="utf-8") as stream:
        for asin, iid in sorted(item_map.items(), key=lambda pair: pair[1]):
            obj = selected.get(asin)
            if obj is None:
                missing.append(asin)
                obj = {"item_id": iid, "asin": asin, "metadata_found": False,
                       "title": "", "brand": "", "category": "", "description": "",
                       "feature": "", "price": "", "text": ""}
            stream.write(json.dumps(obj, ensure_ascii=False) + "\n")
    save_json(OUT / "missing_metadata_asins.json", missing)
    stats = json.loads((OUT / "stats.json").read_text(encoding="utf-8"))
    stats.update({"source_metadata_rows": rows, "matching_metadata_rows": matched_rows,
                  "items_with_metadata": len(selected), "items_without_metadata": len(missing),
                  "items_with_nonempty_text": sum(bool(x["text"]) for x in selected.values()),
                  "metadata_archive_sha256": digest(source)})
    save_json(OUT / "stats.json", stats)
    print(json.dumps({key: stats[key] for key in (
        "source_metadata_rows", "items_with_metadata", "items_without_metadata",
        "items_with_nonempty_text")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "interactions", "metadata"), default="all")
    args = parser.parse_args()
    if args.stage in ("all", "interactions"):
        interactions()
    if args.stage in ("all", "metadata"):
        metadata()
