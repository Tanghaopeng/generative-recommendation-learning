"""Verify processed files and the actual upstream SASRec partition function."""
import ast
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "office2018"
repo = ROOT / "baselines" / "sasrec"
stats = json.loads((OUT / "stats.json").read_text(encoding="utf-8"))
user_map = json.loads((OUT / "user_mapping.json").read_text(encoding="utf-8"))
item_map = json.loads((OUT / "item_mapping.json").read_text(encoding="utf-8"))
assert set(user_map.values()) == set(range(1, stats["users"] + 1))
assert set(item_map.values()) == set(range(1, stats["items"] + 1))

# Extract only the inspected, dependency-free loader; no torch/numpy import or
# training is performed. Its source is preserved exactly as cloned.
tree = ast.parse((repo / "utils.py").read_text(encoding="utf-8"))
loader = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "data_partition")
namespace = {"defaultdict": defaultdict}
exec(compile(ast.Module(body=[loader], type_ignores=[]), str(repo / "utils.py"), "exec"), namespace)
previous_dir = Path.cwd()
try:
    os.chdir(repo)
    train, valid, test, users, items = namespace["data_partition"]("office2018")
finally:
    os.chdir(previous_dir)
assert users == stats["users"] and items == stats["items"]
item_counts = Counter()
user_count = 0
interaction_count = 0
with (OUT / "sequences.jsonl").open(encoding="utf-8") as stream:
    for line in stream:
        row = json.loads(line)
        uid = row["user_id"]
        ids = row["items"]
        assert len(ids) >= 5 and len(ids) == len(set(ids))
        assert row["timestamps"] == sorted(row["timestamps"])
        assert row["train"] == train[uid] == ids[:-2]
        assert row["valid"] == valid[uid] == ids[-2:-1]
        assert row["test"] == test[uid] == ids[-1:]
        item_counts.update(ids)
        interaction_count += len(ids)
        user_count += 1
assert user_count == stats["users"] and interaction_count == stats["interactions"]
assert len(item_counts) == stats["items"] and min(item_counts.values()) >= 5
assert sum(map(len, train.values())) == stats["train_interactions"]
assert hashlib.sha256((OUT / "office2018.txt").read_bytes()).hexdigest() == stats["sasrec_input_sha256"]
assert (OUT / "office2018.txt").read_bytes() == (repo / "data" / "office2018.txt").read_bytes()

metadata_ids = set()
nonempty = 0
with (OUT / "items.jsonl").open(encoding="utf-8") as stream:
    for line in stream:
        row = json.loads(line)
        assert row["item_id"] == item_map[row["asin"]]
        assert row["item_id"] not in metadata_ids
        metadata_ids.add(row["item_id"])
        nonempty += int(bool(row["text"]))
assert metadata_ids == set(item_map.values())
assert nonempty == stats["items_with_nonempty_text"]
report = {"status": "passed", "users": users, "items": items,
          "interactions": interaction_count, "verified_upstream_split": True,
          "checks": ["contiguous positive IDs", "chronological histories", "unique user-item pairs",
                     "user and item 5-core", "upstream train/valid/test equality",
                     "input checksum and copy equality", "item-metadata ID alignment"],
          "training_executed": False}
(OUT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
