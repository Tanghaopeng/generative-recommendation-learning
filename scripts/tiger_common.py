"""Shared artifact and data contracts for the small TIGER adaptation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'baselines/tiger/upstream'
PROCESSED = ROOT / 'data/processed/office2018'
sys.path.insert(0, str(UPSTREAM))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    replace_file(temp, path)


def replace_file(temp, path):
    # Windows readers (including PowerShell/UI/antivirus) can briefly deny
    # FILE_SHARE_DELETE. Preserve the prepared file and retry the atomic rename.
    for attempt in range(40):
        try:
            Path(temp).replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


def configure(seed=2026, threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if os.name == 'nt':
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


def source_hashes():
    files = list((ROOT / 'scripts').glob('*tiger*.py'))
    files += [ROOT / 'scripts/build_item_embeddings.py', ROOT / 'scripts/train_rqvae.py',
              ROOT / 'scripts/build_semantic_ids.py']
    files += list(UPSTREAM.rglob('*.py'))
    return {str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p)
            for p in sorted(set(files)) if p.is_file()}


def sequences():
    rows = [json.loads(line) for line in (PROCESSED / 'sequences.jsonl').open(encoding='utf-8')]
    stats = json.loads((ROOT / 'experiments/data_stats.json').read_text(encoding='utf-8'))
    if len(rows) != stats['users']:
        raise ValueError('User count differs from the SASRec data version')
    if sha256(PROCESSED / 'office2018.txt') != stats['sasrec_input_sha256']:
        raise ValueError('Data fingerprint differs from the SASRec baseline')
    original = {}
    for line in (PROCESSED / 'office2018.txt').open(encoding='utf-8'):
        uid, item = map(int, line.split())
        original.setdefault(uid, []).append(item)
    for uid, row in enumerate(rows, 1):
        if row['user_id'] != uid or row['items'] != row['train'] + row['valid'] + row['test']:
            raise ValueError('Invalid shared sequence split')
        if len(row['valid']) != 1 or len(row['test']) != 1:
            raise ValueError('Expected one held-out item per user')
        if row['items'] != original[uid]:
            raise ValueError('Sequence contents differ from the pinned SASRec input')
    return rows, stats


def load_sid(path):
    path = Path(path)
    info = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    sid = np.load(path / 'semantic_ids.npy')
    if info['status'] != 'completed' or sha256(path / 'semantic_ids.npy') != info['sid_sha256']:
        raise ValueError('Incomplete or modified SID artifact')
    if sid.shape != (info['items'] + 1, 4) or not (sid[0] == -1).all():
        raise ValueError('SID shape or padding row differs from the contract')
    if len(np.unique(sid[1:], axis=0)) != info['items']:
        raise ValueError('Full SIDs must uniquely identify catalog items')
    if sha256(PROCESSED / 'items.jsonl') != info['items_source_sha256']:
        raise ValueError('SID catalog text does not match current items')
    return sid, info


def atomic_checkpoint(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    replace_file(temp, path)


def random_state(rng):
    return {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(),
            'python': random.getstate(), 'generator': rng.bit_generator.state}


def restore_random(state, rng):
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    rng.bit_generator.state = state['generator']
