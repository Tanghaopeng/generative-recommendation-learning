"""Validate the EMA-SID experiment before renting/starting GPU training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EMA_SID_SHA256 = '975075925cc54f903d7bc9069bf3653a96b8c29b613b846bdc4a27a8351a4554'
SEQUENCE_SHA256 = 'a242718d3410cfdf75a0ca411c530a9c4ae3898e293476922fd7a66861e4bcf6'
BASE_MANIFEST_SHA256 = '6e0a9a179409cf2b7dc5ea8115e092310f18611a222225b9740721274e9aab58'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def compare_training_config(config, baseline):
    """Allow batch execution variants while preserving effective batch and scope."""
    batch_keys = {'micro_batch_size', 'gradient_accumulation_steps'}
    if ({k: v for k, v in config.items() if k not in batch_keys}
            != {k: v for k, v in baseline.items() if k not in batch_keys}):
        raise ValueError('EMA config differs from baseline outside batch execution settings')
    pair = (config.get('micro_batch_size'), config.get('gradient_accumulation_steps'))
    if pair not in ((2, 64), (4, 32), (8, 16)) or baseline['micro_batch_size'] * baseline['gradient_accumulation_steps'] != 128:
        raise ValueError('Requires supported micro-batch/accumulation pair with effective batch 128')
    return {k: {'gradient': baseline[k], 'ema': config[k]}
            for k in sorted(batch_keys) if config[k] != baseline[k]}


def check(root=ROOT):
    root = Path(root)
    processed = root/'data/processed/office2018'
    data = processed/'qwen_sft_ema'
    m = read(data/'manifest.json')
    sm = read(processed/'tiger_sids_ema_20261008/manifest.json')
    sid_path = processed/'tiger_sids_ema_20261008/semantic_ids.npy'
    if (m['status'] != 'completed' or sm['status'] != 'completed'
        or digest(sid_path) != EMA_SID_SHA256 or sm['sid_sha256'] != EMA_SID_SHA256
        or m['sid_manifest']['sid_sha256'] != EMA_SID_SHA256
        or sm.get('codebook_update') != 'EMA' or m['tokens_per_namespace'] != 256):
        raise ValueError('Requires the pinned EMA SID version, not gradient SID or a renamed catalog')
    for name, checksum in m['files'].items():
        if Path(name).name != name or digest(data/name) != checksum:
            raise ValueError('Prepared data hash mismatch: '+name)
    if (digest(data/'histories.jsonl') != SEQUENCE_SHA256 or m['source_sequence_sha256'] != SEQUENCE_SHA256
        or (m['users'],m['items'],m['training_examples'],m['max_history']) != (86713,25898,397345,20)):
        raise ValueError('User split or training-position version changed')
    codes = np.load(sid_path,allow_pickle=False)
    catalog = read(data/'catalog.json')
    if (codes.shape != (25899,4) or not (codes[0] == -1).all()
        or set(catalog) != {str(i) for i in range(1,25899)}
        or len(np.unique(codes[1:],axis=0)) != 25898):
        raise ValueError('Catalog completeness, padding or four-token uniqueness failed')
    if any(catalog[str(i)]['sid'] != codes[i].tolist() for i in range(1,25899)):
        raise ValueError('Prepared catalog contains a different SID mapping')
    with (data/'histories.jsonl').open(encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    positions = []
    for user,row in enumerate(rows):
        full = row['train'] + row['valid'] + row['test']
        if (not row['train'] or len(row['valid']) != 1 or len(row['test']) != 1
            or row['items'] != full or len(set(full)) != len(full)
            or min(full) < 1 or max(full) > 25898):
            raise ValueError('Invalid shared temporal split')
        start = max(0,len(row['train'])-1-20)
        positions.extend((user,start,t) for t in range(start+1,len(row['train'])))
    actual = np.load(data/'train_examples.npy',allow_pickle=False)
    np.testing.assert_array_equal(actual,np.asarray(positions,dtype=np.int64))
    config = read(root/'experiments/qwen_sft_ema_20261021.json')
    baseline = read(root/'04_推荐生成/02_Qwen2.5-1.5B-Instruct/configs/q1_full.json')
    config_differences = compare_training_config(config, baseline)
    old_test = read(root/'experiments/qwen_sft_office2018_20261013/test.json')
    if (old_test['prepared_manifest_sha256'] != BASE_MANIFEST_SHA256
        or old_test['code_sha256']['qwen_sft.py'] != digest(root/'scripts/qwen_sft.py')):
        raise ValueError('Core input/decoding implementation changed from the gradient baseline')
    valid_ids = sorted(random.Random(2027).sample(range(len(rows)),5000))
    valid_hash = hashlib.sha256(json.dumps(valid_ids,separators=(',',':')).encode()).hexdigest()
    return dict(status='ema_inputs_verified',users=len(rows),items=len(catalog),training_examples=len(actual),
        prepared_manifest_sha256=digest(data/'manifest.json'),sid_sha256=EMA_SID_SHA256,
        sequence_sha256=SEQUENCE_SHA256,train_examples_sha256=digest(data/'train_examples.npy'),
        validation_users=5000,validation_subset_seed=2027,validation_user_indices_sha256=valid_hash,
        effective_batch_size=config['micro_batch_size']*config['gradient_accumulation_steps'],
        expected_optimizer_steps=int(np.ceil(len(actual)/128)),config_identical_to_gradient_baseline=not config_differences,
        config_differences_from_gradient_baseline=config_differences,
        effective_batch_matches_gradient_baseline=True,
        core_qwen_code_matches_gradient_baseline=True,actual_gpu_training=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    result=check(args.root)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
