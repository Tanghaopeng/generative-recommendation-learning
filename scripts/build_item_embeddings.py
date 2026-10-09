"""Encode the fixed Office2018 catalog with a frozen, pinned MiniLM model."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

from tiger_common import ROOT, PROCESSED, configure, sha256, utc_now, write_json

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'


def catalog_text(row):
    fields = ['title', 'brand', 'category', 'description', 'feature']
    text = '. '.join(f'{key}: {str(row.get(key, "")).strip()}' for key in fields
                     if str(row.get(key, '')).strip())
    return text if text else 'Product information unavailable.'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(PROCESSED / 'tiger_embeddings'))
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--limit', type=int, default=0, help='Separate smoke artifact only')
    args = parser.parse_args()
    if args.limit < 0 or args.batch_size < 1 or not 1 <= args.max_tokens <= 512:
        raise ValueError('Require limit>=0, batch size>0 and max tokens in [1,512]')
    configure(threads=args.threads)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    source = PROCESSED / 'items.jsonl'
    rows = [json.loads(line) for line in source.open(encoding='utf-8')]
    if [r['item_id'] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError('Item IDs must be continuous and ordered')
    if args.limit:
        rows = rows[:args.limit]
    texts = [catalog_text(r) for r in rows]
    contract = {'model': MODEL, 'revision': REVISION, 'dimension': 384,
                'max_tokens': args.max_tokens, 'items': len(rows),
                'items_source_sha256': sha256(source), 'smoke_only': bool(args.limit),
                'normalization': 'attention-mask mean pooling then L2', 'frozen': True}
    manifest_path = out / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    if manifest and manifest['contract'] != contract:
        raise ValueError('Output belongs to a different catalog/model/configuration')
    if manifest.get('status') == 'completed':
        if sha256(out / 'embeddings.npy') != manifest['embedding_sha256']:
            raise ValueError('Completed embeddings changed')
        print('Verified existing completed embeddings.', flush=True)
        return
    model_path = ROOT / 'data/models/all-MiniLM-L6-v2'
    snapshot_download(MODEL, revision=REVISION, local_dir=model_path,
                      allow_patterns=['config.json', 'tokenizer.json', 'tokenizer_config.json',
                                      'special_tokens_map.json', 'vocab.txt', 'model.safetensors'],
                      max_workers=2)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval()
    model.requires_grad_(False)
    if model.config.hidden_size != 384:
        raise ValueError('Unexpected encoder dimension')
    array_path = out / 'embeddings.npy'
    done = int(manifest.get('encoded_items', 0))
    if not 0 <= done <= len(rows) or (done and not array_path.exists()):
        raise ValueError('Embedding resume state and array differ')
    array = np.lib.format.open_memmap(array_path, mode='r+' if array_path.exists() else 'w+',
                                     dtype=np.float32, shape=(len(rows) + 1, 384))
    if array.shape != (len(rows)+1,384) or array.dtype != np.float32:
        raise ValueError('Existing array shape or dtype differs')
    array[0] = 0
    lengths = [len(tokenizer.encode(t, add_special_tokens=True, truncation=False)) for t in texts]
    manifest.update(contract=contract, status='running', encoded_items=done,
                    started_at_utc=manifest.get('started_at_utc', utc_now()),
                    missing_metadata=sum(not r['metadata_found'] for r in rows),
                    placeholder_items=sum(catalog_text(r) == 'Product information unavailable.' for r in rows),
                    duplicate_text_items=sum(n - 1 for n in Counter(texts).values()),
                    truncated_text_items=sum(n > args.max_tokens for n in lengths),
                    batch_size=args.batch_size, threads=args.threads)
    write_json(manifest_path, manifest)
    started = time.perf_counter()
    rss_peak = psutil.Process().memory_info().rss
    with torch.inference_mode():
        for start in range(done, len(rows), args.batch_size):
            end = min(start + args.batch_size, len(rows))
            batch = tokenizer(texts[start:end], padding=True, truncation=True,
                              max_length=args.max_tokens, return_tensors='pt')
            hidden = model(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            vectors = torch.nn.functional.normalize(pooled, dim=1).numpy()
            if not np.isfinite(vectors).all() or (np.linalg.norm(vectors, axis=1) < .99).any():
                raise ValueError('Invalid normalized embedding')
            array[start + 1:end + 1] = vectors
            array.flush()
            rss_peak = max(rss_peak, psutil.Process().memory_info().rss)
            manifest.update(encoded_items=end, elapsed_this_segment_seconds=time.perf_counter() - started,
                            peak_observed_rss_bytes=rss_peak)
            write_json(manifest_path, manifest)
            if start == done or (start // args.batch_size) % 25 == 0 or end == len(rows):
                print(f'encoded {end}/{len(rows)} elapsed={time.perf_counter()-started:.1f}s '
                      f'rss={rss_peak/1024**2:.0f}MiB', flush=True)
    if not np.allclose(np.linalg.norm(array[1:], axis=1), 1, atol=1e-5):
        raise ValueError('Final embedding norm verification failed')
    manifest.update(status='completed', completed_at_utc=utc_now(),
                    embedding_sha256=sha256(array_path), array_shape=list(array.shape))
    write_json(manifest_path, manifest)
    print('Completed and verified catalog embeddings.', flush=True)


if __name__ == '__main__':
    main()
