"""Validate frozen MiniLM-derived SIDs and prepare portable SFT input artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np
from qwen_sft import ROOT, clean_title, positions, read_json, sha256, write_json, validate_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed', type=Path, default=ROOT / 'data/processed/office2018')
    parser.add_argument('--sids', type=Path, default=ROOT / 'data/processed/office2018/tiger_sids')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-history', type=int, default=20)
    args = parser.parse_args()
    if args.max_history < 1 or args.output.exists():
        raise ValueError('Use positive max-history and a NEW output directory to preserve artifacts')
    # Reuse the project's existing full data/split and SID provenance validators.
    from tiger_common import sequences, load_sid, PROCESSED
    if args.processed.resolve() != PROCESSED.resolve():
        raise ValueError('This version validates the shared Office Products directory only')
    rows, stats = sequences()
    sid, sid_manifest = load_sid(args.sids)
    validate_rows(rows, len(sid) - 1)
    catalog = {}
    with (args.processed / 'items.jsonl').open(encoding='utf-8') as handle:
        import json
        for line in handle:
            item = json.loads(line)
            number = item['item_id']
            catalog[str(number)] = {'sid': sid[number].tolist(), 'title': clean_title(item.get('title'))}
    if set(map(int, catalog)) != set(range(1, len(sid))):
        raise ValueError('Catalog is incomplete')
    examples = positions(rows, args.max_history)
    args.output.mkdir(parents=True)
    shutil.copyfile(args.processed / 'sequences.jsonl', args.output / 'histories.jsonl')
    write_json(args.output / 'catalog.json', catalog)
    np.save(args.output / 'train_examples.npy', examples, allow_pickle=False)
    width = max(256, sid_manifest.get('tokens_per_hierarchy', 256), int(sid[1:].max()) + 1)
    manifest = dict(status='completed', prepared_at_utc=datetime.now(timezone.utc).isoformat(),
                    users=len(rows), items=len(catalog), training_examples=len(examples),
                    max_history=args.max_history, split='per-user leave-two-out; unchanged SASRec tail positions',
                    sid_length=4, tokens_per_namespace=width, sid_manifest=sid_manifest,
                    source_sequence_sha256=sha256(args.processed / 'sequences.jsonl'),
                    files={name: sha256(args.output / name) for name in
                           ('histories.jsonl', 'catalog.json', 'train_examples.npy')},
                    fourth_token='collision disambiguation; not a fourth semantic quantizer',
                    catalog_policy='fixed full metadata catalog, as in existing baselines; not global-time cold start')
    write_json(args.output / 'manifest.json', manifest)
    print(f'Prepared {len(examples)} targets, {len(rows)} users, {len(catalog)} items. No model training.')


if __name__ == '__main__':
    main()
