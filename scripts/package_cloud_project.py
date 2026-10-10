"""Package project sources and pinned MiniLM/gradient-RQ-VAE assets for the cloud."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new archive path; preserve published bundles')
    files = set()
    names = subprocess.check_output(['git', 'ls-files', '-co', '--exclude-standard', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    for name in filter(None, names):
        path = ROOT / name
        if path.is_file() and path.suffix not in ('.safetensors', '.pt', '.pth', '.ckpt'):
            files.add(path)
    processed = ROOT / 'data/processed/office2018'
    for directory in ('qwen_sft_adam', 'tiger_sids', 'tiger_embeddings'):
        files.update(p for p in (processed / directory).iterdir() if p.is_file())
    for name in ('items.jsonl', 'sequences.jsonl', 'stats.json', 'item_mapping.json', 'office2018.txt'):
        files.add(processed / name)
    for name in ('best.pth', 'run.json'):
        files.add(ROOT / 'outputs/rqvae_office2018_cpu_20261008' / name)
    files = sorted(files)
    for path in files:
        if not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError(f'Missing/outside-workspace asset: {path}')
    record = {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
              'git_worktree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
              'recipe': 'frozen MiniLM + gradient-updated RQ-VAE, unique four-token SID',
              'external_model': '/mnt/Qwen2.5-1.5B-Instruct',
              'q0_and_project_grpo_ready': False,
              'files': {path.relative_to(ROOT).as_posix(): {'bytes': path.stat().st_size, 'sha256': fingerprint(path)} for path in files}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prefix = 'generative-recommendation-learning/'
    with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, prefix + path.relative_to(ROOT).as_posix())
        archive.writestr(prefix + 'CLOUD_BUNDLE_MANIFEST.json', json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        archive.writestr(prefix + 'CLOUD_START_HERE.txt',
                         'Extract this archive under /mnt. Model weights already live in /mnt/Qwen2.5-1.5B-Instruct.\n'
                         'Do not copy the Windows venv. Install compatible Linux/CUDA dependencies after checking the GPU.\n'
                         'SFT and sampled AUC entries are included; Q0 initialization and project GRPO adaptation remain pending.\n'
                         'The upstream GRPO files are reading references, not the local four-token production entry.\n'
                         'Verify each file against CLOUD_BUNDLE_MANIFEST.json before use.\n')
    # Check both ZIP integrity and every archived payload against its manifest.
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP integrity check failed')
        for name, entry in record['files'].items():
            if hashlib.sha256(archive.read(prefix + name)).hexdigest() != entry['sha256']:
                raise ValueError('Archived fingerprint mismatch: ' + name)
    proof = {'archive': args.output.name, 'bytes': args.output.stat().st_size,
             'sha256': fingerprint(args.output), 'files': len(files), 'git_commit': record['git_commit'],
             'status': 'local_archive_verified_not_yet_uploaded'}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
