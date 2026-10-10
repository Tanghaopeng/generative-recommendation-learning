"""Download and hash-verify the pinned official Qwen3 embedding checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime, timezone

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '02_商品编码/02_Qwen3_Embedding_0.6B'
REPO = 'Qwen/Qwen3-Embedding-0.6B'
REVISION = '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'
WEIGHT_SHA256 = '0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd'
MODELS = {
    '0.6B': (BASE, REPO, REVISION, {'model.safetensors': WEIGHT_SHA256}),
    '4B': (ROOT / '02_商品编码/02_Qwen3_Embedding_4B', 'Qwen/Qwen3-Embedding-4B',
           '5cf2132abc99cad020ac570b19d031efec650f2b', {
               'model-00001-of-00002.safetensors': 'e70bfe3c970523fb7ef4eddffed2254ce3f1e7150c3de2af4342de129dd756f8',
               'model-00002-of-00002.safetensors': 'ed1b87c8e9eb7e535a1a155e4fd00d9f4dba80e58a6db48a4c9f82cede7079c1',
           }),
}


def digest(path, algorithm='sha256', git_blob=False):
    h = hashlib.new(algorithm)
    if git_blob:
        h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def matches(path, record):
    if not path.is_file() or path.stat().st_size != record['size']:
        return False
    if record.get('lfs'):
        return digest(path) == record['lfs']['sha256']
    return digest(path, 'sha1', git_blob=True) == record['blobId']


def download(path, record, weight_source):
    if matches(path, record):
        print(f'Already verified: {record["rfilename"]}', flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.part')
    mirror = record['rfilename'].endswith('.safetensors') and weight_source == 'modelscope'
    url = (f'https://modelscope.cn/api/v1/models/{REPO}/repo' if mirror
           else f'https://huggingface.co/{REPO}/resolve/{REVISION}/{record["rfilename"]}')
    params = {'Revision': 'master', 'FilePath': record['rfilename']} if mirror else None
    for attempt in range(4):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > record['size']:
            raise ValueError('Partial file exceeds expected size')
        if offset == record['size']:
            if not matches(partial, record):
                raise ValueError(f'Partial file hash mismatch: {record["rfilename"]}')
            partial.replace(path)
            return
        try:
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            with requests.get(url, params=params, headers=headers, stream=True,
                              timeout=(20, 60)) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Invalid download resume range')
                    mode = 'ab'
                elif response.status_code == 200:
                    offset, mode = 0, 'wb'
                else:
                    raise ValueError('Unexpected download response')
                start = last = time.monotonic()
                initial = offset
                with partial.open(mode) as stream:
                    for block in response.iter_content(4 * 1024 * 1024):
                        if not block:
                            continue
                        stream.write(block)
                        offset += len(block)
                        now = time.monotonic()
                        if now - last >= 10:
                            print(f'{record["rfilename"]}: {offset/1e6:.1f}/{record["size"]/1e6:.1f} MB, '
                                  f'{(offset-initial)/(now-start)/1e6:.2f} MB/s', flush=True)
                            last = now
            if partial.stat().st_size != record['size']:
                raise requests.ConnectionError('Incomplete file')
            if not matches(partial, record):
                raise ValueError(f'Official file fingerprint mismatch: {record["rfilename"]}')
            partial.replace(path)
            print(f'Verified: {record["rfilename"]}', flush=True)
            return
        except requests.RequestException as error:
            print(f'Retry {attempt+1}: {type(error).__name__}', flush=True)
            if attempt == 3:
                raise


def download_parallel_weight(path, record):
    """Resume independent HTTP ranges, assemble, then check the canonical hash."""
    if matches(path, record):
        return
    chunk_size = 128 * 1024 * 1024
    chunks = path.parent / (path.name + '.chunks')
    chunks.mkdir(parents=True, exist_ok=True)
    url = f'https://modelscope.cn/api/v1/models/{REPO}/repo'
    params = {'Revision': 'master', 'FilePath': record['rfilename']}

    def fetch(index):
        start = index * chunk_size
        end = min(start + chunk_size, record['size']) - 1
        piece = chunks / f'{index:04d}.part'
        for attempt in range(4):
            offset = piece.stat().st_size if piece.exists() else 0
            if offset == end - start + 1:
                return piece
            if offset > end - start + 1:
                raise ValueError('Oversized range file')
            try:
                with requests.get(url, params=params, stream=True, timeout=(20, 60),
                                  headers={'Range': f'bytes={start+offset}-{end}'}) as response:
                    response.raise_for_status()
                    expected = f'bytes {start+offset}-{end}/{record["size"]}'
                    if response.status_code != 206 or response.headers.get('Content-Range') != expected:
                        raise ValueError('Server did not honor the requested byte range')
                    with piece.open('ab') as stream:
                        for block in response.iter_content(4 * 1024 * 1024):
                            if block:
                                stream.write(block)
                if piece.stat().st_size != end - start + 1:
                    raise requests.ConnectionError('Incomplete range')
                print(f'{record["rfilename"]}: chunk {index+1} complete', flush=True)
                return piece
            except requests.RequestException:
                if attempt == 3:
                    raise
        raise RuntimeError('Range retries exhausted')

    with ThreadPoolExecutor(max_workers=4) as executor:
        pieces = list(executor.map(fetch, range((record['size'] + chunk_size - 1)//chunk_size)))
    partial = path.with_name(path.name + '.part')
    with partial.open('wb') as output:
        for piece in pieces:
            with piece.open('rb') as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    output.write(block)
    if not matches(partial, record):
        raise ValueError('Assembled weight fingerprint mismatch')
    partial.replace(path)
    for piece in pieces:
        piece.unlink()
    chunks.rmdir()
    print(f'Verified full weight: {record["rfilename"]}', flush=True)


def main():
    global BASE, REPO, REVISION
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--size', choices=list(MODELS), default='4B')
    p.add_argument('--parallel', action='store_true', help='Four resumable range workers per weight')
    p.add_argument('--weight-source', choices=['huggingface', 'modelscope'], default='modelscope')
    args = p.parse_args()
    BASE, REPO, REVISION, expected_weights = MODELS[args.size]
    response = requests.get(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}',
                            params={'blobs': 'true'}, timeout=30)
    response.raise_for_status()
    metadata = response.json()
    if metadata['sha'] != REVISION:
        raise ValueError('Model revision mismatch')
    files = [r for r in metadata['siblings'] if r['rfilename'] != '.gitattributes']
    for name, expected in expected_weights.items():
        weight = next(r for r in files if r['rfilename'] == name)
        if weight['lfs']['sha256'] != expected:
            raise ValueError('Canonical weight hash mismatch')
    destination = BASE / '模型文件'
    if args.parallel:
        if args.weight_source != 'modelscope':
            raise ValueError('Parallel range mode requires the ModelScope mirror')
        weights = [r for r in files if r['rfilename'].endswith('.safetensors')]
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(lambda r: download_parallel_weight(destination/r['rfilename'], r), weights))
    records = []
    for record in sorted(files, key=lambda r: r['rfilename'].endswith('.safetensors')):
        path = destination / record['rfilename']
        download(path, record, args.weight_source)
        records.append({'path': record['rfilename'], 'bytes': path.stat().st_size,
                        'sha256': digest(path), 'official_git_blob': record['blobId'],
                        'official_lfs_sha256': record.get('lfs', {}).get('sha256')})
    manifest = {'status': 'complete', 'model_id': REPO, 'revision': REVISION,
                'source': f'https://huggingface.co/{REPO}/tree/{REVISION}',
                'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
                'weight_source': args.weight_source,
                'mirror_policy': 'Mirror bytes must match the pinned Hugging Face SHA256',
                'files': records}
    (BASE / '来源').mkdir(parents=True, exist_ok=True)
    (BASE / '来源/manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n',
                                          encoding='utf-8', newline='\n')
    print(json.dumps({'status': 'complete', 'files': len(records), 'output': str(destination)},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
