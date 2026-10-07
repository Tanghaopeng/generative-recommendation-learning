"""Download official Office Products 2018 archives and verify SHA-256.

No credentials or external packages are needed. Large datasets stay outside Git.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(spec, folder, verify_only=False):
    target = folder / spec["name"]
    expected = spec["sha256"]
    if target.exists():
        if target.stat().st_size == spec["bytes"] and sha256(target) == expected:
            print(f"verified: {target.name}", flush=True)
            return
        raise ValueError(f"Existing file does not match sources.json: {target}. Preserve it and investigate before replacing it.")
    if verify_only:
        raise FileNotFoundError(target)
    part = target.with_name(target.name + ".part")
    for attempt in range(3):
        offset = part.stat().st_size if part.exists() else 0
        # A complete partial archive can be finalized without another request.
        if offset == spec["bytes"] and sha256(part) == expected:
            part.replace(target)
            print(f"verified: {target.name}", flush=True)
            return
        headers = {"User-Agent": "generative-recommendation-learning"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            request = urllib.request.Request(spec["url"], headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Server returned an unexpected range.")
                    mode = "ab"
                else:
                    offset = 0
                    mode = "wb"
                received = offset
                last_print = time.monotonic()
                with part.open(mode) as stream:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(chunk)
                        received += len(chunk)
                        if time.monotonic() - last_print >= 5:
                            print(f"{target.name}: {received / spec['bytes']:.1%}", flush=True)
                            last_print = time.monotonic()
            if part.stat().st_size != spec["bytes"] or sha256(part) != expected:
                raise ValueError(f"Downloaded archive does not match sources.json: {part}")
            part.replace(target)
            print(f"downloaded and verified: {target.name}", flush=True)
            return
        except (urllib.error.URLError, OSError) as error:
            if attempt == 2:
                raise
            print(f"download interrupted ({type(error).__name__}); retrying", flush=True)
            time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    folder = ROOT / "data" / "raw"
    folder.mkdir(parents=True, exist_ok=True)
    specs = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))["files"]
    for spec in specs:
        download(spec, folder, args.verify_only)
