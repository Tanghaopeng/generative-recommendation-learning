"""Download the pinned official Qwen checkpoint, then verify it offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from datetime import datetime, timezone

from huggingface_hub import snapshot_download
import requests

REPOSITORY = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
WEIGHT_BYTES = 3087467144
WEIGHT_SHA256 = "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
FILES = [
    "LICENSE", "README.md", "config.json", "generation_config.json",
    "merges.txt", "model.safetensors", "tokenizer.json",
    "tokenizer_config.json", "vocab.json",
]


def download_modelscope_weight(destination: Path) -> None:
    """Use Qwen's public domestic mirror; the canonical HF hash is checked later."""
    final = destination / "model.safetensors"
    if final.is_file():
        return
    partial = destination / "model.safetensors.part"
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > WEIGHT_BYTES:
        raise RuntimeError("Partial download exceeds expected weight size")
    if offset == WEIGHT_BYTES:
        partial.replace(final)
        return
    url = "https://modelscope.cn/api/v1/models/" + REPOSITORY + "/repo"
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(
        url, params={"Revision": "master", "FilePath": "model.safetensors"},
        headers=headers, stream=True, timeout=(15, 60),
    ) as response:
        response.raise_for_status()
        if offset and response.status_code == 206:
            if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise RuntimeError("Unexpected resume range")
            mode = "ab"
        elif response.status_code == 200:
            offset = 0
            mode = "wb"
        else:
            raise RuntimeError("Unexpected download HTTP status")
        started = last_report = time.monotonic()
        initial_offset = offset
        with partial.open(mode) as stream:
            for chunk in response.iter_content(4 * 1024 * 1024):
                if not chunk:
                    continue
                stream.write(chunk)
                offset += len(chunk)
                current = time.monotonic()
                if current - last_report >= 10:
                    rate = (offset - initial_offset) / max(current - started, 0.001) / 1e6
                    print(f"ModelScope: {offset / 1e9:.2f}/{WEIGHT_BYTES / 1e9:.2f} GB, {rate:.2f} MB/s", flush=True)
                    last_report = current
    if partial.stat().st_size != WEIGHT_BYTES:
        raise RuntimeError("Incomplete ModelScope weight download; rerun to resume")
    partial.replace(final)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs/pretrained/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--weight-source", choices=["huggingface", "modelscope"], default="huggingface")
    arguments = parser.parse_args()
    destination = arguments.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "download_manifest.json"
    manifest = {
        "status": "verifying" if arguments.verify_only else "downloading",
        "repository": REPOSITORY,
        "revision": REVISION,
        "source": "https://huggingface.co/" + REPOSITORY,
        "weight_source": arguments.weight_source,
        "weight_source_revision": "master_checked_against_canonical_sha256" if arguments.weight_source == "modelscope" else REVISION,
        "output": str(destination),
        "expected_weight_bytes": WEIGHT_BYTES,
        "expected_weight_sha256": WEIGHT_SHA256,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    def save_manifest() -> None:
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)

    save_manifest()
    try:
        if not arguments.verify_only:
            snapshot_download(
                repo_id=REPOSITORY, revision=REVISION, local_dir=str(destination),
                allow_patterns=[name for name in FILES if arguments.weight_source == "huggingface" or name != "model.safetensors"],
                max_workers=2, token=False,
            )
            if arguments.weight_source == "modelscope":
                download_modelscope_weight(destination)
        missing = [name for name in FILES if not (destination / name).is_file()]
        if missing:
            raise RuntimeError("Missing checkpoint files: " + ", ".join(missing))
        weight = destination / "model.safetensors"
        if weight.stat().st_size != WEIGHT_BYTES:
            raise RuntimeError("Weight file size differs from the pinned official checkpoint")
        print("Checking SHA256 of model.safetensors...", flush=True)
        if digest(weight) != WEIGHT_SHA256:
            raise RuntimeError("Weight SHA256 differs from the pinned official checkpoint")
        configuration = json.loads((destination / "config.json").read_text(encoding="utf-8"))
        if configuration.get("model_type") != "qwen2" or configuration.get("num_hidden_layers") != 28:
            raise RuntimeError("Unexpected checkpoint architecture")
        manifest["files"] = {
            name: {"bytes": (destination / name).stat().st_size, "sha256": digest(destination / name)}
            for name in FILES if name != "model.safetensors"
        }
        manifest["files"]["model.safetensors"] = {"bytes": WEIGHT_BYTES, "sha256": WEIGHT_SHA256}
        manifest.update(status="complete", updated_at_utc=datetime.now(timezone.utc).isoformat())
        save_manifest()
        print(json.dumps({"status": "complete", "output": str(destination), "revision": REVISION}, ensure_ascii=False), flush=True)
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__)
        save_manifest()
        raise


if __name__ == "__main__":
    main()
