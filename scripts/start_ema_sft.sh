#!/usr/bin/env bash
# This template is materialized with the verified archive hash by package_ema_sft.py.
set -Eeuo pipefail
RUN=${EMA_RUN_DIR:-/root/ema-sft-20261021}
SOURCE=$(cd -- "$(dirname -- "$0")" && pwd)
ARCHIVE="$SOURCE/__BUNDLE_NAME__"
STAGE=${EMA_STAGE:-all}
SESSION=rec-ema-sft
mkdir -p "$RUN/logs"
if [ "${1:-}" != "--worker" ]; then
    command -v tmux >/dev/null || { echo 'tmux missing. Install tmux in the selected image before launching.'; exit 2; }
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Session $SESSION already exists; inspect: tail -n 30 $RUN/logs/pipeline.log"
        exit 0
    fi
    printf -v COMMAND 'EMA_RUN_DIR=%q EMA_STAGE=%q bash %q --worker >> %q 2>&1' "$RUN" "$STAGE" "$SOURCE/start_ema_sft_20261021.sh" "$RUN/logs/pipeline.log"
    tmux new-session -d -s "$SESSION" "$COMMAND"
    echo "Started $SESSION; progress: tail -n 30 $RUN/logs/pipeline.log"
    exit 0
fi
trap 'code=$?; echo "EMA_PIPELINE_EXIT_CODE=$code"; printf "%s\n" "$code" > "$RUN/pipeline.exit"' EXIT
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1
echo "Verifying training kit before installing dependencies..."
python - "$ARCHIVE" "$RUN" <<'PY'
import hashlib,json,sys,zipfile
from pathlib import Path
archive,run=Path(sys.argv[1]),Path(sys.argv[2])
with archive.open('rb') as f:
    fingerprint=hashlib.file_digest(f,'sha256').hexdigest()
if fingerprint!='__BUNDLE_SHA256__':
    raise ValueError('Uploaded ZIP SHA256 differs from the reviewed local kit')
with zipfile.ZipFile(archive) as z:
    manifest=json.loads(z.read('project/EMA_BUNDLE_MANIFEST.json'))
    for name,record in manifest['files'].items():
        target=run/'project'/name
        if not target.resolve().is_relative_to((run/'project').resolve()):
            raise ValueError('Unsafe ZIP path')
        data=z.read('project/'+name)
        if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:
            raise ValueError('ZIP member checksum mismatch: '+name)
        if target.exists() and target.read_bytes()!=data:
            raise ValueError('Preserving existing different project file: '+str(target))
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.write_bytes(data)
    (run/'project/EMA_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('EMA_KIT_HASHES_VERIFIED',fingerprint,flush=True)
PY
python -c 'import sys,torch; assert sys.version_info[:2]==(3,11), "Select the Python3.11 image"; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA GPU required"; print("IMAGE_TORCH",torch.__version__,"GPU",torch.cuda.get_device_name(0),flush=True)'
python -m venv --system-site-packages "$RUN/.venv"
PY="$RUN/.venv/bin/python"
"$PY" -m pip install --timeout 30 --retries 2 -r "$RUN/project/requirements-qwen-sft.txt" 2>&1 | tee "$RUN/logs/dependencies.log"
cd "$RUN/project"
"$PY" -u scripts/run_ema_sft.py --run "$RUN" --stage "$STAGE" --model-source /mnt/Qwen2.5-1.5B-Instruct --backup-dir /mnt/ema-sft-20261021
