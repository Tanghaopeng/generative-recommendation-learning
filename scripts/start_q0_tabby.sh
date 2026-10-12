#!/bin/bash
# User runs this through tmux in Tabby; it exports Q0 only, without training/evaluation.
set -Eeuo pipefail
REF="${1:?Supply the full reviewed Git commit SHA}"
[[ "$REF" =~ ^[a-f0-9]{40}$ ]] || { echo 'Expected a full 40-character Git commit SHA'; exit 2; }
RUN=/root/generative-run-20261012
PROJECT="$RUN/generative-recommendation-learning"
PYTHON="$RUN/.venv/bin/python"
test -f "$RUN/setup_ready" || { echo 'Server preparation must pass first'; exit 2; }
test ! -e "$RUN/outputs/qwen_q0_seed2026" || { echo 'Q0 output already exists; preserving it'; exit 2; }
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$RUN/logs" "$RUN/checks/code_backups_q0_$REF"
cd "$PROJECT"
"$PYTHON" - "$REF" "$RUN" <<'PY'
import json
from pathlib import Path
import shutil
import sys
from urllib.request import urlopen

ref, run = sys.argv[1], Path(sys.argv[2])
project = run / 'generative-recommendation-learning'
names = ['qwen_sft.py', 'qwen_initialization.py', 'export_qwen_q0.py', 'train_qwen_sft.py',
         'evaluate_qwen_sft.py', 'evaluate_qwen_auc.py']
payloads = {}
for name in names:
    url = f'https://raw.githubusercontent.com/Tanghaopeng/generative-recommendation-learning/{ref}/scripts/{name}'
    with urlopen(url, timeout=30) as response:
        payloads[name] = response.read()
    compile(payloads[name], name, 'exec')
# Download and syntax-check all files before changing the bundled scripts.
for name, data in payloads.items():
    path = project / 'scripts' / name
    backup = run / f'checks/code_backups_q0_{ref}' / name
    if path.exists() and not backup.exists():
        shutil.copyfile(path, backup)
    path.write_bytes(data)
    print('Updated source:', name, flush=True)
(run / 'q0_code_revision.json').write_text(json.dumps({'git_commit':ref, 'files': names}, indent=2)+'\n')
PY
"$PYTHON" scripts/export_qwen_q0.py \
    --model "$RUN/assets/Qwen2.5-1.5B-Instruct" \
    --data data/processed/office2018/qwen_sft_adam \
    --seed 2026 --output "$RUN/outputs/qwen_q0_seed2026"
echo 'Q0_EXPORT_READY: no SFT, GRPO, or evaluation has been started.'
