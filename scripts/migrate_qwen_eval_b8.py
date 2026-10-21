"""Revalidate retained checkpoints with BF16 batch8, then resume unchanged SFT.

The original trainer must be paused. Until revalidation and preflight pass, an
error resumes that original process. Checkpoints and source records are archived;
optimizer/scheduler/RNG files are hardlinked without modification.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from qwen_eval_runtime import (BATCHED_EXECUTION, contract_digest, project_validation_state,
                               validate_eval_migration)
from qwen_sft import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--paused-pid', type=int, required=True)
    parser.add_argument('--steps', type=int, nargs='+', default=[2000, 2500])
    args = parser.parse_args()
    run = args.run.resolve()
    project = run / 'project'
    out = run / 'outputs/qwen_ema_sft'
    history = run / 'history/single_user_eval_before_b8'
    migration = run / 'migration/eval_b8_20261021'
    migration.mkdir(parents=True, exist_ok=True)
    status_path = migration / 'status.json'
    if status_path.exists():
        raise ValueError('Preserve the existing migration; inspect its status first')
    status = dict(status='preflight', started_at_utc=datetime.now(timezone.utc).isoformat(),
                  evaluation_execution=BATCHED_EXECUTION, source_pid=args.paused_pid)

    def report(**values):
        status.update(values)
        write_json(status_path, status)
        print('EVAL_B8_MIGRATION', json.dumps(status), flush=True)

    def invoke(script, *values):
        subprocess.run([sys.executable, '-u', str(project / 'scripts' / script),
                        *map(str, values)], cwd=project, check=True)

    proc = Path(f'/proc/{args.paused_pid}')
    original_start = (proc / 'stat').read_text().split()[21]
    if b'/root/ema-sft-20261021/project/scripts/train_qwen_sft.py' not in (proc / 'cmdline').read_bytes().split(b'\0'):
        raise ValueError('Unexpected original trainer process')
    original_stopped = False
    try:
        report()
        data = project / 'data/processed/office2018/qwen_sft_ema'
        evaluation_config = project / 'experiments/qwen_eval_bf16_b8.json'
        source_contract = read_json(out / 'contract.json')
        source_state = read_json(out / f'checkpoint-{max(args.steps)}/trainer_state.json')
        if source_state['global_step'] != max(args.steps):
            raise ValueError('Resume step differs from checkpoint state')
        for step in args.steps:
            for name in ('model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json'):
                if not (out / f'checkpoint-{step}' / name).is_file():
                    raise ValueError(f'Incomplete checkpoint {step}: {name}')
        invoke('train_qwen_sft.py', '--config', project / 'experiments/qwen_sft_ema_20261021.json',
               '--data', data, '--model', run / 'assets/Qwen2.5-1.5B-Instruct',
               '--initialization', run / 'outputs/qwen_ema_q0', '--output', migration / 'dry_run',
               '--evaluation-config', evaluation_config, '--dry-run')
        target_code = {name: sha256(project / 'scripts' / name) for name in
                       (*source_contract['code_sha256'], 'qwen_eval_fast.py', 'qwen_eval_runtime.py')}
        record = dict(source_contract_sha256=contract_digest(source_contract),
                      target_code_sha256=target_code,
                      resume_weight_sha256=sha256(out / f'checkpoint-{max(args.steps)}/model.safetensors'),
                      resume_step=max(args.steps), original_progress_retained_through_step=max(args.steps),
                      evaluation_execution=BATCHED_EXECUTION, source_archive=str(history))
        target_contract = dict(source_contract, code_sha256=target_code, evaluation_execution=BATCHED_EXECUTION)
        validate_eval_migration(source_contract, target_contract, record)
        reports = []
        for step in sorted(args.steps):
            report(status='revalidating_retained_checkpoints', current_checkpoint=step)
            result_path = migration / f'checkpoint_{step}_valid5000_b8.json'
            invoke('evaluate_qwen_sft.py', '--checkpoint', out / f'checkpoint-{step}',
                   '--data', data, '--split', 'valid', '--users', 5000, '--seed', 2027,
                   '--beam', 20, '--max-length', 512, '--evaluation-config', evaluation_config,
                   '--output', result_path)
            result = read_json(result_path)
            if result['checkpoint_weight_sha256'] != sha256(out / f'checkpoint-{step}/model.safetensors'):
                raise ValueError('Candidate weight fingerprint mismatch')
            reports.append(dict(step=step, result=result))
        projected = project_validation_state(source_state, reports, out, source_contract['prepared_manifest_sha256'])
        write_json(migration / 'source_trainer_state.json', source_state)
        write_json(migration / 'projected_trainer_state.json', projected)
        record['candidate_reports'] = {f'checkpoint_{entry["step"]}_valid5000_b8.json':
            sha256(migration / f'checkpoint_{entry["step"]}_valid5000_b8.json') for entry in reports}
        # All validation/preflight checks pass before the original in-memory state is discarded.
        report(status='committing_migration')
        if (proc / 'stat').read_text().split()[21] != original_start:
            raise ValueError('Original process identity changed')
        os.kill(args.paused_pid, signal.SIGINT)
        os.kill(args.paused_pid, signal.SIGCONT)
        original_stopped = True
        for _ in range(150):
            if not proc.exists() and subprocess.run(['tmux', 'has-session', '-t', 'rec-ema-sft'],
                                                    capture_output=True).returncode:
                break
            time.sleep(.2)
        else:
            raise RuntimeError('Original pipeline has not exited; no checkpoint paths were moved')
        history.mkdir(parents=True, exist_ok=True)
        for name in ('logs', 'checks', 'outputs/qwen_ema_sft', 'pipeline.exit'):
            source = run / name
            if source.exists():
                target = history / name
                if target.exists():
                    raise ValueError('Preserve existing source archive: ' + str(target))
                target.parent.mkdir(parents=True, exist_ok=True)
                source.rename(target)
        (run / 'logs').mkdir()
        (run / 'checks').mkdir()
        out.mkdir(parents=True)
        source_out = history / 'outputs/qwen_ema_sft'

        def clone(source, target):
            if Path(source).suffix == '.json':
                return shutil.copy2(source, target)
            os.link(source, target)
            if os.stat(source).st_ino != os.stat(target).st_ino:
                raise ValueError('Checkpoint link identity mismatch')
            return target

        for step in args.steps:
            shutil.copytree(source_out / f'checkpoint-{step}', out / f'checkpoint-{step}', copy_function=clone)
        shutil.copy2(source_out / 'contract.json', out / 'contract.json')
        resume = out / f'checkpoint-{max(args.steps)}'
        record['source_trainer_state_sha256'] = sha256(resume / 'trainer_state.json')
        write_json(resume / 'trainer_state.json', projected)
        record['projected_trainer_state_sha256'] = sha256(resume / 'trainer_state.json')
        record['optimizer_scheduler_rng_preserved_by_hardlink'] = True
        record_path = migration / 'migration_record.json'
        write_json(record_path, record)
        for path in migration.glob('*.json'):
            shutil.copy2(path, run / 'checks' / path.name)
        for path in migration.glob('*.json'):
            shutil.copy2(path, Path('/mnt') / ('ema_eval_b8_' + path.name))
        command = (f'export PATH=/root/miniconda3/envs/myconda/bin:/root/miniconda3/bin:$PATH; '
                   f'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; '
                   f'cd {project}; {sys.executable} -u scripts/run_ema_sft.py --run {run} --stage all '
                   f'--backup-dir /mnt/ema-sft-20261021 --resume {resume} '
                   f'--evaluation-config {evaluation_config} --eval-migration-record {record_path} '
                   f'> {run}/logs/pipeline.log 2>&1; code=$?; '
                   f'echo EMA_PIPELINE_EXIT_CODE=$code >> {run}/logs/pipeline.log; echo $code > {run}/pipeline.exit')
        subprocess.run(['tmux', 'new-session', '-d', '-s', 'rec-ema-sft', command], check=True)
        report(status='batch8_pipeline_launched_from_checkpoint', resume_step=max(args.steps),
               migration_record=str(record_path), optimizer_scheduler_rng_preserved=True)
    except Exception as exc:
        report(status='failed', error=repr(exc), original_process_already_stopped=original_stopped)
        if not original_stopped and proc.exists() and (proc / 'stat').read_text().split()[21] == original_start:
            os.kill(args.paused_pid, signal.SIGCONT)
            report(original_training_resumed=True)
        raise


if __name__ == '__main__':
    main()
