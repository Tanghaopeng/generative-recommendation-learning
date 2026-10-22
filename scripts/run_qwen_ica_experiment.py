"""Sequential ICA experiment runner. Never competes with the active HPR GPU job."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def fingerprint(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify_archive(path, manifest):
    verified = set()
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            value = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    value.update(block)
            if member.name not in manifest or value.hexdigest() != manifest[member.name]['sha256']:
                raise RuntimeError('Compressed backup verification failed: ' + member.name)
            verified.add(member.name)
    if verified != set(manifest):
        raise RuntimeError('Compressed backup missing files')


def backup(root, project, destination):
    """Verified inference package; optimizer checkpoints stay on instance disk.

    Never delete existing cloud data to create space. The compressed archive is
    made and fully verified locally before checking available cloud quota.
    """
    package = root / 'ica_selected_and_results.tar.gz'
    members = {}
    groups = [('selected', root / 'formal/selected'), ('checks', root / 'checks')]
    for prefix, directory in groups:
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                members[f'{prefix}/{path.relative_to(directory).as_posix()}'] = path
    for relative in ('formal/run.json', 'formal/contract.json', 'formal/trainer_state.json', 'formal/gradient_audit.json'):
        path = root / relative
        if path.exists():
            members[relative] = path
    for name in ('qwen_ica.py', 'train_qwen_ica.py', 'evaluate_qwen_ica.py', 'run_qwen_ica_experiment.py',
                 'qwen_sft.py', 'qwen_eval_fast.py', 'qwen_hpr.py', 'qwen_hpr_decode.py', 'evaluate_qwen_hpr.py'):
        members['code/scripts/' + name] = project / 'scripts' / name
    members['code/experiments/qwen_ica_full.json'] = project / 'experiments/qwen_ica_full.json'
    members['code/requirements-qwen-sft.txt'] = project / 'requirements-qwen-sft.txt'
    manifest = {name: dict(bytes=path.stat().st_size, sha256=fingerprint(path)) for name, path in members.items()}
    with tarfile.open(package, 'w:gz', compresslevel=1) as archive:
        for name, path in members.items():
            archive.add(path, arcname=name, recursive=False)
    verify_archive(package, manifest)
    write(root / 'backup_manifest.json', dict(files=manifest, compressed_sha256=fingerprint(package),
        compressed_bytes=package.stat().st_size, includes_optimizer=False,
        restore='extract archive; run selected/ with the saved scripts/qwen_ica.py imported'))
    destination.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(destination).free
    if free < package.stat().st_size + 64 * 1024**2:
        write(root / 'backup_status.json', dict(status='cloud_space_insufficient', local_package=str(package),
            required_bytes=package.stat().st_size + 64 * 1024**2, available_bytes=free))
        # Small metrics/configs are still preserved; original cloud files untouched.
        for path in (root / 'checks').glob('*.json'):
            shutil.copy2(path, destination / path.name)
        shutil.copy2(root / 'backup_manifest.json', destination / 'backup_manifest.json')
        return False
    target = destination / package.name
    if target.exists():
        raise ValueError('Refuse to overwrite an existing cloud model package')
    temporary = target.with_suffix(target.suffix + '.partial')
    shutil.copyfile(package, temporary)
    if fingerprint(temporary) != fingerprint(package):
        raise RuntimeError('Cloud archive hash mismatch; do not release instance')
    temporary.replace(target)
    shutil.copy2(root / 'backup_manifest.json', destination / 'backup_manifest.json')
    write(root / 'backup_status.json', dict(status='verified', path=str(target), sha256=fingerprint(target)))
    shutil.copy2(root / 'backup_status.json', destination / 'backup_status.json')
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--sft-record', type=Path, required=True)
    p.add_argument('--wait-for-hpr', type=Path, required=True)
    p.add_argument('--cloud-backup', type=Path, required=True)
    args = p.parse_args()
    root, project = args.root.resolve(), args.project.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'runner_started.json').exists():
        raise ValueError('A runner already owns this directory; do not launch duplicate training')
    write(root / 'runner_started.json', dict(timestamp=time.time(), python=sys.executable))
    (root / 'logs').mkdir(exist_ok=True)
    (root / 'checks').mkdir(exist_ok=True)

    def phase(name, **values):
        value = dict(phase=name, timestamp=time.time(), **values)
        write(root / 'status.json', value)
        print(value, flush=True)

    def run(name, script, arguments):
        phase(name)
        command = [sys.executable, '-u', str(project / 'scripts' / script), *map(str, arguments)]
        with (root / 'logs' / (name + '.log')).open('w', encoding='utf-8') as stream:
            subprocess.run(command, cwd=project, stdout=stream, stderr=subprocess.STDOUT, check=True)

    phase('waiting_for_hpr_final_tests', dependency=str(args.wait_for_hpr))
    polls = 0
    while not (args.wait_for_hpr / 'completed.json').exists():
        exit_file = args.wait_for_hpr / 'pipeline.exit'
        if exit_file.exists():
            raise RuntimeError('HPR pipeline ended without verified completion; ICA will not start automatically')
        time.sleep(30)
        polls += 1
        if polls % 10 == 0:
            phase('waiting_for_hpr_final_tests', dependency=str(args.wait_for_hpr))
    # Confirm there are no GPU compute processes left, including a concurrent user job.
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if active:
        phase('waiting_for_gpu', active_pids=active)
        while active:
            time.sleep(30)
            active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    common = ['--checkpoint', args.checkpoint, '--sft-record', args.sft_record, '--data', args.data]
    run('smoke', 'train_qwen_ica.py', [*common, '--output', root / 'smoke', '--smoke-steps', '3'])
    gradients = json.loads((root / 'smoke/gradient_audit.json').read_text())
    if not gradients or not all(x['ica_grad_norm'] > 0 and x['qwen_grad_norm'] > 0 for x in gradients):
        raise RuntimeError('GPU ICA/Qwen gradient smoke failed')
    # Pilot optimizer state is disposable. Only this new, exactly checked path is removed.
    smoke_dir = (root / 'smoke').resolve()
    if smoke_dir.parent != root or smoke_dir.name != 'smoke':
        raise RuntimeError('Unexpected disposable smoke path')
    for path in smoke_dir.glob('checkpoint-*'):
        if path.resolve().parent != smoke_dir or path.is_symlink():
            raise RuntimeError('Unsafe disposable smoke checkpoint path')
        shutil.rmtree(path)
    run('formal', 'train_qwen_ica.py', [*common, '--output', root / 'formal'])
    selected = root / 'formal/selected'
    run('validation', 'evaluate_qwen_ica.py', ['--checkpoint', selected, '--data', args.data,
        '--split', 'valid', '--users', '5000', '--output', root / 'checks/valid.json'])
    run('test', 'evaluate_qwen_ica.py', ['--checkpoint', selected, '--data', args.data,
        '--split', 'test', '--selection-record', root / 'checks/valid.json',
        '--output', root / 'checks/test.json', '--export-candidates', root / 'checks/test_candidates.json'])
    test = json.loads((root / 'checks/test.json').read_text())
    comparisons = {}
    for name in ('test_sft', 'test_hpr'):
        original = json.loads((args.wait_for_hpr / 'checks' / (name + '.json')).read_text())
        if original['users'] != test['users']:
            raise RuntimeError('Comparison user sets differ')
        comparisons[name] = {key: original[key] for key in ('Recall@10', 'NDCG@10', 'Recall@20', 'NDCG@20')}
    comparisons['test_ica'] = {key: test[key] for key in ('Recall@10', 'NDCG@10', 'Recall@20', 'NDCG@20')}
    write(root / 'checks/comparison.json', dict(status='completed', metrics=comparisons,
        note='ICA includes additional SFT; a matched continued-SFT control is required for causal attribution'))
    phase('backup')
    saved = backup(root, project, args.cloud_backup)
    phase('completed' if saved else 'completed_needs_cloud_space', cloud_model_verified=saved)
    write(root / 'completed.json', dict(cloud_model_verified=saved, root=str(root), timestamp=time.time()))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Log an actionable failure; never release the paid instance automatically.
        print(f'ICA_RUN_FAILED: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        raise
