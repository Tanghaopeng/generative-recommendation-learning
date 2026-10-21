"""Server runner: verify -> original-model Q0 -> matched EMA SFT -> backup -> test."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from check_ema_sft_inputs import ROOT, check, digest, read


def save(path,record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)


def copy_verified(source,target,expected=None):
    source,target=Path(source),Path(target)
    expected=expected or digest(source)
    if target.is_file():
        if digest(target)!=expected:
            raise ValueError('Preserving an existing different file: '+str(target))
        return expected
    target.parent.mkdir(parents=True,exist_ok=True)
    tmp=target.with_name(target.name+'.partial')
    if tmp.exists():
        raise ValueError('Preserving incomplete copy; inspect or use a new run directory: '+str(tmp))
    started=time.monotonic();total=0;next_report=256*1024**2
    with source.open('rb') as src,tmp.open('xb') as dst:
        for block in iter(lambda:src.read(8*1024**2),b''):
            dst.write(block);total+=len(block)
            if total>=next_report:
                print(f'COPY {source.name}: {total}/{source.stat().st_size} bytes, {total/2**20/max(.001,time.monotonic()-started):.1f} MiB/s',flush=True)
                next_report+=256*1024**2
    if digest(tmp)!=expected:
        raise ValueError('Copy checksum failed: '+str(tmp))
    tmp.replace(target)
    return expected


def run_python(script,*args,log_dir):
    command=[sys.executable,'-u',str(ROOT/'scripts'/script),*map(str,args)]
    print('RUN',script,*map(str,args),flush=True)
    with (log_dir/(script.removesuffix('.py')+'.log')).open('a',encoding='utf-8',buffering=1) as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,encoding='utf-8',errors='replace',bufsize=1)
        for line in process.stdout:
            log.write(line);print(line,end='',flush=True)
        code=process.wait()
    if code:
        raise RuntimeError(f'{script} exited {code}; inspect {log_dir}')


def backup(run,cloud,include_resume=False):
    """Save a reloadable selected model first; optional full optimizer checkpoint."""
    selected=run/'outputs/qwen_ema_sft/selected'
    if not (selected/'model.safetensors').is_file():
        raise ValueError('No complete selected model to back up')
    mapping={p:cloud/'selected'/p.relative_to(selected) for p in selected.rglob('*') if p.is_file()}
    data=ROOT/'data/processed/office2018'
    for folder in ('qwen_sft_ema','tiger_sids_ema_20261008'):
        for p in (data/folder).iterdir():
            if p.is_file():mapping[p]=cloud/'data'/folder/p.name
    for p in (run/'outputs/qwen_ema_sft').glob('*.json'):
        mapping[p]=cloud/'training_records'/p.name
    for p in (run/'checks').glob('*.json'):
        mapping[p]=cloud/'checks'/p.name
    mapping[run/'outputs/qwen_ema_q0/q0_initialization.json']=cloud/'training_records/q0_initialization.json'
    mapping[ROOT/'experiments/qwen_sft_ema_20261021.json']=cloud/'training_records/training_config.json'
    for p in (ROOT/'scripts').glob('*.py'):
        mapping[p]=cloud/'code'/p.name
    mapping[ROOT/'requirements-qwen-sft.txt']=cloud/'code/requirements-qwen-sft.txt'
    for p in (run/'logs').glob('*.log'):
        mapping[p]=cloud/'logs'/p.name
    if include_resume:
        candidates=[p for p in (run/'outputs/qwen_ema_sft').glob('checkpoint-*')
            if (p/'optimizer.pt').is_file() and (p/'scheduler.pt').is_file() and (p/'trainer_state.json').is_file()]
        if not candidates:raise ValueError('No complete resume checkpoint found')
        # Retain the latest and validation-best checkpoint, plus the exact Q0
        # record/weights: the Trainer resume contract fingerprints that export.
        for checkpoint in candidates:
            mapping.update({p:cloud/'resume'/checkpoint.name/p.relative_to(checkpoint)
                for p in checkpoint.rglob('*') if p.is_file()})
        initial=run/'outputs/qwen_ema_q0'
        mapping.update({p:cloud/'resume/qwen_ema_q0'/p.relative_to(initial)
            for p in initial.rglob('*') if p.is_file()})
        mapping.update({p:cloud/'resume/project'/p.relative_to(ROOT)
            for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts})
    # Logs may grow during evaluation. Give each backup phase its own metadata/log snapshot.
    phase='test_completed' if (run/'checks/test.json').is_file() else 'sft_completed'
    snapshot=phase+'-'+str(time.time_ns())
    mapping={src:(cloud/'snapshots'/snapshot/dst.relative_to(cloud))
        if dst.relative_to(cloud).parts[0] in ('checks','logs') else dst for src,dst in mapping.items()}
    cloud.mkdir(parents=True,exist_ok=True)
    required=sum(p.stat().st_size for p,target in mapping.items() if not target.exists())
    if shutil.disk_usage(cloud).free<required+256*1024**2:
        raise RuntimeError('Not enough reported persistent disk space for verified backup')
    files={}
    for source,target in mapping.items():
        checksum=copy_verified(source,target)
        files[target.relative_to(cloud).as_posix()]=dict(bytes=target.stat().st_size,sha256=checksum)
    result=dict(status='backup_verified',phase=phase,files=files,
        selected_weight_sha256=digest(selected/'model.safetensors'),
        resume_checkpoint_included=include_resume,
        note='Selected model, SID tokenizer, mapping, data and code are reusable; optimizer resume requires the optional resume folder.')
    save(cloud/f'backup_{snapshot}.json',result)
    save(cloud/'LATEST_BACKUP.json',dict(manifest=f'backup_{snapshot}.json',phase=phase))
    print('PERSISTENT_BACKUP_VERIFIED',cloud,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT.parent)
    p.add_argument('--model-source',type=Path,default=Path('/mnt/Qwen2.5-1.5B-Instruct'))
    p.add_argument('--backup-dir',type=Path,default=Path('/mnt/ema-sft-20261021'))
    p.add_argument('--stage',choices=['prepare','train','test','backup','all'],default='all')
    p.add_argument('--resume',type=Path)
    p.add_argument('--backup-resume',action='store_true')
    args=p.parse_args()
    run=args.run.resolve(); logs=run/'logs';checks=run/'checks'
    logs.mkdir(parents=True,exist_ok=True);checks.mkdir(exist_ok=True)
    os.environ.update(TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_TELEMETRY='1',PYTHONUNBUFFERED='1')
    # Set before importing torch; use the allocator verified in the batch-16 probe.
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    evidence=check();save(checks/'input_verification.json',evidence)
    data=ROOT/'data/processed/office2018/qwen_sft_ema'
    q0=run/'outputs/qwen_ema_q0';out=run/'outputs/qwen_ema_sft'
    config=ROOT/'experiments/qwen_sft_ema_20261021.json'
    model=run/'assets/Qwen2.5-1.5B-Instruct'
    if args.stage in ('prepare','train','all'):
        args.backup_dir.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(args.backup_dir).free<4*1024**3:
            raise RuntimeError('Reserve at least 4 GiB persistent space for selected-model backup before starting training')
        import torch
        import importlib.metadata
        if (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
            or torch.cuda.device_count()!=1):
            raise RuntimeError('Expose one BF16 CUDA GPU; do not install/replace image PyTorch blindly')
        tensor=torch.randn(128,128,device='cuda',dtype=torch.bfloat16)
        assert torch.isfinite(tensor@tensor.T).all()
        save(checks/'environment.json',dict(gpu=torch.cuda.get_device_name(0),
            total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,cuda=torch.version.cuda,
            cuda_allocator_config=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
            versions={n:importlib.metadata.version(n) for n in ('torch','transformers','accelerate','numpy','safetensors')},
            bf16_smoke_test='passed'))
        source=read(ROOT/'experiments/ema_sft_20261021/model_source_manifest.json')
        for name,entry in source['files'].items():
            origin=args.model_source/name if name=='model.safetensors' else ROOT/'assets/qwen_base_metadata'/name
            copy_verified(origin,model/name,entry['sha256'])
        if not q0.exists():
            run_python('export_qwen_q0.py','--model',model,'--data',data,'--seed',2026,'--output',q0,log_dir=logs)
        from qwen_initialization import check_initialization
        check_initialization(q0,data,2026)
        dry=checks/'sft_dry_run'
        if dry.exists():
            previous=read(dry/'dry_run.json')
            if previous['prepared_manifest_sha256']!=evidence['prepared_manifest_sha256'] or previous['config']!=read(config):
                raise ValueError('Existing dry run differs; preserve it and inspect')
        else:
            run_python('train_qwen_sft.py','--config',config,'--data',data,'--model',model,
                '--initialization',q0,'--output',dry,'--dry-run',log_dir=logs)
        save(checks/'prepare_ready.json',dict(status='ready_for_formal_ema_sft',optimizer_steps=0,
            sid_sha256=evidence['sid_sha256'],prepared_manifest_sha256=evidence['prepared_manifest_sha256']))
        print('EMA_PREPARE_READY; original Qwen + deterministic untrained SID rows, no optimizer step.',flush=True)
    if args.stage in ('train','all'):
        if not (out/'run.json').is_file() or read(out/'run.json')['status']!='sft_completed_requires_full_validation_before_test':
            extra=['--resume',args.resume] if args.resume else []
            run_python('train_qwen_sft.py','--config',config,'--data',data,'--model',model,
                '--initialization',q0,'--output',out,*extra,log_dir=logs)
    if args.stage in ('train','test','all'):
        selection=checks/'selection_valid5000.json'
        if not selection.exists():
            run_python('freeze_qwen_training_selection.py','--run',out,'--data',data,'--output',selection,log_dir=logs)
        if read(selection)['checkpoint_weight_sha256']!=digest(out/'selected/model.safetensors'):
            raise ValueError('Selected model fingerprint differs')
        backup(run,args.backup_dir,args.backup_resume)
    if args.stage=='backup':
        backup(run,args.backup_dir,args.backup_resume)
    if args.stage in ('test','all'):
        result=checks/'test.json'
        if not result.exists():
            run_python('evaluate_qwen_sft.py','--checkpoint',out/'selected','--data',data,
                '--split','test','--users',0,'--seed',2027,'--beam',20,'--max-length',512,
                '--selection-record',checks/'selection_valid5000.json','--output',result,log_dir=logs)
        new=read(result);old=read(ROOT/'experiments/qwen_sft_office2018_20261013/test.json')
        if (new['users']!=86713 or new['split']!='test' or new['diagnostic_only']
            or new['checkpoint_weight_sha256']!=digest(out/'selected/model.safetensors')
            or new['prepared_manifest_sha256']!=evidence['prepared_manifest_sha256']):
            raise ValueError('Test artifact fingerprint/scope mismatch')
        keys=('Recall@10','NDCG@10','Recall@20','NDCG@20')
        comparison=dict(status='test_comparison_completed',baseline='gradient RQ-VAE Qwen Q1 SFT',
            candidate='EMA RQ-VAE Qwen Q1 SFT',
            same_training_config=evidence['config_identical_to_gradient_baseline'],
            config_differences_from_gradient_baseline=evidence['config_differences_from_gradient_baseline'],
            effective_batch_matches_gradient_baseline=evidence['effective_batch_matches_gradient_baseline'],
            single_seed=2026,
            metrics={k:dict(gradient=old[k],ema=new[k],absolute_delta=new[k]-old[k],
                relative_delta=(new[k]-old[k])/old[k]) for k in keys},
            gradient_weight_sha256=old['checkpoint_weight_sha256'],ema_weight_sha256=new['checkpoint_weight_sha256'])
        save(checks/'gradient_vs_ema.json',comparison)
        backup(run,args.backup_dir,args.backup_resume)
        save(run/'completed.json',dict(status='ema_sft_test_and_backup_completed',backup=str(args.backup_dir)))
        print('EMA_SFT_TEST_AND_BACKUP_COMPLETED',json.dumps(comparison),flush=True)


if __name__=='__main__':
    main()
