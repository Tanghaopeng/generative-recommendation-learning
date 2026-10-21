"""Build a small offline EMA training kit, with prepared samples and no Qwen weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from check_ema_sft_inputs import ROOT, check, digest, read


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,default=ROOT/'outputs/pretrained/Qwen2.5-1.5B-Instruct')
    args=p.parse_args()
    if args.output.exists():raise ValueError('Use a new archive path')
    proof=check()
    source=read(ROOT/'experiments/ema_sft_20261021/model_source_manifest.json')
    if digest(args.model/'model.safetensors')!=source['files']['model.safetensors']['sha256']:
        raise ValueError('Local original Qwen weight hash differs from pinned source')
    names=['requirements-qwen-sft.txt','experiments/qwen_sft_ema_20261021.json',
        'experiments/qwen_eval_bf16_b8.json',
        'experiments/ema_sft_20261021/model_source_manifest.json',
        'experiments/qwen_sft_office2018_20261013/test.json',
        '04_推荐生成/02_Qwen2.5-1.5B-Instruct/configs/q1_full.json',
        '04_推荐生成/02_Qwen2.5-1.5B-Instruct/EMA_SFT运行说明.md']
    names += ['scripts/'+n for n in ('qwen_sft.py','qwen_initialization.py','export_qwen_q0.py',
        'train_qwen_sft.py','prepare_qwen_sft.py','evaluate_qwen_sft.py','freeze_qwen_training_selection.py',
        'check_ema_sft_inputs.py','run_ema_sft.py','qwen_eval_fast.py','qwen_eval_runtime.py',
        'migrate_qwen_eval_b8.py')]
    payload={name:(ROOT/name).read_bytes() for name in names}
    for directory in ('qwen_sft_ema','tiger_sids_ema_20261008'):
        for f in (ROOT/'data/processed/office2018'/directory).iterdir():
            if f.is_file():payload[f.relative_to(ROOT).as_posix()]=f.read_bytes()
    for name,record in source['files'].items():
        if name=='model.safetensors':continue
        data=(args.model/name).read_bytes()
        if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:
            raise ValueError('Base model metadata differs: '+name)
        payload['assets/qwen_base_metadata/'+name]=data
    manifest=dict(status='ema_bundle_verified',recipe='original Qwen Q1 full SFT with pinned EMA SID',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        preparation=proof,external_weight='/mnt/Qwen2.5-1.5B-Instruct/model.safetensors',
        external_weight_bytes=source['files']['model.safetensors']['bytes'],
        external_weight_sha256=source['files']['model.safetensors']['sha256'],
        files={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in payload.items()})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in payload.items():z.writestr('project/'+name,data)
        z.writestr('project/EMA_BUNDLE_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(args.output) as z:
        assert z.testzip() is None
        for name,record in manifest['files'].items():
            data=z.read('project/'+name)
            assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256']
    print('Validating the actual extracted SFT entry point on CPU; no 1.5B model is loaded...',flush=True)
    with tempfile.TemporaryDirectory(prefix='ema_sft_check_',dir=args.output.parent) as folder:
        stage=Path(folder).resolve()
        assert stage.is_relative_to(args.output.parent.resolve())
        with zipfile.ZipFile(args.output) as z:
            if any(not (stage/name).resolve().is_relative_to(stage) for name in z.namelist()):
                raise ValueError('Unsafe archive path')
            z.extractall(stage)
        project=stage/'project'
        result=subprocess.run([sys.executable,'-u',str(project/'scripts/train_qwen_sft.py'),
            '--config',str(project/'experiments/qwen_sft_ema_20261021.json'),
            '--data',str(project/'data/processed/office2018/qwen_sft_ema'),
            '--model',str(args.model.resolve()),'--output',str(stage/'dry_run'),'--dry-run',
            '--evaluation-config',str(project/'experiments/qwen_eval_bf16_b8.json')],
            cwd=project,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',timeout=300)
        if result.returncode:
            raise RuntimeError('Extracted training entry point failed:\n'+result.stdout[-6000:])
        check_record=read(stage/'dry_run/dry_run.json')
        assert check_record['train_rows']==397345 and check_record['validation_users']==5000
        assert check_record['actual_gpu_training'] is False
        assert check_record['evaluation_execution']['batch_size']==8
    receipt=dict(status='local_bundle_verified_cloud_upload_not_verified',archive=args.output.name,
        bytes=args.output.stat().st_size,sha256=digest(args.output),files=len(payload),
        external_weight_sha256=manifest['external_weight_sha256'],extracted_sft_dry_run='passed')
    args.output.with_suffix('.manifest.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    launcher=ROOT/'scripts/start_ema_sft.sh'
    text=launcher.read_text(encoding='utf-8').replace('__BUNDLE_SHA256__',receipt['sha256']).replace('__BUNDLE_NAME__',args.output.name)
    target=args.output.parent/f'start_{args.output.stem}.sh'
    with target.open('w',encoding='utf-8',newline='\n') as stream:stream.write(text)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    main()
