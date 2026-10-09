"""One-off dependent experiment: original retriever completes, then EMA SID run.

No scheduled recurrence, no replacement of the original SID/checkpoint artifacts.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--wait-for',required=True)
    p.add_argument('--sids',required=True)
    p.add_argument('--codec',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--name',default='tiger_ema_cpu_20261008')
    args=p.parse_args()
    out=Path(args.output)
    status_path=out.with_suffix('.job.json')
    if status_path.exists() or (out.exists() and any(out.iterdir())):
        raise ValueError('Use new EMA experiment and job paths')
    status_path.parent.mkdir(parents=True,exist_ok=True)
    original=Path(args.wait_for)
    info=dict(arguments=vars(args),launcher_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def record(stage,**fields):
        info.update(status=stage,updated_at_utc=datetime.now(timezone.utc).isoformat(),**fields)
        temp=status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        for attempt in range(40):
            try:
                temp.replace(status_path)
                return
            except PermissionError:
                if attempt==39:
                    raise
                time.sleep(.05)
    try:
        sid=read(Path(args.sids)/'manifest.json')
        codec=read(Path(args.codec)/'run.json')
        if sid['status']!='completed' or codec['status']!='completed' or codec.get('codebook_update','').split(';')[0]!='EMA':
            raise ValueError('Requires completed EMA quantizer and SID artifacts')
        record('waiting_for_original_retriever',sid_sha256=sid['sid_sha256'])
        print('Queued EMA SID retriever; waiting for original full run.',flush=True)
        deadline=time.monotonic()+8*3600
        while True:
            original_job=original.with_suffix('.job.json')
            if original_job.exists() and read(original_job)['status']=='failed':
                raise RuntimeError('Original job failed; EMA run has not been started')
            if (original/'run.json').exists():
                state=read(original/'run.json')
                if state['status']=='completed' and 'test' in state:
                    break
            if time.monotonic()>deadline:
                raise TimeoutError('Original retriever did not complete within 8 hours')
            time.sleep(15)
        # Inherit every learning/search setting, including requested total
        # epochs (the same early-stop rule remains active independently).
        a=state['arguments']
        command=[sys.executable,str(ROOT/'scripts/run_tiger.py'),'--sids',args.sids,'--output',str(out)]
        for key in ('epochs','batch_size','eval_batch_size','maxlen','beam','hidden','layers','heads','ff','lr','threads','seed','patience'):
            command.extend(['--'+key.replace('_','-'),str(a[key])])
        child=subprocess.Popen(command,cwd=ROOT)
        record('training',child_pid=child.pid,command=command,original_best_epoch=state['best_epoch'])
        if child.wait()!=0:
            raise RuntimeError('EMA SID retriever failed; inspect stderr')
        record('exporting_report',child_pid=None)
        subprocess.run([sys.executable,str(ROOT/'scripts/report_tiger.py'),'--run',str(out),
                        '--codec',args.codec,'--name',args.name],cwd=ROOT,check=True)
        report_path=ROOT/'experiments'/f'{args.name}.md'
        text=report_path.read_text(encoding='utf-8')
        text=text.replace('# 小型 TIGER 首轮 CPU 实验','# 使用EMA语义ID的小型TIGER对照实验',1)
        text=text.replace('| 小型 TIGER 首轮 |','| 小型 TIGER（EMA SID） |')
        baseline=state['test']
        text+='\n原版生成模型（Adam码本SID，同样的数据/训练/搜索配置）测试对照：\n\n'
        text+=f'- 原版 Recall@10={baseline["Recall@10"]:.6f}，NDCG@10={baseline["NDCG@10"]:.6f}。\n'
        text+='- 两版均从随机初始化独立训练，分别按各自验证NDCG@10选模；不将新SID直接替换进旧模型。\n'
        report_path.write_text(text,encoding='utf-8')
        record('completed',report=f'experiments/{args.name}.md')
    except BaseException as error:
        record('failed',error=str(error))
        raise


if __name__=='__main__':
    main()
