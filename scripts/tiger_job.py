"""One-off local job: train, then export the completed aggregate experiment.

This is not a scheduler. Status and child PID are stored alongside local logs.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--codec', required=True)
    p.add_argument('--name', default='tiger_cpu_20261008')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--ff', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--eval-batch-size', type=int, default=32)
    args = p.parse_args()
    output = Path(args.output)
    status_path = output.with_suffix('.job.json')
    if status_path.exists() or (output.exists() and any(output.iterdir())):
        raise ValueError('Choose new job and experiment paths')
    status_path.parent.mkdir(parents=True,exist_ok=True)
    def record(stage, **fields):
        info.update(status=stage,updated_at_utc=datetime.now(timezone.utc).isoformat(),**fields)
        temp = status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        temp.replace(status_path)
    info = dict(arguments=vars(args),python=sys.executable)
    command = [sys.executable, str(ROOT/'scripts/run_tiger.py'), '--output',str(output),
               '--epochs',str(args.epochs),'--threads','2','--hidden',str(args.hidden),
               '--layers',str(args.layers),'--heads',str(args.heads),'--ff',str(args.ff),
               '--batch-size',str(args.batch_size),'--eval-batch-size',str(args.eval_batch_size)]
    try:
        child = subprocess.Popen(command,cwd=ROOT)
        record('training',child_pid=child.pid,command=command)
        if child.wait() != 0:
            raise RuntimeError('Training failed; inspect the local stderr log')
        record('exporting_report',child_pid=None)
        subprocess.run([sys.executable,str(ROOT/'scripts/report_tiger.py'),'--run',str(output),
                        '--codec',args.codec,'--name',args.name],cwd=ROOT,check=True)
        record('completed',report=f'experiments/{args.name}.md')
    except BaseException as error:
        record('failed',error=str(error))
        raise


if __name__ == '__main__':
    main()
