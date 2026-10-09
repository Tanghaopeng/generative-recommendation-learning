"""Synthetic CPU load benchmark only; no recommendation accuracy is computed."""
import argparse
from pathlib import Path
import time

import numpy as np
import psutil
import torch

from tiger_common import configure, write_json
from tiger_model import TigerModel, pack_histories
from build_semantic_ids import disambiguate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--hidden',type=int,default=128)
    p.add_argument('--layers',type=int,default=2)
    p.add_argument('--heads',type=int,default=4)
    args = p.parse_args()
    configure(threads=2)
    rng = np.random.default_rng(2026)
    full, counts = disambiguate(rng.integers(0,256,size=(25898,3)))
    sid = np.concatenate([np.full((1,4),-1),full])
    config = dict(vocabulary=256,hidden=args.hidden,layers=args.layers,heads=args.heads,ff=args.hidden*2)
    model = TigerModel(sid,config)
    optimizer = torch.optim.Adam(model.parameters(),lr=.001)
    report = dict(smoke_only=True,synthetic=True,threads=2,config=config,
                  parameters=sum(p.numel() for p in model.parameters()),timings={})
    for length in (3,20):
        histories = rng.integers(1,len(sid),size=(128,length)).tolist()
        tokens,mask = pack_histories(histories,sid)
        targets = torch.from_numpy(sid[rng.integers(1,len(sid),size=128)])
        model.train()
        durations = []
        for step in range(4):
            before = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            model(tokens,mask,targets).backward()
            optimizer.step()
            durations.append(time.perf_counter()-before)
        report['timings'][f'train_128_history{length}_median_seconds'] = float(np.median(durations[1:]))
    for length in (3,20):
        histories = rng.integers(1,len(sid),size=(32,length)).tolist()
        tokens,mask = pack_histories(histories,sid)
        model.eval()
        durations = []
        for step in range(3):
            before = time.perf_counter()
            model.retrieve(tokens,mask,[set(h) for h in histories],beam=20,topk=20)
            durations.append(time.perf_counter()-before)
        report['timings'][f'retrieve_32_history{length}_median_seconds'] = float(np.median(durations[1:]))
    report['observed_rss_bytes'] = psutil.Process().memory_info().rss
    write_json(Path(args.output),report)
    print(report,flush=True)


if __name__ == '__main__':
    main()
