"""Local inference against the frozen SID catalog; no held-out labels are needed."""
import argparse
import json
import torch

from tiger_common import PROCESSED, configure, load_sid
from tiger_model import TigerModel, pack_histories


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--sids', default=str(PROCESSED/'tiger_sids'))
    p.add_argument('--history', required=True, help='Comma-separated processed item IDs, oldest first')
    p.add_argument('--topk', type=int, default=10)
    p.add_argument('--beam', type=int, default=20)
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    configure(threads=args.threads)
    sid, info = load_sid(args.sids)
    saved = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if saved['sid_sha256'] != info['sid_sha256']:
        raise ValueError('Checkpoint and SID catalog differ')
    history = [int(v.strip()) for v in args.history.split(',')]
    if not history or any(i <= 0 or i >= len(sid) for i in history):
        raise ValueError('History must contain valid processed catalog IDs')
    if args.topk < 1 or args.beam < args.topk:
        raise ValueError('Require beam >= topk >= 1')
    model = TigerModel(sid,saved['config'])
    model.load_state_dict(saved['state_dict'])
    model.eval()
    tokens, mask = pack_histories([history],sid,saved['maxlen'])
    ids, scores = model.retrieve(tokens,mask,[set(history)],beam=args.beam,topk=args.topk)
    catalog = {r['item_id']:r for r in map(json.loads,(PROCESSED/'items.jsonl').open(encoding='utf-8'))}
    result = [dict(item_id=i,title=catalog[i]['title'],semantic_id=sid[i].tolist(),log_probability=s)
              for i,s in zip(ids[0].tolist(),scores[0].tolist()) if i]
    print(json.dumps(dict(smoke_checkpoint=saved['smoke_only'],recommendations=result),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
