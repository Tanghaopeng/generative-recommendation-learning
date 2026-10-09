"""Export aggregate experiment evidence; never publish data mappings or weights."""
import argparse
import json
from pathlib import Path

from tiger_common import ROOT, PROCESSED, write_json, utc_now


def relative(value):
    if isinstance(value, dict):
        return {k:relative(v) for k,v in value.items()}
    if isinstance(value, list):
        return [relative(v) for v in value]
    if isinstance(value, str):
        return value.replace(str(ROOT), '.').replace('\\','/')
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, help='Directory containing the completed run.json')
    p.add_argument('--codec', required=True)
    p.add_argument('--embeddings', default=str(PROCESSED/'tiger_embeddings'))
    p.add_argument('--name', default='tiger_cpu_20261008')
    args = p.parse_args()
    run = json.loads((Path(args.run)/'run.json').read_text(encoding='utf-8'))
    if run['status'] != 'completed' or run['smoke_only'] or 'test' not in run:
        raise ValueError('Formal reports require a completed non-smoke full test')
    codec = json.loads((Path(args.codec)/'run.json').read_text(encoding='utf-8'))
    embeddings = json.loads((Path(args.embeddings)/'manifest.json').read_text(encoding='utf-8'))
    if run['sid_manifest']['embedding_sha256'] != embeddings['embedding_sha256']:
        raise ValueError('Embedding versions differ')
    if codec['embedding_sha256'] != embeddings['embedding_sha256']:
        raise ValueError('Codec versions differ')
    report = relative(dict(exported_at_utc=utc_now(),retriever=run,codec=codec,embeddings=embeddings))
    write_json(ROOT/'experiments'/f'{args.name}.json',report)
    metric = run['test']
    lines = ['# 小型 TIGER 首轮 CPU 实验', '',
             '这是一轮个人简化实现实验：复用固定第三方 RQ-VAE 和 T5 模型，独立适配四段 SID、共同样本、确定性检索和评测。',
             '不声称复现 TIGER 论文成绩。模型按验证 NDCG@10 选择，测试在选定之后执行一次。', '',
             f'- 参数量：{run["parameters"]:,}；训练轮数：{len(run["history"])}；最佳轮：{run["best_epoch"]}。',
             f'- 停止原因：`{run["stop_reason"]}`；总运行时间：{run["elapsed_seconds"]/60:.1f} 分钟。',
             f'- 每轮有效下一物品样本：{run["training_examples"]:,}；CPU 线程：{run["arguments"]["threads"]}。',
             f'- 文本编码：冻结 MiniLM，384 维，最多 {embeddings["contract"]["max_tokens"]} token；缺元数据 {embeddings["missing_metadata"]} 件。',
             f'- 超出截断上限的商品文本：{embeddings["truncated_text_items"]:,}件，占 {embeddings["truncated_text_items"]/embeddings["contract"]["items"]:.2%}；优先保留标题、品牌和分类。',
             f'- RQ-VAE：3×256 码本，32 维隐空间；第 {codec["best_epoch"]} 轮重建验证最佳。',
             f'- 三段 ID 重复比例：{run["sid_manifest"]["raw_duplicate_fraction"]:.2%}；最大碰撞组：{run["sid_manifest"]["max_collision_group"]}；最终 SID 全部唯一。', '',
             '| 方法 | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |',
             '|---|---:|---:|---:|---:|',
             '| SASRec 第23轮基线 | 0.057650 | 0.035554 | 0.074406 | 0.039775 |',
             f'| 小型 TIGER 首轮 | {metric["Recall@10"]:.6f} | {metric["NDCG@10"]:.6f} | {metric["Recall@20"]:.6f} | {metric["NDCG@20"]:.6f} |', '',
             '每人一个下一商品目标，所以 HR=Recall。相同用户、目录、用户内时间切分、完整历史过滤和最近20件商品输入。',
             f'生成模型使用全目录前缀树、固定 beam={metric["beam"]}；属于有限 beam 近似搜索，SASRec 属于逐商品精确打分。',
             '生成概率在限制合法 token 之前计算，不按合法分支重新归一化；训练四段 token 交叉熵，不另采商品负例。',
             f'测试 {metric["users"]:,} 用户；输出不足20件的用户 {metric["shortfall_users"]:,}；返回商品合法率 {metric["legal_item_fraction"]:.2%}。',
             f'推理批大小 {metric["timing_batch_size"]}，预热后批次 P50/P95 {metric["warm_batch_seconds_p50"]:.3f}/{metric["warm_batch_seconds_p95"]:.3f} 秒；这不是单请求线上延迟。', '',
             '单种子、首轮小模型，未做 beam 敏感性、超参数搜索或严格全局时间评测。商品文本使用固定完整目录，不使用留出评论或行为共购图。',
             '完整配置、逐轮验证、指纹和编码/RQ-VAE 统计见同名 JSON；用户记录、SID映射、原始数据和权重均留在忽略目录。', '',
             '| 轮次 | 训练 loss | 验证 Recall@10 | 验证 NDCG@10 | 训练秒 | 验证秒 |',
             '|---|---:|---:|---:|---:|---:|']
    for row in run['history']:
        v = row['validation']
        lines.append(f'| {row["epoch"]} | {row["training_loss"]:.5f} | {v["Recall@10"]:.6f} | {v["NDCG@10"]:.6f} | {row["training_seconds"]:.1f} | {v["seconds"]:.1f} |')
    (ROOT/'experiments'/f'{args.name}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
