"""Compare completed, matched retrievers trained on Adam and EMA semantic IDs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = ('Recall@10', 'NDCG@10', 'Recall@20', 'NDCG@20')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def relative(value):
    if isinstance(value, dict):
        return {key: relative(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relative(item) for item in value]
    if isinstance(value, str):
        return value.replace(str(ROOT), '.').replace('\\', '/')
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adam', default='outputs/tiger_office2018_cpu_20261008')
    parser.add_argument('--ema', default='outputs/tiger_office2018_ema_cpu_20261008')
    parser.add_argument('--name', default='semantic_retrievers_comparison_20261008')
    args = parser.parse_args()
    runs = [read(ROOT / directory / 'run.json') for directory in (args.adam, args.ema)]
    adam, ema = runs
    if any(run['status'] != 'completed' or run['smoke_only'] or 'test' not in run for run in runs):
        raise ValueError('Both formal retrievers must finish before exporting the comparison')
    for key in ('config', 'training_examples', 'parameters', 'data_sha256', 'sequence_sha256',
                'source_sha256', 'checkpoint_selection', 'training', 'evaluation', 'negative_sampling'):
        if adam[key] != ema[key]:
            raise ValueError(f'Unmatched experiments: {key}')
    matched = ('epochs', 'batch_size', 'eval_batch_size', 'maxlen', 'beam', 'hidden', 'layers',
               'heads', 'ff', 'lr', 'threads', 'seed', 'patience')
    if any(adam['arguments'][key] != ema['arguments'][key] for key in matched):
        raise ValueError('Training/search settings differ')
    if adam['sid_manifest']['sid_sha256'] == ema['sid_manifest']['sid_sha256']:
        raise ValueError('Expected two different SID mappings')
    if adam['sid_manifest']['embedding_sha256'] != ema['sid_manifest']['embedding_sha256']:
        raise ValueError('Text embeddings differ')
    for run in runs:
        best = max(run['history'], key=lambda row: row['validation']['NDCG@10'])
        if best['epoch'] != run['best_epoch']:
            raise ValueError('Selected checkpoint does not match validation history')
        if any(row['validation']['users'] != run['test']['users'] for row in run['history']):
            raise ValueError('Validation/test user counts differ')
        if run['test']['smoke_only'] or run['test']['beam'] != run['arguments']['beam']:
            raise ValueError('Test/search protocol differs')
        if digest(ROOT / run['arguments']['sids'] / 'semantic_ids.npy') != run['sid_manifest']['sid_sha256']:
            raise ValueError('Frozen SID mapping changed')
        if digest(ROOT / run['arguments']['output'] / 'best.pth') != run['checkpoint_sha256']:
            raise ValueError('Selected model checkpoint changed')
    if any(digest(ROOT / path) != fingerprint for path, fingerprint in adam['source_sha256'].items()):
        raise ValueError('Retriever sources changed since training')
    baseline = read(ROOT / 'experiments/sasrec_cpu_extend40_20261007.json')
    if baseline['data_sha256'] != adam['data_sha256'] or baseline['test']['users'] != adam['test']['users']:
        raise ValueError('SASRec comparison uses a different data version')
    delta = {metric: ema['test'][metric] - adam['test'][metric] for metric in METRICS}
    report = relative(dict(
        exported_at_utc=datetime.now(timezone.utc).isoformat(),
        adam=adam, ema=ema, sasrec_test=baseline['test'],
        ema_minus_adam=delta, matched_arguments={key: adam['arguments'][key] for key in matched},
        checked_live_source_and_sid_fingerprints=True,
        limitation='single seed, three epochs; finite-beam retrieval versus exact SASRec scoring',
        report_script_sha256=digest(__file__)))
    lines = ['# 两套语义ID的独立Transformer训练对照', '',
             '两版均已完成正式训练、全用户验证选模及一次测试。共享冻结MiniLM商品向量，分别使用Adam与EMA码本生成的SID。',
             '生成模型均从随机初始化开始，未微调预训练T5权重；RQ-VAE不参与生成交叉熵的反向传播。', '',
             '| 方法 | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | 最佳轮 |',
             '|---|---:|---:|---:|---:|---:|']
    for label, result, epoch in (
        ('SASRec', baseline['test'], baseline['best_epoch']),
        ('TIGER式召回（Adam SID）', adam['test'], adam['best_epoch']),
        ('TIGER式召回（EMA SID）', ema['test'], ema['best_epoch'])):
        lines.append(f'| {label} | ' + ' | '.join(f'{result[key]:.6f}' for key in METRICS) + f' | {epoch} |')
    lines += ['', 'EMA减原版的绝对变化：' + '，'.join(f'{key}={delta[key]:+.6f}' for key in METRICS) + '。',
              '量化重建改善与推荐指标改善分别判断；本次只支持当前配置和单随机种子的实验结论。', '',
              '## 固定训练与评测配置', '',
              f'- 参数{adam["parameters"]:,}；hidden{adam["config"]["hidden"]}，编码器/解码器各{adam["config"]["layers"]}层，{adam["config"]["heads"]}头，FF{adam["config"]["ff"]}。',
              f'- 每轮{adam["training_examples"]:,}个有效下一商品目标；batch{adam["arguments"]["batch_size"]}，Adam lr{adam["arguments"]["lr"]}，seed{adam["arguments"]["seed"]}，CPU{adam["arguments"]["threads"]}线程，最多{adam["arguments"]["epochs"]}轮。',
              '- 相同用户内leave-two-out划分；每人一个验证/测试目标；仅验证NDCG@10选模。',
              '- 连续两次验证低于最佳且训练loss低于最佳轮时暂停；测试不参与暂停或选模。',
              '- 输入最近20件商品的四段SID；解码器输入BOS,a,b,c，预测a,b,c,d；四位置交叉熵均值。',
              '- 不采商品负样本；每个位置在该层256个码字上做softmax，其他码字参与分类竞争。',
              '- 测试86,713用户、25,898商品；完整目录前缀树beam20，过滤完整已知历史；有限beam是近似检索。',
              '- SASRec对目录逐商品精确打分，检索算法不同；每人一个目标时HR@K=Recall@K。', '',
              '## 逐轮验证', '',
              '| SID版本 | 轮 | 训练loss | 验证Recall@10 | 验证NDCG@10 | 过拟合预警次数 |',
              '|---|---:|---:|---:|---:|---:|']
    for label, run in (('Adam', adam), ('EMA', ema)):
        for row in run['history']:
            lines.append(f'| {label} | {row["epoch"]} | {row["training_loss"]:.6f} | '
                         f'{row["validation"]["Recall@10"]:.6f} | {row["validation"]["NDCG@10"]:.6f} | '
                         f'{row["consecutive_overfit_warnings"]} |')
    lines += ['', '## 检索检查', '',
              '| SID版本 | 返回不足20件的用户 | 商品合法率 | 停止原因 |', '|---|---:|---:|---|']
    for label, run in (('Adam', adam), ('EMA', ema)):
        result = run['test']
        lines.append(f'| {label} | {result["shortfall_users"]} | {result["legal_item_fraction"]:.2%} | {run["stop_reason"]} |')
    lines += ['', '## 上游流程与本地差异', '',
              '固定第三方仓库的train_decoder.py先加载已训练RQ-VAE，预计算商品SID，然后训练T5编码器/解码器。',
              '第三方模型forward仅监督前三段语义码，去掉消歧列；本地TigerModel复用原T5模块，额外监督第4段，确保可以唯一映射商品。',
              '本地还适配了共同SASRec样本、时间切分、确定性前缀树beam、完整历史过滤和全用户评测。',
              '15份上游原文件保持不变。两版生成代码相同，SID不同；源码、SID、数据指纹和配置在同名JSON中。', '',
              '[原版报告](tiger_cpu_20261008.md) · [EMA报告](tiger_ema_cpu_20261008.md) · '
              '[量化与死码审计](rqvae_adam_vs_ema_20261008.md)', '',
              '原始数据、用户记录、SID映射和权重保留在本地Git忽略目录。']
    directory = ROOT / 'experiments'
    (directory / f'{args.name}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (directory / f'{args.name}.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(report=f'experiments/{args.name}.md', ema_minus_adam=delta), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
