"""Audit codebook perplexity for the four main SID versions, without retraining."""
import argparse
import json
from pathlib import Path
import numpy as np

from tiger_common import ROOT, PROCESSED, sha256, utc_now, write_json, load_sid
from sid_usage import semantic_usage


SOURCE = 'https://github.com/google-deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def render(report):
    lines=['# 四方案SID质量与码本困惑度对比','',
        '同一批25,898件商品；每件商品计一次，不按交互频率加权。只统计前三段语义码，排除padding及第四段消歧编号。冻结32维编码器后重聚类版本不纳入正式比较。','',
        '| 方案 | SC↑ | PD↓ | 三层活跃码字 | 三层有效码字数 / 困惑度 | 三段碰撞余量↓ | 最大同码组 |',
        '|---|---:|---:|---|---|---:|---:|']
    for r in report['rows']:
        active=' / '.join(str(l['active_codes']) for l in r['layers'])
        effective=' / '.join(f"{l['effective_codes']:.2f}" for l in r['layers'])
        lines.append(f"| {r['model']} | {r['SC']:.6f} | {r['PD']:.6f} | {active} | {effective} | {r['excess_fraction']:.2%} | {r['largest_group']} |")
    lines += ['','## 分配集中程度','',
        '| 方案 | 三层有效码字数 / 256 | 三层最大单码字商品占比 | 三层死码 |',
        '|---|---|---|---|']
    for r in report['rows']:
        normalized=' / '.join(f"{l['normalized_effective_codes']:.2%}" for l in r['layers'])
        top=' / '.join(f"{l['top1_fraction']:.2%}" for l in r['layers'])
        dead=' / '.join(str(l['dead_codes']) for l in r['layers'])
        lines.append(f"| {r['model']} | {normalized} | {top} | {dead} |")
    lines += ['','## 定义与来源','',
        '`p_k = 第k个码字分配到的商品数 / 商品总数`',
        '`H = -Σ(p_k × ln(p_k))，只对p_k>0求和`',
        '`有效码字数 = 码本困惑度 = exp(H)`；归一化有效码字数为`exp(H)/256`。','',
        f'定义参考[DeepMind Sonnet官方VQ-VAE实现]({SOURCE})，VectorQuantizer与VectorQuantizerEMA均返回perplexity。官方使用输入张量展平后的one-hot分配均值；本项目使用完整商品目录的整数频次，并排除零概率项，数学定义一致，统计范围不同。',
        '本指标衡量整个码本分配的多样性，不给单个码字设“有效/无效”阈值。范围为1至活跃码字数；均匀分配到m个码字时等于m，频次集中时更接近1。',
        '这是码本使用分布的困惑度，不是语言模型预测困惑度；此处更高表示分配更均匀，不表示推荐一定更好。它也不是完整三段SID组合数，不能将三个有效码字数直接相乘作为商品容量。','',
        '## 本次结果解释','',
        'EMA后两层有效码字数约244，分配最接近均匀；首层仍有139个死码。R³-VAE首层虽有120个活跃码字，却只有约54.23个有效码字，不能据活跃数略高就认定其分配更好。',
        '直接384维RQ-KMeans第三层有254个活跃码字，但最大码字承载65.29%的商品，有效码字数只有3.61；它几乎没有死码，仍存在严重分配集中。',
        '当前离线质量更均衡的是EMA；已有Qwen SFT结果仍使用梯度RQ-VAE。R³-VAE与RQ-KMeans未测对应下游推荐效果，不将几何指标等同于推荐收益。','',
        'RQ-KMeans使用384维，其他三种使用32维潜空间；方法、维度、训练预算不同，且只跑一个种子。旧模型、训练记录和原始SC/PD报告不作改写。','',
        '## 证据与复现','',
        '- [完整逐层频次、指标与SID指纹](sid_usage_comparison_20261016.json)',
        '- [SID质量验证指标](../03_离散语义ID/SID质量验证指标.md)',
        '- [实现](../scripts/sid_usage.py) / [报告入口](../scripts/report_sid_usage.py)',
        '- [RQ-KMeans实现及训练记录](../03_离散语义ID/03_MiniLM_RQKMeans/README.md)',
        '', '```powershell',
        'python scripts/report_sid_usage.py --output outputs/sid_usage_comparison_new.json',
        'python scripts/report_sid_usage.py --evidence experiments/sid_usage_comparison_20261016.json --output outputs/sid_usage_from_evidence.json',
        '```','',
        '第一条需要本地四套SID及其清单；第二条仅用公开聚合频次重算困惑度并重建报告，不依赖权重、用户数据或SID映射。']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--evidence',type=Path)
    args=p.parse_args()
    if args.output.exists() or args.output.with_suffix('.md').exists():
        raise ValueError('Use a new output path')
    if args.evidence:
        from sid_usage import codebook_usage
        report=read(args.evidence)
        for row in report['rows']:
            for layer in row['layers']:
                recalculated=codebook_usage(np.asarray(layer['counts'],dtype=np.int64))
                if abs(recalculated['effective_codes']-layer['effective_codes'])>1e-10:
                    raise ValueError('Published frequency / perplexity mismatch')
                layer.update(recalculated)
    else:
        paths=[('RQ-VAE',PROCESSED/'tiger_sids','experiments/sid_quality_adam_20261016.json'),
            ('RQ-VAE EMA',PROCESSED/'tiger_sids_ema_20261008','experiments/sid_quality_ema_20261016.json'),
            ('R3-VAE',ROOT/'outputs/r3vae_minilm_metrics_only_20261016/sids','experiments/sid_quality_r3vae_20261016.json'),
            ('RQ-KMeans MiniLM384',ROOT/'outputs/rqkmeans_minilm384_20261016/sids','outputs/rqkmeans_minilm384_20261016/sid_quality.json')]
        rows=[]
        for label,path,qp in paths:
            codes,sm=load_sid(path)
            q=read(ROOT/qp)
            if (q['sid_sha256']!=sm['sid_sha256'] or q['embedding_sha256']!=sm['embedding_sha256']
                or q['protocol']!='fixed_minilm_cosine_macro_sc_centroid_pd_v1'):
                raise ValueError('Quality / SID contracts differ')
            layers=semantic_usage(codes[1:,:3],sm['codebook_size'])
            if [l['dead_codes'] for l in layers]!=q['dead_codes']:
                raise ValueError('Frequency audit differs from historical quality report')
            _,sizes=np.unique(codes[1:,:3],axis=0,return_counts=True)
            row=dict(model=label,items=sm['items'],SC=q['SC_layer_macro'],PD=q['PD_layer_macro'],
                layers=layers,unique_three_sid=len(sizes),excess_fraction=1-len(sizes)/sm['items'],
                largest_group=int(sizes.max()),collision_groups=int((sizes>1).sum()),
                items_in_collision_groups=int(sizes[sizes>1].sum()),full_unique_sid=sm['final_unique_ids'],
                embedding_sha256=sm['embedding_sha256'],sid_sha256=sm['sid_sha256'],
                prefix_quality=[r for r in q['rows'] if r['grouping']=='prefix'])
            # Independent legacy calculation guards against accidentally changing the metric.
            legacy={'RQ-VAE':'rqvae_adam_codebook_audit_20261008.json',
                'RQ-VAE EMA':'rqvae_ema_codebook_audit_20261008.json'}
            if label in legacy:
                old=read(ROOT/'experiments'/legacy[label])['full_catalog']['layers']
                assert all(abs(a['effective_codes']-b['effective_codes'])<1e-10 for a,b in zip(layers,old))
            rows.append(row)
        if len({r['embedding_sha256'] for r in rows})!=1:
            raise ValueError('Frozen feature versions differ')
        report=dict(date='2026-10-16',recorded_at_utc=utc_now(),
            protocol='catalog_hard_assignment_entropy_perplexity_v1',
            sc_pd_protocol='fixed_minilm_cosine_macro_sc_centroid_pd_v1',
            definition='effective_codes = codebook_perplexity = exp(-sum(p * ln(p))), p=count/N, zero terms omitted',
            source=SOURCE,aggregation='per layer; one count per unique catalog product; full catalog',
            padding_excluded=True,collision_counter_excluded=True,rows=rows,
            source_sha256={n:sha256(ROOT/'scripts'/n) for n in ('sid_usage.py','report_sid_usage.py')})
    write_json(args.output,report)
    args.output.with_suffix('.md').write_text(render(report),encoding='utf-8')
    print('Four-codec perplexity report generated; historical Adam/EMA values verified.')


if __name__=='__main__':
    main()
