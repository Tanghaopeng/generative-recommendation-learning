"""Rebuild the published three-codec SID comparison from aggregate evidence."""
import argparse
import hashlib
import json
from pathlib import Path

from tiger_common import ROOT, write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def text_sha256(path):
    """Match Git text artifacts on Windows and Unix despite CRLF checkout."""
    return hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.md').exists():
        raise ValueError('Use a new output path to preserve previous reports')
    base=ROOT/'experiments'
    collisions=read(base/'sid_collision_comparison_20261016.json')
    collision_map={row['model']:row for row in collisions['rows']}
    rows=[];sources={}
    for label,name in [('RQ-VAE','sid_quality_adam_20261016.json'),
                       ('RQ-VAE EMA','sid_quality_ema_20261016.json'),
                       ('R3-VAE','sid_quality_r3vae_20261016.json')]:
        quality=read(base/name);collision=collision_map[label]
        if (quality['protocol']!='fixed_minilm_cosine_macro_sc_centroid_pd_v1'
            or quality['sid_sha256']!=collision['sid_sha256']
            or quality['embedding_sha256']!=collision['embedding_sha256']):
            raise ValueError('Quality and collision evidence differ')
        rows.append(dict(model=label,SC=quality['SC_layer_macro'],PD=quality['PD_layer_macro'],
                         used_codes=[round(v*256) for v in quality['codebook_utilization']],
                         codebook_utilization=quality['codebook_utilization'],
                         dead_codes=quality['dead_codes'],**{k:collision[k] for k in
                         ('items','unique_three_sid','collision_groups','items_in_collision_groups',
                          'collision_item_fraction','excess_fraction','largest_group','full_unique_sid',
                          'sid_sha256','embedding_sha256')}))
        sources[label]=dict(quality_report=name,sha256=text_sha256(base/name))
    if len({row['embedding_sha256'] for row in rows})!=1:
        raise ValueError('Comparison requires the same frozen feature space')
    training_path=base/'r3vae_minilm_20261016/training_result.json'
    training=read(training_path)
    if training['config']['sc_weight']!=0 or training['config']['pd_weight']!=0:
        raise ValueError('This publication records the reconstruction-trained R3-VAE configuration')
    report=dict(date='2026-10-16',protocol='fixed_minilm_cosine_macro_sc_centroid_pd_v1',
                scope='25898 full-catalog frozen MiniLM features; geometric SID evaluation',
                semantic_layers=3,codebook_size=256,collision_counter_excluded=True,
                rows=rows,sources=sources,
                evidence_text_hash_convention='SHA256 of UTF-8 file bytes with CRLF normalized to LF; binary artifact hashes unchanged',
                collision_report=dict(file='sid_collision_comparison_20261016.json',sha256=text_sha256(base/'sid_collision_comparison_20261016.json')),
                r3vae_training=dict(record='r3vae_minilm_20261016/training_result.json',
                    sha256=text_sha256(training_path),parameters=training['parameters'],
                    epochs=training['best_epoch'],seconds=training['elapsed_seconds'],config=training['config']),
                conclusions=['R3-VAE has higher SC and lower PD in this fixed-feature protocol',
                             'R3-VAE also has more three-segment collisions; this is a tradeoff',
                             'No new downstream recommendation Recall/NDCG has been measured for R3-VAE'])
    write_json(args.output,report)
    text=['# SID质量对比 2026年10月16日','',
          '同一批25,898件商品、冻结MiniLM 384维向量，比较RQ-VAE、EMA与R³-VAE适配版。SC和PD分别按三个独立语义层宏平均；逐层与前缀详情保留在原始JSON。','',
          '| 量化器 | SC↑ | PD↓ | 三层利用率 | 三层死码 | 三段碰撞余量↓ | 最大同码组↓ |',
          '|---|---:|---:|---|---|---:|---:|']
    for row in rows:
        usage=' / '.join(f'{v:.2%}' for v in row['codebook_utilization'])
        dead=' / '.join(map(str,row['dead_codes']))
        text.append(f"| {row['model']} | {row['SC']:.6f} | {row['PD']:.6f} | {usage} | {dead} | {row['excess_fraction']:.2%} | {row['largest_group']} |")
    text+=['','## 碰撞与唯一性','',
           '| 量化器 | 不同三段SID数 | 碰撞组数 | 参与碰撞商品数 | 参与碰撞占比 | 完整四段唯一SID数 |',
           '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        text.append(f"| {row['model']} | {row['unique_three_sid']:,} | {row['collision_groups']:,} | {row['items_in_collision_groups']:,} | {row['collision_item_fraction']:.2%} | {row['full_unique_sid']:,} |")
    text+=['','碰撞余量=(商品数−不同三段SID数)/商品数。参与碰撞占比=至少两件商品同码的组所包含商品数/商品数。两者不是同一指标，完整四段SID仍一一对应商品。','',
           '## R³ VAE训练记录','',
           f"模型{training['parameters']:,}参数，30轮CPU训练约{training['elapsed_seconds']:.1f}秒。编码器384→256→128→32，三层256码字，参考向量投影与top1余弦评分量化。用训练商品向量的残差MiniBatchKMeans初始化，以5%目录向量验证集的余弦重构损失选模。",
           '训练目标为余弦重构；SC/PD正则权重为0。AdamW学习率5e-4、weight decay 1e-5、batch size 512、seed 2026。','',
           '## 结果解释','',
           'R³-VAE的SC、PD和首层利用率改善，但三段SID碰撞余量上升到44.36%，第二、三层利用率也未达到100%。这些结果描述语义聚类与区分性的取舍，不等同于推荐效果提升。',
           '当前R³-VAE是参考官方top1评分运算的32维MLP适配，论文softmax描述与公开代码存在差异。参考投影系数和评分不写入SID，因此重构表现不能单独证明离散编码保留了全部信息。',
           '本次没有重新训练R³-VAE SID对应的TIGER/Qwen推荐器；单种子、30轮预算与不同优化目标也限制了方法优劣结论。','',
           '## 原始证据','',
           '- [RQ-VAE质量指标](sid_quality_adam_20261016.json)',
           '- [EMA质量指标](sid_quality_ema_20261016.json)',
           '- [R³-VAE质量指标](sid_quality_r3vae_20261016.json)',
           '- [碰撞分布与指纹](sid_collision_comparison_20261016.json)',
           '- [R³-VAE训练结果](r3vae_minilm_20261016/training_result.json)',
           '- [SID验证指标说明](../03_离散语义ID/SID质量验证指标.md)']
    args.output.with_suffix('.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('SID comparison generated for RQ-VAE, EMA and R3-VAE')


if __name__=='__main__':
    main()
