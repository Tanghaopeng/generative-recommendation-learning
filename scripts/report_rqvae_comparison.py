"""Aggregate Adam/EMA quantizer evidence; downstream recall remains separate."""
import argparse
import ast
import json
from pathlib import Path
import sys
import statistics

import numpy as np
import torch

from tiger_common import ROOT,PROCESSED,write_json,utc_now,sha256,configure,load_sid


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--adam-audit',default='experiments/rqvae_adam_codebook_audit_20261008.json')
    p.add_argument('--ema-audit',default='experiments/rqvae_ema_codebook_audit_20261008.json')
    p.add_argument('--adam-run',default='outputs/rqvae_office2018_cpu_20261008')
    p.add_argument('--ema-run',default='outputs/rqvae_office2018_ema_cpu_20261008')
    p.add_argument('--ema-sids',default=str(PROCESSED/'tiger_sids_ema_20261008'))
    args=p.parse_args()
    configure(threads=1)
    adam,ema=read(args.adam_audit),read(args.ema_audit)
    ar,er=read(Path(args.adam_run)/'run.json'),read(Path(args.ema_run)/'run.json')
    if (ar['status']!='completed' or er['status']!='completed' or ar['config']!=er['config'] or
            ar['embedding_sha256']!=er['embedding_sha256']):
        raise ValueError('Quantizer experiments are not comparable')
    for key in ('seed','batch_size','lr','epochs','patience'):
        if ar['arguments'][key]!=er['arguments'][key]:
            raise ValueError(f'Matched setting differs: {key}')
    if adam['checkpoint_epoch']!=ar['best_epoch'] or ema['checkpoint_epoch']!=er['best_epoch']:
        raise ValueError('Audit did not use selected checkpoints')
    old=np.load(PROCESSED/'tiger_sids/semantic_ids.npy')
    new=np.load(Path(args.ema_sids)/'semantic_ids.npy')
    changed=int((old[1:]!=new[1:]).any(axis=1).sum())
    saved=torch.load(Path(args.ema_run)/'best.pth',map_location='cpu',weights_only=False)
    buffers=[]
    for layer in range(3):
        count=saved['ema_state_dict'][f'layers.{layer}.ema_count'].numpy()
        buffers.append(dict(layer=layer+1,update_steps=int(saved['ema_state_dict'][f'layers.{layer}.update_steps']),
                            min_count=float(count.min()),median_count=float(statistics.median(count.tolist())),max_count=float(count.max()),
                            counts_below_epsilon=int((count<er['ema_config']['epsilon']).sum()),
                            counts_below_one=int((count<1).sum()),
                            meaning='EMA assignments per minibatch, with initial pseudocount; not corpus frequency'))
    # Verify the actual inference entry point refuses accidental SID switching.
    # Execute the unchanged inference entry-point function in isolation, as in
    # the dataset loader audit. The version guard runs before model creation;
    # importing the T5 backend is unnecessary for this negative test.
    filename=ROOT/'scripts/recommend_tiger.py'
    tree=ast.parse(filename.read_text(encoding='utf-8'))
    entry=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    def unexpected_model(*args,**kwargs):
        raise AssertionError('Mismatched SID reached model initialization')
    namespace=dict(argparse=argparse,torch=torch,PROCESSED=PROCESSED,configure=configure,
                   load_sid=load_sid,TigerModel=unexpected_model)
    exec(compile(ast.Module(body=[entry],type_ignores=[]),str(filename),'exec'),namespace)
    original_argv=sys.argv
    rejected=False
    try:
        sys.argv=['recommend_tiger.py','--checkpoint','outputs/tiger_office2018_cpu_smoke_20261008/best.pth',
                  '--sids',args.ema_sids,'--history','1,2,3','--threads','1']
        try:
            namespace['main']()
        except ValueError as error:
            if 'Checkpoint and SID catalog differ' not in str(error):
                raise
            rejected=True
    finally:
        sys.argv=original_argv
    if not rejected:
        raise AssertionError('SID version mismatch was accepted')
    info=read(Path(args.ema_sids)/'manifest.json')
    info.update(codebook_update='EMA',ema_config=er['ema_config'],
                upstream_inference_compatible=True,training_state_contains_ema_buffers=True)
    write_json(Path(args.ema_sids)/'manifest.json',info)
    report=dict(recorded_at_utc=utc_now(),adam_audit=adam,ema_audit=ema,
                adam_best_epoch=ar['best_epoch'],ema_best_epoch=er['best_epoch'],
                adam_validation_reconstruction=ar['best_validation_reconstruction'],
                ema_validation_reconstruction=er['best_validation_reconstruction'],
                matched_configuration=ar['config'],ema_configuration=er['ema_config'],
                ema_history=er['history'],ema_buffers=buffers,
                changed_full_sid_items=changed,changed_full_sid_fraction=changed/(len(old)-1),
                sid_mismatch_rejected=True,ema_sid_manifest=info,
                mismatch_test='unchanged inference main() extracted with AST; guard before model initialization',
                core_sources_unchanged=True,ema_source_sha256=er['source_sha256'],
                report_script_sha256=sha256(__file__),dead_code_reset=False,
                downstream_result=None,downstream_plan='new independent retriever after original completes')
    # Recheck the live original run source and frozen SID; do not just assert.
    live=read('outputs/tiger_office2018_cpu_20261008/run.json')
    if any(sha256(ROOT/path)!=digest for path,digest in live['source_sha256'].items()):
        raise ValueError('Original retriever sources changed')
    if sha256(PROCESSED/'tiger_sids/semantic_ids.npy')!=live['sid_manifest']['sid_sha256']:
        raise ValueError('Original SID table changed')
    write_json(ROOT/'experiments/rqvae_adam_vs_ema_20261008.json',report)
    lines=['# RQ-VAE 码字审计与EMA对照','',
           '2026-10-08的真实全量商品实验；原版和EMA版分别完成30轮，按相同5%商品向量验证重建误差选模。',
           '使用同一384维冻结文本向量、初始化、数据划分、种子2026、3×256码本、隐空间32、batch512、Adam学习率0.001。',
           'EMA版只更换码本更新方式与对应损失项，衰减0.99、初始伪计数1、epsilon1e-5，不做死码重置；残差STE图与原版相同。','',
           '| 层 | 原版活跃/256 | 原版死码 | 原版有效码字 | EMA活跃/256 | EMA死码 | EMA有效码字 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for a,e in zip(adam['full_catalog']['layers'],ema['full_catalog']['layers']):
        lines.append(f'| {a["layer"]} | {a["active_codes"]}（{a["utilization"]:.2%}） | {a["dead_codes"]} | {a["effective_codes"]:.2f} | {e["active_codes"]}（{e["utilization"]:.2%}） | {e["dead_codes"]} | {e["effective_codes"]:.2f} |')
    lines += ['', '以上死码定义为选定checkpoint在全部25,898商品上的零次硬分配，不是“训练一生从未更新”。',
              '只统计前三层量化码本；第4段是碰撞组计数，不是语义量化码本。',
              '利用率=有命中的码字/256；有效码字数=exp(分配概率熵)。即使所有码字都用到，分布极偏时有效数仍很小。',
              '另有train/validation分别统计，完整256个频次与死码ID在审计JSON。验证只有1,295件商品，零命中不能单独证明码字失效。','',
              '| 指标 | 原版 | EMA |','|---|---:|---:|',
              f'| 最佳轮 | {ar["best_epoch"]} | {er["best_epoch"]} |',
              f'| 验证重建误差（越低越好） | {ar["best_validation_reconstruction"]:.6f} | {er["best_validation_reconstruction"]:.6f} |',
              f'| 三段SID唯一组 | {adam["full_catalog"]["raw_unique_ids"]} | {ema["full_catalog"]["raw_unique_ids"]} |',
              f'| 超出每组第一件的重复商品比例 | {adam["full_catalog"]["extra_duplicate_fraction"]:.2%} | {ema["full_catalog"]["extra_duplicate_fraction"]:.2%} |',
              f'| 最大碰撞组 | {adam["full_catalog"]["max_collision_group"]} | {ema["full_catalog"]["max_collision_group"]} |','',
              'EMA首层覆盖和分配均匀性有所改善，后两层也更均匀；仍有139个首层死码，不能称为已消除塌缩。',
              '重建误差仅小幅改善；这不是推荐HR提升证据。两版训练总loss目标不同，不能用总loss大小直接比较。','',
              '## 损失与梯度实测','',
              '原版：L=重建 + Σ||sg(r)-e||² + 0.25Σ||r-sg(e)||²。',
              '码本项更新码本；承诺项约束编码器；重建项通过STE更新编码器/解码器。',
              'EMA版：L=重建 + 0.25Σ||r-sg(e)||²；码本向量不参加Adam，不用码本损失反传，而由EMA计数与残差向量和更新。',
              'N←0.99N+0.01n；M←0.99M+0.01Σr；e←M/N，数值耗尽的未用中心保留原值，不主动重启。',
              '[VQ-VAE论文 §3.2/附录A.1](https://arxiv.org/pdf/1711.00937)说明EMA替代码本损失更新，承诺项仍保留。','',
              '实测源实现存在一个梯度限制：逐层e_ST=r+sg(e-r)，随后r_next=r-e_ST，使后两层承诺损失对编码器的梯度为0。',
              '原版第1层承诺项有编码器梯度，后两层为0；码本项对各自码本有梯度。EMA码本项梯度全为0，权重由EMA更新。',
              'EMA对照保留同一残差图，便于单独比较更新方式；不能误写为“所有层承诺项都在约束编码器”。',
              '后续若改为减去detach后的真实码字，应独立比较，不能把梯度结构改变混入本次EMA结果。','',
              '## 与后续生成模型的连接','',
              '当前是两阶段训练，符合[TIGER论文 §3](https://arxiv.org/html/2305.05065v3#S3)中的内容SID生成→行为生成模型路线。',
              '1. 商品文本→冻结MiniLM→RQ-VAE，用重建/量化目标学离散表示，冻结商品SID表。',
              '2. 用户历史商品查SID→Transformer，用下一商品四token交叉熵训练，梯度不回传RQ-VAE。',
              '生成模型的64维SID embedding是独立可训练参数，不是直接取RQ-VAE的32维码本向量。',
              '因此优化器和梯度解耦，但商品ID↔SID接口与版本绑定；修改码本可能改变标签含义。',
              f'两版完整SID不同的商品：{changed:,}/{len(old)-1:,}（{changed/(len(old)-1):.2%}，只表示整数ID变化，未做簇对齐）。',
              '已实测旧checkpoint配新SID会被推理入口拒绝。旧版源码和SID指纹不变，正在训练的旧模型继续使用旧映射。',
              'EMA使用新目录和新checkpoint独立训练同配置生成模型，排在旧版完成后开始；下游结果尚无，完成后自动生成tiger_ema_cpu_20261008报告。','',
              '## 代码与复跑','',
              '新增scripts/rqvae_ema.py、train_rqvae_ema.py、audit_codebooks.py、report_rqvae_comparison.py。上游15份原文件未修改。',
              'EMA训练state保存EMA计数、向量和、步数、Adam与RNG；额外导出原RqVae可加载的推理权重。验证与SID导出均不会更新EMA。',
              '```powershell',
              'python scripts/train_rqvae_ema.py --output outputs/rqvae_ema_new --epochs 30 --decay 0.99 --threads 2',
              'python scripts/audit_codebooks.py --checkpoint outputs/rqvae_ema_new/best.pth --output outputs/ema_audit_new.json --threads 2',
              'python scripts/build_semantic_ids.py --checkpoint outputs/rqvae_ema_new/best.pth --output data/processed/office2018/tiger_sids_ema_new --threads 2',
              'python scripts/run_tiger.py --sids data/processed/office2018/tiger_sids_ema_new --output outputs/tiger_ema_new --epochs 3 --hidden 64 --layers 1 --heads 2 --ff 128 --threads 2',
              '```','',
              '完整匹配配置、EMA逐轮利用率、时间、梯度数值与版本指纹见同名JSON；raw/用户数据/向量/SID映射/权重仍在Git忽略目录。']
    (ROOT/'experiments/rqvae_adam_vs_ema_20261008.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Comparison exported. Changed SID items={changed}/{len(old)-1}; version mismatch rejected.',flush=True)


if __name__=='__main__':
    main()
