# RQ-VAE 码字审计与EMA对照

2026-10-08的真实全量商品实验；原版和EMA版分别完成30轮，按相同5%商品向量验证重建误差选模。
使用同一384维冻结文本向量、初始化、数据划分、种子2026、3×256码本、隐空间32、batch512、Adam学习率0.001。
EMA版只更换码本更新方式与对应损失项，衰减0.99、初始伪计数1、epsilon1e-5，不做死码重置；残差STE图与原版相同。

| 层 | 原版活跃/256 | 原版死码 | 原版有效码字 | EMA活跃/256 | EMA死码 | EMA有效码字 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 84（32.81%） | 172 | 60.25 | 117（45.70%） | 139 | 84.98 |
| 2 | 256（100.00%） | 0 | 236.57 | 256（100.00%） | 0 | 244.13 |
| 3 | 256（100.00%） | 0 | 239.09 | 256（100.00%） | 0 | 244.82 |

以上死码定义为选定checkpoint在全部25,898商品上的零次硬分配，不是“训练一生从未更新”。
只统计前三层量化码本；第4段是碰撞组计数，不是语义量化码本。
利用率=有命中的码字/256；有效码字数=exp(分配概率熵)。即使所有码字都用到，分布极偏时有效数仍很小。
另有train/validation分别统计，完整256个频次与死码ID在审计JSON。验证只有1,295件商品，零命中不能单独证明码字失效。

| 指标 | 原版 | EMA |
|---|---:|---:|
| 最佳轮 | 30 | 29 |
| 验证重建误差（越低越好） | 0.426134 | 0.423728 |
| 三段SID唯一组 | 20332 | 21474 |
| 超出每组第一件的重复商品比例 | 21.49% | 17.08% |
| 最大碰撞组 | 34 | 33 |

EMA首层覆盖和分配均匀性有所改善，后两层也更均匀；仍有139个首层死码，不能称为已消除塌缩。
重建误差仅小幅改善；这不是推荐HR提升证据。两版训练总loss目标不同，不能用总loss大小直接比较。

## 损失与梯度实测

原版：L=重建 + Σ||sg(r)-e||² + 0.25Σ||r-sg(e)||²。
码本项更新码本；承诺项约束编码器；重建项通过STE更新编码器/解码器。
EMA版：L=重建 + 0.25Σ||r-sg(e)||²；码本向量不参加Adam，不用码本损失反传，而由EMA计数与残差向量和更新。
N←0.99N+0.01n；M←0.99M+0.01Σr；e←M/N，数值耗尽的未用中心保留原值，不主动重启。
[VQ-VAE论文 §3.2/附录A.1](https://arxiv.org/pdf/1711.00937)说明EMA替代码本损失更新，承诺项仍保留。

实测源实现存在一个梯度限制：逐层e_ST=r+sg(e-r)，随后r_next=r-e_ST，使后两层承诺损失对编码器的梯度为0。
原版第1层承诺项有编码器梯度，后两层为0；码本项对各自码本有梯度。EMA码本项梯度全为0，权重由EMA更新。
EMA对照保留同一残差图，便于单独比较更新方式；不能误写为“所有层承诺项都在约束编码器”。
后续若改为减去detach后的真实码字，应独立比较，不能把梯度结构改变混入本次EMA结果。

## 与后续生成模型的连接

当前是两阶段训练，符合[TIGER论文 §3](https://arxiv.org/html/2305.05065v3#S3)中的内容SID生成→行为生成模型路线。
1. 商品文本→冻结MiniLM→RQ-VAE，用重建/量化目标学离散表示，冻结商品SID表。
2. 用户历史商品查SID→Transformer，用下一商品四token交叉熵训练，梯度不回传RQ-VAE。
生成模型的64维SID embedding是独立可训练参数，不是直接取RQ-VAE的32维码本向量。
因此优化器和梯度解耦，但商品ID↔SID接口与版本绑定；修改码本可能改变标签含义。
两版完整SID不同的商品：25,898/25,898（100.00%，只表示整数ID变化，未做簇对齐）。
已实测旧checkpoint配新SID会被推理入口拒绝。旧版源码和SID指纹不变，正在训练的旧模型继续使用旧映射。
EMA使用新目录和新checkpoint独立训练同配置生成模型。两版现已完成3轮训练，全用户验证均选中第3轮，未触发暂停；推荐结果见[生成模型对照](semantic_retrievers_comparison_20261008.md)。EMA测试前10指标略高、前20指标略低，未呈现全面提升。同名量化JSON保留当时的阶段快照，下游完整数据在新的生成模型对照JSON中。

## 代码与复跑

新增scripts/rqvae_ema.py、train_rqvae_ema.py、audit_codebooks.py、report_rqvae_comparison.py。上游15份原文件未修改。
EMA训练state保存EMA计数、向量和、步数、Adam与RNG；额外导出原RqVae可加载的推理权重。验证与SID导出均不会更新EMA。
```powershell
python scripts/train_rqvae_ema.py --output outputs/rqvae_ema_new --epochs 30 --decay 0.99 --threads 2
python scripts/audit_codebooks.py --checkpoint outputs/rqvae_ema_new/best.pth --output outputs/ema_audit_new.json --threads 2
python scripts/build_semantic_ids.py --checkpoint outputs/rqvae_ema_new/best.pth --output data/processed/office2018/tiger_sids_ema_new --threads 2
python scripts/run_tiger.py --sids data/processed/office2018/tiger_sids_ema_new --output outputs/tiger_ema_new --epochs 3 --hidden 64 --layers 1 --heads 2 --ff 128 --threads 2
```

完整匹配配置、EMA逐轮利用率、时间、梯度数值与版本指纹见同名JSON；raw/用户数据/向量/SID映射/权重仍在Git忽略目录。
