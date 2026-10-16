# 03 离散语义 ID

输入02层的连续商品向量，输出固定商品目录的唯一离散 SID。此步骤与04的推荐生成训练解耦。

| 版本 | 输入 | 码本更新 | 状态 |
|---|---|---|---|
| [01_MiniLM_RQVAE](01_MiniLM_RQVAE/README.md) | MiniLM 384维 | Adam＋码本损失＋承诺损失 | 全量已完成 |
| [02_MiniLM_RQVAE_EMA](02_MiniLM_RQVAE_EMA/README.md) | 同一 MiniLM 向量 | EMA 码本更新＋承诺损失 | 全量已完成 |
| [03_MiniLM_RQKMeans](03_MiniLM_RQKMeans/README.md) | MiniLM 384维 | 残差 K-means | 预留，尚未实现 |
| [04_MiniLM_R3VAE](04_MiniLM_R3VAE/README.md) | 同一 MiniLM 向量 | 参考向量＋rating量化，余弦重构训练 | 30轮CPU量化和SID验证已完成 |

三套已完成版本均为三层×256码字，再追加按商品ID确定的重码组内编号：`[a,b,c,collision]`。
第四段只保证唯一性，不是第四层语义聚类。前缀可能共享，完整四段不能重复。
码本利用率和死码报告：[Adam/EMA对照](../experiments/rqvae_adam_vs_ema_20261008.md)。
SID质量协议：[SC/PD、码本利用率、死码与碰撞定义](SID质量验证指标.md)。
统一结果：[RQ-VAE / EMA / R³-VAE对比](../experiments/sid_quality_comparison_20261016.md)。R³-VAE有SC/PD改善与碰撞增加的取舍，尚无对应的下游推荐结果。
生成模型的 CE 梯度不会回传到这层；切换 SID 方案必须重新准备样本并独立训练生成器。
