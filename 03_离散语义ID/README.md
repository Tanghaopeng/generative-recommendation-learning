# 03 离散语义 ID

输入02层的连续商品向量，输出固定商品目录的唯一离散 SID。此步骤与04的推荐生成训练解耦。

| 版本 | 输入 | 码本更新 | 状态 |
|---|---|---|---|
| [01_MiniLM_RQVAE](01_MiniLM_RQVAE/README.md) | MiniLM 384维 | Adam＋码本损失＋承诺损失 | 全量已完成 |
| [02_MiniLM_RQVAE_EMA](02_MiniLM_RQVAE_EMA/README.md) | 同一 MiniLM 向量 | EMA 码本更新＋承诺损失 | 全量已完成 |
| [03_MiniLM_RQKMeans](03_MiniLM_RQKMeans/README.md) | MiniLM 384维 | 残差 MiniBatchKMeans | 全量SID验证已完成 |
| [04_MiniLM_R3VAE](04_MiniLM_R3VAE/README.md) | 同一 MiniLM 向量 | 参考向量＋rating量化，余弦重构训练 | 30轮CPU量化和SID验证已完成 |

已完成版本均为三层×256码字，再追加按商品ID确定的重码组内编号：`[a,b,c,collision]`。
第四段只保证唯一性，不是第四层语义聚类。前缀可能共享，完整四段不能重复。
码本利用率和死码报告：[Adam/EMA对照](../experiments/rqvae_adam_vs_ema_20261008.md)。
SID质量协议：[SC/PD、码本利用率、死码与碰撞定义](SID质量验证指标.md)。
统一结果：[RQ-VAE / EMA / R³-VAE / RQ-KMeans对比](../experiments/sid_usage_comparison_20261016.md)。新增三层有效码字数（码本困惑度）、归一化值及最大码字占比；各方案有聚类、分配均匀性与碰撞的取舍，R³-VAE及RQ-KMeans尚无对应下游推荐结果。
生成模型的 CE 梯度不会回传到这层；切换 SID 方案必须重新准备样本并独立训练生成器。

正式对照仅保留四种SID方案；冻结原编码器后重聚类的条件实验不纳入主表。
