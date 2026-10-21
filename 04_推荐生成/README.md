# 04 推荐生成

| 版本 | 初始化 | 训练目标 | 状态 |
|---|---|---|---|
| [01_小型TIGER](01_小型TIGER/README.md) | 随机初始化小型T5 | 历史SID→下一商品SID | Adam/EMA两版已训练 |
| [02_Qwen2.5-1.5B-Instruct](02_Qwen2.5-1.5B-Instruct/README.md) | 已训练好的Qwen Instruct权重 | Q1主任务、Q2可选文本对齐，全参数SFT | Q1梯度/EMA SID两版均完成GPU训练与全量测试；Q2未训练 |

SASRec 对照保留在 `baselines/sasrec/`。当前公开实验截至Q1 SFT；后续偏好优化将另行发布。

[Qwen SFT 码本对照与原始结果](../experiments/ema_sft_20261021/README.md)。当前梯度 SID 版四项推荐指标略高，继续作为主基线。
