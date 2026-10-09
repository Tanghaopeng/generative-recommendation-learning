# 04 推荐生成

| 版本 | 初始化 | 训练目标 | 状态 |
|---|---|---|---|
| [01_小型TIGER](01_小型TIGER/README.md) | 随机初始化小型T5 | 历史SID→下一商品SID | Adam/EMA两版已训练 |
| [02_Qwen2.5-1.5B-Instruct](02_Qwen2.5-1.5B-Instruct/README.md) | 已训练好的Qwen Instruct权重 | Q1主任务、Q2可选文本对齐，全参数SFT | 代码与CPU通路检查；GPU训练待执行 |

SASRec 对照保留在 `baselines/sasrec/`。GRPO 尚未实现，本次不能把SFT代码写成完成了OneRec强化学习。
