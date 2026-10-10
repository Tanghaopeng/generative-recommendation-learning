# 07 组合结果统计

同一数据划分、商品目录、历史过滤；每人一个测试目标，所以HitRate@K=Recall@K。SASRec全目录精确打分，TIGER有限beam20近似搜索；差异必须保留。

2026-10-10 已整理 [Q0→SFT→GRPO 多角度评测方案](../01_知识库/03_数据与评测/Q0_SFT_GRPO多角度评测方案_20261010.md)：主线固定 MiniLM＋梯度码本 RQ-VAE；扩展覆盖、长尾、多样性、生成质量、分组表现和成本。方案尚未全部接入评测代码，没有产生新指标。

| 编码器 | SID | 生成/排序模型 | 状态 | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
|---|---|---|---|---:|---:|---:|---:|
| 协同行为学习 | 无 | SASRec，第23轮 | 已完成 | 0.057650 | 0.035554 | 0.074406 | 0.039775 |
| MiniLM | RQ-VAE Adam | 小型TIGER，第3轮 | 已完成 | 0.049566 | 0.036792 | 0.060637 | 0.039582 |
| MiniLM | RQ-VAE EMA | 小型TIGER，第3轮 | 已完成 | 0.049820 | 0.037539 | 0.057096 | 0.039382 |
| MiniLM | RQ-VAE Adam | Qwen1.5B Q1全参SFT | 已实现，GPU未训练 | — | — | — | — |
| MiniLM | RQ-VAE Adam | Qwen1.5B Q2文本对齐SFT | 已实现，GPU未训练 | — | — | — | — |
| MiniLM | RQ-VAE EMA | Qwen1.5B Q1全参SFT | 同一代码可切换，GPU未训练 | — | — | — | — |
| MiniLM | RQ-KMeans | Qwen1.5B | SID方法尚未实现 | — | — | — | — |

前三行来源：[真实Transformer/SASRec汇总](../experiments/semantic_retrievers_comparison_20261008.json)。同单随机种子、短轮次，不代表稳定论文结论。
新增编码器、量化器、生成器组合单独新增一行，同时链接每日实验记录和完整JSON；同一组合多次运行保留全部run_id。
机器可读表：[combinations.json](combinations.json)。当前没有任何Qwen推荐指标。
