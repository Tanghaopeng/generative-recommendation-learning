# 07 组合结果统计

同一数据划分、商品目录、历史过滤；每人一个测试目标，所以HitRate@K=Recall@K。SASRec全目录精确打分，TIGER有限beam20近似搜索；差异必须保留。

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
