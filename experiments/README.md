# 实验记录

已完成 SASRec 首次本地 CPU 训练：原运行完成 11 轮后中断，从权重恢复至总计 20 轮，
仅用验证 NDCG@10 选择第 20 轮权重，再完成一次全商品测试与推理检查。
旧检查点没有 Adam/RNG 状态，恢复时重新初始化，不能视为原配置不中断训练。

随后从第 20 轮完整训练状态继续，原计划最多再训练 20 轮。
第 24、25 轮训练 loss 更低而验证 NDCG@10 低于最佳，连续两轮触发暂停。
实际新增 5 轮，最佳模型为第 23 轮；训练进程已退出，完整检查点保留。

- [data_stats.json](data_stats.json)：数据统计与处理协议。
- [data_verification.json](data_verification.json)：早期数据核验快照，`training_executed=false` 仅表示核验脚本不训练。
- [sasrec_cpu_20261007.md](sasrec_cpu_20261007.md)：真实实验报告，含配置、选模、测试、延迟和恢复说明。
- [sasrec_cpu_20261007.json](sasrec_cpu_20261007.json)：完整聚合指标、每轮记录、环境、数据与代码 SHA-256。
- [sasrec_cpu_extend40_20261007.md](sasrec_cpu_extend40_20261007.md)：续训报告，包含暂停规则和逐轮验证走势。
- [sasrec_cpu_extend40_20261007.json](sasrec_cpu_extend40_20261007.json)：续训聚合指标和暂停证据。
- [tiger_setup_20261008.md](tiger_setup_20261008.md)：TIGER全量编码、30轮量化、唯一SID、真实数据通路和首轮训练启动记录。
- [tiger_setup_20261008.json](tiger_setup_20261008.json)：准备阶段聚合统计与版本指纹，不包含正式召回成绩。
- [tiger_cpu_20261008.md](tiger_cpu_20261008.md)：原版SID生成模型的3轮正式训练、全用户验证与测试。
- [tiger_cpu_20261008.json](tiger_cpu_20261008.json)：原版生成模型完整配置、逐轮记录、测试与指纹。
- [tiger_ema_cpu_20261008.md](tiger_ema_cpu_20261008.md)、[JSON](tiger_ema_cpu_20261008.json)：EMA SID独立生成模型的正式训练、验证与测试。
- [两套SID的Transformer对照](semantic_retrievers_comparison_20261008.md)、[完整对照JSON](semantic_retrievers_comparison_20261008.json)：匹配配置、版本核验、逐轮验证、测试指标与SASRec比较。
- [rqvae_adam_vs_ema_20261008.md](rqvae_adam_vs_ema_20261008.md)：死码、码字分布、EMA对照、梯度实测与两阶段连接说明。
- [rqvae_adam_vs_ema_20261008.json](rqvae_adam_vs_ema_20261008.json)：两版量化器的完整聚合证据与EMA逐轮码字使用情况。
- [原版码字审计](rqvae_adam_codebook_audit_20261008.json)、[EMA码字审计](rqvae_ema_codebook_audit_20261008.json)：全目录/训练/验证频次、死码ID、有效码字数与梯度去向。

## 结果表

| 方法 | 数据版本 | Recall@10 | NDCG@10 | 状态 |
|---|---|---|---|---|
| Popularity | Office2018 去重后 5-core | 0.008949 | 0.004577 | 训练集商品计数，相同测试协议 |
| SASRec（首次 20 轮） | Office2018 去重后 5-core | 0.053660 | 0.031269 | 首次本地实验；第 20 轮最佳 |
| SASRec（继续训练） | 同上 | 0.057650 | 0.035554 | 第 25 轮暂停；第 23 轮最佳 |
| 小型 TIGER（Adam SID） | 与 SASRec 相同 | 0.049566 | 0.036792 | 已完成3轮；第3轮验证最佳 |
| 小型 TIGER（EMA SID） | 与 SASRec 相同 | 0.049820 | 0.037539 | 已完成3轮；第3轮验证最佳 |
| 小型 LLM 推荐 SFT | 与前阶段相同 | — | — | 待实现 |
| 简化 OneRec | 列表任务，另建协议 | — | — | 待实现 |

结果由真实运行填入，不引用上游其他数据集的成绩替代。
以上为全部 86,713 个用户、25,898 个商品的全目录测试，每人一个目标，过滤完整已知历史。
单目标下HitRate@10=Recall@10；SASRec的0.057650表示约5.76%的用户，其真实下一件商品进入前10。当前两版生成召回的Recall@10较低、NDCG@10较高；三轮小模型与25轮SASRec的训练预算不同，不能当成方法优劣定论。
SASRec训练每个有效位置采1个负例；TIGER训练四段SID交叉熵，不采商品负例。两者评测均不抽负候选；TIGER采用全目录前缀树beam20近似检索。不能与上游101个采样候选的指标直接比较。
每次记录代码与数据版本、超参数、随机种子、软硬件环境、评测口径、
验证集选择依据、测试指标、训练和推理耗时及限制。
权重与完整日志保存在本地忽略目录，不提交普通 Git。

## 2026-10-09 Qwen SFT工程验证

[实现核验JSON](qwen_sft_implementation_20261009.json)记录Adam-Q1、Adam-Q2、EMA-Q1真实tokenizer检查；未进行Qwen GPU训练，无推荐指标。新记录入口见[每日记录](../06_每日效果记录/README.md)，汇总见[组合表](../07_组合结果统计/README.md)。
