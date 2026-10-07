# 实验记录

当前完成数据准备与核验，未执行完整模型训练，没有模型指标或权重。

- [data_stats.json](data_stats.json)：数据统计与处理协议。
- [data_verification.json](data_verification.json)：核验结果，`training_executed=false`。

## 结果表

| 方法 | 数据版本 | Recall@10 | NDCG@10 | 状态 |
|---|---|---|---|---|
| SASRec | Office2018 去重后 5-core | — | — | 源码已整理，未训练 |
| 小型 TIGER 式召回 | 与 SASRec 相同 | — | — | 待实现 |
| 小型 LLM 推荐 SFT | 与前阶段相同 | — | — | 待实现 |
| 简化 OneRec | 列表任务，另建协议 | — | — | 待实现 |

结果由真实运行填入，不引用上游其他数据集的成绩替代。
每次记录代码与数据版本、超参数、随机种子、软硬件环境、评测口径、
验证集选择依据、测试指标、训练和推理耗时及限制。
权重与完整日志保存在本地忽略目录，不提交普通 Git。
