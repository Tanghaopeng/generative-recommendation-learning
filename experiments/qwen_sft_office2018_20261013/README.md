# Qwen Q1 SFT正式结果（2026-10-13）

MiniLM冻结商品编码 → 梯度码本RQ-VAE → 三段语义码加一段消歧 → Qwen2.5-1.5B-Instruct全参数SFT。Qwen生成下一商品SID，MiniLM/RQ-VAE不参与这一阶段的梯度更新。

| 项目 | 设置 |
|---|---|
| 数据 | Amazon Reviews 2018 Office Products；86713用户，25898商品 |
| 划分 | 每用户按时间leave-two-out；每人一个验证/测试目标 |
| 训练 | 397345个位置，Q1下一SID任务，1轮，约3105次更新 |
| 选模 | 固定5000人验证子集，按NDCG@10选中checkpoint-3000 |
| 验证成绩 | Recall@10=0.1216；NDCG@10=0.0969749937；eval_loss=1.811376 |
| 硬件与精度 | A100-PCIE-40GB；BF16权重与混合精度计算；SDPA；激活检查点 |
| 原训练配置 | microbatch2、累积64、有效batch128；学习率5e-5；seed2026 |
| 评测 | sid_beam_v2；beam20；全目录有限beam近似检索；完整已知历史排除 |

验证成绩来自已保留终端记录；本次没有取得原始完整SFT run.json/trainer_state.json，因此不伪造其完整逐步历史或精确训练总时长。正式测试JSON具有数据、权重及评测代码指纹。

| 正式测试 | 数值 |
|---|---:|
| Recall/HitRate@10 | 0.09367684199600983 |
| NDCG@10 | 0.07549556623872561 |
| Recall/HitRate@20 | 0.1088187469006954 |
| NDCG@20 | 0.07931915657505524 |
| 用户 | 86713 |
| 返回不足20件的用户 | 13 |
| 检索计时 | 10406.69秒，约2小时53分钟，**不是训练时间** |

每人一个正目标，HitRate=Recall。AUC/GAUC尚未实测；未计算项不写0。SASRec是全目录精确打分，本模型是有限beam近似；容量、预算与硬件也不同，不是严格消融。

## 模型内部结构

1544870912个参数；28层Decoder；hidden1536；Q头12、KV头2、每头128；FFN8960；RoPE、RMSNorm、SiLU；扩展词表152689，含1024个SID token；输入输出embedding共享。实际导出配置见[model_config.json](model_config.json)，全模型对照见[模型参数表](../../07_组合结果统计/模型参数_20261013.json)。

## 候选抽样与性能短测

33人（32人seed2027随机样本加示例用户1）共660候选，仅5/33人的前20件只有2～3种第一段SID；平均5.88种。详见[candidate_summary.json](candidate_summary.json)及[逐用户统计](../../07_组合结果统计/SID抽样统计_20261013.json)。抽样不能代表全量分布。

同一128条样本的A100短测：batch2/累积64平均10.922秒每更新；batch4/累积32平均6.577秒。只计时2步、预热1步；预先完成分词/组batch，排除评测和保存；batch4并未用于上述已完成正式训练。原始记录在本目录，不能冒充新模型结果。

## 权重与复用

按照用户要求，本次只公开代码、配置、结构和结果，SFT权重仍在私有持久网盘。权重SHA256、归档SHA256及内部路径见[model_artifact.json](model_artifact.json)。Git中没有完整Qwen检查点；clone仓库本身不等于下载已训练模型。还原selected后还需相同SID映射、扩展分词器和数据。恢复训练另需优化器、scheduler与随机状态。

复现入口：`scripts/prepare_qwen_sft.py`、`export_qwen_q0.py`、`train_qwen_sft.py`、`freeze_qwen_training_selection.py`、`evaluate_qwen_sft.py`。原数据和大权重继续按原约定保存在Git以外。

抽样的33名用户及其完整20个候选、四段SID和路径分数见[candidates_33_users.jsonl](candidates_33_users.jsonl)，不是全量候选导出。
