# 项目实现、结果与技术取舍

[返回首页](README.md) · [正式 SFT 实验](experiments/qwen_sft_office2018_20261013/README.md) · [运行步骤](REPRODUCING.md)

## 要解决的问题

将传统“历史序列 → 商品打分”的推荐，逐步改造成“历史语义 ID → 自回归生成下一商品 ID”，并在同一处理版本的公开数据上比较结果。项目以 Amazon Reviews 2018 Office Products 为起点，先完成 SASRec 基线，再独立验证商品编码、残差量化、小型生成器和预训练 Qwen 适配。

数据保留每个用户—商品的首次交互，再重新进行 5-core；最终 86,713 用户、25,898 商品、677,247 交互。Qwen 下一物品训练展开为 397,345 个位置，每个位置仅使用当时可见历史。

## 实现贡献与代码位置

### 1. 建立共同数据和评测协议

保留上游 SASRec 模型结构，新增独立的数据准备、采样、训练、恢复与评测入口。上游采样候选指标不直接搬入项目结果表；当前主表使用完整目录和固定历史过滤，仅验证 NDCG@10 用于选模。

- 数据处理与核验：[prepare_office.py](scripts/prepare_office.py)、[verify_dataset.py](scripts/verify_dataset.py)。
- 训练与推荐：[run_sasrec.py](scripts/run_sasrec.py)、[recommend_sasrec.py](scripts/recommend_sasrec.py)。
- 记录：[首次训练](experiments/sasrec_cpu_20261007.md)、[续训与暂停](experiments/sasrec_cpu_extend40_20261007.md)。

续训在第 25 轮触发连续两次过拟合预警，保留验证选中的第 23 轮权重。该预警是工程暂停条件，不代表已从理论上证明过拟合。

### 2. 商品向量、残差量化与 EMA 对照

使用外部预训练、冻结的 MiniLM 对商品文本编码，得到归一化 384 维向量。RQ-VAE 的可训练 MLP 将其映射到 32 维，三层残差量化各使用 256 个码字，解码器重构原向量；三段语义码有碰撞时追加第四段消歧码，保证完整 SID 唯一。

- 向量构建：[build_item_embeddings.py](scripts/build_item_embeddings.py)。
- 梯度量化：[train_rqvae.py](scripts/train_rqvae.py)；EMA 对照：[train_rqvae_ema.py](scripts/train_rqvae_ema.py)。
- 结果：[码本审计报告](experiments/rqvae_adam_vs_ema_20261008.md)。

梯度版以重构、码本、承诺损失训练；EMA 版通过 EMA 更新码本，梯度仍更新编码器和解码器。第一层活跃码字由 84 增至 117、死码由 172 降至 139；这项变化只证明本次码本利用率改善，不能替代推荐指标。

### 3. 生成召回与合法路径约束

小型 TIGER 使用随机初始化的 Encoder–Decoder 模型，Qwen 则从外部预训练 Instruct 权重开始。两者都将用户历史表示为商品 SID 序列，生成四段 SID，再精确映射到商品。

前缀树记录目录中实际存在的完整路径，在每一步屏蔽不存在的后续 token。beam20 保留多条候选路径，形成 Top-K 商品列表；它不等于对整个目录做精确全概率排序。第四段不是连续向量近邻查找，而是唯一 SID 查表的一部分。

- 小型模型：[tiger_model.py](scripts/tiger_model.py)、[TIGER 实施路线](01_知识库/04_TIGER/小型TIGER实施路线.md)。
- Qwen 数据、训练、检索共用逻辑：[qwen_sft.py](scripts/qwen_sft.py)。
- 独立评测与候选导出：[evaluate_qwen_sft.py](scripts/evaluate_qwen_sft.py)、[export_qwen_candidates.py](scripts/export_qwen_candidates.py)。

### 4. Qwen 全参数 SFT 适配

参考固定版本的 MiniOneRec 实现思路，在本项目数据与四段 SID 上编写适配入口；具体来源与未修改源码见[参考目录](baselines/minionerec_reference/README.md)。

1. 扩展 4×256 个 SID token，固定随机种子导出 Q0，作为无推荐训练的共同初始化；Q0 不表示额外预训练。
2. 通过 `apply_chat_template` 将系统说明、用户任务和历史 SID 编成输入；训练时拼接目标商品四段 SID 与结束标记。
3. 提示与 padding 的 `labels=-100`，只在目标答案计算交叉熵；提示仍参与注意力，梯度可经答案回传到提示相关表示与模型参数。
4. 全参数更新 Qwen 的 embedding、注意力、FFN、归一化等可训练参数。MiniLM、RQ-VAE 与 SID 映射保持离线固定。
5. 按验证 NDCG@10 选择模型，再对留出的测试目标评测。

入口：[prepare_qwen_sft.py](scripts/prepare_qwen_sft.py)、[export_qwen_q0.py](scripts/export_qwen_q0.py)、[train_qwen_sft.py](scripts/train_qwen_sft.py)、[freeze_qwen_training_selection.py](scripts/freeze_qwen_training_selection.py)。真实分词样本见 [sample_tokenization.json](experiments/qwen_sft_office2018_20261013/sample_tokenization.json)。

## 如何解读结果

| 观察 | 实测依据 | 可支持的结论 |
|---|---|---|
| Qwen SFT 高于当前 SASRec | Recall@10：0.057650→0.093677；NDCG@10：0.035554→0.075496 | 当前方案组合的实测差异；容量、预训练、预算与搜索方式未控制，不能全部归因于生成式方法 |
| 小型 TIGER 的 Recall@10 回退 | 梯度版 0.049566，低于 SASRec 0.057650 | 通路跑通不等于优于传统推荐；还需研究容量、训练预算、SID 质量与检索误差 |
| EMA 提升码本利用率，但 Qwen 推荐指标略降 | 第一层活跃码字 84→117；Qwen Recall@10：0.093677→0.092870，Recall@20：0.108819→0.106466 | 本次码本分配改善未转化为推荐收益，梯度 SID 继续作为主基线 |
| Top20 不只共享 2–3 个首段码 | 33 人、660 候选；首段平均 5.88 种 | 反驳该抽样中的直觉假设；不能据此推断所有用户分布 |

原始证据统一链接于[首页结果表](README.md#项目成果)和[SFT 码本对照](experiments/ema_sft_20261021/README.md)。原 SFT 完整逐步日志本次未取回，不编造精确训练总时长；测试记录中的约 2 小时 53 分钟是检索评测时长。

## 规模、部署与可复用资产

Qwen 扩展后共 1,544,870,912 参数：28 层 Decoder、hidden1536、12 个 Q 头 / 2 个 KV 头、FFN8960、词表152689、输入输出权重共享。SASRec 1,709,440 参数，小型 TIGER 328,320 参数；MiniLM 22,713,216 参数与 RQ-VAE 294,912 参数属于离线阶段。

公开资产包含代码、配置、模型结构、正式指标、抽样候选、上游来源和 SHA256 指纹。Qwen SFT 权重与完整训练归档存私有持久网盘；部署需要权重、扩展分词器、配置与相同 SID 映射，精确续训还需优化器、调度器和随机状态。

## 当前边界与下一步

当前只报告一个数据处理版本和单随机种子，没有显著性检验、线上 A/B 或全局时间冷启动结论。不同方案参数规模悬殊，结果表不能替代同容量消融。

后续优先保持数据与检索协议固定，比较商品编码、量化质量与生成器适配的影响；新增方案需先检查训练信号、实际参数更新、显存与验证效果，再进行独立测试和发布。已有 AUC / GAUC 入口尚未实测；其他编码模型资产准备也不等于完成对应推荐实验。
