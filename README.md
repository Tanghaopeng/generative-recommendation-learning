# Generative Recommendation Learning

个人生成式推荐学习与实践项目，按“理解方法 → 阅读实现 → 准备数据 → 建立基准 → 逐步迭代”的顺序推进。

本仓库与 [推荐系统学习知识库](https://github.com/Tanghaopeng/recommendation-system-learning) 配套：知识库侧重概念与专题整理，这里侧重代码、公开数据、实验协议和实现过程。

当前主线：**Amazon Reviews 2018 Office Products → SASRec → MiniLM＋RQ-VAE／EMA＋小型TIGER → Qwen2.5-1.5B-Instruct全参数SFT → 后续GRPO**。

## 按阶段学习与运行

| 编号 | 目录 | 内容 |
|---|---|---|
| 01 | [知识库](01_知识库/README.md) | 原理、数据协议、SFT输入输出与梯度流程 |
| 02 | [商品编码](02_商品编码/README.md) | MiniLM等连续内容向量版本 |
| 03 | [离散语义ID](03_离散语义ID/README.md) | RQ-VAE Adam、EMA；RQ-KMeans为预留版本 |
| 04 | [推荐生成](04_推荐生成/README.md) | 已训练的小型TIGER、Qwen全参数SFT实现 |
| 06 | [每日效果记录](06_每日效果记录/README.md) | 每天的实现、工程验证与真实实验状态 |
| 07 | [组合结果统计](07_组合结果统计/README.md) | 编码器×SID×生成器的效果总表 |

根目录05暂未使用。已有`baselines/`、`scripts/`、`experiments/`保留原路径，编号目录提供版本说明、配置和入口链接，保持旧实验指纹可核验。

**2026-10-09新增：** [Qwen全参数SFT运行说明](04_推荐生成/02_Qwen2.5-1.5B-Instruct/README.md)与[原理讲解](01_知识库/05_SFT与生成式推荐/Qwen全参数SFT流程.md)。
复用已有MiniLM及四段SID，提供Q1主任务、Q2可选标题对齐、GPU训练和约束商品检索代码；CPU检查通过不等于完成1.5B训练，当前没有Qwen推荐指标。参考源码已固定版本下载并保留在[MiniOneRec参考目录](baselines/minionerec_reference/README.md)。

**MiniLM文件已随仓库提供：** [模型权重、分词器与架构说明](02_商品编码/01_MiniLM/README.md)，固定官方版本，权重约90.87 MB；可分别离线加载，不需自行训练MiniLM。

**第二套商品编码模型：** [Qwen3-Embedding-4B](02_商品编码/02_Qwen3_Embedding_4B/README.md)，默认输出2560维，使用最后有效token池化；准备本地与矩阵云模型资产后，再独立训练对应的量化器与推荐模型。模型资产准备不等于已完成商品向量生成或推荐训练。

**Q0→SFT→GRPO 评测计划：** [多角度评测协议](01_知识库/03_数据与评测/Q0_SFT_GRPO多角度评测方案_20261010.md)。已增加独立 [AUC/GAUC 评分入口](scripts/evaluate_qwen_auc.py)：固定每用户1正例＋100个无重复代理负例，与完整目录的HR/NDCG分开报告。新增指标单元核验通过，GPU评分未执行；Q0初始化导出和本项目GRPO适配仍待完成。

> 当前 SASRec 最佳权重为第 23 轮：测试 Recall@10=0.057650，NDCG@10=0.035554。
> 续训在第 25 轮因连续过拟合迹象自动暂停；配置、验证走势和检查点见 [续训报告](experiments/sasrec_cpu_extend40_20261007.md)。
> [首次 20 轮实验](experiments/sasrec_cpu_20261007.md) 保留作历史对照，包含一次优化器重置后的恢复。
> 小型 TIGER 原版SID已完成3轮生成训练，验证选中第3轮：测试Recall@10=0.049566、NDCG@10=0.036792，见[正式报告](experiments/tiger_cpu_20261008.md)。
> 首轮采用hidden64、编码器/解码器各1层、2头；沿用[共同评测口径](experiments/tiger_cpu_20261008.md)，全目录前缀树beam20属于近似检索。
> Qwen全参数SFT已有实现，正式GPU训练待执行；GRPO与OneRec后续实验仍为计划。
> 两套SID的独立生成模型均已完成3轮训练，全用户验证都选中第3轮，未触发过拟合暂停；完整结果见[Transformer对照报告](experiments/semantic_retrievers_comparison_20261008.md)。
> [码本审计与EMA对照](experiments/rqvae_adam_vs_ema_20261008.md)显示第一层活跃码字84→117、死码172→139；EMA召回测试前10指标略高、前20指标略低，量化利用率改善不等于推荐效果全面改善。

## 已完成的基线与评测口径

2026-10-07至08日的真实本地CPU实验；以下均为测试集结果，按验证NDCG@10选取模型。

| 方法 | HitRate@10 / Recall@10 | NDCG@10 | HitRate@20 / Recall@20 | NDCG@20 |
|---|---:|---:|---:|---:|
| Popularity（训练集商品计数） | 0.008949 | 0.004577 | 0.016572 | 0.006475 |
| SASRec（首次 20 轮，第 20 轮最佳） | 0.053660 | 0.031269 | 0.070785 | 0.035575 |
| SASRec（续训，第 23 轮最佳） | 0.057650 | 0.035554 | 0.074406 | 0.039775 |
| 小型 TIGER（Adam SID，第3轮最佳） | 0.049566 | 0.036792 | 0.060637 | 0.039582 |
| 小型 TIGER（EMA SID，第3轮最佳） | 0.049820 | 0.037539 | 0.057096 | 0.039382 |

**如何理解 5.76%：** 对每个用户推荐 10 件商品，检查真实下一件商品是否在其中。
HitRate@10 是命中用户数除以全部测试用户数；每人仅一个测试目标，因此与 Recall@10 相等。
这不是推荐商品的准确率，也不是从 10 件候选中挑选正样本。NDCG@10 还衡量命中目标在前 10 中的排位。

下表为SASRec规则。TIGER式召回使用四段SID交叉熵，不抽商品负例；候选检索使用全目录前缀树的有限beam，详见正式报告。

| 环节 | SASRec实际使用的规则 |
|---|---|
| 数据划分 | 每用户按时间排序，最后一项测试、倒数第二项验证、更早部分训练 |
| 训练负采样 | 每个有效下一物品位置抽 1 个均匀随机负例；排除完整训练历史，每轮重新抽；不读取验证/测试目标进行过滤 |
| 训练损失 | 正例标签 1、负例标签 0 的 BCE 相加；padding 不参与损失 |
| 评测用户 | 全部 86,713 个用户，每人 1 个真实目标 |
| 评测候选 | 全部 25,898 件商品，排除已知历史与 padding；不随机抽负候选 |
| 测试输入 | 训练历史加验证商品，模型编码最近 20 项，过滤时仍使用完整已知历史 |
| 排序与选模 | 分数降序、同分时商品 ID 升序；仅验证 NDCG@10 选模与判断暂停，测试不参与 |

上游采样评测在“1 个正样本 + 100 个随机负候选”中排序，**不能与这里的全目录指标直接比较**。
未观测商品也不等于用户明确不喜欢。当前成绩是单随机种子的起步基线，
使用用户内划分与固定商品目录，不等于论文精确复现、严格全局时间实验或线上效果。

续训计划新增最多 20 轮，实际新增 5 轮：第 24、25 轮训练 loss 低于第 23 轮，
但验证 NDCG@10 均低于第 23 轮最佳值，触发连续两次过拟合预警，暂停于第 25 轮。
这是保守的暂停信号，不能表述为验证成绩逐轮下降或已经证明过拟合。
报告、逐轮记录与配置见 [实验总表](experiments/README.md) 和 [续训报告](experiments/sasrec_cpu_extend40_20261007.md)。
训练代码、依赖、测试与聚合报告公开；数据、用户映射、训练权重及完整本机日志保留在本地忽略目录。指定的公开预训练MiniLM权重和配套分词器另在02目录提供。

**已完成首轮：** [小型 TIGER 实施路线](01_知识库/04_TIGER/小型TIGER实施路线.md)：商品文本 → 冻结MiniLM向量 → RQ-VAE → 四段唯一SID → 下一物品生成。原版与EMA的[正式对照](experiments/semantic_retrievers_comparison_20261008.md)已导出。
沿用相同数据划分、用户、目录与历史过滤；TIGER有限beam属于近似检索，不能表述为SASRec那样的全目录精确打分。

## 学习导航

| 目录 / 文档 | 内容 |
|---|---|
| [项目路线](01_知识库/01_项目路线/路线图.md) | 阶段目标、实现顺序、验收标准 |
| [SASRec 代码流程讲解](01_知识库/02_SASRec/SASRec代码流程讲解.md) | 数据、样本、自注意力、损失、预测与评估 |
| [本地训练与推理](01_知识库/02_SASRec/本地训练与推理.md) | 已验证的 CPU 环境、完整训练、恢复和推荐调用 |
| [数据与评测协议](01_知识库/03_数据与评测/数据与评测协议.md) | 清洗、共同划分、元数据缺失与评测口径 |
| [小型 TIGER 实施路线](01_知识库/04_TIGER/小型TIGER实施路线.md) | 从商品文本、RQ-VAE、语义 ID 到生成与评测的实施顺序 |
| [SASRec 源码](baselines/sasrec/) | 固定版本的第三方 PyTorch 实现 |
| [数据目录](data/README.md) | 下载与处理文件的用途 |
| [实验记录](experiments/README.md) | 数据统计、核验与真实模型指标 |
| [第三方来源](THIRD_PARTY_NOTICES.md) | 代码来源、固定 commit、许可证与引用 |

## 当前进度

- [x] 下载 Amazon Reviews 2018 Office Products 官方 5-core 交互和元数据。
- [x] 去重后重新进行用户—物品 5-core，导出共同数据划分。
- [x] 核验数字 ID、时序、5-core、SASRec 划分和元数据对齐。
- [x] 整理 SASRec PyTorch 源码、中文讲解与运行入口。
- [x] 配置训练环境，完成小规模训练通路检查。
- [x] 适配全物品评估，仅用验证集选模型，完成首次 SASRec 基线训练及推理。
- [x] 固定并保留第三方TIGER实现来源，完成25,898件商品向量与30轮RQ-VAE。
- [x] 生成完整四段唯一语义ID，验收生成训练、检索与权重加载通路。
- [x] 完成原版SID的小型TIGER正式训练、全用户验证选模与一次测试。
- [x] 审计三层死码、频次与有效码字数，完成30轮EMA量化对照。
- [x] 完成EMA SID的独立生成训练、全用户验证选模、测试与原版/SASRec对照。
- [x] 接入Qwen预训练权重、实现四段SID全参数SFT、样本检查和检索评测入口。
- [ ] 在GPU运行Qwen pilot、正式SFT及全量验证/测试。
- [ ] 探索简化 OneRec 的列表生成与偏好优化。

## 项目结构

```text
generative-recommendation-learning/
├── 01_知识库/                  原理、路线、数据协议、SFT讲解
├── 02_商品编码/                连续商品向量版本
├── 03_离散语义ID/              Adam / EMA；RQ-KMeans预留
├── 04_推荐生成/                小型TIGER / Qwen全参SFT配置与入口
├── 06_每日效果记录/            每日状态与实验证据
├── 07_组合结果统计/            编码器×SID×生成器效果总表
├── baselines/                  保留原始第三方源码和许可证
├── scripts/                    实际可运行入口
├── tests/                      协议与模型通路检查
├── experiments/                聚合报告，保留历史路径
├── data/                       本地忽略：原始、处理、SID数据
└── outputs/                    本地忽略：权重、训练状态与日志
```

## 数据准备

脚本只依赖 Python 标准库。在项目根目录执行：

```bash
python scripts/download_data.py
python scripts/prepare_office.py
python scripts/verify_dataset.py
```

下载脚本检查 SHA-256，复用已有完整文件。只核验已下载数据时执行：

```bash
python scripts/download_data.py --verify-only
```

原始压缩文件共约 597MiB；处理需要额外磁盘空间。
预处理生成 `data/processed/office2018/`，并把两列输入复制到 `baselines/sasrec/data/office2018.txt`。
原始评论、用户映射、训练数据和训练权重不上传 GitHub。

## 当前数据统计

| 项目 | 数量 |
|---|---:|
| 官方 5-core 评论 | 800,357 |
| 清洗后用户 | 86,713 |
| 清洗后物品 | 25,898 |
| 清洗后交互 | 677,247 |
| 训练交互 | 503,821 |
| 验证 / 测试目标 | 各 86,713 |
| 有元数据的物品 | 25,870 |
| 缺少元数据的物品 | 28 |

统计属于本仓库的处理版本：保留每个用户—物品的首次交互，再重新进行 5-core。
详细协议见 [数据与评测说明](01_知识库/03_数据与评测/数据与评测协议.md)，统计见 [data_stats.json](experiments/data_stats.json)。

## SASRec 起步运行

已验证独立 Python 3.12.13 环境：PyTorch 2.7.1+cpu、NumPy 2.2.6。
完整依赖见 `requirements-cpu-lock.txt`，CPU 环境可按下面步骤创建：

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-cpu-lock.txt
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux / macOS: source .venv/bin/activate
```

GPU 用户需按硬件安装相应 PyTorch 构建，Intel 显卡不能直接使用 CUDA。
激活虚拟环境后，从项目根目录检查数据、协议和训练通路：

```bash
python scripts/verify_dataset.py
python -m unittest discover -s tests -p test_sasrec_runner.py -v
python scripts/run_sasrec.py --output outputs/sasrec_smoke_new --smoke-only --smoke-steps 20 --threads 2
```

小规模检查只跑 20 个 batch，并核对权重保存和重新加载一致，不提供正式指标。
完整训练与推理：

```bash
python scripts/run_sasrec.py --output outputs/sasrec_office2018_cpu_new --epochs 20 --maxlen 20 --batch-size 128 --eval-batch-size 128 --eval-every 5 --patience 2 --threads 2
python scripts/recommend_sasrec.py --checkpoint outputs/sasrec_office2018_cpu_new/best.pth --user-id 1 --k 10
```

不同实验使用新输出目录，避免覆盖旧权重。当前上游源码仍保留供学习；
实际运行入口使用原模型，独立实现采样、全商品评测、验证集选模、恢复和结果记录。
训练可能持续占用 CPU，本机恢复阶段使用 2 个线程和 BelowNormal 优先级。

## 当前正式评测适配

上游使用“1 个真实目标 + 100 个随机候选”评估，会随机抽评测用户；保存 checkpoint 的条件还参考测试集。
这些代码保留供学习，实际入口 `scripts/run_sasrec.py` 不使用该评测和选模逻辑。
当前已统一全物品、全部用户评估、完整已知历史过滤、固定随机种子和同分排序，
仅按验证 NDCG@10 选模型，最终报告一次测试结果。

## 实现与结果的边界

- 模型代码来自 [pmixer/SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch)，已保留原作者和许可证。
- 本项目新增数据下载、处理、核验、训练入口、全商品评测、恢复、推理、实验记录和讲解。
- 指标在真实训练前保持“未训练”，不引用上游其他数据集成绩替代本项目结果。
- 论文方法、个人简化设计和真实实验结果分别标注。

## 主要参考

- [SASRec 作者原始实现](https://github.com/kang205/SASRec)
- [TIGER 论文](https://arxiv.org/abs/2305.05065)
- [RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender)
- [OneRec 论文](https://arxiv.org/abs/2502.18965)
- [MiniOneRec](https://github.com/AkaliKong/MiniOneRec)
- [官方 OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec)

## 许可证

代码采用 [Apache-2.0](LICENSE)。第三方数据与内容保留各自原始权利，见 [第三方说明](THIRD_PARTY_NOTICES.md)。
