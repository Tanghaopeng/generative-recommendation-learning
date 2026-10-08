# Generative Recommendation Learning

个人生成式推荐学习与实践项目，按“理解方法 → 阅读实现 → 准备数据 → 建立基准 → 逐步迭代”的顺序推进。

本仓库与 [推荐系统学习知识库](https://github.com/Tanghaopeng/recommendation-system-learning) 配套：知识库侧重概念与专题整理，这里侧重代码、公开数据、实验协议和实现过程。

当前主线：**Amazon Reviews 2018 Office Products → SASRec → 小型 TIGER 式生成式召回 → 简化 OneRec**。

> 当前 SASRec 最佳权重为第 23 轮：测试 Recall@10=0.057650，NDCG@10=0.035554。
> 续训在第 25 轮因连续过拟合迹象自动暂停；配置、验证走势和检查点见 [续训报告](experiments/sasrec_cpu_extend40_20261007.md)。
> [首次 20 轮实验](experiments/sasrec_cpu_20261007.md) 保留作历史对照，包含一次优化器重置后的恢复。
> TIGER、预训练大模型和 OneRec 是后续计划，当前没有这些模型的指标或权重。

## 已完成的基线与评测口径

2026-10-07 的真实本地 CPU 实验；以下均为测试集结果，按验证 NDCG@10 选取模型。

| 方法 | HitRate@10 / Recall@10 | NDCG@10 | HitRate@20 / Recall@20 | NDCG@20 |
|---|---:|---:|---:|---:|
| Popularity（训练集商品计数） | 0.008949 | 0.004577 | 0.016572 | 0.006475 |
| SASRec（首次 20 轮，第 20 轮最佳） | 0.053660 | 0.031269 | 0.070785 | 0.035575 |
| SASRec（续训，第 23 轮最佳） | **0.057650** | **0.035554** | **0.074406** | **0.039775** |

**如何理解 5.76%：** 对每个用户推荐 10 件商品，检查真实下一件商品是否在其中。
HitRate@10 是命中用户数除以全部测试用户数；每人仅一个测试目标，因此与 Recall@10 相等。
这不是推荐商品的准确率，也不是从 10 件候选中挑选正样本。NDCG@10 还衡量命中目标在前 10 中的排位。

| 环节 | 本项目实际使用的规则 |
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
训练代码、依赖、测试与聚合报告公开；数据、用户映射、权重及完整本机日志保留在本地忽略目录。

**下一步：** 先完成 [小型 TIGER 实施路线](01_知识库/04_TIGER/小型TIGER实施路线.md) 的商品文本与向量准备，
再依次实现 RQ-VAE、语义 ID 和下一物品生成，沿用本节评测口径。

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
- [ ] 构建物品向量、RQ-VAE 和语义 ID，训练小型 TIGER 式召回。
- [ ] 引入小型预训练模型，开展推荐 SFT。
- [ ] 探索简化 OneRec 的列表生成与偏好优化。

## 项目结构

```text
generative-recommendation-learning/
├── 01_知识库/
│   ├── 01_项目路线/
│   ├── 02_SASRec/
│   ├── 03_数据与评测/
│   └── 04_TIGER/
├── baselines/sasrec/       模型源码与上游说明
├── scripts/               下载、预处理、训练、评测与推理
├── data/                  本地数据，原始和处理文件不入 Git
├── experiments/           数据统计与实验记录
├── requirements.txt
├── sources.json           官方地址、固定版本和 SHA-256
└── THIRD_PARTY_NOTICES.md
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
python -m unittest discover -s tests -v
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
