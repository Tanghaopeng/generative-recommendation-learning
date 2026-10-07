# Generative Recommendation Learning

个人生成式推荐学习与实践项目，按“理解方法 → 阅读实现 → 准备数据 → 建立基准 → 逐步迭代”的顺序推进。

本仓库与 [推荐系统学习知识库](https://github.com/Tanghaopeng/recommendation-system-learning) 配套：知识库侧重概念与专题整理，这里侧重代码、公开数据、实验协议和实现过程。

当前主线：**Amazon Reviews 2018 Office Products → SASRec → 小型 TIGER 式生成式召回 → 简化 OneRec**。

> 当前完成数据下载、预处理、核验和 SASRec 源码整理，尚未完成模型训练。
> TIGER、预训练大模型和 OneRec 是后续计划，当前没有这些模型的指标或权重。

## 学习导航

| 目录 / 文档 | 内容 |
|---|---|
| [项目路线](01_知识库/01_项目路线/路线图.md) | 阶段目标、实现顺序、验收标准 |
| [SASRec 代码流程讲解](01_知识库/02_SASRec/SASRec代码流程讲解.md) | 数据、样本、自注意力、损失、预测与评估 |
| [数据与评测协议](01_知识库/03_数据与评测/数据与评测协议.md) | 清洗、共同划分、元数据缺失与评测口径 |
| [SASRec 源码](baselines/sasrec/) | 固定版本的第三方 PyTorch 实现 |
| [数据目录](data/README.md) | 下载与处理文件的用途 |
| [实验记录](experiments/README.md) | 数据统计、核验结果与待填的模型指标 |
| [第三方来源](THIRD_PARTY_NOTICES.md) | 代码来源、固定 commit、许可证与引用 |

## 当前进度

- [x] 下载 Amazon Reviews 2018 Office Products 官方 5-core 交互和元数据。
- [x] 去重后重新进行用户—物品 5-core，导出共同数据划分。
- [x] 核验数字 ID、时序、5-core、SASRec 划分和元数据对齐。
- [x] 整理 SASRec PyTorch 源码、中文讲解与运行入口。
- [ ] 配置训练环境，完成小规模训练通路检查。
- [ ] 适配全物品评估，仅用验证集选模型，训练正式 SASRec 基线。
- [ ] 构建物品向量、RQ-VAE 和语义 ID，训练小型 TIGER 式召回。
- [ ] 引入小型预训练模型，开展推荐 SFT。
- [ ] 探索简化 OneRec 的列表生成与偏好优化。

## 项目结构

```text
generative-recommendation-learning/
├── 01_知识库/
│   ├── 01_项目路线/
│   ├── 02_SASRec/
│   └── 03_数据与评测/
├── baselines/sasrec/       模型源码与上游说明
├── scripts/               下载、预处理与核验
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

建议使用独立的 Python 3.11/3.12 环境。依赖范围是起步建议，尚未验证完整训练环境。

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux / macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

GPU 用户需按硬件安装相应 PyTorch 构建，Intel 显卡不能直接使用 CUDA。
准备好环境和数据后，检查训练通路：

```bash
cd baselines/sasrec
python main.py --dataset office2018 --train_dir smoke_cpu --device cpu --maxlen 50 --hidden_units 64 --num_blocks 2 --num_heads 2 --batch_size 32 --num_epochs 1
```

命令在完整数据上训练 1 epoch，CPU 可能耗时较长。上游每 20 epoch 才评估，因此不会产生正式 HR/NDCG 指标。
命令示例不等于已经验证的训练结果。

## 正式对比前需要完成的适配

上游使用“1 个真实目标 + 100 个随机候选”评估，会随机抽评测用户；保存 checkpoint 的条件还参考测试集。
这些代码保留供学习，不能直接作为与 TIGER 公平比较的最终协议。
下一步统一全物品评估、历史过滤、随机种子与 K 值，仅验证集选模型，最终报告一次测试结果。

## 实现与结果的边界

- 模型代码来自 [pmixer/SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch)，已保留原作者和许可证。
- 本项目新增数据下载、处理、核验、学习路线与讲解。
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
