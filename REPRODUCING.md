# 复现与运行

本页保留数据准备与 SASRec 起步命令。所有命令均在项目根目录执行；使用新的输出目录，避免覆盖旧实验。

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


## 生成式推荐运行入口

| 阶段 | 文档与代码 |
|---|---|
| MiniLM 商品向量 | [模型与分词器](02_商品编码/01_MiniLM/README.md) · `scripts/build_item_embeddings.py` |
| 梯度 / EMA 量化 | [离散语义 ID](03_离散语义ID/README.md) · `scripts/train_rqvae.py` · `scripts/train_rqvae_ema.py` |
| 小型 TIGER | [实施路线](01_知识库/04_TIGER/小型TIGER实施路线.md) · [实际对照报告](experiments/semantic_retrievers_comparison_20261008.md) |
| Qwen 全参数 SFT | [训练配置与步骤](04_推荐生成/02_Qwen2.5-1.5B-Instruct/README.md) · [正式实验](experiments/qwen_sft_office2018_20261013/README.md) |
| 固定初始化 | `scripts/export_qwen_q0.py` · `scripts/qwen_initialization.py` |
| 检索与候选分析 | `scripts/evaluate_qwen_sft.py` · `scripts/export_qwen_candidates.py` |

Qwen 实验环境为 Python 3.11、PyTorch 2.6.0+cu124、Transformers 4.57.6，硬件 A100 40GB。CPU 基线环境与 GPU 环境分别锁定，不将 CPU requirements 直接当作 GPU 安装方案。

## 复用 SFT 检查点

已训练 Qwen 权重继续存放于私有持久网盘，未上传 GitHub。推理需要 `selected` 中的权重、扩展分词器、模型配置，以及对应的 SID 映射和数据 manifest。检查权重与数据 SHA256 后，再按原评测协议运行。完整续训还需要 optimizer、scheduler 和 RNG 状态，仅有部署权重不能精确恢复训练。详见[权重记录](experiments/qwen_sft_office2018_20261013/model_artifact.json)。

Git 中的正式测试 JSON 保留当时评测代码 SHA256。当前入口后续可能有修复；复核历史结果时应匹配记录的代码版本，不能仅用最新脚本就声称逐字节复现。
