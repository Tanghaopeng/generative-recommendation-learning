# 第三方代码与数据来源

## SASRec PyTorch 实现

- 上游：[pmixer/SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch)
- 作者：Zan Huang；论文作者：Wang-Cheng Kang、Julian McAuley。
- 固定版本：`b253407b3c5d6ec3201a39767a572f44fd5ef6d0`。
- 本仓库位置：`baselines/sasrec/`。
- 代码来源：上游 `python/main.py`、`python/model.py`、`python/utils.py`。
- 这三个源码文件按原样复制，没有声称为本项目原创。
- 已保留上游 LICENSE、README、CITATION.cff、Result_Norm.md。
- 许可证：Apache-2.0，见 [保留的许可证](baselines/sasrec/LICENSE)。
- 作者原始实现：[kang205/SASRec](https://github.com/kang205/SASRec)，本仓库只链接参考。

## Amazon Reviews 2018

- 官方页面：<https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/>
- 数据作者：Jianmo Ni、Jiacheng Li、Julian McAuley。
- 引用：Justifying recommendations using distantly-labeled reviews and fine-grained aspects，EMNLP 2019。
- 原始交互和元数据由下载脚本从官方地址获取，不作为本仓库源码发布。
- 数据使用应遵循原始来源的说明；本仓库的代码许可证不覆盖第三方数据。
- 下载地址、字节数、SHA-256 和读取记录数见 [sources.json](sources.json)。

## 本项目新增部分

数据下载、清洗、核验脚本，学习路线，代码讲解和实验记录由本项目整理维护。
`scripts/run_sasrec.py` 调用保留的上游模型，新增固定种子的单进程采样、全商品评测、
仅验证集选模、实验记录；`recommend_sasrec.py` 与 `report_sasrec.py` 提供推理与汇总。
训练目标与模型结构沿用上游，新增入口不声称为 SASRec 原始方法或精确论文复现。
仓库根目录 LICENSE 采用 Apache-2.0；第三方内容保留各自原有权利和署名。

## 小型 TIGER 的第三方实现

- 来源：[EdoardoBotta/RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender)。
- 固定版本：`957d32bda43958ba641ae919abb1092ac6738798`；下载日期：2026-10-08。
- 作者：Edoardo Botta；MIT，见 [原许可证](baselines/tiger/upstream/LICENSE)。
- 本仓库保留15份未修改原文件，含 RQ-VAE、量化器、MLP、T5序列模型和所需辅助模块。
- 完整文件指纹：[upstream_source.json](baselines/tiger/upstream_source.json)。
- 这不是 TIGER 论文作者官方实现，不声称精确复现论文。
- 新增适配：[说明](baselines/tiger/README.md)。本项目更改采样/初始化/训练入口、四段SID、检索与评测；原文件不变。
- EMA更新模块 `scripts/rqvae_ema.py` 为本项目独立实现，参考 [VQ-VAE论文附录A.1](https://arxiv.org/pdf/1711.00937) 的计数/向量和移动平均公式；不是原仓库的EMA源码，也不声称官方TIGER EMA实现。

## 冻结商品文本编码器

- 模型：[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)。
- 固定 revision：`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`；模型卡标注 Apache-2.0。
- 本项目使用 Transformers 加载预训练权重，masked mean pooling、L2归一化，离线产出384维向量。
- 不训练此编码器，CPU版本实际截断128 token。按用户要求，2026-10-09将这份固定版本的实际权重和tokenizer发布到[02商品编码/MiniLM](02_商品编码/01_MiniLM/README.md)，保留官方模型卡、Apache-2.0许可说明、逐文件来源与校验值；其他训练权重仍保留本地。

## Qwen SFT 的 MiniOneRec 单卡参考

- 来源：[wbn11/minionerec-single-gpu](https://github.com/wbn11/minionerec-single-gpu)，固定提交 `bccd7ef70291c5b4b0c30b6222bc491039b4cd18`。
- 未修改参考源码、README、依赖和上游提供的 Apache-2.0 `LICENSE-MiniOneRec.txt` 保存在 [baselines/minionerec_reference](baselines/minionerec_reference/README.md)，校验值见该目录 `upstream_source.json`。
- 参考其全参数BF16 SFT、SID词表扩展、回答监督与多任务对齐设计。本地 `qwen_sft.py`、`prepare_qwen_sft.py`、`train_qwen_sft.py`、`evaluate_qwen_sft.py` 为独立适配，未直接调用上游三段SID训练入口；不是官方OneRec或精确论文复现。
- 模型：[Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)，固定 revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。本地已验证模型权重，权重和数据均不提交Git；遵循模型自带许可证。
