# MiniOneRec 单卡实现参考

来源：[wbn11/minionerec-single-gpu](https://github.com/wbn11/minionerec-single-gpu/tree/bccd7ef70291c5b4b0c30b6222bc491039b4cd18)，固定提交见 `upstream_source.json`。
本目录保存未修改的 SFT、GRPO 训练入口及它们依赖的项目源码，并保留上游 README、A6000 依赖清单及上游提供的 Apache-2.0 许可证。SFT 文件于 2026-10-09 下载，GRPO 及所需依赖于 2026-10-10 补齐；逐文件 SHA-256 见 `upstream_source.json`。这是两个训练入口所需的源码集合，不是整个上游仓库的完整镜像。

## 两个阶段的代码位置

| 阶段 / 功能 | 未修改的上游文件 | 作用 |
|---|---|---|
| SFT 启动入口 | [scripts/train_sft.py](upstream/scripts/train_sft.py) | 解析模型、数据和训练参数，启动全参数 SFT |
| SFT 训练 | [sft_training.py](upstream/src/minionerec/training/sft_training.py) | 加载 Qwen、扩展 SID 词表、组织多任务数据、构建 Trainer |
| SFT 样本 | [sft_datasets.py](upstream/src/minionerec/data/sft_datasets.py) | 构造输入、回答和仅回答部分的监督标签 |
| SID 词表 | [sft_vocab.py](upstream/src/minionerec/training/sft_vocab.py) | 添加 SID token、初始化对应向量 |
| GRPO 启动入口 | [scripts/train_grpo.py](upstream/scripts/train_grpo.py) | 从 SFT 检查点加载模型，启动奖励优化 |
| GRPO 训练 | [grpo_training.py](upstream/src/minionerec/training/grpo_training.py) | 策略和参考模型、候选生成、优势计算与参数更新 |
| GRPO 样本 | [grpo_datasets.py](upstream/src/minionerec/data/grpo_datasets.py) | 推荐和文本对齐 prompt、训练目标映射 |
| GRPO 候选生成 | [grpo_generation.py](upstream/src/minionerec/generation/grpo_generation.py) | SID 前缀树约束的 beam 候选生成；默认 16 个候选，启用采样 |
| GRPO 奖励 | [ranking_grpo.py](upstream/src/minionerec/rewards/ranking_grpo.py) | 精确命中、排名惩罚及组内标准化优势 |
| GRPO 损失 | [grpo_loss.py](upstream/src/minionerec/training/grpo_loss.py) | 生成 token 的策略项、KL 项和有效长度归一化 |

GRPO 的策略模型采用全参数更新；参考模型不参与梯度更新，但按配置定期混合同步。命中候选获得精确奖励 1；有命中的组内，错误候选受排名惩罚；整组无命中则全部奖励为零。这里采用上游推荐场景的 beam 候选改编，不能直接当作标准独立随机采样 GRPO，也不能把损失描述成带 PPO clip 的实现。

## 如何理解“已下载”和“可运行”

上面的源码和项目内导入依赖已经下载，文件指纹和 Python 语法已核验；没有在本地执行它们的 GPU 训练。本目录不能直接当成本项目运行入口。运行上游脚本仍需 Linux/CUDA 环境、兼容依赖、上游格式的数据、模型路径，以及 GRPO 所需的 SFT 检查点；依赖清单是上游环境记录，尚未在我们的服务器验证。

本项目当前已适配的 SFT 入口是根目录的 `scripts/train_qwen_sft.py`；本项目 GRPO 的四段 SID、数据与评测适配尚未实现或训练。下载上游 GRPO 源码不代表已经完成这部分适配。

本项目的 `scripts/qwen_sft.py`、`train_qwen_sft.py` 等为独立适配实现，参考其“全参数 BF16＋SID 词表扩展＋仅回答监督＋可选文本对齐”流程。
差异：上游三段 SID、本地四段唯一 SID；上游 CSV/其他类目、本地 Office Products 共同划分；上游按验证 loss 选模、本地按验证商品 NDCG@10；本地使用 Qwen chat template，并提供冻结前缀树检索和恢复契约。
不是官方 OneRec 实现，也不是严格论文复现。原始 MiniOneRec 项目：[AkaliKong/MiniOneRec](https://github.com/AkaliKong/MiniOneRec)。
