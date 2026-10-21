# 生成式推荐评测加速审计：2026-10-21

原型测量时的状态：已完成独立原型、一致性测试和A100短测，尚未部署。后续用户明确选择BF16、batch8验证与测试；正式版本新增独立execution配置、迁移保护及混合评测记录拒绝机制，详见EMA SFT运行说明与迁移运行记录。原型与正式训练进程隔离，未重置或修改正式权重、优化器及训练进度。

## 发现的开销

1. `qwen_sft.evaluate_retrieval`逐用户调用`retrieve`。训练batch16只影响训练与teacher-forcing验证loss，不改变逐用户的Recall/NDCG召回循环。
2. `CatalogConstraint.__call__`每条beam分别调用GPU张量的`.tolist()`，再逐行写掩码；输出阶段又逐条读取序列和分数。这些设备到主机的数据读取会引入同步。[PyTorch CUDA说明](https://docs.pytorch.org/docs/stable/notes/cuda.html)
3. 训练结束加载最佳检查点后，`trainer.evaluate()`再次评估相同5000用户。可以在核对最佳模型、数据指纹、用户子集、解码配置后复用选模时的评测记录，避免重复生成；当前代码尚未实施这项改动。
4. KV cache与`torch.inference_mode`已经开启。本次优先处理请求批量和数据传输，不能把已经启用的选项重复计为加速改进。

## 实现的两个独立原型

- **bulk_io_b1**：每层一次读取全部beam前缀，合并掩码索引写入；最终一次读取全部SID、一次读取全部分数。仍逐用户调用同一模型，不改变前向batch形状。
- **bucket_b4 / bucket_b8**：在上述实现上，按完整提示的token长度分桶，每次生成4或8个用户。每个用户仍保留20条beam，独立排除其完整历史，原始顺序在输出时恢复。仅合并相同长度，避免补齐或位置编号改变。混合长度的一般LLM批量生成需要正确处理left padding，[Hugging Face说明](https://huggingface.co/docs/transformers/v4.57.1/llm_tutorial)。

两种原型均保持四段SID、全词表log-softmax之后再做合法路径掩码、`renormalize_logits=False`、无采样、beam20及完整历史排除。没有只在SID词表内重新归一化，也没有缩小目录或用户数来宣称等价加速。

代码：[qwen_eval_fast.py](../../scripts/qwen_eval_fast.py)、[benchmark_qwen_eval.py](../../scripts/benchmark_qwen_eval.py)。本地3项测试覆盖各层掩码、beam与用户的对应、候选顺序、完全排除的目录，以及不同长度提示拒绝合并。

## BF16实测

使用EMA数据对应的未训练Q0，seed2027抽取128个验证用户。每种模式预热8个用户，再计时；同一模型、用户、beam和设备。原始报告：[eval_prototype_benchmark.json](eval_prototype_benchmark.json)。

| 执行模式 | 用户/秒 | 相对原版吞吐 | Top20商品及顺序完全一致 | 商品集合存在差异的用户 |
|---|---:|---:|---:|---:|
| 原版逐用户 | 8.25 | 1.00× | 128/128 | 0 |
| 合并数据读取，逐用户 | 9.11 | 1.10× | 128/128 | 0 |
| 同长度分桶，batch4 | 19.63 | 2.38× | 7/128 | 88 |
| 同长度分桶，batch8 | 23.08 | 2.80× | 4/128 | 90 |

分桶batch8的测试进程实际CUDA分配峰值约3.92GiB。该值不包含暂挂的正式训练进程，不应当作整卡占用。

按短测吞吐简单外推，86,713用户约需2.92小时、2.64小时、1.23小时、1.04小时；5000用户约需10.1、9.1、4.2、3.6分钟。外推不含模型加载、备份等开销，不能作为已完成全量测量的结果。

## 数值一致性核查

原型在CPU的小型真实Qwen模型上与原版的各层掩码及候选顺序一致。另使用相同Q0、32个验证用户、FP32且TF32关闭，对照各自逐用户FP32结果：bulk、batch4、batch8均为32/32个完整候选列表一致。原始报告：[eval_fp32_benchmark.json](eval_fp32_benchmark.json)。

这些结果支持BF16批量运算的舍入路径变化是差异的重要来源，尚不足以断言训练后模型的指标差异可忽略。本次Q0尚未学习SID推荐目标，不能用它的排序差异推断正式SFT的效果。FP32对照用于诊断，正式SFT仍为BF16。

两个短测分别暂挂正式进程约60.8秒、67.3秒，结束后恢复；恢复证据：[BF16进程记录](eval_prototype_probe.json)、[FP32进程记录](eval_fp32_probe.json)。

## 建议采用顺序

1. 保留本轮已经运行的正式训练和验证，避免为更换评测执行方式重跑训练。
2. 在训练后的选中检查点上，对bulk逐用户模式再做候选级审计；它目前是与原版一致、约10%增益的候选替代实现。
3. 若要采用约2.8倍吞吐的batch8模式，在训练后检查点上量化排序及Recall/NDCG差异，并对旧梯度SID模型和EMA模型使用同一执行后端重新评测。旧单用户结果与新批量结果需要明确标记执行方式，不能直接把差值都归因于SID改进。
4. 未来训练版本在严格核对指纹后复用最佳检查点已完成的验证结果，省去末尾重复的一轮5000人生成评测。训练中每500步的验证频率与用户数仍保持原设定。

可复现命令（短测，不执行训练）：

```bash
python scripts/benchmark_qwen_eval.py --checkpoint /path/to/q0_or_trained_model --data data/processed/office2018/qwen_sft_ema --users 128 --precision bf16 --output /path/to/new_eval_probe.json
```

本次原始BF16原型源码包、测试报告和进程恢复记录已写入16区网盘；仓库源代码增加了可选FP32诊断参数。本节性能结论限于单台A100、单次小样本测试；部署后的正式效果单独记录。
