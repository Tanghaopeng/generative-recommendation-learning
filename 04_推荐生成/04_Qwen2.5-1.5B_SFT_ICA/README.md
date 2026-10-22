# 已有 SFT → 加入 ICA → 继续监督微调

本版从已训练好的 **梯度更新 RQ-VAE SID 对应的 Qwen SFT** 权重继续训练。
MiniLM、RQ-VAE、商品 SID 映射和分词器保持固定；**新增 ICA 与原 Qwen 一起更新**。
不是从原始 Qwen 或 Q0 重新开始，也不是 LoRA 或强化学习。

参考：[BARGE, Bridging the Structural Gap: Adapting Autoregressive Generation for Recommendation](https://arxiv.org/abs/2607.21028v1)。
这是按论文思路编写的 decoder-only Qwen 适配实现，没有使用作者官方实现。
论文公式中的商品内聚合、MLP 和门控残差被保留；这里使用 128 维注意力投影以减少额外参数。
我们的第四段用于碰撞消歧，因此 **只聚合前三段语义 token，将摘要融合进历史商品的全部四段 token**。

## 输入在哪里改变

原版：

```text
历史 A/B/C 的四段 SID → Qwen token embedding → 28 层 Qwen → 预测 D 的四段 SID
```

ICA 版：

```text
历史 A/B/C 的四段 SID → Qwen token embedding
    → 按商品分组，每组 [x0, x1, x2, x3]
    → 可训练 query 对 x0/x1/x2 做商品内注意力聚合
    → LayerNorm + MLP 得到商品摘要 z
    → 每个历史 token 使用 sigmoid 门控，将摘要作为残差加回
    → 原来的 28 层 Qwen → 预测 D 的四段 SID
```

对一件商品，计算为：

```text
u = Project([x0, x1, x2])                      # 1536 → 128
a = CrossAttention(learned_query, u, u)        # 4 heads, 不跨商品
z = MLP(LayerNorm(Restore(a)))                # 128 → 1536 → 256 → 1536
g_l = sigmoid(W_gate · concat(x_l, z) + b)     # l = 0,1,2,3
x'_l = x_l + g_l * z
```

这里的 x 是 **Qwen 的可训练 token embedding**，不是 RQ-VAE 的 32 维码本向量。
ICA 不改变输出空间，仍然由 Qwen 自回归生成四段 SID，再通过 SID → ItemID 映射得到商品。
前缀树、历史商品排除、Top-20 候选生成都复用原来的逻辑。

## 为什么不能直接处理输入里的全部 SID

训练输入同时包含提示与答案，例如：

```text
... user ... A的SID / B的SID / C的SID ...
<|im_start|>assistant\n
D0 D1 D2 D3 EOS
```

如果把 D0/D1/D2/D3 也提前聚合，预测 D0 时就会看到 D1/D2/D3，发生答案泄露。
代码通过当前 tokenizer 的 `apply_chat_template` 得到 assistant 分隔符，在这个位置之前识别完整历史 SID 四元组。
**答案 token、普通提示词、角色标记、换行和 padding 都不接受 ICA 融合。**
历史商品内部可以聚合，因为线上预测时该商品的四段编号已全部可见。
训练只监督答案 token 与 EOS；Qwen 的因果注意力继续工作。
本版只支持 Q1 单轮下一商品样本，不支持将多个对话 packing 到一条样本。

生成时仅在 prompt prefill 上执行 ICA；使用 KV cache 的后续单 token 前向不再聚合生成中的答案。
不使用 KV cache 时，重新计算也始终以 assistant 分隔符截断历史范围。

## 初始化、训练与保存

- 从经过 SHA256 与 SID 数据绑定核验的原 SFT 加载全部 Qwen 参数；不扩词表、不重置 SID embedding。
- ICA 的 MLP 最后一层初始化为零，使插入模块时输出与原模型一致。第一步先更新这层，之后梯度进入整个 ICA 分支。
- 使用答案交叉熵，继续全参数 SFT。默认较小学习率 `1e-5`；配置见 `experiments/qwen_ica_full.json`。
- `selected/` 保存 **完整 Qwen＋ICA 权重、模型配置、分词器、model_artifact.json**，可以直接推理或作为后续训练起点。
- `checkpoint-*` 额外保存优化器、调度器与 RNG，用于精确断点续训；不要用仅有权重的 `selected/` 冒充断点。
- 模型类型为 `qwen2_ica`。必须先导入 `qwen_ica`，或使用本目录对应的训练/评测脚本；普通 Qwen 加载器不会静默忽略 ICA。

| 组件 | 参数量 |
|---|---:|
| 原 Qwen SID SFT | 1,544,870,912 |
| 新增 ICA | 5,970,816 |
| Qwen＋ICA 合计 | 1,550,841,728 |

BF16 权重约 3.10 GB，另需少量配置与分词器文件。自动备份会先生成并完整校验压缩包，再检查网盘空间。
容量不足时保留实例上的完整模型与压缩包，并记录 `cloud_space_insufficient`；不会删除原网盘文件腾空间。
自动备份保存可推理、可继续新一轮训练的完整模型；精确断点续训所需的优化器状态仍在实例 `checkpoint-*` 中。

## 需要看的代码

| 文件 | 作用 |
|---|---|
| `scripts/qwen_ica.py` | 商品分组、历史边界、ICA、Qwen backbone 接入及完整模型保存重载 |
| `scripts/train_qwen_ica.py` | 真正的训练入口，已有 SFT 初始化、全参数更新、梯度审计与断点恢复 |
| `scripts/evaluate_qwen_ica.py` | 载入 Qwen＋ICA，在匹配的 SID 数据上生成 Top-20 并计算指标 |
| `scripts/run_qwen_ica_experiment.py` | 等当前 HPR 实验结束，执行 GPU 试跑、正式 ICA SFT、评测与备份 |
| `tests/test_qwen_ica.py` | 用真实微型 Qwen 验证答案隔离、生成、梯度、保存及恢复 |

## 运行示例

在项目根目录，使用已安装 torch/transformers 的 Python：

```bash
python scripts/train_qwen_ica.py \
  --checkpoint /path/to/original_sft/selected \
  --sft-record experiments/qwen_sft_office2018_20261013/model_artifact.json \
  --data data/processed/office2018/qwen_sft_adam \
  --output outputs/qwen_sft_ica
```

先检查环境与梯度可额外加 `--smoke-steps 3` 并使用单独的输出目录。
只核对分词器、样本、来源而不加载大模型：加 `--dry-run`。
断点恢复：同一命令加 `--resume outputs/qwen_sft_ica/checkpoint-N`。

评测：

```bash
python scripts/evaluate_qwen_ica.py --checkpoint outputs/qwen_sft_ica/selected \
  --data data/processed/office2018/qwen_sft_adam --split valid --users 5000 \
  --output outputs/ica_valid.json
python scripts/evaluate_qwen_ica.py --checkpoint outputs/qwen_sft_ica/selected \
  --data data/processed/office2018/qwen_sft_adam --split test \
  --selection-record outputs/ica_valid.json --output outputs/ica_test.json \
  --export-candidates outputs/ica_test_candidates.json
```

可另训一个 `--control` 版本：同样从原 SFT 继续训练，但关闭 ICA。
它有助于区分“ICA 的收益”和“额外继续训练的收益”。尚未运行的实验不填写提升数字。

## 与当前 HPR 的关系

先训练独立 **SFT＋ICA**，与原 SFT 及 SFT＋HPR 比较。
下一版可以在冻结的 SFT＋ICA 上重新提取上下文特征，重新训练 HPR，得到 **SFT＋ICA＋HPR**。
原 HPR 特征和头绑定旧 Qwen 的权重、配置及商品表示，**不能直接挂到更新后的 ICA 模型上**。
本版不改正在执行的 HPR 测试文件，也不覆盖原 SFT/HPR 的权重和评测结果。

当前状态以服务器任务记录和 `run.json` 为准。代码检查通过不代表完整模型已训练，也不代表已取得指标提升。
