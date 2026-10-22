# 原 SFT 基础上的 ICA＋HPR 联合训练

本版按“**已有 SFT → 同时加入 ICA 与 HPR → 一起继续训练**”实现。
不是先把 ICA 训练完再冻结主干、单独拟合 HPR。
现有 SFT＋ICA、冻结 SFT＋HPR 实验保留为独立对照；这里尚无完整实验指标，不填提升数字。

## 联合训练与分阶段训练的区别

| 方案 | Qwen | ICA | HPR | 训练目标 |
|---|---|---|---|---|
| 已有 SFT＋ICA 实验 | 更新 | 更新 | 无 | 下一商品交叉熵 |
| 已有冻结 SFT＋HPR 实验 | 冻结 | 无 | 更新 | HPR 对比损失 |
| 先 ICA、再冻结训练 HPR | 第二阶段冻结 | 第二阶段冻结 | 更新 | 第二阶段只有 HPR 对比损失 |
| **本版 ICA＋HPR 联合训练** | **更新** | **更新** | **更新** | **交叉熵＋HPR 对比损失** |

前向计算仍然依次经过模块。这是网络的计算顺序，与是否分阶段更新参数是两回事。

```text
原 SFT 权重 + 新 ICA + 新 HPR
                  ↓
同一批用户历史 SID → token embedding → ICA → Qwen（一次主干前向）
                                               ├→ 答案位置的 logits → L_CE
                                               └→ 最后一个提示位置的 h0
目标商品前三段 SID → 当前 Qwen embedding 累加路径 ───────→ HPR → L_HPR
                                                            ↓
                                             L = L_CE + λ × L_HPR
                                                            ↓
                                             一次 backward / optimizer.step
                                                            ↓
                                               同时更新 Qwen、ICA、HPR
```

**不会缓存冻结的 h0，也不会把路径 embedding 从计算图中 detach。**
因此 HPR 梯度能够通过 h0 更新 Qwen、ICA，也能通过路径表示更新 SID token embedding。
MiniLM、RQ-VAE、商品 SID 映射与 tokenizer 均保持固定；这是监督与对比目标的联合优化，不是强化学习。

## 两个损失各自做什么

**L_CE：** 只监督下一商品的四个 SID token 与 EOS。提示位置 `labels=-100`。
它让自回归生成器继续学习正确商品的输出概率。

**L_HPR：** 分别比较一段、两段、三段语义前缀。
上下文 h0 是 assistant 回答开始前的最后一个提示位置的隐藏状态；正确路径是目标商品的累计 token embedding。
HPR 头将两者映射到 64 维并计算相似度，对正确路径拉近、负路径推远。
同前缀样本作为多个正例，不互相当作负例；额外负例来自目录中同父前缀的其他路径，不够时从全目录补齐。
第四段只是消歧编号，延续现有 HPR 的前三段语义建模方式。

当前默认 `λ=0.05` 是本地实验起始配置，尚未证明最优，也不声称是论文原始参数。
Qwen＋ICA 默认学习率 `1e-5`；新 HPR 小头使用 `1e-3`。
HPR 保持 FP32 参数和对比计算，Qwen＋ICA 使用 BF16。

HPR 的批内对比范围是当前 micro-batch；梯度累积不会把分开的批次自动合并成一个大对比批。
代码另外采样目录负例，以补充批内候选。
CE 与 HPR 都按 micro-batch 均值计算，合并后由 Trainer 统一做梯度累积归一化，避免仅对 CE 使用全局 token 数导致 λ 被放大。

## 如何避免答案泄露

一个训练样本仍为“提示＋历史 A/B/C＋assistant 分隔符＋D 的四段 SID＋EOS”。

- ICA 只聚合 assistant 分隔符之前的完整历史商品，D 不参与商品内聚合。
- HPR 从 `first_supervised_position - 1` 取 h0，并校验标签开始位置紧接真实 assistant 分隔符。
- Qwen 因果注意力使 h0 看不到后面的答案 D。
- D 的 SID 仅作为监督目标与对比正例使用，不能加入历史。

同一个主干前向可以同时提供 h0 与答案 logits，不必重复跑一次 Qwen，也不必保留全部 28 层隐藏状态。

## 核心代码

| 文件 | 作用 |
|---|---|
| `scripts/qwen_ica_hpr_joint.py` | 一体化模型；同一前向同时计算 CE 与 HPR；保留所有梯度路径 |
| `scripts/train_qwen_ica_hpr_joint.py` | 真正训练入口；原 SFT 初始化、分组学习率、梯度审计、断点恢复 |
| `scripts/evaluate_qwen_ica_hpr_joint.py` | 加载一体化权重，使用其内置 HPR 头进行受约束生成与指标计算 |
| `experiments/qwen_ica_hpr_joint.json` | 联合训练与验证配置 |
| `tests/test_qwen_ica_hpr_joint.py` | HPR 单独反传到 ICA/Qwen 的验证、答案隔离、混合精度保存与精确恢复 |

这里是参考 [BARGE](https://arxiv.org/abs/2607.21028v1) 结构思路的本地 Qwen 适配，保留已有低维 ICA、三层 HPR 和前缀树解码方式，不是作者官方代码的原样复现。

## 训练命令

初始化使用原始已训练 SFT 的 `selected/`。为清晰比较四组方法，不接着单独 ICA 的权重初始化联合版。

```bash
python scripts/train_qwen_ica_hpr_joint.py \
  --checkpoint /path/to/original_sft/selected \
  --sft-record experiments/qwen_sft_office2018_20261013/model_artifact.json \
  --data data/processed/office2018/qwen_sft_adam \
  --output outputs/qwen_ica_hpr_joint
```

先核对来源、分词器和样本：加 `--dry-run`，不加载大模型、不训练。
GPU 试跑：使用单独输出目录并加 `--smoke-steps 3`。
精确恢复：保持配置不变，加 `--resume outputs/qwen_ica_hpr_joint/checkpoint-N`。

`selected/` 一份模型包同时保存 Qwen、ICA、HPR 参数及 tokenizer；不要拆接旧实验的 HPR 头。
`checkpoint-*` 另含 optimizer/scheduler/RNG，适用于断点恢复。
先导入 `qwen_ica_hpr_joint` 或使用对应脚本；其模型类型为 `qwen2_ica_hpr_joint`。

## 评测命令

训练中固定解码融合权重 0.25 监测验证 NDCG。选定模型后，只在验证集比较 0、0.1、0.25 等解码权重，再冻结选择。
训练损失系数 λ 与解码融合权重 α 是两个不同参数，不能混为一谈。

```bash
python scripts/evaluate_qwen_ica_hpr_joint.py \
  --checkpoint outputs/qwen_ica_hpr_joint/selected \
  --data data/processed/office2018/qwen_sft_adam \
  --split valid --users 5000 --strength 0.25 --output outputs/joint_valid_a025.json

python scripts/freeze_qwen_hpr_selection.py \
  --reports outputs/joint_valid_a025.json --output outputs/joint_selection.json

python scripts/evaluate_qwen_ica_hpr_joint.py \
  --checkpoint outputs/qwen_ica_hpr_joint/selected \
  --data data/processed/office2018/qwen_sft_adam \
  --split test --strength 0.25 --selection-record outputs/joint_selection.json \
  --output outputs/joint_test.json --export-candidates outputs/joint_candidates.json
```

若比较多个 α，应把对应的全部验证报告传给 `freeze_qwen_hpr_selection.py --reports`。
测试传入冻结记录中的 α，不能看测试结果后挑选。
最终比较原 SFT、SFT＋ICA、冻结 SFT＋HPR、本版联合模型；效果以真实测试结果为准。
