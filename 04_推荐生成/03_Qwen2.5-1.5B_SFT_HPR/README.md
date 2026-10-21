# 已有 Qwen SFT + HPR：实现与代码导读

本版在**已有梯度 SID 版 Qwen SFT**上增加 Hierarchical Path Reranking（层次路径重排序）。冻结 Qwen，只训练三组小型路径匹配双塔；生成每段 SID 时，在 beam 淘汰候选前加入匹配分。没有再次全参数 SFT，没有 ICA，也没有 RL/GRPO。

状态：代码及本地小模型检查已完成；**尚未在真实 1.5B SFT 权重上训练 HPR，没有新增正式推荐指标，也不声明已有提升**。现有 SFT、SID 和结果不覆盖。

## 1. 文件和阅读顺序

| 文件 | 作用 | 是否执行训练 |
|---|---|---|
| [prepare_qwen_hpr.py](../../scripts/prepare_qwen_hpr.py) | 加载已有 SFT，缓存历史提示最后位置的隐藏状态及冻结 SID token 向量 | 否，Qwen 前向推理 |
| [qwen_hpr.py](../../scripts/qwen_hpr.py) | HPR 双塔、三段累计向量、前缀负例、多正例对比损失、保存加载 | 定义被训练的模块和损失 |
| **[train_qwen_hpr.py](../../scripts/train_qwen_hpr.py)** | **创建优化器、计算损失、反向传播、更新 HPR、验证并保存** | **这是真正的训练入口** |
| [qwen_hpr_decode.py](../../scripts/qwen_hpr_decode.py) | 带 KV cache 的逐段 beam 搜索和 HPR 重评分 | 否，推理 |
| [evaluate_qwen_hpr.py](../../scripts/evaluate_qwen_hpr.py) | 验证/测试、Recall/NDCG、真实评测候选导出 | 否，评测 |
| [freeze_qwen_hpr_selection.py](../../scripts/freeze_qwen_hpr_selection.py) | 根据验证报告固定评分器和解码配置，供测试核对 | 否，选模记录 |
| [qwen_hpr_v1.json](../../experiments/qwen_hpr_v1.json) | HPR 的训练超参数 | 配置 |
| [test_qwen_hpr.py](../../tests/test_qwen_hpr.py) | 真正小型 Qwen 的前向、梯度、beam、保存和来源一致性检查 | 本地测试，不是正式实验 |

建议先看 `HPRScorer`，再看 `contrastive_loss`，然后看 `fit_heads` 中的优化器循环，最后看 `retrieve_hpr_group`。

## 2. 哪些参数训练，哪些复用

| 部分 | 参数与来源 | 是否更新 |
|---|---|---|
| MiniLM、RQ-VAE、四段 SID | 完全复用现有梯度版本 | 否 |
| Qwen2.5-1.5B-Instruct | 已完成推荐 SFT 的 1,544,870,912 参数 | 否 |
| HPR 用户塔 | 每个语义深度一个 `Linear(1536,64,bias=False)`，共三个 | 是 |
| HPR 路径塔 | 每个语义深度一个 `Linear(1536,64,bias=False)`，共三个 | 是 |
| HPR 温度 | 每个语义深度一个可学习缩放参数 | 是 |

HPR 共 **589,827 个参数**，FP32 权重约 2.25MiB；推理时需同时加载原 SFT 和 HPR。Qwen 按 BF16 提取特征和推理；小型双塔用 FP32 计算对比损失和路径分数。`train_qwen_hpr.py` 本身不加载 Qwen，所以也可以单独用 CPU 训练头。

## 3. 一条样本怎么变成训练数据

假设训练交互是 `[A,B,C,D,E]`，其中一个训练位置是：

```text
历史：A,B,C
目标：D
```

第一步，完全复用原 `prepare_request` 和 chat template，输入 Qwen 的仍是系统提示、用户任务、历史 A/B/C 四段 SID，以及结尾的 assistant 生成提示。**D 不拼进 Qwen 输入。**

第二步，读取最后一个提示 token 的最终层隐藏状态 `h0`，它是 1536 维用户上下文。右侧 padding 不参与表示，使用每条真实提示的最后位置。

第三步，设 D 的 SID 为 `[d1,d2,d3,d4]`，从**已有 Qwen 的输入 embedding 表**取得前三段向量：

```text
p1 = E(d1)
p2 = E(d1) + E(d2)
p3 = E(d1) + E(d2) + E(d3)
```

这里没有使用 RQ-VAE 的连续码本向量。`d4` 是消歧编号，不进入路径向量，也不新增第四个语义评分头。

特征缓存保存：`train_contexts.npy`、`train_targets.npy`、`valid_contexts.npy`、`valid_targets.npy`、`sid_embeddings.npy`、`catalog_sids.npy`、`train_positions.npy` 和 `manifest.json`。上下文以 FP32 缓存，原 BF16 特征可精确表示；全部训练上下文约 2.27GiB，加验证和 SID 表约 **2.31GiB**，实际依样本数变化。缓存不含测试目标。

训练位置来自原准备好的 `train_examples.npy`；验证特征来自原用户的验证目标。缓存可用于修改 HPR 的投影维度、学习率、负例数量等实验，避免反复运行 1.5B Qwen。改变 SFT 权重、SID 映射、历史截断或提示模板时，必须重建缓存。

## 4. HPR 怎么学习

在语义深度 l，将 `h0` 和 `pl` 分别投影、L2 归一化，计算余弦相似度，再乘可学习温度缩放：

```text
h0 → 用户投影塔 → 归一化 ─┐
                         ├→ 余弦相似度 × 温度缩放
pl → 路径投影塔 → 归一化 ─┘
```

每层采用双向 InfoNCE，最后对三层取平均。正例是当前样本真实目标的前缀。一个 batch 内其他真实前缀提供负例；另外从完整合法目录中优先采样“共享父前缀、下一码不同”的候选，不足时从其他合法前缀补充。全程排除当前真实前缀，负例内去重。

**共享前缀不是负例。** 如果 D、F 的第一段都是 54，则 batch 内这两列对于对应第一段监督都属于正例。代码以真实前缀相等建立多正例 mask，对正例概率求和，而非只认相似度矩阵的对角线。额外目录负例仅进入用户→路径方向的分母，因为这些额外路径没有对应的正例用户锚点。

真正参数更新在 `fit_heads`：

```python
optimizer = torch.optim.AdamW(scorer.parameters(), ...)
optimizer.zero_grad(set_to_none=True)
loss, diagnostics = contrastive_loss(scorer, contexts, targets, ...)
loss.backward()
torch.nn.utils.clip_grad_norm_(scorer.parameters(), ...)
optimizer.step()
```

优化器里只有 `scorer.parameters()`。不计算新的 Qwen token 交叉熵，也不更新 Qwen embedding、Transformer 或 RQ-VAE。

日志包含逐层 loss、正例总概率、不同前缀数量及梯度范数。`selected/`先按验证对比损失保留头，**这不是推荐收益结论**；还需实际检索验证决定是否使用 HPR 及其融合强度。

## 5. 解码具体改在哪里

原先的检索入口没有修改。新增 `retrieve_hpr_group`，一次 prefill 得到用户表示及 KV cache，之后逐 token 扩展，并在 beam 改变时重排 cache。

每段流程：

1. 通过原完整目录前缀树列出合法后续 token；最后一段排除完整历史中的商品。
2. 计算**完整词表归一化**的 token log-probability，加上该路径原始累计生成分。
3. 按累计生成分保留 Top-N 待评分池。
4. 对池中路径计算当前深度的 HPR 分数。
5. 用 `累计LM分 + strength × log_softmax(HPR分)` 选择继续保留的 beam。

原始累计 LM 分和融合分分开保存；不会将前几段 HPR 奖励重复加进后续累计 LM 分。第四段沿用前三段累计路径的第三组匹配头，加入四段完整生成概率，不把消歧 embedding 当作语义。最终四段仍按原唯一 SID 查表映射到商品。

这是**每一段淘汰前的路径重评分**，不是只给已生成的20件商品做事后排序。Top-N 池外路径无法被 HPR 挽救。`strength=0`直接调用原有 SFT 批量检索，是可核对的关闭开关。

## 6. 运行命令

以下在已配置 GPU 环境和已恢复梯度 SFT 权重的服务器上，从项目根目录运行。先把 `SFT` 改成实际的**已训练梯度版 selected 目录**，不能填写原始 Qwen 权重目录或 EMA 模型目录。数据沿用 `qwen_sft_adam`。

```bash
SFT=/root/generative-run-20261012/outputs/qwen_q1_from_q0_full/selected
DATA=data/processed/office2018/qwen_sft_adam
CACHE=outputs/qwen_hpr_features
RUN=outputs/qwen_hpr_v1
```

只做一次特征提取，不训练 Qwen。默认使用公开梯度 SFT 资产记录校验权重和 SID 数据的 SHA256，防止两者配错：

```bash
python scripts/prepare_qwen_hpr.py --checkpoint "$SFT" --data "$DATA" --output "$CACHE"
```

真正训练 HPR，复用以上缓存：

```bash
python scripts/train_qwen_hpr.py --cache "$CACHE" --output "$RUN"
```

验证关闭 HPR 的对照，以及两个候选融合强度：

```bash
python scripts/evaluate_qwen_hpr.py --checkpoint "$SFT" --hpr "$RUN/selected" --data "$DATA" --users 5000 --strength 0 --output "$RUN/valid_a0.json"
python scripts/evaluate_qwen_hpr.py --checkpoint "$SFT" --hpr "$RUN/selected" --data "$DATA" --users 5000 --strength 0.1 --output "$RUN/valid_a01.json"
python scripts/evaluate_qwen_hpr.py --checkpoint "$SFT" --hpr "$RUN/selected" --data "$DATA" --users 5000 --strength 0.25 --output "$RUN/valid_a025.json"
```

依据验证 NDCG@10 冻结选择；若关闭 HPR 胜出，保留原 SFT，不能强行发布提升：

```bash
python scripts/freeze_qwen_hpr_selection.py --reports "$RUN/valid_a0.json" "$RUN/valid_a01.json" "$RUN/valid_a025.json" --output "$RUN/selection.json"
```

测试使用选中报告中的 `strength` 和对应 HPR 目录。下面仅举选中 0.25 的例子；选中其他值必须替换，脚本会核对。可同时导出本次评测实际使用的前33条候选，含商品标题、四段 SID 和路径分，不会另跑一遍改变候选。

```bash
python scripts/evaluate_qwen_hpr.py --checkpoint "$SFT" --hpr "$RUN/selected" --data "$DATA" --split test --strength 0.25 --selection-record "$RUN/selection.json" --output "$RUN/test.json" --export-candidates "$RUN/candidates_33.json"
```

权重、数据、HPR 参数、代码、前缀树相关设置或融合参数与冻结记录不符时拒绝测试。只读验证报告的选择脚本不读取测试集。

## 7. 保存和恢复

`selected/`保存小型 `model.safetensors` 和 `config.json`，后者绑定原 SFT 和 SID 数据指纹。`last.pt`保存 HPR、AdamW 和已完成 epoch 的进度；`contract.json`约束来源和配置，`history.json`记录诊断。恢复：

```bash
python scripts/train_qwen_hpr.py --cache "$CACHE" --output "$RUN" --resume
```

当前在 epoch 结束时保存，意外中断的未完成 epoch 会重跑。恢复不需要原 SFT 优化器。部署仍必须保留原 SFT、对应 tokenizer/配置、SID 目录和 HPR；小评分器不能单独替代 Qwen。这里没有自动上传、租机或开机操作。

## 8. 与论文的对应和差异

参考 [BARGE](https://arxiv.org/abs/2607.21028v1) 的累计路径、分层双塔、对比训练和淘汰前重评分思想；全部代码为本项目适配实现，没有下载或直接调用作者源码。

论文联合训练主模型与 HPR；本版先冻结已有 Qwen、单独训练小头，以较低成本验证路径评分是否有用。论文用户锚点来自 Encoder–Decoder 的初始 Decoder 状态；本版用 Qwen 最后提示状态。论文四层均为语义，本版只有前三层参与语义对比。论文还讨论生成分布高概率错误前缀和曝光未点击负例，本版采用批内多正例与合法目录结构负例，**尚未实现模型挖掘高概率负例，也不把未观察商品认定为用户真实不喜欢的商品**。

本地检查覆盖：共享前缀正例、合法负例、提示末状态与padding、冻结 Qwen 的真实小模型训练、非零且有限梯度、保存加载/恢复、HPR在淘汰前挽救路径、KV cache搜索与全路径穷举对照、零强度复用原检索、真实批次候选导出，以及验证/测试来源隔离。正式收益仍需服务器实验确认。
