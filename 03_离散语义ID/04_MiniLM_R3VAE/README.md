# MiniLM R³ VAE

在冻结MiniLM商品向量上，加入可训练参考方向与rating残差量化，生成三段语义码和一段消歧码。当前适配已完成30轮CPU量化训练，模型294,944参数。

[SID结果对比](../../experiments/sid_quality_comparison_20261016.md) · [验证指标说明](../SID质量验证指标.md) · [训练配置](../../experiments/r3vae_minilm.json) · [完整训练证据](../../experiments/r3vae_minilm_20261016/training_result.json)

## 计算流程

```text
冻结MiniLM 384维商品向量
→ 可训练MLP 384→256→128→32
→ 参考向量投影与初始残差
→ 三层余弦评分量化 每层256个32维码字
→ 累加参考投影和三层贡献
→ 解码器32→128→256→384
→ 余弦重构损失
```

对归一化编码表示z和单位参考方向r，先计算 `anchor=(z·r)r`，初始残差为 `z−anchor`。每层将残差和码字归一化，选余弦评分s最高的单位码字c，用 `s×c` 作为贡献，从归一化残差中减去它后进入下一层。三次选中的码字编号组成语义SID。

当前训练目标为 `1−cosine(重构向量,原始MiniLM向量)`，SC/PD正则权重为0。没有传统VQ码本损失、承诺损失或EMA；MLP、参考方向、被选中码字及评分通过重构梯度更新。top1索引本身不可微，也没有使用STE。

初始化采用最多4096个量化训练商品的逐层残差MiniBatchKMeans。三段重码按商品ID递增追加第四段计数，完整SID唯一。参考投影系数及rating评分没有写入SID；下游推荐器消费编号，不直接使用这些连续量或原码本向量。

## 与原RQ VAE的区别

| 环节 | 原梯度RQ-VAE | R³-VAE适配 |
|---|---|---|
| 初始残差 | MLP编码表示 | 编码表示减去参考投影 |
| 选码方式 | 欧氏距离最近 | 余弦评分最高 |
| 每层贡献 | 选中的码字向量 | 评分乘单位码字向量 |
| 主要梯度路径 | STE和码本承诺损失 | 可微评分与选中码字的重构梯度 |
| 重构目标 | 归一化向量平方误差 | 余弦重构 |

两者都使用冻结MiniLM输入、32维MLP、三层256码字和K-means初始化，但优化目标与量化运算不同，不把结果解释为只改变一个组件的消融。

## 训练与结果

| 设置 | 当前记录 |
|---|---|
| 商品数 | 25,898 |
| 训练与目录向量验证 | 24,603 / 1,295，固定seed 2026 |
| 模型参数 | 294,944 |
| 训练预算与选中轮次 | 30轮，第30轮 |
| 优化器 | AdamW，学习率5e-4，weight decay 1e-5 |
| batch size与精度 | 512，FP32 CPU，2线程 |
| 实测耗时 | 约191.5秒，含初始化、训练、导出和质量统计 |
| 选模依据 | 5%目录向量验证集的余弦重构损失 |

SC层宏平均为0.445852，PD层宏平均为−0.809742。三层使用码字120、240、208个，利用率46.88%、93.75%、81.25%，死码136、16、48个。

共有14,409种三段SID，3,713个碰撞组，最大组97件；三段碰撞余量44.36%，参与碰撞商品15,202件，占58.70%。第四段消歧后25,898个完整SID均唯一。

相较原RQ-VAE，SC/PD和首层利用率改善，但碰撞更多。当前没有重新训练此SID对应的TIGER/Qwen推荐器，因此没有新的Recall/NDCG结论，也没有替换原SFT的SID映射。

## 运行

从项目根目录执行，使用新的输出目录：

```powershell
& .venv/Scripts/python.exe scripts/train_r3vae.py --config experiments/r3vae_minilm.json --output outputs/r3vae_minilm --threads 2
```

代码：[模型](../../scripts/r3vae.py)、[训练与导出](../../scripts/train_r3vae.py)、[SID质量验证](../../scripts/sid_quality.py)。每次保存best/last检查点、AdamW与随机状态、run.json、sids目录和sid_quality.json。向量、权重和SID数组保存在Git忽略目录。

新推荐样本准备：

```powershell
& .venv/Scripts/python.exe scripts/prepare_qwen_sft.py --sids outputs/r3vae_minilm/sids --output data/processed/office2018/qwen_sft_r3vae
```

切换SID需要独立准备样本并重新训练推荐模型。恢复量化器可用 `--resume PATH/last.pth` 并指定新的output，保持源码、数据和配置一致，仅增加epochs预算。

## 来源与适配边界

- [R³-VAE论文v3](https://arxiv.org/html/2604.11440v3)。
- [作者仓库](https://github.com/wwqq/R3-VAE)，固定提交 `87f1c0c21cecd16a1c44870dabe29d970475b5e1`；[源码指纹](source.json)。
- 论文描述softmax加权和K-means初始化；固定官方代码实际使用随机初始化和top1余弦评分加权。本版参考其top1运算，并采用本项目的32维MLP和残差K-means初始化。
- 公开代码的SC/PD训练代理与本地冻结向量评测定义不同；当前配置以重构为训练目标。
- `paper_softmax`是另一个量化运算开关，当前公开结果采用 `official_top1`，不混用两者的结果。
- 量化核心已与固定官方版本核对前向和输入梯度。本版为独立编写的项目适配，不声称严格复现论文架构或收益。
