# MiniLM → RQ-KMeans

2026-10-16完成直接量化384维MiniLM向量的全量商品实验，作为与梯度RQ-VAE、EMA和R³-VAE并列的第四种SID方案。

每层拟合256个中心：`r1=x → 选c1 → r2=x-c1 → 选c2 → r3=x-c1-c2 → 选c3`。
前三段为三个中心的索引，第四段仍为同码组内按商品ID顺序追加的消歧编号。各层在所有上层簇的残差上共享一个码本，不是为每个父簇分别建256个中心。

MiniLM冻结，没有可学习MLP、重构梯度、承诺损失或EMA训练；MiniBatchKMeans直接拟合残差中心。3×256×384码本共有294,912个标量；它与使用32维潜空间的其他方案在维度、表示来源上不同，不是单变量消融。

| SC↑ | PD↓ | 活跃码字 | 死码 | 有效码字数 | 三段碰撞余量↓ | 最大同码组 |
|---:|---:|---|---|---|---:|---:|
| 0.667177 | -1.132093 | 256/256/254 | 0/0/2 | 190.38/105.03/3.61 | 52.92% | 46 |

25,898件商品，24,603件训练商品拟合，1,295件目录验证商品只诊断；seed2026，与其他方案同一划分。SC/PD在原始384维MiniLM空间计算。

参数：k-means++，n_init=3，max_iter=100，max_no_improvement=20，batch1024，reassignment_ratio=0.01，2个CPU线程。残差使用欧氏距离，不逐层重新归一化。MiniBatchKMeans的n_init用于选择初始化，不是三次完整聚类训练；提前停止后实际迭代少于上限。

三层实际n_iter为6/11/2；聚类拟合约33.95秒，包含评测等总计约41.91秒，另有Python启动/导入时间。单随机种子，尚未训练这套SID的下游推荐模型。

**不能只看活跃数：** 第三层65.29%的商品使用同一码字，有效码字数仅3.61，即使254个码字有过命中，也仍存在严重集中。有效码字数即码本困惑度`exp(-Σp ln p)`，来源于[DeepMind Sonnet官方VQ-VAE](https://github.com/google-deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py)，本项目按全商品目录逐层计算。

聚合指标与原始运行记录见[run.json](../../experiments/rqkmeans_minilm384_20261016/run.json)、[SID清单](../../experiments/rqkmeans_minilm384_20261016/sid_manifest.json)。目录中不含商品SID映射、向量或码本数组；这些二进制工件保存在本地outputs。

## 运行

先按项目文档准备冻结MiniLM全目录向量，再运行：

```powershell
python scripts/train_rqkmeans.py --output outputs/rqkmeans_minilm384_new --threads 2
```

输出包含`codebooks.npz`、`sids/semantic_ids.npy`、SID清单指纹、`sid_quality.json`及含训练/目录验证/全目录频次的`run.json`。不覆盖旧SID；新SID用于Qwen时需要重新构建训练样本并训练生成器。

代码中的冻结编码器入口保留为辅助分析，正式四方案对照不纳入该条件实验。

- [四方案完整对比](../../experiments/sid_usage_comparison_20261016.md)
- [完整聚合频次与困惑度](../../experiments/sid_usage_comparison_20261016.json)
- [实现入口](../../scripts/train_rqkmeans.py)
- [指标定义](../SID质量验证指标.md)
- [MiniBatchKMeans官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)
