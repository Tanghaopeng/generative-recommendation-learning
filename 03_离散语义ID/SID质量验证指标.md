# SID质量验证指标

本项目使用同一批25,898件商品、冻结MiniLM 384维向量，验证不同量化器产生的SID。统一协议为 `fixed_minilm_cosine_macro_sc_centroid_pd_v1`。指标用于检查语义聚类、码本占用和商品区分性，后续再用相同推荐评测协议检验下游收益。

[三种量化器实测对比](../experiments/sid_quality_comparison_20261016.md) · [评测代码](../scripts/sid_quality.py) · [碰撞分布](../experiments/sid_collision_comparison_20261016.json)

## 验证范围

- 排除数组第0行padding；验证向量、SID与商品目录的SHA256和商品数量。
- 语义SID为前三段，每层256个码字；第四段是重码组内消歧编号。
- SC/PD同时报告每层独立码字分组和累计一至三段前缀分组。主表为三个独立语义层的宏平均。
- 所有量化器都在相同冻结输入特征上评估，避免不同模型在自己改变后的空间中自评。
- 此处PD是内容向量的几何代理，不直接使用用户点击、验证或测试目标。

## 语义内聚度 SC

SC为Semantic Cohesion，不是轮廓系数。先将商品向量归一化，计算同簇不同商品的平均余弦相似度，再对簇宏平均。越高通常表示同码商品的内容更接近。

每簇以无序对i<j计算，范围为[-1,1]。实现用单位向量之和的平方减去商品数，再除以n(n−1)，等价于有序对均值，避免构造商品两两相似度矩阵。

单例簇没有商品对，SC未定义：排除并报告单例簇数和有效商品覆盖率；全部单例时输出null。不能将单例SC填为1。

## 偏好区分度 PD

PD为Preference Discrimination。先求每簇单位商品向量的均值，再归一化得到簇中心；对不同中心a<b计算：

`PD = log(mean(exp(-2 × (1 − cosine(center_a, center_b)))))`

在相同口径下，越低、越负通常表示簇中心更分离。不足两个非零中心时输出null；零中心排除并记录数量。

超过200,000个簇对时，以固定随机种子无放回抽样，并注明近似、总对数和评估对数。独立语义层最多256个活跃码字，主表各层PD均为精确计算。前缀分组可能使用抽样。

SC/PD借鉴[R³-VAE论文§3.4](https://arxiv.org/html/2604.11440v3#S3.SS4)。本地明确采用冻结MiniLM特征、宏平均和单例处理规则，不能将数值与论文表格直接比较。

## 码本利用率与死码

`利用率 = 完整商品目录中至少被选中过一次的码字数 / 256`

`死码数 = 256 − 被选中过的码字数`

每层分别报告。这里的死码是冻结检查点在当前目录中的零分配码字，不等于训练过程中从未更新。利用率高也不意味着分配均衡，因此还应查看频次分布、有效码字数和最大码字占比；原RQ-VAE审计保留了这些统计。

## 三段SID碰撞与四段唯一性

**碰撞余量比例**计每个同码组保留一件之后多出的商品：

`碰撞余量比例 = (商品数 − 不同三段SID数) / 商品数`

**参与碰撞占比**计所有至少两件商品同码的组包含的商品：

`参与碰撞占比 = 同码组规模≥2的商品总数 / 商品数`

例如5件商品的三段SID为A、A、A、B、C：余量比例为2/5=40%，参与碰撞占比为3/5=60%。报告另含不同三段SID数、碰撞组数、最大组规模和精确组规模直方图。

消歧时按商品ID递增，为同组三段SID追加0、1、2等第四段编号。完整四段SID必须与商品一一对应；第四段不纳入SC/PD，否则唯一SID会把每件商品都变成单例。

## 如何解释结果

SC改善可能伴随更粗的商品分组；PD改善也可能伴随更少的活跃簇。应同时检查利用率、死码、碰撞与单例覆盖，再用下游Recall/NDCG等验证。当前R³-VAE的SC/PD改善而三段碰撞增加，正体现了这种取舍。

SC/PD作为验证指标不会改变训练目标。将指标加入损失需要可微代理或明确松弛；把冻结向量与硬SID算出的NumPy浮点数直接加到loss，不会自动提供所需梯度。当前发布的R³-VAE配置用余弦重构训练，SC/PD正则权重为0。

## 复现验证

```powershell
& .venv/Scripts/python.exe scripts/sid_quality.py --sids data/processed/office2018/tiger_sids --output outputs/sid_quality_rqvae.json
& .venv/Scripts/python.exe scripts/sid_quality.py --sids data/processed/office2018/tiger_sids_ema_20261008 --output outputs/sid_quality_ema.json
& .venv/Scripts/python.exe scripts/sid_quality.py --sids outputs/r3vae_minilm/sids --output outputs/sid_quality_r3vae.json
```

使用新output保留历史结果。原始向量与SID数组需要先按项目流程生成；公开Git保留聚合指标和指纹。只从已发布证据重建对照表可运行：

```powershell
& .venv/Scripts/python.exe scripts/report_sid_comparison.py --output outputs/sid_comparison.json
```
