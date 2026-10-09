# 小型 TIGER 式召回

源码：[run_tiger.py](../../scripts/run_tiger.py)、[tiger_model.py](../../scripts/tiger_model.py)。
采用保留的第三方 T5 模型，经本地四段SID适配。两套 SID 分别从头训练生成器，同一共享划分和 beam20。
完整配置和结果：[Transformer 对照](../../experiments/semantic_retrievers_comparison_20261008.md)。
