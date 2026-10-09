# MiniOneRec 单卡实现参考

来源：[wbn11/minionerec-single-gpu](https://github.com/wbn11/minionerec-single-gpu/tree/bccd7ef70291c5b4b0c30b6222bc491039b4cd18)，固定提交见 `upstream_source.json`。
本目录保存未修改的 SFT 训练、数据集、词表处理源码及上游提供的 Apache-2.0 许可证；这些是阅读参考，不能直接当成本项目运行入口。

本项目的 `scripts/qwen_sft.py`、`train_qwen_sft.py` 等为独立适配实现，参考其“全参数 BF16＋SID 词表扩展＋仅回答监督＋可选文本对齐”流程。
差异：上游三段 SID、本地四段唯一 SID；上游 CSV/其他类目、本地 Office Products 共同划分；上游按验证 loss 选模、本地按验证商品 NDCG@10；本地使用 Qwen chat template，并提供冻结前缀树检索和恢复契约。
不是官方 OneRec 实现，也不是严格论文复现。原始 MiniOneRec 项目：[AkaliKong/MiniOneRec](https://github.com/AkaliKong/MiniOneRec)。
