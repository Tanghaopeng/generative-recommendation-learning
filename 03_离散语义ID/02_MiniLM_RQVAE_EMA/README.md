# MiniLM → RQ-VAE（EMA）

源码：[量化训练](../../scripts/train_rqvae_ema.py)、[生成唯一SID](../../scripts/build_semantic_ids.py)、[码本审计](../../scripts/audit_codebooks.py)。
已完成30轮CPU量化实验；量化器由验证重建损失选取，不能据此判断推荐一定更好。
输入为相同冻结 MiniLM 向量，输出 `data/processed/office2018/tiger_sids_ema_20261008/semantic_ids.npy`，第0行是padding，完整商品SID长度4。
使用已有SID时，直接把这个目录传给 `prepare_qwen_sft.py --sids`。
需要重新量化时先执行训练入口 `--help`，使用新的 `--output` 保存每次实验，再由 `build_semantic_ids.py --checkpoint ... --output ...` 导出。
