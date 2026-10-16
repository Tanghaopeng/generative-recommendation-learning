# MiniLM → RQ-VAE（Adam）

源码：[量化训练](../../scripts/train_rqvae.py)、[生成唯一SID](../../scripts/build_semantic_ids.py)、[码本审计](../../scripts/audit_codebooks.py)。
已完成30轮CPU量化实验；量化器由验证重建损失选取，不能据此判断推荐一定更好。
初始化已使用逐层残差 MiniBatchKMeans：从量化器训练子集抽取最多4096个商品，先经初始MLP编码到32维，再逐层聚类并减去选中中心。配置里的 `codebook_kmeans_init=False` 关闭的是上游首次前向的自动初始化，本项目在训练前显式调用 `initialize_codebooks`，不代表未用K-means。初始化后使用Adam更新码本。
输入为相同冻结 MiniLM 向量，输出 `data/processed/office2018/tiger_sids/semantic_ids.npy`，第0行是padding，完整商品SID长度4。
使用已有SID时，直接把这个目录传给 `prepare_qwen_sft.py --sids`。
需要重新量化时先执行训练入口 `--help`，使用新的 `--output` 保存每次实验，再由 `build_semantic_ids.py --checkpoint ... --output ...` 导出。
