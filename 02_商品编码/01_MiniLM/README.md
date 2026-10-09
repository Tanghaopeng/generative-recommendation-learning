# 冻结 MiniLM 商品向量 v1

源码：[build_item_embeddings.py](../../scripts/build_item_embeddings.py)。
预训练模型 `sentence-transformers/all-MiniLM-L6-v2`，固定 revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`。

`title + brand + category + description + feature` → 模型自带 WordPiece tokenizer → 最多128 token → 冻结 Transformer → 有效 token masked mean pooling → L2 归一化 → 384维向量。
padding 不参与平均；商品ID从1开始，第0行全零。完整输出 `data/processed/office2018/tiger_embeddings/embeddings.npy`，shape 为 `(25899,384)`。
已存在并通过 manifest 校验的全量结果可以直接复用，无须为了 Qwen SFT 重跑编码。

```bash
python scripts/build_item_embeddings.py --output data/processed/office2018/tiger_embeddings --batch-size 32 --max-tokens 128 --threads 2
```

在仓库根目录运行。`version.json` 是版本登记；实际 CLI 参数见脚本 `--help`，不是直接可执行的训练配置。
