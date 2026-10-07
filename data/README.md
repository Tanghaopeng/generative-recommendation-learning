# 数据目录

在项目根目录运行：

```bash
python scripts/download_data.py
python scripts/prepare_office.py
python scripts/verify_dataset.py
```

原始文件下载到 `data/raw/`，处理文件写到 `data/processed/office2018/`。
这两个目录，以及用户/物品映射与本机记录，均不提交 Git。
下载脚本检查 `sources.json` 中记录的字节数和 SHA-256；已有正确文件会直接复用。
该版本原始压缩数据共约 597MiB，解压处理需要额外磁盘空间。

处理结果包括：

- `office2018.txt`：SASRec 的两列有序交互输入。
- `sequences.jsonl`：共同的用户训练、验证、测试划分。
- `interactions.csv`：规范化交互表。
- `user_mapping.json`、`item_mapping.json`：ID 映射。
- `items.jsonl`：为后续 TIGER 准备的物品文本。
- `stats.json`、`verification.json`：本次数据处理统计与核验。
- `missing_metadata_asins.json`：缺少物品元数据的列表。

预处理会把 SASRec 输入复制到 `baselines/sasrec/data/office2018.txt`。
缺少的 28 个物品文本需要在 TIGER 阶段补齐或统一处理。
