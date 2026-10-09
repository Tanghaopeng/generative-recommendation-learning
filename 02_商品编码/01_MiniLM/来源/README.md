# 模型与分词器来源

- 官方：[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41)。发布组织为Sentence Transformers，基础模型为`nreimers/MiniLM-L6-H384-uncased`。
- 固定revision：`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，与已有商品编码实验一致。
- 整理日期：2026-10-09。复用原先官方Hub缓存，并补充下载官方原始模型卡。
- 模型和tokenizer未修改，也没有在本项目商品数据上微调。仅重新组织存放目录，保留原文件字节。
- 校验：[manifest.json](manifest.json)登记逐文件来源、大小、SHA256。权重核对官方LFS SHA256，普通文件核对官方Git blob SHA1。
- 模型卡声明Apache-2.0，见[保留的原始模型卡](上游模型卡.md)。附带[Apache-2.0标准文本](LICENSE-Apache-2.0.txt)，不声称官方仓库存在同名LICENSE文件。
- `sentence_transformers配置/`只保存上游modules、pooling和长度等参考配置，不是独立完整模型。实际使用时分别加载`模型权重/`与`分词器/`，显式池化和归一化。

中文讲解、版本登记和验证脚本为本项目新增内容，不声称训练了这份MiniLM权重或编写了原始分词器。
