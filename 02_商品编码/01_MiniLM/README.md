# 冻结 MiniLM 商品向量 v1

本目录保存**实际预训练权重、配套分词器和来源说明**。直接调用发布好的模型，不在本项目商品数据上训练MiniLM。
商品文本参与forward计算向量，MiniLM参数始终冻结；后面的RQ-VAE和推荐生成器才需要本项目训练。

## 文件导航

```text
01_MiniLM/
├── 模型权重/
│   ├── model.safetensors     实际权重，90,868,376字节
│   └── config.json           网络结构配置
├── 分词器/
│   ├── tokenizer.json        fast tokenizer的规则和词表
│   ├── vocab.txt             30,522个WordPiece词条
│   ├── tokenizer_config.json 小写化等分词设置
│   └── special_tokens_map.json 特殊token映射
├── 来源/
│   ├── README.md             官方来源、版本、许可说明
│   ├── 上游模型卡.md          官方原始模型卡
│   ├── manifest.json         逐文件字节数、来源与SHA256
│   ├── 离线验证.json          参数量及与原缓存的向量一致性
│   ├── LICENSE-Apache-2.0.txt 标准许可文本
│   └── sentence_transformers配置/ 上游pooling等结构参考
├── 模型架构与编码流程.md      中文原理说明
└── version.json             本项目版本登记
```

权重约90.87 MB（86.66 MiB），采用普通Git保存，是真实文件而非下载占位符或Git LFS指针。克隆仓库即可获得，不需要Git LFS。
仅这一份指定MiniLM权重解除忽略，其他大模型权重和用户数据不入Git。

## 来源与架构

发布模型：[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41)。固定revision：`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，与此前商品编码实验相同。
基础为`nreimers/MiniLM-L6-H384-uncased`；发布方进一步用句子对做对比学习，让文本向量适合语义相似度计算。

| 项目 | 当前模型 |
|---|---|
| 结构 | BertModel，双向Transformer encoder |
| Transformer层数 | 6 |
| 隐藏维度／商品向量维度 | 384 |
| 每层注意力头数 | 12，每头32维 |
| FFN中间维度／激活 | 1536／GELU |
| 参数量 | 22,713,216，约22.7M（0.023B） |
| 分词器 | uncased WordPiece，词表30,522 |
| 位置编码上限 | 512，可学习绝对位置embedding |
| 本项目实际长度 | 最多128token，含特殊token |
| 汇总方式 | attention-mask mean pooling＋L2归一化 |

Sentence-Transformers封装默认截断256token，网络位置上限512，本项目显式截断128，三者不是同一个设置。
详细解释见[模型架构与编码流程](模型架构与编码流程.md)，许可证和官方文件校验见[来源说明](来源/README.md)。

## 离线加载：权重与tokenizer分别读取

在仓库根目录运行以下示例。两个目录分别交给AutoModel和AutoTokenizer，不能把整个`01_MiniLM`目录直接交给SentenceTransformer。

```python
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

base = Path("02_商品编码/01_MiniLM")
torch.set_num_threads(2)
tokenizer = AutoTokenizer.from_pretrained(base / "分词器", local_files_only=True)
model = AutoModel.from_pretrained(base / "模型权重", local_files_only=True)
model.eval().requires_grad_(False)
texts = ["title: Blue ballpoint pen. category: Office Products"]
batch = tokenizer(texts, padding=True, truncation=True, max_length=128,
                  return_tensors="pt")
with torch.inference_mode():
    hidden = model(**batch).last_hidden_state  # [B,L,384]
    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
    vectors = F.normalize(pooled, p=2, dim=1)   # [B,384]
print(vectors.shape)
```

`local_files_only=True`只读取本次发布的本地文件，不需要再次下载或训练。
核验命令：`python scripts/verify_minilm_assets.py --output outputs/minilm_verify.json`。

## 已有商品向量实验

[build_item_embeddings.py](../../scripts/build_item_embeddings.py)将`title + brand + category + description + feature`拼接后按上述流程编码。
第0行全零代表padding；25,898件真实商品的完整结果为`data/processed/office2018/tiger_embeddings/embeddings.npy`，shape为`(25899,384)`。
padding不参与平均；与官方示例一致，`[CLS]`和`[SEP]`属于有效位置，也参与平均。
已有完整向量可以直接复用。为保持旧实验源码指纹，原批量编码脚本仍使用原缓存目录`data/models/all-MiniLM-L6-v2`，包含固定版本下载/缓存逻辑；本次不改动该入口。

```bash
python scripts/build_item_embeddings.py --output data/processed/office2018/tiger_embeddings --batch-size 32 --max-tokens 128 --threads 2
```

完全离线使用本次分目录文件时，采用前面的加载示例。`version.json`仅为版本登记，不是可直接执行的CLI配置。
