# Qwen3-Embedding-4B 商品编码版本

本版本准备独立于 MiniLM 的第二套商品内容编码。先下载官方预训练模型并冻结使用，后续再单独比较领域微调。下载模型不代表已经生成商品向量或完成推荐训练。

## 来源与文件

- 官方模型：[Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B)。
- 固定 revision：`5cf2132abc99cad020ac570b19d031efec650f2b`。
- 官方代码：[QwenLM/Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding)。
- 模型卡声明 Apache-2.0；原始模型卡保留于 `模型文件/README.md`。
- `模型文件/` 保存标准 Hugging Face 目录结构：两个 BF16 safetensors 权重分片、权重索引、模型配置、分词器及 Sentence Transformers 池化配置。
- 权重合计 8,043,592,088 字节，约 8.04 GB / 7.49 GiB；分词器等文件另计。
- 小文件来自官方 Hugging Face 固定版本，权重可来自 Qwen 的 ModelScope 镜像；所有文件必须匹配固定 HF revision 的 Git blob 或 LFS SHA256。
- 下载完成后，`来源/manifest.json` 记录每个文件的大小与 SHA256；`来源/离线验证.json` 记录离线文件结构、权重头和分词器验证。

```powershell
.venv/Scripts/python.exe scripts/download_qwen3_embedding.py --size 4B --parallel
.venv/Scripts/python.exe scripts/verify_qwen3_embedding_assets.py
```

权重按项目规则保留在本地与矩阵云网盘；GitHub 保存说明、下载与验证脚本、来源指纹。不要把两个 GB 级权重加入普通 Git。

2026-10-09 本地下载完成，13 个官方文件均通过来源指纹验证。离线验证覆盖文件哈希、BF16 权重头及索引、参数数量、分词器加载；不包含完整模型前向或商品编码。

2026-10-10 已确认矩池云网盘 `1区 /Qwen3-Embedding-4B` 的两个权重分片均显示绿色上传成功标记，分词器、配置与 `1_Pooling/config.json` 也已上传。同区实例挂载后路径为 `/mnt/Qwen3-Embedding-4B`。

第一分片曾网络异常，重新提交后复用已有分块并上传成功。因此网页历史队列保留一条失败记录，最终同名续传任务为成功状态；不能只看历史队列的总完成比例。详细核验边界见 `来源/云盘上传完成.json`。根目录中的 `HTTP导入测试_已取消_不可用` 是已取消的重复导入测试，不属于模型资产，不要加载其中的文件。

## 模型结构与编码方式

这是在 Qwen3-4B-Base 基础上训练的文本 embedding 模型。骨干是带因果注意力的 Transformer，使用其最后一个有效 token 的隐藏状态作为文本向量，再做 L2 归一化；不同于 MiniLM 的 masked mean pooling。

配置为 36 层、隐藏维度 2560、32 个 query 注意力头、8 个 KV 头、每头 128 维、SwiGLU 中间维度 9728、RMSNorm、RoPE。GQA 的 query 投影总维度不必等于隐藏维度，因此不要用 `2560 / 32` 推断每头维度。词表配置为 151665。

官方模型卡标称最大上下文 32K；配置中的位置上限 40960 不应被当成已经验证的更长使用长度。商品编码起步可限定 128 或 256 token，正式版本需固定并记录，长文本成本随长度增加。

默认输出 2560 维，支持 MRL 输出截断；这与模型有 4B 参数、RQ-VAE 隐空间为 32 维是三个独立概念。第一版建议先使用完整 2560 维输出，再由单独训练的 RQ-VAE 映射到 32 维。不能直接使用 MiniLM 384 维版本的量化器权重。

```text
商品文本 → 冻结 Qwen3-Embedding-4B → 2560 维向量
         → 新训练的 RQ-VAE → 三段语义码 + 消歧码
         → 独立训练的 Qwen2.5-1.5B-Instruct 推荐模型
```

商品作为文档编码时，官方示例不加查询 instruction；若后续编码搜索查询，可使用查询任务 instruction。不得无记录地更改文本字段、截断、池化或输出维度。

## 离线加载示例

```python
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

path = Path('02_商品编码/02_Qwen3_Embedding_4B/模型文件')
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, padding_side='left')
model = AutoModel.from_pretrained(
    path, local_files_only=True, torch_dtype=torch.bfloat16,
    attn_implementation='sdpa',
).to('cuda').eval().requires_grad_(False)
batch = tokenizer(
    ['title: A4 printer paper. category: Office Products.'],
    padding=True, truncation=True, max_length=128, return_tensors='pt',
).to('cuda')
with torch.inference_mode():
    hidden = model(**batch, use_cache=False).last_hidden_state
    # 左补齐时最后一列就是每个样本的最后有效 token。
    vectors = F.normalize(hidden[:, -1].float(), p=2, dim=-1)
assert vectors.shape == (1, 2560)
```

这个示例是供后续 GPU 使用的接口说明；本次本地下载验证不宣称运行了 4B 全模型前向，也不宣称生成了全商品向量。

## 后续优化

先比较冻结 MiniLM 和冻结 Qwen3-Embedding-4B 的完整推荐流程；之后可用训练期交互构造正例与负例，对 embedding 模型做对比学习微调，优先考虑 LoRA。embedding 微调通常优化向量之间的关系，不能直接照搬后续 SID 生成器的下一 token 交叉熵。

改变商品编码模型后，必须重新生成商品向量、训练量化器和生成 SID，再训练对应的推荐模型；保存旧版结果用于对照。
