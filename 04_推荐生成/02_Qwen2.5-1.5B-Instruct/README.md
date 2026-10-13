# Qwen2.5-1.5B-Instruct 全参数 SFT v1

流程：已有商品文本 → 冻结MiniLM向量 → 已有Adam/EMA四段SID → Qwen推荐样本 → 全参数SFT → 商品级检索验证 → 固定选模后测试。
Qwen 在这一版负责生成 SID，不负责重新生成商品内容向量。

## 先保存 Q0，再从同一检查点训练 Q1

2026-10-12 增加 [export_qwen_q0.py](../../scripts/export_qwen_q0.py) 和 [qwen_initialization.py](../../scripts/qwen_initialization.py)。
Q0 在 CPU 上加载固定官方 BF16 权重，固定 seed=2026，增加 4×256 个普通 SID token，初始化其输入/输出向量后保存完整检查点；没有反向传播或优化器更新。
Qwen 原权重中有 tokenizer 尚未使用的词表行，SID 会占用其中一部分。因此初始化覆盖全部 1024 个 SID 对应行，而非仅覆盖 resize 新增的行。
原 tokenizer 可表达的 token 向量及 Transformer 层参数不变；若输入/输出权重绑定，仍保持绑定。

以下命令在项目根目录和已配置的 Python 环境内执行；模型和 Q0 完整检查点放在实例硬盘，避免把 5GB 网盘塞满：

```bash
python scripts/export_qwen_q0.py --model /root/generative-run-20261012/assets/Qwen2.5-1.5B-Instruct --data data/processed/office2018/qwen_sft_adam --seed 2026 --output /root/generative-run-20261012/outputs/qwen_q0_seed2026
python scripts/evaluate_qwen_sft.py --checkpoint /root/generative-run-20261012/outputs/qwen_q0_seed2026 --data data/processed/office2018/qwen_sft_adam --split valid --users 64 --seed 2027 --output outputs/q0_valid_u64_seed2027.json
```

64 人结果仅为通路与耗时诊断，不能当成正式效果；正式比较固定同一用户集合、SID、提示模板和解码配置。Q0 新 SID 未经过商品语义对齐，不代表已对齐的大模型零样本推荐能力。
完整验证、冻结选择和测试沿用下方入口；评测报告中标记 `model_stage=q0_untrained_sid`。

2026-10-12 解码修正为 `sid_beam_v2`：独立 GenerationConfig，关闭模型默认值回填，不采样、重复惩罚 1.0；避免继承原 Instruct 的文本生成惩罚，保证 SID 分数口径一致。原 Q0 权重可直接复用，但旧解码配置的指标须保留为旧协议，统一重评后比较。
Qwen tokenizer 明确保持原正则，规避 Transformers 4.57.6 的 Mistral 误判；不要给 Qwen 盲目设置 `fix_mistral_regex=True`。检索报告增加耗时、吞吐和峰值显存。

Q1 显式传入已保存的 Q0，直接加载其权重与 tokenizer，不重新随机初始化 SID：

```bash
python scripts/train_qwen_sft.py --config 04_推荐生成/02_Qwen2.5-1.5B-Instruct/configs/q1_pilot.json --data data/processed/office2018/qwen_sft_adam --model /root/generative-run-20261012/assets/Qwen2.5-1.5B-Instruct --initialization /root/generative-run-20261012/outputs/qwen_q0_seed2026 --output outputs/qwen_q1_from_q0_pilot --dry-run
```

Q0 记录绑定原模型、SID 数据 manifest、seed、全部导出文件指纹；校验失败会拒绝加载。`--initialization` 同样适用于正式 Q1、Q2 和续训命令。
下方不带 `--initialization` 的旧命令仍兼容，但属于重新扩词初始化的旧方案，不能作为“与 Q0 相同起点”的主线结果。
导出目录已存在时拒绝覆盖；若导出中断，应保留现场并使用新的目录，而不是把半成品当成完整检查点。

[start_q0_tabby.sh](../../scripts/start_q0_tabby.sh) 是本次实例的人工操作辅助入口：按指定完整 Git 提交下载适配代码，备份旧脚本，然后只导出 Q0；不自动执行评测、SFT 或 GRPO。

## 入口与配置

| 文件 | 作用 |
|---|---|
| [prepare_qwen_sft.py](../../scripts/prepare_qwen_sft.py) | 校验现有目录、SID来源和划分，导出可搬到服务器的样本索引 |
| [qwen_sft.py](../../scripts/qwen_sft.py) | chat prompt、词表、回答mask、四任务数据集、前缀树、Trainer商品指标 |
| [train_qwen_sft.py](../../scripts/train_qwen_sft.py) | tokenizer dry-run、全参数BF16训练、完整训练状态恢复 |
| [evaluate_qwen_sft.py](../../scripts/evaluate_qwen_sft.py) | 独立全目录约束生成评测 |
| [freeze_qwen_selection.py](../../scripts/freeze_qwen_selection.py) | 从完整验证报告中选择并冻结最佳权重指纹 |
| [q1_pilot.json](configs/q1_pilot.json) | 10,000训练目标，20次更新，64人检索验证，仅验证通路 |
| [q1_full.json](configs/q1_full.json) | 主任务全量样本、1轮、5000人快速验证 |
| [q2_alignment.json](configs/q2_alignment.json) | 从同一原始权重独立训练，增加标题↔SID、历史SID→下一标题 |

当前已实现代码并进行本地小模型验证；没有正式 Qwen 推荐指标。首个pilot将验证速度和显存；64人是快速工程检查，正式验证样本不能据此更改。
所有命令在仓库根目录执行。Linux建议 Python3.11/3.12；CUDA驱动与PyTorch匹配后安装：

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-qwen-sft.txt
python scripts/prepare_qwen_sft.py --sids data/processed/office2018/tiger_sids --output data/processed/office2018/qwen_sft_adam
python scripts/train_qwen_sft.py --config 04_推荐生成/02_Qwen2.5-1.5B-Instruct/configs/q1_pilot.json --data data/processed/office2018/qwen_sft_adam --model /mnt/Qwen2.5-1.5B-Instruct --output outputs/qwen_q1_pilot --dry-run
```

Windows本地把 `--model` 替换为 `outputs/pretrained/Qwen2.5-1.5B-Instruct`，用 `.venv/Scripts/python.exe` 执行。
dry-run只加载tokenizer，不加载1.5B权重。GPU正式训练使用新的输出目录并移除 `--dry-run`：

```bash
python scripts/train_qwen_sft.py --config 04_推荐生成/02_Qwen2.5-1.5B-Instruct/configs/q1_pilot.json --data data/processed/office2018/qwen_sft_adam --model /mnt/Qwen2.5-1.5B-Instruct --output outputs/qwen_q1_gpu_pilot
python scripts/train_qwen_sft.py --config 04_推荐生成/02_Qwen2.5-1.5B-Instruct/configs/q1_full.json --data data/processed/office2018/qwen_sft_adam --model /mnt/Qwen2.5-1.5B-Instruct --output outputs/qwen_q1_full
```

全参训练：全部Qwen层、输入embedding、输出头都更新；无LoRA、无量化训练。BF16、梯度检查点、microbatch2、累积64，即单GPU每次更新128条样本；AdamW、lr5e-5、cosine、3%warmup、梯度裁剪1.0。
先跑pilot测峰值显存再确定正式预算；当前实例为 A100 40GB，BF16 运算和样本 dry-run 已通过，正式训练显存与耗时尚未测量。
Q1约397345主任务/轮。Q2每轮访问同样的主任务，再加每件有标题商品的两条对齐样本和每条有目标标题的历史→标题样本，按拼接数据集均匀shuffle；它的算力预算更大，要记录总样本/token/更新次数，不能归因于方法一定更好。
Q2也是从原始Qwen初始化，不能偷偷接着Q1训练再与Q1当等预算比较。短pilot不用于报告算法优劣。

## 恢复、选模和测试

每500次更新保存 `checkpoint-*` 的权重、tokenizer、optimizer、scheduler、RNG和Trainer state；保留最佳和最新等最多2个检查点。恢复参数：

```bash
python scripts/train_qwen_sft.py --config 04_推荐生成/02_Qwen2.5-1.5B-Instruct/configs/q1_full.json --data data/processed/office2018/qwen_sft_adam --model /mnt/Qwen2.5-1.5B-Instruct --output outputs/qwen_q1_full --resume outputs/qwen_q1_full/checkpoint-500
```

配置、代码、数据manifest或SID词表改变会拒绝恢复。`selected/` 是验证选中的推理权重；完整续训必须使用 `checkpoint-*`，仅有selected无法恢复优化器。
中断发生在首次保存前时，仍需从头开始；pilot首次20步保存，正式首次500步保存。网盘需保留完整输出目录，单份Adam训练状态可能超过20GB，100GB应控制检查点数量并观察空余。

训练只读验证目标，按固定seed随机5000人商品NDCG@10选模，连续3次无提升早停；验证loss仅作诊断。
1轮不足500步的修改配置须相应调整eval_steps。当前标准配置均会触发定期保存。
保留的候选检查点需要分别跑全量86,713人验证后再冻结最终版本；runner只自动选择快速验证最佳，`selected/`不代表已完成全量最终选模。
beam20从完整25,898商品前缀树近似搜索，过滤完整已知历史，输出最多20件商品；不能描述为全目录精确打分。当前检索逐用户执行，首版偏重易读性，正式全量验证可能很慢，先pilot测量，后续可批量化优化。

```bash
python scripts/evaluate_qwen_sft.py --checkpoint outputs/qwen_q1_full/selected --data data/processed/office2018/qwen_sft_adam --split valid --output outputs/qwen_q1_full/full_validation.json
```

候选检查点各自完成全量验证后，用脚本按NDCG@10选取最佳并冻结选择记录；检查点、数据、beam和长度参数都会绑定。可以给`--reports`传入多个完整验证报告，同分按输入报告顺序选取。

```bash
python scripts/freeze_qwen_selection.py --reports outputs/qwen_q1_full/full_validation.json --output outputs/qwen_q1_full/selection.json
```

测试命令必须提供这份JSON；`--checkpoint`要对应冻结选中的权重，如果另一个候选胜出，应替换为那个checkpoint目录：

```bash
python scripts/evaluate_qwen_sft.py --checkpoint outputs/qwen_q1_full/selected --data data/processed/office2018/qwen_sft_adam --split test --selection-record outputs/qwen_q1_full/selection.json --output outputs/qwen_q1_full/test.json
```

输出已存在时拒绝覆盖；测试只在最终配置固定后执行一次，不用于调参。若检查点为多分片格式，本版单文件权重指纹检查需要扩展；本次1.5B默认保存为单文件。

## 来源

[固定版本原源码](../../baselines/minionerec_reference/README.md) 已下载保留。使用其全参SFT和四方向任务的设计参考；本地四段SID、数据、prompt、检索和选模代码是适配实现，未直接调用上游三段SID脚本。上游结果不能当成本项目结果。
详细原理：[从样本到全参数SFT](../../01_知识库/05_SFT与生成式推荐/Qwen全参数SFT流程.md)。本次公开Q1 SFT正式测试；后续偏好优化另行发布。

## 独立 AUC / GAUC 评测

[evaluate_qwen_auc.py](../../scripts/evaluate_qwen_auc.py) 对固定候选逐一 teacher forcing 评分，默认每人 1 正例＋100 个均匀无放回代理负例，保存候选指纹和逐用户分数；GAUC 按用户候选数加权。分数为四段 SID 在未屏蔽词表下的 log-probability 之和，不是点击概率，也不是仅对 beam Top20 计算 AUC。

```bash
python scripts/evaluate_qwen_auc.py --checkpoint outputs/qwen_q1_full/selected --data data/processed/office2018/qwen_sft_adam --split valid --users 1000 --negatives 100 --seed 2026 --output outputs/qwen_q1_full/auc_valid_s100_u1000
```

默认子集仅为诊断；`--users 0` 为全用户。测试额外要求 `--selection-record`，不覆盖旧输出。GPU 评分尚未执行；其他新增维度与统一列表导出仍待接入。详见 [多角度协议](../../01_知识库/03_数据与评测/Q0_SFT_GRPO多角度评测方案_20261010.md)。

## 2026-10-13 正式SFT结果

Q1已完成A100 40GB全参数训练，以固定5000人验证NDCG@10选中第3000步，完成86713人全量测试。见[结果、模型结构与权重指纹](../../experiments/qwen_sft_office2018_20261013/README.md)及[精度设置](../../01_知识库/05_SFT与生成式推荐/FP32_BF16与本项目精度设置.md)。

正式测试沿用sid_beam_v2、beam20与完整历史排除。冻结训练期验证选模证据可使用新增入口 `scripts/freeze_qwen_training_selection.py`；`evaluate_qwen_sft.py`接受这种固定验证子集的选模记录，不要求额外重复全用户验证。
