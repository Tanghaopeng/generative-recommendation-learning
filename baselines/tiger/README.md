# 小型 TIGER 式生成式召回

本目录保留第三方实现，并由项目自己的脚本适配 Office Products。
第三方来源是 [EdoardoBotta/RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender)，
固定 commit `957d32bda43958ba641ae919abb1092ac6738798`，MIT 许可证。
这属于第三方实现，不是 TIGER 论文作者官方代码，也不是论文成绩复现。
`upstream/` 中选取的15份原文件逐字保留，哈希见 `upstream_source.json`，完整下载在本地忽略的 `repos/`。

## 实际复用与本地适配

| 环节 | 沿用内容 | 本项目适配 |
|---|---|---|
| RQ-VAE | 上游 MLP、残差量化、STE 与量化损失 | 有限次数 MiniBatchKMeans 初始化；归一化文本重建损失；不调用 torch.compile 的 forward |
| 序列生成模型 | 上游 T5 编码器/解码器、共享 SID embedding、BOS、分层输出头 | 输入与预测保留第4段消歧；不加用户哈希或商品分隔符；teacher forcing |
| 解码 | 使用上游 encoder/decoder 方法与 HF KV cache | 自建前缀树、确定性 beam、完整历史过滤；不使用上游随机采样 generate |
| 数据与评测 | 本仓库已冻结的 SASRec 数据 | 相同有效正例位置和尾部窗口；全用户验证选模；独立测试 |

上游 `forward` 会移除消歧列，本项目 `scripts/tiger_model.py` 的子类覆盖训练入口，
预测完整四段 ID；不会修改原文件来掩盖差异。分层 softmax 先计算，再施加合法分支限制，
保留原始累积 log probability。有限 beam 是近似检索，与逐商品打分有算法差异。

## 运行顺序

从项目根目录安装 CPU 依赖，然后执行。每次模型训练使用新的输出目录：

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-tiger-lock.txt --index-strategy unsafe-best-match
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe scripts/build_item_embeddings.py --threads 2 --batch-size 32
.venv/Scripts/python.exe scripts/train_rqvae.py --output outputs/rqvae_new --epochs 30 --threads 2
.venv/Scripts/python.exe scripts/build_semantic_ids.py --checkpoint outputs/rqvae_new/best.pth --threads 2
.venv/Scripts/python.exe scripts/run_tiger.py --output outputs/tiger_smoke_new --smoke-steps 20 --threads 2
.venv/Scripts/python.exe scripts/run_tiger.py --output outputs/tiger_formal_new --epochs 3 --threads 2
.venv/Scripts/python.exe scripts/report_tiger.py --run outputs/tiger_formal_new --codec outputs/rqvae_new --name tiger_new
.venv/Scripts/python.exe scripts/recommend_tiger.py --checkpoint outputs/tiger_formal_new/best.pth --history 1,2,3 --topk 10
```

编码使用冻结 MiniLM（384维），固定 revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，
截断128 token，masked mean pooling 后 L2 归一化；128是本项目 CPU 配置，模型卡默认256。
商品文本拼接标题、品牌、分类、描述和特征；28件无元数据商品使用中性占位文本。
RQ-VAE 起步为隐空间32维、3×256码本，随机5%商品向量验证重建，其余训练。
这不是行为的验证集；完整固定商品目录的文本用于离线编码，不读取留出评论。

默认召回模型 hidden128、编码/解码各2层、4头、FF256；上游 T5 默认每头 d_kv64、dropout0.1。
这是随机初始化小模型，不是微调 MiniLM，MiniLM只离线产出商品向量。
每个用户训练历史的有效预测位置各成为一个样本，最近20件商品窗口与 SASRec 对齐，
全部397,345个正例位置每轮遍历一次。按长度分桶减少 padding，桶内和批次顺序由种子控制。
目标固定4 token，损失是四位置 CE 的均值，不另抽商品负样本。

验证与测试均用完整目录前缀树，默认 beam20，输出不足20件如实记录，
不使用答案补候选、不自动重归一化概率。本首轮固定 beam，无验证集 beam 搜索；
之后可以在验证集单独比较20/50/100再选定测试配置。
仅验证 NDCG@10 选模；连续两次验证低于最佳且训练 loss 低于最佳轮时暂停。
正式运行结束后加载验证最佳模型，仅执行一次全用户测试。
`--smoke-steps` 产物明确标为检查用途，不能当作正式指标或正式续训起点。

完整恢复示例（使用新目录，保持训练配置与源码一致）：

```powershell
.venv/Scripts/python.exe scripts/run_tiger.py --resume outputs/tiger_formal_new/last.pth --output outputs/tiger_continue_new --epochs 6 --threads 2
```

保存模型、Adam、Torch/Python/NumPy随机状态，逐轮 checkpoint；
训练尚未完成一轮而中断时，该轮不算完成，恢复会从上轮 checkpoint 重新执行。
原始数据、用户映射、向量、SID与权重保存在 Git 忽略目录，只发布源码和聚合实验报告。

2026-10-08首轮根据CPU负载测试采用更小的 hidden64、各1层、2头、FF128（约32.8万参数），
计划最多3轮。128维两层仍是脚本默认配置，属于后续扩容选项。
本机实际使用命令需补上 `--hidden 64 --layers 1 --heads 2 --ff 128`；这些区别都写入 `run.json`。
`scripts/tiger_job.py` 可单次执行训练并在完成后自动导出聚合报告，状态见实验目录旁的 `.job.json`。
训练真正进程ID在 `run.json` 的 `process_id` 中；它与Windows虚拟环境启动器PID可能不同。

流程讲解见 [实施与学习路线](../../01_知识库/04_TIGER/小型TIGER实施路线.md)。

## 码本审计与EMA版本

2026-10-08另完成一版EMA更新，详见 [死码审计与对照](../../experiments/rqvae_adam_vs_ema_20261008.md)。
原版用重建+码本损失+0.25承诺损失，Adam更新码本；EMA版用重建+0.25承诺损失，码本由计数/残差向量和的EMA更新，衰减0.99。
相同初始化、商品向量、5%验证划分、种子与训练轮数，不启用死码重置。
EMA最佳第29轮：首层117/256活跃、139死码；原版最佳第30轮：84活跃、172死码。
两版后两层均256活跃，但分布不同，不能只看“全部用过”。

新增 `scripts/rqvae_ema.py` / `train_rqvae_ema.py` / `audit_codebooks.py`，额外依赖无需安装。
EMA保存完整EMA buffer、优化器和随机状态，同时导出原RqVae兼容的推理权重，原15份上游文件不变。
源实现逐层STE使后两层承诺项对编码器梯度为0；本次EMA保留同一残差图，另列为实现限制。

量化器与召回Transformer分别训练，但通过SID版本绑定。Transformer有自己的SID embedding，不直接复用RQ码本向量。
新EMA SID位于 `data/processed/office2018/tiger_sids_ema_20261008/`；旧表保持不变。
新生成模型输出为 `outputs/tiger_office2018_ema_cpu_20261008/`，通过一次性 `rqvae_ema_job.py` 在旧版结束后启动。两版均已完成3轮正式生成训练，各自验证选中第3轮，未触发过拟合暂停。
两版各自使用验证NDCG@10选模和相同提前停止规则；EMA的重建改善不等于推荐指标改善。
全用户测试与源码/SID/权重指纹核验见[两版Transformer对照](../../experiments/semantic_retrievers_comparison_20261008.md)。本次EMA前10指标略高、前20指标略低；三轮单种子结果不支持全面优于原版的结论。

```powershell
python scripts/train_rqvae_ema.py --output outputs/rqvae_ema_new --epochs 30 --decay 0.99 --threads 2
python scripts/audit_codebooks.py --checkpoint outputs/rqvae_ema_new/best.pth --output outputs/ema_audit_new.json --threads 2
python scripts/build_semantic_ids.py --checkpoint outputs/rqvae_ema_new/best.pth --output data/processed/office2018/tiger_sids_ema_new --threads 2
python scripts/run_tiger.py --sids data/processed/office2018/tiger_sids_ema_new --output outputs/tiger_ema_new --epochs 3 --hidden 64 --layers 1 --heads 2 --ff 128 --threads 2
```

恢复EMA时使用 `train_rqvae_ema.py --resume .../last.pth --output 新目录 --epochs 更大的总轮数`，保持EMA和优化器配置一致。
不能只用导出的推理weights恢复EMA训练，因为计数与向量和也是训练状态。
