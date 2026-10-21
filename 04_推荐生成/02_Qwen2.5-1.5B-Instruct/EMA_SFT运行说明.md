# EMA SID → Qwen 全参数SFT：2026-10-21

本实验从原始`Qwen2.5-1.5B-Instruct`重新做Q1下一商品SID监督微调，输入改为已冻结的EMA RQ-VAE SID。不加载旧梯度SID的SFT权重，也不重新训练MiniLM或量化器。

## 固定比较条件

| 项目 | 设置 |
|---|---|
| 数据 | 同一Office Products目录，86,713用户，25,898商品 |
| SID | EMA映射SHA256 `975075925cc54f903d7bc9069bf3653a96b8c29b613b846bdc4a27a8351a4554` |
| 训练样本 | 397,345个位置，历史最多20件；按原用户内时序划分 |
| 训练 | 全参数、BF16、SDPA、激活检查点、1轮、seed2026 |
| 批次 | microbatch16 × 累积8 = 128；原正式SFT为2 × 64 |
| 优化 | AdamW，学习率5e-5，cosine，warmup3%，梯度裁剪1 |
| 验证选模 | 每500步固定5,000用户，seed2027，NDCG@10；早停耐心3 |
| 测试 | 冻结已选模型后，86,713用户全量测试一次；sid_beam_v2、beam20 |

除microbatch及梯度累积外，训练配置逐字段与旧Q1正式配置核对。两版有效batch均为128，但浮点计算与padding分组变化，不能保证数值相同；结果对照会明确记录这两项执行配置差异。核心`qwen_sft.py`也核对旧正式测试记录的源码SHA256。重跑仍只有一个随机种子，硬件及库版本等差异保留在环境记录，不把结果描述为多种子显著性结论。

EMA数据已经在本地准备好，启动不需要在计费服务器重新生成商品向量、训练码本或构造样本位置。运行时仍按原实现动态分词；这次没有顺带改变训练数据加载或缓存机制。

## 网盘要有的文件

放到**16区网盘根目录**，实例中映射为`/mnt`：

1. `ema_sft_20261021_v5.zip`：小型离线训练包，含代码、配置、准备好的数据、EMA SID、分词器和原模型配置；不含3GB Qwen权重。
2. `start_ema_sft_20261021_v5.sh`：绑定该ZIP SHA256的启动器。
3. `ema_sft_20261021_v5.manifest.json`：上传核对清单。
4. 已有的`Qwen2.5-1.5B-Instruct/model.safetensors`：3,087,467,144字节，SHA256为`dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`。

本地包校验完成不等于已上传或网盘校验完成。网盘上传状态以交付时的核对记录为准；根目录权重仍会在开机复制时验证完整SHA256。输入ZIP及启动器不依赖服务器访问GitHub，避免之前raw.githubusercontent.com连接失败的问题。

V5使用batch16/累积8，默认采用短测验证的`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，并记录前10次参数更新的耗时、CUDA实际分配峰值及缓存预留峰值。V4为batch8/累积16；该任务在152步时按用户要求切换至batch16，尚无检查点，从相同Q0重新开始；旧记录保存在`history/batch8_stopped_before_checkpoint/`。V3为batch4/累积32；该任务在54步时按用户要求切换至batch8，同样未产生检查点，记录保存在`history/batch4_stopped_before_checkpoint/`。10月21日旧batch2任务在87步时按用户要求停止，尚无检查点；日志保存在实例的`history/batch2_stopped_before_checkpoint/`，本轮从完全相同的Q0重新训练。网盘另有`ema_batch4_overlay_20261021.zip`及`ema_batch4_switch_20261021.json`保存配置变更证据。旧V2包为batch2，V3为batch4，V4为batch8；当前batch16使用V5。网盘另有`ema_batch8_overlay_20261021.zip`及`ema_batch8_switch_20261021.json`保存batch8变更。另有`ema_batch16_overlay_20261021.zip`及`ema_batch16_switch_20261021.json`保存当前正式配置变更。

V2补齐了训练入口用于源码指纹记录的`prepare_qwen_sft.py`。V1首次在检查阶段因缺少该文件退出，尚未执行优化器更新。V2已在解压后的独立目录执行真实SFT dry-run；打包脚本以后也会自动执行该检查。运行记录、实际上传校验及当前训练状态分别保存，旧的本地准备记录不作为云端运行成功证据。

## 租用与启动

沿用已验证的单卡A100 40GB，选择16区、Ubuntu22.04/Python3.11/PyTorch2.6.0/CUDA12.4镜像。本地磁盘预留至少50GB；持久网盘需要**至少4GiB可用空间**用于默认模型备份，额外完整优化器检查点另需空间。运行器会先检查报告的空闲空间，但部分FUSE网盘无法准确反映账户配额，仍需在客户端确认实际可用容量。

Tabby连接后，复制这一条：

```bash
bash /mnt/start_ema_sft_20261021_v5.sh
```

它在tmux的`rec-ema-sft`会话中执行：校验ZIP与所有成员 → 新建环境并安装固定依赖（复用镜像PyTorch）→ 校验并复制原模型到本地磁盘 → 导出EMA专用未训练SID初始化Q0 → tokenizer/sample检查 → 正式SFT → 冻结5000人验证集选出的模型 → 持久备份 → 全量测试 → 输出梯度版与EMA版对照 → 再次备份。

若只想先准备环境和初始化、不开始训练：

```bash
EMA_STAGE=prepare bash /mnt/start_ema_sft_20261021_v5.sh
```

准备会话结束后，再运行不带`EMA_STAGE`的那条即可继续。启动器会拒绝重复创建同名活动tmux会话。Q0初始化不做优化器更新；正式训练开始后会打印Trainer的loss/epoch。

查看进度：

```bash
tail -n 30 /root/ema-sft-20261021/logs/pipeline.log
```

退出码：`/root/ema-sft-20261021/pipeline.exit`。成功完成全部流程后才生成`completed.json`。训练与全量测试都需要时间；tmux防止断线中断，但不会自动停止平台GPU计费。

## 结果与备份

- 本地训练输出：`/root/ema-sft-20261021/outputs/qwen_ema_sft/`。
- 全量测试：`/root/ema-sft-20261021/checks/test.json`。
- 自动对照：`/root/ema-sft-20261021/checks/gradient_vs_ema.json`。
- 网盘备份：`/mnt/ema-sft-20261021/`，先保存已训练模型，再跑耗时测试。
- `selected/`含完整模型权重、模型配置、生成配置和扩展分词器；`data/`含EMA SID与训练/评测映射；`code/`和训练记录用于复现。
- `LATEST_BACKUP.json`指向最近一次完整校验的清单，每个文件记录大小和SHA256。只有打印`PERSISTENT_BACKUP_VERIFIED`才算本轮备份完成。

默认只把可直接推理/评测/继续新任务微调的selected模型、数据、代码和记录写到网盘，不重复存Q0及原始模型，不包含优化器。训练期本地`checkpoint-*`仍有优化器、scheduler及随机状态，可在同一完整环境中恢复。

若需要把完整断点也备份，预留足够额外空间后执行。此选项额外保留最近与验证最佳检查点（最多两个）、精确Q0权重/记录及原工程，远大于默认约3.1GB模型备份：

```bash
/root/ema-sft-20261021/.venv/bin/python /root/ema-sft-20261021/project/scripts/run_ema_sft.py --stage backup --backup-resume
```

断点恢复需要原输出目录的contract、相同代码/数据/Q0及完整checkpoint，不能只拿`selected/`当成精确续训。旧实例释放后先按备份记录恢复路径和环境，再明确指定`--resume`；本套启动器不自动猜测或覆盖失败断点。

## 本地准备和验证

```powershell
python scripts/check_ema_sft_inputs.py --output outputs/ema_sft_input_check.json
python scripts/package_ema_sft.py --output outputs/cloud_transfer/ema_sft_20261021_v5/ema_sft_20261021_v5.zip
```

不需要在本机加载1.5B权重做训练；本地检查和dry-run只能证明数据/tokenizer/入口可用，GPU训练是否成功由新实例的真实日志和结果判定。
