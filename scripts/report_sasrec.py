"""Export aggregate results from a completed local run; no user records."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from run_sasrec import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--name", default="sasrec_cpu_20261007")
    args = parser.parse_args()
    if Path(args.name).name != args.name or any(c in args.name for c in "/\\:"):
        parser.error("name must be a filename stem")
    run_dir = args.run.resolve()
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if run["status"] != "completed":
        parser.error("Run is not completed")
    assert run["test"]["users"] == run["popularity_test"]["users"] == run["users"]
    candidates = [e for e in run["history"] if "validation" in e]
    expected = max(candidates, key=lambda e: e["validation"]["NDCG@10"])
    assert run["best_epoch"] == expected["epoch"]
    run["artifacts"] = {"checkpoint": str((run_dir / "best.pth").relative_to(ROOT)).replace("\\", "/"),
                        "full_record": str((run_dir / "run.json").relative_to(ROOT)).replace("\\", "/")}
    hashes = run_dir / "source_hashes.json"
    if hashes.exists():
        run["source_sha256"] = json.loads(hashes.read_text(encoding="utf-8"))
    if run.get("resume"):
        parent_hashes = (ROOT / run["resume"]["checkpoint"]).parent / "source_hashes.json"
        if parent_hashes.exists():
            run["resume"]["parent_source_sha256"] = json.loads(parent_hashes.read_text(encoding="utf-8"))
    output = ROOT / "experiments"
    write_json(output / f"{args.name}.json", run)
    metrics = ["Recall@10", "NDCG@10", "Recall@20", "NDCG@20"]
    rows = ["| 方法 | " + " | ".join(metrics) + " |",
            "|---|" + "---:|" * len(metrics)]
    for label, key in [("SASRec", "test"), ("Popularity（训练集计数）", "popularity_test")]:
        rows.append("| " + label + " | " + " | ".join(f"{run[key][m]:.6f}" for m in metrics) + " |")
    config = run["config"]
    resumed = run.get("resume")
    resume_note = ""
    if resumed:
        resume_note = (f"\n本次包含中断恢复：此前完成 {resumed['checkpoint_epoch']} 轮，"
                       f"恢复模式为 `{resumed['mode']}`。"
                       f"原阶段 batch={resumed['parent_arguments']['batch_size']}、"
                       f"threads={resumed['parent_environment']['threads']}；"
                       f"恢复阶段 batch={run['arguments']['batch_size']}、"
                       f"threads={run['environment']['threads']}。"
                       + ("旧 checkpoint 缺少优化器与 RNG 状态，恢复时重新初始化，"
                          "本结果不能声称等同于原配置连续训练。\n"
                          if not resumed["optimizer_and_rng_restored"] else
                          "已恢复优化器与 RNG；若更改 batch 等配置，则不能声称与原配置连续训练逐步一致。\n"))
    active_training_seconds = sum(e["training_seconds"] for e in run["history"])
    china_zone = timezone(timedelta(hours=8))
    def china_time(value):
        return datetime.fromisoformat(value).astimezone(china_zone).strftime("%Y-%m-%d %H:%M:%S")
    initial_start = run["started_at_utc"]
    ancestor = resumed
    reset_ancestors = []
    while ancestor:
        initial_start = ancestor["parent_started_at_utc"]
        if not ancestor["optimizer_and_rng_restored"]:
            reset_ancestors.append(ancestor["checkpoint_epoch"])
        ancestor = ancestor.get("parent_resume")
    if resumed and reset_ancestors and resumed["optimizer_and_rng_restored"]:
        reset_epochs = "、".join(str(e) for e in reset_ancestors)
        resume_note += f"\n历史阶段曾从第 {reset_epochs} 轮的旧权重恢复并重置优化器；本段恢复包含完整状态。\n"
    batch_description = ("恢复阶段 " if resumed else "") + f"batch={run['arguments']['batch_size']}"
    stage_start = run.get("training_start_epoch", resumed["checkpoint_epoch"] + 1 if resumed else 1)
    continued = stage_start > 20
    title = "SASRec 继续训练实验" if continued else "SASRec 首次本地 CPU 实验"
    stop_note = ""
    if run.get("training_paused"):
        stop_note = (f"第 {run['training_stopped_at_epoch']} 轮出现连续过拟合迹象，已暂停训练。"
                     "按验证集保存最佳权重，同时保留最后一轮完整检查点，后续可恢复。")
    elif continued:
        stop_note = "达到本次指定的最大轮数，未触发连续过拟合迹象的暂停规则。"
    stage_note = (f"本次从第 {stage_start} 轮开始，"
                  f"最多到第 {run.get('requested_last_epoch', len(run['history']))} 轮；"
                  f"实际新增 {len([e for e in run['history'] if e['epoch'] >= stage_start])} 轮。")
    trajectory = ""
    comparison = ""
    if continued:
        trajectory_rows = ["## 续训的验证走势", "",
                           "| 轮数 | 训练 loss | 验证 Recall@10 | 验证 NDCG@10 | 连续过拟合警告 |",
                           "|---|---:|---:|---:|---:|"]
        for e in run["history"]:
            if e["epoch"] >= run["training_start_epoch"] - 1 and "validation" in e:
                v = e["validation"]
                warnings = e.get("overfit_monitor", {}).get("consecutive_warnings", 0)
                trajectory_rows.append(f"| {e['epoch']} | {e['loss']:.6f} | {v['Recall@10']:.6f} | "
                                       f"{v['NDCG@10']:.6f} | {warnings} |")
        trajectory = "\n".join(trajectory_rows)
        parent_record = (ROOT / resumed["checkpoint"]).parent / "run.json"
        parent_run = json.loads(parent_record.read_text(encoding="utf-8"))
        if "test" in parent_run:
            comparison = ("\n续训前测试 Recall@10=" + f"{parent_run['test']['Recall@10']:.6f}，"
                          + f"NDCG@10={parent_run['test']['NDCG@10']:.6f}。"
                          "只在本段选定权重后比较测试结果，测试成绩不参与暂停或选模。\n")
    record = f"""# {title}

真实运行记录：{china_time(initial_start)} 至 {china_time(run['completed_at_utc'])}（北京时间，含中断等待）。
本段运行开始：{china_time(run['started_at_utc'])}；JSON 原始时间保留为 UTC。
Amazon Reviews 2018 Office Products，去重并重新做 5-core，
{run['users']:,} 个用户、{run['items']:,} 个商品，数据指纹见同名 JSON。

## 测试结果

{chr(10).join(rows)}

全商品候选、全部用户、排除完整已知历史，每人一个目标；单目标下 HR@K=Recall@K。
这些分数不能与上游 101 个采样候选的结果直接比较。
{comparison}

## 训练与选模

{resume_note}

{stage_note}
{stop_note}
暂停规则：{run['protocol'].get('overfitting_pause_rule') or '普通验证集无提升 early stopping'}。
暂停规则是保守的过拟合信号检测，不是数学意义上证明模型已经过拟合。

- 模型：原上游 SASRec，{run['parameters']:,} 个参数，hidden={config['hidden_units']}，
  {config['num_blocks']} 层、{config['num_heads']} 个注意力头、maxlen={config['maxlen']}，
  dropout={config['dropout_rate']}，norm_first={config['norm_first']}。
- 实际训练 {len(run['history'])} 轮，每轮完整遍历用户，{batch_description}，
  Adam lr={run['arguments']['lr']}，seed={run['arguments']['seed']}。
- 每轮 {run['supervised_positions_per_epoch']:,} 个有效监督位置；训练历史保留最近一段。
- 最佳权重：第 {run['best_epoch']} 轮，仅验证 NDCG@10 选模，
  最佳验证 NDCG@10={run['best_validation_ndcg10']:.6f}。
- 第 1 轮平均 loss={run['history'][0]['loss']:.6f}，
  最后一轮平均 loss={run['history'][-1]['loss']:.6f}。
- 当前运行段耗时 {run['elapsed_seconds']/60:.2f} 分钟，包含数据加载、训练、评测和推理检查。
  所有已完成轮次的训练累计耗时 {active_training_seconds/60:.2f} 分钟，不含中断等待。
- 环境：Python {run['environment']['python']}、PyTorch {run['environment']['torch']}、
  NumPy {run['environment']['numpy']}，CPU {run['environment']['threads']} 个计算线程。

{trajectory}

## 本机推理

加载模型并预热后，单用户全商品 Top-10，30 次测量：
P50={run['inference']['single_user_full_catalog_top10_p50_ms']:.2f} ms，
P95={run['inference']['single_user_full_catalog_top10_p95_ms']:.2f} ms。
不包含磁盘、网络和并发队列，只是这台电脑上一个用户的参考延迟。

本地权重：`{run['artifacts']['checkpoint']}`。
权重、用户示例和完整日志留在被 Git 忽略的 outputs 目录。
汇总记录：[{args.name}.json]({args.name}.json)。

## 限制与下一步

这是单随机种子、有限轮数的起步基线，尚未做超参数搜索或多次运行统计，
不是 SASRec 的最优成绩或论文精确复现。
评论作为交互代理、同时间戳顺序不确定、用户内划分和全数据过滤，
不等同于严格全局时间实验或线上效果。
后续 TIGER 使用同一商品目录、划分、候选过滤和评测规则。
"""
    (output / f"{args.name}.md").write_text(record, encoding="utf-8")
    print(f"Wrote experiments/{args.name}.json and .md")


if __name__ == "__main__":
    main()
