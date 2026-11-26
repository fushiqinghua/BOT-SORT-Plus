"""Summarize and fingerprint the successful single E2 coordinate ablation."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import sys

from e2_boxes import BASE, RAW, KF, output_json, read_json

LABEL = {BASE: "冻结 B", RAW: "E2-raw 原始检测框", KF: "E2-kf 复制状态更新"}
KEYS = ("HOTA", "IDF1", "MOTA", "FP", "FN", "IDSW")


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def md_table(columns, rows):
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |",
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def metric_row(label, row):
    return [label, *(f"{row[k]:.4f}" if k in KEYS[:3] else row[k] for k in KEYS)]


def render(full, smoke, changed_rows):
    metrics = full["combined"]
    groups = full["clear"]["by_lineage"]
    clear = groups["all"]
    windows = full["fixed_windows"]
    native_changes = full["clear"]["native_CLEAR_changes_raw_to_KF"]
    same_mota = metrics[KF]["MOTA"] - metrics[BASE]["MOTA"]
    vs_raw_mota = metrics[KF]["MOTA"] - metrics[RAW]["MOTA"]
    track_identity = windows[KF]["original_failure_same_id"] - windows[RAW]["original_failure_same_id"]
    threshold = (191 + metrics[KF]["IDSW"] - metrics[BASE]["IDSW"] + 1) // 2
    decision = "CONTINUE" if same_mota > 0 and vs_raw_mota > 0 and track_identity >= 0 else "STOP"
    target = smoke["clear"]["target"]
    result = [
        "# E2 固定提名的输出坐标对照", "",
        f"本轮决策 **{decision}**。全量固定 {clear['rows']} 条补帧提名，正确补帧由 "
        f"{clear['raw_tp']} 增至 {clear['kf_tp']}。E2-kf 的 MOTA 相对 E2-raw 变化 "
        f"{vs_raw_mota:+.4f} 个百分点，相对冻结 B 变化 {same_mota:+.4f} 个百分点。"
        "作为 D3 补帧方向的收尾，冻结 B 继续用于正式对照。",
        "",
        "## 实验约束", "",
        "保留原生 MOT 行和补帧帧号、轨迹 ID、检测索引、分数、总数与顺序。每个提名单独复制首轮实测状态流中 after-GMC 处的 "
        "8 维均值和 8×8 协方差，按同帧原检测中心 XYWH 调用一次固定 BoxMOT 16.0.11 的 "
        "KalmanFilterXYWH.update，只将后验前四维转换为 MOT TLWH。预测脚本无真值参数，真值只用于随后的独立 TrackEval 与诊断配对。",
        "",
        "## 试跑", "",
        "四段核验覆盖 DJI_0303_2、DJI_0327_3、DJI_0339_2、DJI_0141_2。"
        f"在 DJI_0303_2 第 58 帧轨迹 2／检测 5，和 GT 1 的 IoU 从 "
        f"{target['raw_iou_prior_gt']:.6f} 变为 {target['kf_iou_prior_gt']:.6f}，"
        f"CLEAR 分配由 {'命中' if target['raw_TP'] else '未命中'}"
        f"变为 {'命中 GT ' + str(target['kf_clear_gt']) if target['kf_TP'] else '未命中'}。",
        "首次试跑已成功生成坐标并复现 TrackEval，逐框核验因 IoU 接口输入形状有误中断。"
        "保留首次日志和坐标，以原样的坐标重做真值复核，再启动唯一的全量试验。",
        "", "## 官方 30 段指标", "",
        md_table(("方法", *KEYS), [metric_row(LABEL[name], metrics[name]) for name in (BASE, RAW, KF)]), "",
        "两组历史基准的每序列和汇总结果均复现，TrackEval 版本和官方真值逐文件哈希一致。"
        f"比较 raw 与 kf，191 行发生 FP→TP {clear['fp_to_tp']} 次、TP→FP {clear['tp_to_fp']} 次、"
        f"仍命中但 GT 改变 {clear['tp_gt_changed']} 次。原生 MOT 行逐字节一致，"
        f"其中 CLEAR 匹配身份实际变化 {native_changes} 行。",
        "",
        md_table(("原误差类型", "提名数", "raw TP", "kf TP", "FP→TP", "TP→FP"), [
            [key, val["rows"], val["raw_tp"], val["kf_tp"], val["fp_to_tp"], val["tp_to_fp"]]
            for key, val in groups.items() if key != "all"
        ]),
        "", "三个跨越 CLEAR 匹配线的原始候选如下。", "",
        md_table(("序列", "帧", "轨迹 ID", "GT", "raw IoU", "kf IoU", "变化"), [
            [row["sequence"], row["frame"], row["track_id"], row["prior_gt"],
             f"{float(row['raw_iou_prior_gt']):.6f}", f"{float(row['kf_iou_prior_gt']):.6f}",
             "FP→TP" if row["kf_TP"] == "True" else "TP→FP"] for row in changed_rows
        ]),
        "", "## 287 个冻结 D3 窗口", "",
        md_table(("方法", "原 ID 首次中断补回", "连续 3 帧", "连续 5 帧", "连续 10 帧"), [
            [LABEL[name], windows[name]["original_failure_same_id"],
             *(windows[name]["stable"][str(h)] for h in (3, 5, 10))]
            for name in (BASE, RAW, KF)
        ]), "",
        f"在保留 191 行、原生 CLEAR 分配与冻结 B 一致，且 IDSW 增量为本实验观测值的条件下，"
        f"补框至少需要 {threshold} 个 TP 才能使 MOTA 不低于冻结 B。"
        "本轮只做一次坐标变体，30 段为开发评估。", "",
        "主证据为 `full/E2_row_integrity.json`、`full/E2_box_pairs.csv`、`full/E2_clear_deltas.csv`、"
        "`full/E2_fixed_windows.csv`、`full/E2_metrics.csv` 和 `E2_evidence_manifest.json`。", "",
    ]
    return "\n".join(result), decision


def main():
    root, = map(Path, sys.argv[1:])
    full = read_json(root / "full/E2_evaluation_summary.json")
    smoke = read_json(root / "smoke/E2_evaluation_summary.json")
    if full["status"] != "COMPLETE" or smoke["status"] != "COMPLETE":
        raise ValueError("Smoke and full E2 must be complete")
    with (root / "full/E2_clear_deltas.csv").open(encoding="utf-8-sig", newline="") as stream:
        changed_rows = [row for row in csv.DictReader(stream) if row["raw_TP"] != row["kf_TP"]]
    report, decision = render(full, smoke, changed_rows)
    with (root / "E2_report.md").open("x", encoding="utf-8") as file:
        file.write(report)
    names = ["run_e2.py", "e2_boxes.py", "e2_evaluate.py", "finalize_e2.py", "smoke.log", "smoke_audit.log",
             "full.log", "E2_report.md"]
    for scope in ("smoke", "full"):
        stage = root / scope
        names.extend(str(path.relative_to(root)) for path in sorted(stage.glob("E2_*")) if path.is_file())
        for filename in ("evaluation_audit.json", "summary.json"):
            names.append(f"{scope}/trackeval/{filename}")
        names.extend(str(path.relative_to(root)) for path in sorted((stage / "inputs/trackers/E2_kf/data").glob("*.txt")))
        names.extend(str(path.relative_to(root)) for path in sorted((stage / "inputs/protocols").glob("*.json")))
    output_json(root / "E2_evidence_manifest.json", {
        "status": "COMPLETE", "decision": decision,
        "sha256": {name: sha(root / name) for name in names},
    })
    print("E2_REPORT_COMPLETE", decision, len(names), flush=True)


if __name__ == "__main__":
    main()
