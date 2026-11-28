"""Verify a downloaded native-W0 benchmark and generate its local comparison."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
R41 = HERE.parent / "vttiny_wem_r41" / "results"
NATIVE = (HERE.parent / "vttiny_wem_r31_checkout" / "results/vttiny/iterations"
          / "v0.4-motion-blur/gmc/02-factorial")
REFERENCE_PROTOCOL_SHA = "7c18c28f2272c6fcf497bc6234c867a3472e0acbab23b73804d9744f315d2819"
BASES = ("native", "B", "BG")
TOGGLES = ("off", "W0")
ARMS = tuple(f"{base}_{mode}" for base in BASES for mode in TOGGLES)
POOLED = "COMBINED_SEQ"
PERCENT = ("HOTA", "DetA", "AssA", "IDF1", "MOTA")
COUNTS = ("FP", "FN", "IDSW")
METRICS = (*PERCENT, *COUNTS)
MAIN = ("HOTA", "IDF1", "MOTA", *COUNTS)
BASE_DESCRIPTIONS = {
    "native": "原始 BoT-SORT，原生 SOF GMC 和 RGB ReID",
    "B": "B_early_moderate，原生 SOF GMC",
    "BG": "evidence_both，B、自适应 GMC、低分双重确认与关联证据",
}


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked(path, expected):
    require(sha(path) == expected, f"SHA-256 mismatch: {path}")
    return read(path)


def index_metrics(summary):
    require(summary["status"] == "PASS" and summary["modality_reference"] == "rgb"
            and summary["split"] == "test", "The official evaluation did not complete in the frozen scope")
    index = {}
    for row in summary["results"]:
        key = row["tracker"], row["sequence"]
        require(key not in index, f"Duplicate metric row: {key}")
        for metric in PERCENT:
            require(type(row[metric]) in (int, float) and math.isfinite(row[metric]), f"Invalid {key}/{metric}")
        for metric in COUNTS:
            require(type(row[metric]) is int and row[metric] >= 0, f"Invalid {key}/{metric}")
        index[key] = row
    return index


def verify_native(root, plan, complete, lengths, frozen, camera):
    receipt = checked(root / "full/verified.json", complete["full_verification_sha256"])
    smoke = read(root / "smoke/verified.json")
    commits = {}
    for stage, record, expected in (("smoke", smoke, {seq: lengths[seq] for seq in plan["smoke"]}),
                                    ("full", receipt, lengths)):
        require(record["status"] == "PASS" and record["native_MOT_parity"] is True
                and record["sequences"] == expected and set(record["results"]) == set(expected),
                f"Unexpected native prediction scope in {stage}")
        for seq, length in expected.items():
            row = record["results"][seq]
            off, on = (row[arm] for arm in ARMS[:2])
            require(off["mot"] == frozen[f"full/trackers/native/data/{seq}.txt"] and off["commits"] == 0,
                    f"Native off differs from the historical original: {stage}/{seq}")
            for arm, detail in zip(ARMS[:2], (off, on)):
                require(detail["frames"] == length and detail["warp_sha256"] == camera[seq],
                        f"Native arm frame count or SOF mismatch: {stage}/{arm}/{seq}")
            start = on["first_commit"]
            require(row["prefix_verified"] == (length if start is None else start - 1)
                    and (start is not None or on["mot"] == off["mot"]), f"Native state prefix mismatch: {stage}/{seq}")
            if stage == "full":
                for arm, detail in zip(ARMS[:2], (off, on)):
                    path = root / "full" / arm / "wem_events/native_botsort" / (seq + ".jsonl")
                    require(sha(path) == detail["weak_events"], f"Weak event receipt mismatch: {arm}/{seq}")
                    events = [item for line in path.read_bytes().splitlines()
                              if (item := json.loads(line))["event"] == "committed"]
                    require(len(events) == detail["commits"]
                            and min((event["frame"] for event in events), default=None) == detail["first_commit"]
                            and all(1 <= event["frame"] <= length for event in events), f"Invalid weak commits: {arm}/{seq}")
                    commits[arm, seq] = len(events)
    return receipt["results"], commits


def verify(root, r41, historical):
    plan = read(root / "plan.json")
    complete = read(root / "complete.json")
    old_inputs = read(r41 / "inputs_verified.json")
    old_complete = read(r41 / "full/complete.json")
    old_eval = read(r41 / "full/evaluation_complete.json")
    old_verified = checked(r41 / "full/verified.json", old_complete["checks"])
    old_metrics = index_metrics(checked(r41 / "full/metrics/summary.json", old_eval["files"]["metrics/summary.json"]))
    lengths = plan["sequences"]
    require(len(lengths) == complete["sequence_count"] == 30
            and sum(lengths.values()) == complete["frames_per_arm"] == 12590
            and lengths == old_inputs["sequences"] and plan["toggles"] == list(TOGGLES)
            and len(plan["smoke"]) == 3 and len(set(plan["smoke"])) == 3
            and set(plan["smoke"]) <= set(lengths)
            and plan["native_reference_protocol_sha256"] == REFERENCE_PROTOCOL_SHA,
            "The plan differs from the locked scope or the historical native protocol")
    require(complete["status"] == "COMPLETE" and complete["native_trajectory_parity"] == "PASS"
            and complete["prior_metric_parity"] == "PASS" and complete["new_replays"] == list(ARMS[:2])
            and complete["retained_references"] == list(ARMS[2:]), "The native-W0 benchmark is incomplete")
    original = read(historical / "MANIFEST.json")
    metrics = index_metrics(checked(root / "metrics/summary.json", complete["metric_sha256"]))
    old_native = index_metrics(read(historical / "full/metrics/summary.json"))
    require(set(metrics) == {(arm, seq) for arm in ARMS for seq in (*lengths, POOLED)},
            "Official evaluation must cover 6 controls × 30 sequences and six COMBINED_SEQ rows")
    native_result, commits = verify_native(root, plan, complete, lengths, original["members"],
                                           old_inputs["input_sha256"]["warps_native"]["files"])
    audit = read(root / "metrics/evaluation_audit.json")
    audit_previous = read(r41 / "full/metrics/evaluation_audit.json")
    require(audit["status"] == "COMPLETE" and audit["summary_sha256"] == complete["metric_sha256"]
            and audit["ground_truth"] == audit_previous["ground_truth"]
            and audit["evaluator"] == audit_previous["evaluator"], "GT or TrackEval differs from the archived evaluation")
    predictions = audit["validation"]["prediction_files"]
    require(audit["validation"]["sequences"] == lengths and set(predictions) == set(ARMS),
            "Prediction evaluator scope mismatch")
    for arm in ARMS:
        for seq in (*lengths, POOLED):
            row = metrics[arm, seq]
            if arm == ARMS[0]:
                require(row == dict(old_native["native", seq], tracker=arm), f"Historical native metric mismatch: {seq}")
            if arm in ARMS[2:]:
                require(row == old_metrics[arm, seq], f"Archived B/BG metric mismatch: {arm}/{seq}")
            if seq != POOLED:
                if arm in ARMS[:2]:
                    expected = native_result[seq][arm]["mot"]
                else:
                    archived = old_verified["arms"][arm]["per_sequence"][seq]
                    expected = archived["mot"]
                    commits[arm, seq] = archived["counts"].get("committed", 0)
                require(predictions[arm][seq]["sha256"] == expected, f"MOT was not the declared evaluation input: {arm}/{seq}")
        for metric in COUNTS:
            require(sum(metrics[arm, seq][metric] for seq in lengths) == metrics[arm, POOLED][metric],
                    f"Pooled {metric} does not sum to sequence counts: {arm}")
        commits[arm, POOLED] = sum(commits[arm, seq] for seq in lengths)
    return lengths, complete, metrics, commits, native_result


def comparison_rows(metrics, commits, sequences):
    for seq in sequences:
        for arm in ARMS:
            base, _ = arm.split("_", 1)
            ref = f"{base}_off"
            current, baseline = metrics[arm, seq], metrics[ref, seq]
            yield dict(base=base, arm=arm, reference=ref, sequence=seq,
                       **{key: current[key] for key in METRICS},
                       **{f"delta_{key}": current[key] - baseline[key] for key in METRICS},
                       weak_commits=commits[arm, seq])


def sequence_directions(paired):
    direction = {}
    for base in BASES:
        direction[base] = {}
        for metric in MAIN:
            values = [row[f"delta_{metric}"] for row in paired if row["arm"] == f"{base}_W0"]
            if metric in COUNTS:
                values = [-value for value in values]
            better = sum(value > 1e-9 for value in values)
            worse = sum(value < -1e-9 for value in values)
            direction[base][metric] = dict(better=better, tied=len(values) - better - worse, worse=worse)
    return direction


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown(report):
    lines = ["# 原始 BoT-SORT 开关 W0 与三底座对照", "",
             "原始 BoT-SORT 的 W0 关闭、开启组各独立回放了官方开发评测的 30 段、12,590 帧。"
             "关闭组 30 份轨迹均与归档原始 BoT-SORT 逐字节一致。新实验保留 W0 的原锚点约束和正式状态反馈。"
             "检测、常规 RGB ReID、官方真值和 TrackEval 固定。两套旧底座使用 R4.1 的已核验轨迹，同版统一重评。", "",
             "## 完整指标", "",
             "HOTA、DetA、AssA、IDF1、MOTA 为 TrackEval COMBINED_SEQ 百分制值。", "",
             "| 底座 | 组别 | HOTA | DetA | AssA | IDF1 | MOTA | FP | FN | IDSW | 弱框提交 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["comparison"]:
        values = [f"{row[key]:.4f}" for key in PERCENT] + [f"{row[key]:,d}" for key in COUNTS]
        lines.append("| " + " | ".join([row["base"], row["arm"], *values, f"{row['weak_commits']:,d}"]) + " |")
    lines += ["", "## 各自底座的 W0 增量", "",
              "每行是同一底座上的 W0 减关闭组。百分制指标差值以百分点计。", "",
              "| 底座 | ΔHOTA | ΔIDF1 | ΔMOTA | ΔFP | ΔFN | ΔIDSW | HOTA 改善／持平／下降段数 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in report["comparison"]:
        if not row["arm"].endswith("_W0"):
            continue
        counts = report["sequence_direction"][row["base"]]["HOTA"]
        values = [f"{row['delta_'+key]:+.4f}" for key in ("HOTA", "IDF1", "MOTA")]
        values += [f"{row['delta_'+key]:+,d}" for key in COUNTS]
        lines.append("| " + " | ".join([row["base"], *values,
                                       "／".join(str(counts[key]) for key in ("better", "tied", "worse"))]) + " |")
    lines += ["", "原始组保留 BoT-SORT 自带的 SOF GMC。原始组的配对结果检验 W0 在没有 B 与 BG 联合保护时的表现。"
              "BG 相对 B 同时包含自适应 GMC、低分双重确认和关联证据，增量差反映这套联合设置。"
              "逐序列涨跌是描述性统计。官方 30 段此前已用于开发。", "",
              "[总体与配对增量](comparison.csv) · [180 条逐序列记录](paired_per_sequence.csv) · [核验和增量差](summary.json)", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="Version directory written by transfer.py fetch")
    parser.add_argument("--r41", type=Path, default=R41, help="Original local R4.1 result archive")
    parser.add_argument("--historical-native", type=Path, default=NATIVE, help="Existing native GMC factorial archive")
    args = parser.parse_args()
    root = args.result.resolve()
    lengths, complete, metrics, commits, native = verify(root, args.r41, args.historical_native)
    pooled = list(comparison_rows(metrics, commits, (POOLED,)))
    paired = list(comparison_rows(metrics, commits, lengths))
    row_for_base = {row["base"]: row for row in pooled if row["arm"].endswith("_W0")}
    differences = []
    for current, reference in (("B", "native"), ("BG", "B"), ("BG", "native")):
        differences.append(dict(base=current, reference=reference, **{
            key: row_for_base[current][f"delta_{key}"] - row_for_base[reference][f"delta_{key}"] for key in METRICS
        }))
    report = dict(status="VERIFIED", evaluated_arms=list(ARMS), sequence_lengths=lengths,
                  base_descriptions=BASE_DESCRIPTIONS, comparison=pooled,
                  sequence_direction=sequence_directions(paired), incremental_differences=differences,
                  native_sequences_with_same_trajectory=sum(row["native_off"]["mot"] == row["native_W0"]["mot"]
                                                            for row in native.values()),
                  completion=complete)
    save_csv(root / "comparison.csv", pooled)
    save_csv(root / "paired_per_sequence.csv", paired)
    (root / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "REPORT.md").write_text(markdown(report), encoding="utf-8")
    print("PASS: native off byte parity, frozen inputs, six pooled metrics, and 180 sequence rows")
    print(root / "REPORT.md")


if __name__ == "__main__":
    main()
