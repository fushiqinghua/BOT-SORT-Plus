"""Audit and report the actual W R4.1 closed-loop TrackEval results.

All percentage metrics come from TrackEval's COMBINED_SEQ row. They are never
averaged over sequences. Labels and ground truth are read only by the separate
evaluation/attribution processes; this module consumes their completed outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile


ROLES = ("B", "BG")
POLICIES = ("off", "W0", "W1", "W2", "W3")
DEFAULT_ARMS = tuple(f"{role}_{policy}" for role in ROLES for policy in POLICIES)
PERCENT = ("HOTA", "DetA", "AssA", "IDF1", "MOTA")
COUNTS = ("FP", "FN", "IDSW")
METRICS = PERCENT + COUNTS
DETAIL_COLUMNS = {"HOTA": "HOTA___AUC", "DetA": "DetA___AUC", "AssA": "AssA___AUC",
                  "IDF1": "IDF1", "MOTA": "MOTA", "FP": "CLR_FP", "FN": "CLR_FN", "IDSW": "IDSW"}
OUTCOMES = ("unmatched_FP", "TP_same_old_identity", "TP_other_identity", "TP_past_identity_unconfirmed")
EPSILON = 1e-8  # Numerical equality only; not a statistical significance threshold.


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _number(value, context, integer=False, nonnegative=False):
    if isinstance(value, bool):
        raise ValueError(f"Boolean metric/count: {context}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value: {context}") from exc
    if not math.isfinite(result) or (integer and result != int(result)) or (nonnegative and result < 0):
        raise ValueError(f"Invalid numeric value: {context}")
    return int(result) if integer else result


def _close(left, right, context):
    if not math.isclose(left, right, rel_tol=0, abs_tol=1e-7):
        raise ValueError(f"Metric or ledger mismatch: {context}: {left} != {right}")


def _unique_strings(values, context):
    result = list(values)
    if not result or any(not isinstance(x, str) or not x for x in result) or len(set(result)) != len(result):
        raise ValueError(f"Invalid or duplicated {context}")
    return result


def _reference(arm):
    role, sep, policy = arm.partition("_")
    if role not in ROLES or not sep or policy not in (*POLICIES, "W_old"):
        raise ValueError(f"Unknown R4.1 arm: {arm}")
    return role + "_off"


def _metric_rows(batch, arms, sequences):
    source = batch / "metrics/summary.json"
    document = read(source)
    if document.get("status") != "PASS":
        raise ValueError("TrackEval summary is not PASS")
    result = {arm: {} for arm in arms}
    for row in document["results"]:
        arm, sequence = row["tracker"], row["sequence"]
        if arm not in result:
            raise ValueError(f"Unexpected evaluated arm: {arm}")
        if sequence in result[arm]:
            raise ValueError(f"Repeated evaluated row: {arm}/{sequence}")
        values = {name: _number(row[name], f"{arm}/{sequence}/{name}", name in COUNTS,
                                name != "MOTA") for name in METRICS}
        if any(values[name] > 100 + 1e-7 for name in PERCENT):
            raise ValueError(f"Percentage outside TrackEval scale: {arm}/{sequence}")
        result[arm][sequence] = values
    for arm, rows in result.items():
        if set(rows) != {*sequences, "COMBINED_SEQ"}:
            raise ValueError(f"Incomplete or unexpected TrackEval sequence scope: {arm}")
        for metric in COUNTS:
            _close(rows["COMBINED_SEQ"][metric], sum(rows[s][metric] for s in sequences),
                   f"{arm}/COMBINED_SEQ/{metric} sum")

    # The canonical evaluator normally writes both files. If supplied, they
    # must agree with the summary, including each sequence, not just totals.
    for arm in arms:
        path = batch / "metrics" / arm / "object_detailed.csv"
        if not path.exists():
            continue
        details = {}
        for row in csv_rows(path):
            sequence = "COMBINED_SEQ" if row["seq"] == "COMBINED" else row["seq"]
            if sequence in details:
                raise ValueError(f"Repeated detailed metric row: {arm}/{sequence}")
            details[sequence] = row
        if set(details) != set(result[arm]):
            raise ValueError(f"Detailed TrackEval coverage mismatch: {arm}")
        for sequence, row in details.items():
            for metric, field in DETAIL_COLUMNS.items():
                value = _number(row[field], f"detailed/{arm}/{sequence}/{metric}", metric in COUNTS,
                                metric != "MOTA")
                _close(result[arm][sequence][metric], value * (100 if metric in PERCENT else 1),
                       f"detailed/{arm}/{sequence}/{metric}")
    path = batch / "metrics/comparison.json"
    if path.exists():
        comparison = read(path)
        if "sequences" in comparison and set(comparison["sequences"]) != set(sequences):
            raise ValueError("Official comparison sequence scope mismatch")
        seen = set()
        for row in comparison["results"]:
            arm = row["tracker"]
            if arm not in result or arm in seen:
                raise ValueError("Official comparison arm coverage mismatch")
            seen.add(arm)
            for metric in METRICS:
                _close(result[arm]["COMBINED_SEQ"][metric], _number(row[metric], metric),
                       f"official comparison/{arm}/{metric}")
        if seen != set(arms):
            raise ValueError("Official comparison is incomplete")
    return result


def _verified(batch, arms, sequences):
    verified = read(batch / "verified.json")
    if verified.get("status") != "PASS" or set(verified.get("scope", ())) != set(sequences):
        raise ValueError("Prediction verification is missing or covers another scope")
    if set(verified.get("arms", {})) != set(arms):
        raise ValueError("Prediction verification arm coverage mismatch")
    for arm, record in verified["arms"].items():
        if set(record.get("per_sequence", {})) != set(sequences):
            raise ValueError(f"Prediction receipts incomplete: {arm}")
        totals = Counter()
        for sequence, detail in record["per_sequence"].items():
            counts = detail["counts"]
            if _number(counts.get("frames", 0), f"{arm}/{sequence}/frames", True, True) <= 0:
                raise ValueError(f"Empty prediction receipt: {arm}/{sequence}")
            for key, value in counts.items():
                totals[key] += _number(value, f"{arm}/{sequence}/{key}", nonnegative=True)
        for key in set(totals) | set(record["totals"]):
            _close(totals[key], _number(record["totals"].get(key, 0), key, nonnegative=True),
                   f"receipt total/{arm}/{key}")
    lengths = [{seq: row["per_sequence"][seq]["counts"]["frames"] for seq in sequences}
               for row in verified["arms"].values()]
    if any(value != lengths[0] for value in lengths[1:]):
        raise ValueError("Prediction arms cover different frame counts")
    return verified


def _attribution(batch, arms, sequences, metrics, verified):
    active = {arm for arm in arms if arm != _reference(arm)}
    document = read(batch / "attribution/summary.json")
    if document.get("status") != "COMPLETE" or set(document.get("variants", {})) != active:
        raise ValueError("CLEAR attribution is incomplete")
    records = {}
    for row in csv_rows(batch / "attribution/per_sequence.csv"):
        key = row["arm"], row["sequence"]
        if key in records or key[0] not in active or key[1] not in sequences:
            raise ValueError(f"Repeated or unexpected attribution row: {key}")
        records[key] = {k: _number(v, f"attribution/{key}/{k}", True,
                                    k not in ("fp_delta", "idsw_delta"))
                        for k, v in row.items() if k not in ("arm", "sequence")}
    if set(records) != {(arm, sequence) for arm in active for sequence in sequences}:
        raise ValueError("Per-sequence CLEAR attribution is incomplete")
    commits, commit_keys, committed = [], set(), Counter()
    for row in csv_rows(batch / "attribution/committed_clear.csv"):
        arm, sequence = row["arm"], row["sequence"]
        if arm not in active or sequence not in sequences or row["outcome"] not in OUTCOMES:
            raise ValueError("Unexpected committed CLEAR row")
        frame = _number(row["frame"], "committed/frame", True, True)
        if not 1 <= frame <= verified["arms"][arm]["per_sequence"][sequence]["counts"]["frames"]:
            raise ValueError("Committed event outside sequence")
        key = arm, sequence, frame, row["track_id"]
        if key in commit_keys:
            raise ValueError("A track has multiple committed weak updates in one frame")
        commit_keys.add(key)
        committed[arm, sequence] += 1
        committed[arm, sequence, row["outcome"]] += 1
        newly = _number(row["newly_covered"], "newly_covered", True, True)
        if newly not in (0, 1) or newly and row["outcome"] == "unmatched_FP":
            raise ValueError("Invalid newly-covered event")
        committed[arm, sequence, "newly_covered"] += newly
        commits.append(row)
    for arm in active:
        totals = Counter()
        for sequence in sequences:
            row = records[arm, sequence]
            current, old = metrics[arm][sequence], metrics[_reference(arm)][sequence]
            _close(row["gained"] - row["lost"], old["FN"] - current["FN"], f"coverage/{arm}/{sequence}")
            _close(row["fp_delta"], current["FP"] - old["FP"], f"FP/{arm}/{sequence}")
            _close(row["idsw_delta"], current["IDSW"] - old["IDSW"], f"IDSW/{arm}/{sequence}")
            expected = verified["arms"][arm]["per_sequence"][sequence]["counts"].get("committed", 0)
            _close(committed[arm, sequence], expected, f"commits/{arm}/{sequence}")
            for outcome in (*OUTCOMES, "newly_covered"):
                _close(row[outcome], committed[arm, sequence, outcome], f"outcome/{arm}/{sequence}/{outcome}")
            totals.update(row)
        archived = document["variants"][arm]
        if set(totals) != set(archived):
            raise ValueError(f"Attribution summary fields differ: {arm}")
        for key in totals:
            _close(totals[key], _number(archived[key], key, True), f"attribution sum/{arm}/{key}")
    for arm in set(arms) - active:
        if verified["arms"][arm]["totals"].get("committed", 0):
            raise ValueError(f"Off reference committed a weak update: {arm}")
    return document, records, commits


def _deltas(current, reference):
    return {"delta_" + metric: current[metric] - reference[metric] for metric in METRICS}


def _hint(row):
    """A directional reading, never an acceptance gate or significance test."""
    if all(abs(row["delta_" + key]) <= EPSILON for key in METRICS):
        return "NO_METRIC_CHANGE"
    if (row["delta_HOTA"] > EPSILON and row["delta_IDF1"] > EPSILON
            and row["delta_MOTA"] >= -EPSILON and row["delta_IDSW"] <= 0):
        return "FAVORABLE_DIRECTION_REVIEW_DISTRIBUTION"
    if (row["delta_HOTA"] < -EPSILON and row["delta_IDF1"] < -EPSILON
            and row["delta_MOTA"] <= EPSILON):
        return "UNFAVORABLE_DIRECTION"
    return "MIXED_TRADEOFF_REVIEW_COSTS"


def _csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _markdown(summary, comparison, components, interactions):
    lines = ["# W R4.1 完整跟踪链路评测", "",
             f"状态：`{summary['status']}`。覆盖 {summary['sequence_count']} 段、{summary['frames']:,} 帧、{len(comparison)} 个独立闭环组。", "",
             "指标来自 TrackEval 的 COMBINED_SEQ 聚合；HOTA、DetA、AssA、IDF1、MOTA 使用百分制，差值为百分点。",
             "每个 B 变体与 B_off 比较，每个 BG 变体与 BG_off 比较。逐序列百分数不用于重算总体指标。", "",
             "## 实测指标与各自参考的变化", "",
             "| 组别 | HOTA | IDF1 | MOTA | FP | FN | IDSW | ΔHOTA | ΔIDF1 | ΔMOTA | ΔFP | ΔFN | ΔIDSW |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    keys = ("HOTA", "IDF1", "MOTA", "FP", "FN", "IDSW")
    for row in comparison:
        cells = [row["arm"]]
        cells += [f"{row[k]:.4f}" if k in PERCENT else str(row[k]) for k in keys]
        cells += [f"{row['delta_'+k]:+.4f}" if k in PERCENT else f"{row['delta_'+k]:+d}" for k in keys]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## 组件差值", "",
              "同一参考内比较 W1−W0、W2−W1、W3−W2。闭环状态会随接纳决定分叉，差值包括后续间接影响。", "",
              "| 对比 | ΔHOTA | ΔIDF1 | ΔMOTA | ΔFP | ΔFN | ΔIDSW |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in components:
        lines.append("| " + row["comparison"] + " | " + " | ".join(
            f"{row['delta_'+k]:+.4f}" if k in PERCENT else f"{row['delta_'+k]:+d}" for k in keys) + " |")
    lines += ["", "## 与相机模块组合的数值变化", "",
              "计算 (BG_W − BG_off) − (B_W − B_off)，用于观察弱证据模块在两套基础上的增量差。",
              "这项数值对比本身不证明具体的协同机制；需要结合提交事件和 B 恢复事件解释。", "",
              "| 策略 | HOTA 增量差 | IDF1 增量差 | MOTA 增量差 | FP 增量差 | FN 增量差 | IDSW 增量差 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in interactions:
        lines.append("| " + row["policy"] + " | " + " | ".join(
            f"{row['interaction_'+k]:+.4f}" if k in PERCENT else f"{row['interaction_'+k]:+d}" for k in keys) + " |")
    lines += ["", "## 如何决定继续或停止", "",
              "先看完整评测的 HOTA、IDF1，再结合 MOTA、FP、FN、IDSW 和逐序列分布判断代价。",
              "报告中的方向提示是描述性辅助，不是自动科研准入规则，不会因为指标下降而阻止其他预定组完成。",
              "若指标完全不变，应结合 exposure.csv 检查规则是否改变候选资格或选择；不能把代码开关存在当作模块贡献。",
              "若仅个别序列贡献增益，需要保留完整配对表和失败案例。若主要指标下降或额外错误抵消收益，应停止当前固定方案。", "",
              "## 统计边界", "",
              "- 本批为已经多次查看的官方 30 段开发评测；没有把它重新声明为未触碰的盲测。",
              "- 当帧未匹配 FP 是官方 CLEAR 结果，不等价于人工确认背景。",
              "- 当帧提交、全程覆盖变化与后续 B 恢复变化分别记录，不能只用提交数量判断收益。",
              "- 既有 R3.1 的 unknown 不会在本报告中被自动改写成正确或错误。",
              "- W_H 是同状态历史可用性影子诊断，不是本批额外闭环实验组；其曝光计数不能冒充新的 TrackEval 指标。", "",
              "[总体指标](comparison.csv) · [逐序列配对](paired_per_sequence.csv) · [组件差值](component_deltas.csv) · [组件逐序列差值](component_per_sequence.csv) · [组合增量差](interactions.csv) · [诊断计数](exposure.csv) · [提交与覆盖归因](attribution.csv) · [核验摘要](summary.json)", ""]
    return "\n".join(lines)


def report(batch, *, expected_sequences=None, expected_arms=None, full=False):
    """Validate all prescribed runs and atomically write ``batch/report``.

    Re-running against unchanged inputs returns the existing, hash-verified
    summary. Existing reports with changed inputs are never silently replaced.
    ``full`` requires exactly 30 sequences; the runner also pins their names.
    """
    batch = Path(batch).resolve()
    raw_verified = read(batch / "verified.json")
    sequences = _unique_strings(expected_sequences if expected_sequences is not None else raw_verified["scope"], "sequences")
    arms = _unique_strings(expected_arms if expected_arms is not None else DEFAULT_ARMS, "arms")
    if full and len(sequences) != 30:
        raise ValueError("Full R4.1 reporting requires all 30 frozen sequences")
    if not set(DEFAULT_ARMS) <= set(arms):
        raise ValueError("All ten main R4.1 arms are required for a completed report")
    for arm in arms:
        if _reference(arm) not in arms:
            raise ValueError(f"Missing own reference for {arm}")
    verified = _verified(batch, arms, sequences)
    metrics = _metric_rows(batch, arms, sequences)
    attribution, attr_sequences, commits = _attribution(batch, arms, sequences, metrics, verified)

    comparison, paired, components, component_sequences, interactions, exposure = [], [], [], [], [], []
    for arm in arms:
        reference = _reference(arm)
        values = metrics[arm]["COMBINED_SEQ"]
        row = dict(arm=arm, reference=reference, **values,
                   **_deltas(values, metrics[reference]["COMBINED_SEQ"]))
        row["direction_hint"] = "REFERENCE" if arm == reference else _hint(row)
        comparison.append(row)
        for sequence in sequences:
            current, old = metrics[arm][sequence], metrics[reference][sequence]
            paired.append(dict(arm=arm, reference=reference, sequence=sequence, **current, **_deltas(current, old)))
            counts = verified["arms"][arm]["per_sequence"][sequence]["counts"]
            for counter, value in sorted(counts.items()):
                exposure.append(dict(arm=arm, sequence=sequence, counter=counter, value=value))
        for counter, value in sorted(verified["arms"][arm]["totals"].items()):
            exposure.append(dict(arm=arm, sequence="COMBINED_SEQ", counter=counter, value=value))
    for role in ROLES:
        for before, after in (("W0", "W1"), ("W1", "W2"), ("W2", "W3")):
            older, newer = f"{role}_{before}", f"{role}_{after}"
            identity = dict(comparison=f"{newer}_minus_{older}", reference=older, arm=newer)
            row = dict(**identity, **_deltas(metrics[newer]["COMBINED_SEQ"], metrics[older]["COMBINED_SEQ"]))
            row["direction_hint"] = _hint(row)
            components.append(row)
            component_sequences.extend(dict(**identity, sequence=sequence,
                                            **_deltas(metrics[newer][sequence], metrics[older][sequence]))
                                       for sequence in sequences)
    indexed = {row["arm"]: row for row in comparison}
    for policy in POLICIES[1:]:
        b, bg = indexed[f"B_{policy}"], indexed[f"BG_{policy}"]
        interactions.append(dict(policy=policy, **{"interaction_" + key: bg["delta_" + key] - b["delta_" + key]
                                                   for key in METRICS}))
    sequence_direction = {}
    for arm in arms:
        values = [row["delta_HOTA"] for row in paired if row["arm"] == arm]
        sequence_direction[arm] = dict(HOTA_improved=sum(v > EPSILON for v in values),
                                        HOTA_tied=sum(abs(v) <= EPSILON for v in values),
                                        HOTA_degraded=sum(v < -EPSILON for v in values))
    summary = dict(schema="wem_r41_report", status="FULL_EVALUATION_COMPLETE" if full else "SUBSET_EVALUATION_COMPLETE",
                   sequence_count=len(sequences), sequences=sequences, frames=verified["arms"][arms[0]]["totals"]["frames"],
                   arms=arms, aggregation="TrackEval COMBINED_SEQ; never average sequence percentages",
                   units=dict(percent_metrics="percentage points (0-100 scale)", count_metrics="official CLEAR counts",
                              exposure="counter-specific; raw edges and selected updates are distinct"),
                   comparison=comparison, component_deltas=components, interactions=interactions,
                   sequence_direction=sequence_direction, attribution=attribution["variants"],
                   checks=dict(prediction_receipts="PASS", metric_scope="PASS", combined_count_sums="PASS",
                               CLEAR_coverage_and_counts="PASS", committed_event_ledgers="PASS", own_reference_pairing="PASS"),
                   scientific_acceptance="NOT_AUTOMATICALLY_DECIDED", online_default="unchanged",
                   limitations=["Official evaluation split has already been used for development.",
                                "Directional hints are descriptive, not significance tests or automatic promotion gates.",
                                "CLEAR unmatched FP does not establish that the crop is human-confirmed background.",
                                "No frozen-state unknown label is relabeled by this report."])
    sources = ["verified.json", "metrics/summary.json", "attribution/summary.json", "attribution/per_sequence.csv",
               "attribution/committed_clear.csv"]
    sources += [str(path.relative_to(batch)) for path in sorted((batch / "metrics").glob("*/object_detailed.csv"))]
    sources += [path for path in ("metrics/comparison.json", "attribution/changed_B_events.csv", "plan.json")
                if (batch / path).exists()]
    input_hashes = {name: sha(batch / name) for name in sources}
    destination = batch / "report"
    if destination.exists():
        manifest = read(destination / "manifest.json")
        if manifest.get("inputs") != input_hashes or read(destination / "summary.json") != summary:
            raise ValueError("Existing report has different inputs or reporting settings; use a fresh run directory")
        for name, digest in manifest["outputs"].items():
            if sha(destination / name) != digest:
                raise ValueError(f"Existing report was changed: {name}")
        return summary
    with tempfile.TemporaryDirectory(prefix=".wem-report-", dir=batch) as temporary:
        staging = Path(temporary) / "report"
        staging.mkdir()
        _csv(staging / "comparison.csv", comparison, comparison[0].keys())
        _csv(staging / "paired_per_sequence.csv", paired, paired[0].keys())
        _csv(staging / "component_deltas.csv", components, components[0].keys())
        _csv(staging / "component_per_sequence.csv", component_sequences, component_sequences[0].keys())
        _csv(staging / "interactions.csv", interactions, interactions[0].keys())
        _csv(staging / "exposure.csv", exposure, ("arm", "sequence", "counter", "value"))
        attr_rows = [dict(arm=arm, **values) for arm, values in attribution["variants"].items()]
        _csv(staging / "attribution.csv", attr_rows, attr_rows[0].keys())
        _json(staging / "summary.json", summary)
        (staging / "REPORT.md").write_text(_markdown(summary, comparison, components, interactions), encoding="utf-8")
        _json(staging / "manifest.json", dict(schema="wem_r41_report_manifest", inputs=input_hashes,
                                             outputs={path.name: sha(path) for path in sorted(staging.iterdir())}))
        staging.rename(destination)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    result = report(args.batch, full=args.full)
    print(json.dumps({key: result[key] for key in ("status", "sequence_count", "frames", "arms")}, ensure_ascii=False))
