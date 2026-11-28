"""Report the five fixed W R4.2 arms after completed online evaluation.

This module consumes official TrackEval and offline CLEAR outputs. It cannot
read ground truth during tracking. It deliberately distinguishes an emitted
weak observation from a committed tracker-state update.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import tempfile

from tools.research.wem_r41.reporting import (
    COUNTS, EPSILON, METRICS, OUTCOMES, PERCENT, _close, _csv, _deltas, _hint,
    _json, _metric_rows, _number, _unique_strings, _verified, csv_rows, read, sha,
)


ROOT = Path(__file__).resolve().parents[3]
REFERENCE = "BG_off"
MAIN = "BG_W0"
RELAXED = "BG_W0_relaxed_anchor"
EMIT_ONLY = "BG_W0_emit_only"
LOWER_LOW = "BG_lower_low"
DEFAULT_ARMS = (REFERENCE, MAIN, LOWER_LOW, RELAXED, EMIT_ONLY)
CONTRASTS = (
    (MAIN, REFERENCE, "complete_W0_effect"),
    (MAIN, LOWER_LOW, "constrained_vs_global_low_threshold"),
    (RELAXED, MAIN, "relaxed_anchor_support"),
    (MAIN, EMIT_ONLY, "future_tracker_state_feedback"),
)
ANCHOR_COUNTERS = (
    "exposure_anchor_not_ready_bypassed_tracks",
    "exposure_anchor_prediction_bypassed_tracks",
    "exposure_anchor_box_bypassed_edges",
    "exposure_anchor_support_bypassed_edges",
    "anchor_support_bypassed_selected",
    "anchor_support_bypassed_committed",
    "exposure_anchor_guard_added_selected",
    "exposure_anchor_guard_removed_selected",
    "exposure_anchor_guard_changed_track_frames",
    "exposure_anchor_guard_added_eligible",
    "exposure_anchor_guard_removed_eligible",
)


def _full_scope(verified, sequences):
    """Require the actual frozen names and lengths, not just 30 arbitrary rows."""
    lengths = read(ROOT / "configs/research/wem_r4_1_protocol.json")["sequences"]
    if set(sequences) != set(lengths) or len(sequences) != 30 or sum(lengths.values()) != 12590:
        raise ValueError("Full R4.2 reporting requires the exact frozen 30 sequences / 12,590 frames")
    for arm, record in verified["arms"].items():
        actual = {seq: detail["counts"]["frames"] for seq, detail in record["per_sequence"].items()}
        if actual != lengths:
            raise ValueError(f"Frozen sequence lengths differ: {arm}")


def _weak_counts(arm, counts):
    """Old imported W0 receipts predate explicit weak-state counters."""
    emitted = _number(counts.get("committed", 0), f"{arm}/committed", True, True)
    expected_updates = emitted if arm in (MAIN, RELAXED) else 0
    if arm in (REFERENCE, LOWER_LOW) and emitted:
        raise ValueError(f"Arm without weak continuation emitted a weak output: {arm}")
    for name, expected in (("weak_outputs_emitted", emitted), ("weak_state_updates", expected_updates)):
        if name in counts:
            _close(_number(counts[name], f"{arm}/{name}", True, True), expected, f"{arm}/{name}")
    return dict(weak_outputs_emitted=emitted, weak_state_updates=expected_updates,
                count_basis="recorded_and_checked" if {"weak_outputs_emitted", "weak_state_updates"} <= set(counts)
                else "derived_from_committed_count_and_fixed_arm_semantics")


def _state_checks(verified, sequences):
    """Output-only must retain the reference canonical future state and events."""
    for arm, record in verified["arms"].items():
        _weak_counts(arm, record["totals"])
        for sequence, detail in record["per_sequence"].items():
            _weak_counts(arm, detail["counts"])
    for sequence in sequences:
        left = verified["arms"][REFERENCE]["per_sequence"][sequence]
        right = verified["arms"][EMIT_ONLY]["per_sequence"][sequence]
        for key in ("native_events", "states"):
            if not left.get(key) or not right.get(key) or left[key] != right[key]:
                raise ValueError(f"Output-only canonical native {key} differs from BG_off: {sequence}")
    return dict(emit_only_native_events="PASS", emit_only_canonical_native_state_chain="PASS",
                weak_output_vs_state_counts="PASS",
                state_hash_basis="emit_only hashes canonical native output, not its augmented emitted MOT")


def _attribution(batch, arms, sequences, metrics, verified):
    active = set(arms) - {REFERENCE}
    document = read(batch / "attribution/summary.json")
    if document.get("status") != "COMPLETE" or set(document.get("variants", {})) != active:
        raise ValueError("CLEAR attribution is incomplete")
    records = {}
    for row in csv_rows(batch / "attribution/per_sequence.csv"):
        key = row["arm"], row["sequence"]
        if key in records or key[0] not in active or key[1] not in sequences:
            raise ValueError(f"Repeated or unexpected attribution row: {key}")
        records[key] = {name: _number(value, f"attribution/{key}/{name}", True,
                                     name not in ("fp_delta", "idsw_delta"))
                        for name, value in row.items() if name not in ("arm", "sequence")}
    if set(records) != {(arm, seq) for arm in active for seq in sequences}:
        raise ValueError("Per-sequence CLEAR attribution is incomplete")
    committed, state_updates, track_keys, detection_keys = Counter(), Counter(), set(), set()
    for row in csv_rows(batch / "attribution/committed_clear.csv"):
        arm, sequence = row["arm"], row["sequence"]
        if arm not in active or sequence not in sequences or row["outcome"] not in OUTCOMES:
            raise ValueError("Unexpected weak-output CLEAR row")
        frame = _number(row["frame"], "weak-output/frame", True, True)
        length = verified["arms"][arm]["per_sequence"][sequence]["counts"]["frames"]
        if not 1 <= frame <= length:
            raise ValueError("Weak output outside sequence")
        track_key = arm, sequence, frame, row["track_id"]
        detection_key = arm, sequence, frame, row["det_index"]
        if track_key in track_keys or detection_key in detection_keys:
            raise ValueError("Weak-output ledger violates same-frame one-to-one matching")
        track_keys.add(track_key)
        detection_keys.add(detection_key)
        newly = _number(row["newly_covered"], "newly_covered", True, True)
        if newly not in (0, 1) or newly and row["outcome"] == "unmatched_FP":
            raise ValueError("Invalid newly-covered weak output")
        committed[arm, sequence] += 1
        committed[arm, sequence, row["outcome"]] += 1
        committed[arm, sequence, "newly_covered"] += newly
        if arm == EMIT_ONLY and (row.get("state_committed") != "0" or
                                 row.get("output_kind") != "emitted_measurement_no_state_feedback"):
            raise ValueError("Output-only CLEAR ledger must explicitly identify no state feedback")
        flag = _number(row.get("state_committed", 1), "state_committed", True, True)
        if flag not in (0, 1) or flag != (arm != EMIT_ONLY):
            raise ValueError("Weak-output state commitment disagrees with its arm")
        state_updates[arm, sequence] += flag
    for arm in active:
        total = Counter()
        for sequence in sequences:
            row = records[arm, sequence]
            old, new = metrics[REFERENCE][sequence], metrics[arm][sequence]
            _close(row["gained"] - row["lost"], old["FN"] - new["FN"], f"coverage/{arm}/{sequence}")
            _close(row["fp_delta"], new["FP"] - old["FP"], f"FP/{arm}/{sequence}")
            _close(row["idsw_delta"], new["IDSW"] - old["IDSW"], f"IDSW/{arm}/{sequence}")
            expected = verified["arms"][arm]["per_sequence"][sequence]["counts"].get("committed", 0)
            _close(committed[arm, sequence], expected, f"commits/{arm}/{sequence}")
            _close(state_updates[arm, sequence], 0 if arm == EMIT_ONLY else expected,
                   f"weak state updates/{arm}/{sequence}")
            for outcome in (*OUTCOMES, "newly_covered"):
                _close(row[outcome], committed[arm, sequence, outcome], f"outcome/{arm}/{sequence}/{outcome}")
            total.update(row)
        archived = document["variants"][arm]
        if set(total) != set(archived):
            raise ValueError(f"Attribution summary fields differ: {arm}")
        for key in total:
            _close(total[key], _number(archived[key], key, True), f"attribution sum/{arm}/{key}")
    return document


def _decision_evidence(verified, metrics, arm, reference, sequences):
    comparable = same = changed_rows = 0
    for sequence in sequences:
        old = verified["arms"][reference]["per_sequence"][sequence]
        new = verified["arms"][arm]["per_sequence"][sequence]
        if old.get("mot") and new.get("mot"):
            comparable += 1
            same += old["mot"] == new["mot"]
        changed_rows += any(abs(metrics[arm][sequence][key] - metrics[reference][sequence][key]) > EPSILON
                            for key in METRICS)
    if comparable == len(sequences):
        state = "TRAJECTORIES_IDENTICAL" if same == comparable else "TRAJECTORIES_DIFFER"
    else:
        state = "TRAJECTORY_HASH_COVERAGE_INCOMPLETE"
    return dict(trajectory_comparable_sequences=comparable, trajectory_identical_sequences=same,
                trajectory_changed_sequences=comparable - same, sequence_metric_rows_changed=changed_rows,
                decision_evidence=state)


def _markdown(summary):
    lines = ["# W R4.2 定型实验结果", "",
             f"状态：`{summary['status']}`。覆盖 {summary['sequence_count']} 段、每组 {summary['frames']:,} 帧、5 个实验组。", "",
             "总体指标采用 TrackEval COMBINED_SEQ；百分制指标差值为百分点，不取逐序列百分数平均。",
             "BG 是冻结的 evidence_both 复合底座；本批不把 BG−B 解释为纯 GMC 贡献。", "",
             "## 总体结果", "",
             "所有组均相对 BG_off。W0 为既有完整受约束续接；lower_low 只降低原生低分阈值；relaxed_anchor 放宽独立锚点支持；emit_only 仅输出弱观测，保留原生未来状态。", "",
             "| 组别 | HOTA | IDF1 | MOTA | ΔHOTA | ΔIDF1 | ΔMOTA | ΔFP | ΔFN | ΔIDSW |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["comparison"]:
        cells = [row["arm"]] + [f"{row[key]:.4f}" for key in ("HOTA", "IDF1", "MOTA")]
        cells += [f"{row['delta_'+key]:+.4f}" for key in ("HOTA", "IDF1", "MOTA")]
        cells += [f"{row['delta_'+key]:+d}" for key in COUNTS]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## 定型对照", "",
              "差值方向固定如下。每个组独立运行，后续候选和关联可随状态分叉；W0−emit_only 衡量状态反馈带来的完整影响，不单独证明某一 KF 公式的贡献。", "",
              "| 对比 | ΔHOTA | ΔIDF1 | ΔMOTA | ΔFP | ΔFN | ΔIDSW | 轨迹发生变化的序列 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["component_deltas"]:
        cells = [row["comparison"]]
        cells += [f"{row['delta_'+key]:+.4f}" for key in ("HOTA", "IDF1", "MOTA")]
        cells += [f"{row['delta_'+key]:+d}" for key in COUNTS]
        cells.append(str(row["trajectory_changed_sequences"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## 弱输出与状态更新", "",
              "旧事件名称 committed 在 emit_only 中表示弱观测输出已接纳，不表示未来跟踪状态被更新。下表把两者明确分开。", "",
              "| 组别 | 弱观测输出数 | 弱观测写入未来状态数 | 计数依据 |",
              "| --- | ---: | ---: | --- |"]
    for arm, value in summary["weak_activity"].items():
        lines.append(f"| {arm} | {value['weak_outputs_emitted']} | {value['weak_state_updates']} | {value['count_basis']} |")
    lines += ["", "## 放宽锚点后是否真的改变行为", "",
              "该消融移除 ready 条件、锚点与正式预测的一致性门、弱框与锚点的一致性门；保留最近常规观测、预算、所有权及其他安全约束。三个被移除条件作为一组解释，不将其效果分别认领。", "",
              "| 诊断计数 | 数量 |",
              "| --- | ---: |"]
    for key, value in summary["anchor_intervention"].items():
        lines.append(f"| {key} | {'未记录' if value is None else value} |")
    lines += ["", "anchor_guard 对照在 relaxed_anchor 当前状态下重算严格锚点策略，属于同状态反事实；它与独立运行的 W0 全程序列对比有不同含义。",
              "若放宽条件没有引入额外提交且轨迹相同，应记录本批未激活该差异；若轨迹改变而指标持平，应检查变化的收益与代价是否相抵。零差异不自动等于约束无用，也不能作为有效贡献。", "",
              "## 结果如何使用", "",
              "- W0 与 lower_low 需要同时看覆盖、FP、IDSW 和主指标，保留更高召回与更少误报之间的真实取舍。",
              "- relaxed_anchor 劣于 W0 支持整组锚点约束的价值；若优于 W0，应报告现有约束过严的证据，不用阈值搜索掩盖结果。",
              "- W0 优于 emit_only 支持未来状态反馈的价值；若相同，不能把当帧补框的收益全部解释成运动状态改善。",
              "- 指标下降不会阻止整批报告生成。方向提示是描述性辅助，没有统计显著或自动科研准入含义。", "",
              "## 复用与统计边界", "",
              "- BG_off、BG_W0 可以来自经过输入、代码和文件核验的 R4.1 完整运行；来源保存在 summary.json 的 reference_sources。复用不是新的重复实验。",
              "- emit_only 的 states 文件以原生输出计算，专用于核对未来状态与 BG_off 一致；不能把该哈希称为实际增强输出的状态哈希。",
              "- 导入参考与新运行的耗时条件可能不同，本报告不据此比较 FPS；部署速度应另行关闭诊断后测量。",
              "- 官方 30 段已经用于开发。本批没有新增独立盲测证据。CLEAR 未匹配 FP 不等价于人工确认背景。", "",
              "[总体指标](comparison.csv) · [逐序列配对](paired_per_sequence.csv) · [组件差值](component_deltas.csv) · [组件逐序列差值](component_per_sequence.csv) · [候选诊断](exposure.csv) · [归因](attribution.csv) · [核验摘要](summary.json)", ""]
    return "\n".join(lines)


def report(batch, *, expected_sequences=None, expected_arms=None, full=True):
    """Validate completed evaluation and publish an idempotent report directory."""
    batch = Path(batch).resolve()
    raw = read(batch / "verified.json")
    sequences = _unique_strings(expected_sequences if expected_sequences is not None else raw["scope"], "sequences")
    arms = _unique_strings(expected_arms if expected_arms is not None else DEFAULT_ARMS, "arms")
    if set(arms) != set(DEFAULT_ARMS):
        raise ValueError("All five fixed R4.2 arms, without extra arms, are required")
    arms = list(DEFAULT_ARMS)
    verified = _verified(batch, arms, sequences)
    if full:
        _full_scope(verified, sequences)
        for arm in arms:
            if not (batch / "metrics" / arm / "object_detailed.csv").is_file():
                raise ValueError(f"Full report requires original TrackEval detailed CSV: {arm}")
    state_checks = _state_checks(verified, sequences)
    metrics = _metric_rows(batch, arms, sequences)
    attribution = _attribution(batch, arms, sequences, metrics, verified)
    comparison, paired, exposure, components, component_sequences = [], [], [], [], []
    for arm in arms:
        values = metrics[arm]["COMBINED_SEQ"]
        row = dict(arm=arm, reference=REFERENCE, **values, **_deltas(values, metrics[REFERENCE]["COMBINED_SEQ"]))
        row["direction_hint"] = "REFERENCE" if arm == REFERENCE else _hint(row)
        comparison.append(row)
        for sequence in sequences:
            current = metrics[arm][sequence]
            paired.append(dict(arm=arm, reference=REFERENCE, sequence=sequence, **current,
                               **_deltas(current, metrics[REFERENCE][sequence])))
            for counter, value in sorted(verified["arms"][arm]["per_sequence"][sequence]["counts"].items()):
                exposure.append(dict(arm=arm, sequence=sequence, counter=counter, value=value))
        for counter, value in sorted(verified["arms"][arm]["totals"].items()):
            exposure.append(dict(arm=arm, sequence="COMBINED_SEQ", counter=counter, value=value))
    for arm, reference, question in CONTRASTS:
        identity = dict(comparison=f"{arm}_minus_{reference}", arm=arm, reference=reference, question=question)
        row = dict(**identity, **_deltas(metrics[arm]["COMBINED_SEQ"], metrics[reference]["COMBINED_SEQ"]),
                   **_decision_evidence(verified, metrics, arm, reference, sequences))
        row["direction_hint"] = _hint(row)
        components.append(row)
        component_sequences.extend(dict(**identity, sequence=seq, **_deltas(metrics[arm][seq], metrics[reference][seq]))
                                   for seq in sequences)
    weak_activity = {arm: _weak_counts(arm, verified["arms"][arm]["totals"]) for arm in arms}
    anchor_intervention = {key: verified["arms"][RELAXED]["totals"].get(key) for key in ANCHOR_COUNTERS}
    direction = {}
    for arm in arms:
        direction[arm] = {}
        for metric in ("HOTA", "IDF1", "MOTA"):
            values = [row["delta_" + metric] for row in paired if row["arm"] == arm]
            direction[arm][metric] = dict(increased=sum(value > EPSILON for value in values),
                                          tied=sum(abs(value) <= EPSILON for value in values),
                                          decreased=sum(value < -EPSILON for value in values))
    summary = dict(
        schema="wem_r42_report", status="FULL_EVALUATION_COMPLETE" if full else "SUBSET_EVALUATION_COMPLETE",
        sequence_count=len(sequences), sequences=sequences, frames=verified["arms"][REFERENCE]["totals"]["frames"],
        arms=arms, aggregation="TrackEval COMBINED_SEQ; never average sequence percentages",
        units=dict(percent_metrics="percentage points (0-100 scale)", count_metrics="official CLEAR counts",
                   exposure="counter-specific; do not sum sequence rows together with COMBINED_SEQ"),
        comparison=comparison, component_deltas=components, sequence_direction=direction,
        weak_activity=weak_activity, anchor_intervention=anchor_intervention,
        attribution=attribution["variants"],
        reference_sources={arm: verified["arms"][arm].get("provenance", {"kind": "not_recorded"}) for arm in (REFERENCE, MAIN)},
        checks=dict(prediction_receipts="PASS", metric_scope="PASS", combined_count_sums="PASS",
                    CLEAR_coverage_and_counts="PASS", weak_output_ledgers="PASS", BG_reference_pairing="PASS", **state_checks),
        scientific_acceptance="NOT_AUTOMATICALLY_DECIDED", production_default_changed=False,
        timing_comparison="NOT_COMPARABLE_WITH_IMPORTED_REFERENCE_TIMINGS",
        reporting_source_sha256=sha(__file__),
        limitations=["Official evaluation split has already been used for development.",
                     "Directional hints are descriptive, not significance tests or automatic promotion gates.",
                     "Relaxed anchor support removes a bundle of three guards, not three separate ablations.",
                     "W0 minus emit_only measures the entire future-state feedback effect, not one Kalman formula.",
                     "CLEAR unmatched FP does not establish that the crop is human-confirmed background.",
                     "Imported reference results are reused observations, not independent experimental repetitions."])
    sources = ["verified.json", "metrics/summary.json", "attribution/summary.json", "attribution/per_sequence.csv",
               "attribution/committed_clear.csv"]
    sources += [str(path.relative_to(batch)) for path in sorted((batch / "metrics").glob("*/object_detailed.csv"))]
    sources += [name for name in ("metrics/comparison.json", "attribution/changed_B_events.csv", "plan.json", "reference_import.json")
                if (batch / name).exists()]
    input_hashes = {name: sha(batch / name) for name in sources}
    destination = batch / "report"
    if destination.exists():
        manifest = read(destination / "manifest.json")
        if manifest.get("inputs") != input_hashes or read(destination / "summary.json") != summary:
            raise ValueError("Existing report has different inputs or reporting settings; use a fresh output directory")
        for name, digest in manifest["outputs"].items():
            if sha(destination / name) != digest:
                raise ValueError(f"Existing report was changed: {name}")
        return summary
    with tempfile.TemporaryDirectory(prefix=".wem-r42-report-", dir=batch) as temporary:
        staging = Path(temporary) / "report"
        staging.mkdir()
        for filename, rows in (("comparison.csv", comparison), ("paired_per_sequence.csv", paired),
                               ("component_deltas.csv", components), ("component_per_sequence.csv", component_sequences)):
            _csv(staging / filename, rows, rows[0].keys())
        _csv(staging / "exposure.csv", exposure, ("arm", "sequence", "counter", "value"))
        attr_rows = [dict(arm=arm, **attribution["variants"][arm], **weak_activity[arm]) for arm in arms if arm != REFERENCE]
        _csv(staging / "attribution.csv", attr_rows, attr_rows[0].keys())
        _json(staging / "summary.json", summary)
        (staging / "REPORT.md").write_text(_markdown(summary), encoding="utf-8")
        _json(staging / "manifest.json", dict(schema="wem_r42_report_manifest", inputs=input_hashes,
                                             outputs={path.name: sha(path) for path in sorted(staging.iterdir())}))
        staging.rename(destination)
    return summary
