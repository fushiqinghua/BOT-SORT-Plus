"""Close the frozen E3 sample review without propagating labels to trajectories.

This standard-library helper is deliberately independent of the historical E3
scripts and their local ``audit_native_fp`` import.  It reads human labels; the
old ``analyze_signals.py`` used proxy labels and did not read that review CSV.
Nothing here trains a classifier, computes a background prevalence estimate,
or changes a tracker.  ``uncertain`` is a completed review, not a negative label.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


LABELS = (
    "confirmed_background", "possible_unlabelled_object", "true_object", "uncertain",
)
EDITABLE = ("human_label", "human_reviewer", "human_reason")
IDENTITY = ("case_id", "sequence", "original_video", "frame", "track_id", "det_index")
MACHINE_MAPPING = {
    "background_like": "confirmed_background",
    "object_like": "true_object",
    "possible_unlabelled_object": "possible_unlabelled_object",
    "uncertain": "uncertain",
}
SIGNALS = (
    "output_short_side", "output_score", "detection_short_side", "detection_score",
    "past15_observations", "past15_score_median", "past15_high_observations",
    "past15_recoveries", "frames_since_previous_measured", "gmc_speed_diag_per_frame",
    "rgb_valid", "rgb_quality", "rgb_cosine_to_previous_real_feature", "observed_now",
)
DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "results/vttiny/iterations/v0.4-motion-blur/round2-e1-e2-e3/E3"
)
SCOPE = "sampled_target_at_center_frame; adjacent_frames_are_context_only"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f"Missing or duplicate CSV columns: {path}")
        rows = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError(f"CSV row width does not match its header: {path}")
    return list(fields), rows


def _write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict]) -> None:
    # BOM is intentional for direct reading in Windows spreadsheet software.
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _output(root: Path, source: Path, names: Iterable[str]) -> None:
    root = root.resolve()
    if root == source or source in root.parents:
        raise ValueError("Output must be outside the frozen E3 source archive")
    for name in names:
        if (root / name).exists():
            raise FileExistsError(f"Refusing to overwrite {root / name}; choose a new output directory")
    root.mkdir(parents=True, exist_ok=True)


def _reference(source: Path) -> tuple[list[str], list[dict[str, str]], dict]:
    path = source / "B/E3_manual_review.csv"
    fields, rows = _rows(path)
    required = set(IDENTITY) | set(EDITABLE) | {"audit_stratum", "card_png"}
    if not required.issubset(fields):
        raise ValueError(f"Reference review is missing columns: {sorted(required - set(fields))}")
    ids = [row["case_id"] for row in rows]
    if not rows or any(not key for key in ids) or len(ids) != len(set(ids)):
        raise ValueError("Reference review must contain unique nonempty case IDs")
    contract_path = source / "B/review_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8-sig"))
    if int(contract["samples"]) != len(rows):
        raise ValueError("Reference sample count differs from review_contract.json")
    if int(contract["source_videos"]) != len({r["original_video"] for r in rows}):
        raise ValueError("Reference original-video count differs from review_contract.json")
    expected = contract.get("manual_review_sha256")
    if expected and _sha(path) != expected:
        raise ValueError("Frozen reference review checksum differs from review_contract.json")
    if set(contract["labels"]) != set(LABELS):
        raise ValueError("Reference label vocabulary differs from the supported E3 contract")
    return fields, rows, contract


def validate_review(fields: list[str], rows: list[dict[str, str]],
                    source_fields: list[str], source_rows: list[dict[str, str]]) -> dict:
    """Validate human declarations, never infer whether a named person is human."""
    errors = []
    if set(fields) != set(source_fields):
        errors.append({"type": "columns_changed", "expected": source_fields, "observed": fields})
    reference = {row["case_id"]: row for row in source_rows}
    ids = [row.get("case_id", "") for row in rows]
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        errors.append({"type": "duplicate_case_ids", "case_ids": duplicates})
    missing, extra = set(reference) - set(ids), set(ids) - set(reference)
    if missing or extra:
        errors.append({"type": "case_set_changed", "missing": sorted(missing), "extra": sorted(extra)})
    completed, pending = [], []
    counts = Counter()
    strata: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        case = row.get("case_id", "")
        if case not in reference:
            continue
        changed = [key for key in source_fields if key not in EDITABLE and row.get(key) != reference[case][key]]
        if changed:
            errors.append({"type": "frozen_fields_changed", "case_id": case, "fields": changed})
        label = row.get("human_label", "").strip()
        if label not in LABELS and label != "pending":
            errors.append({"type": "invalid_label", "case_id": case, "value": label})
            continue
        if label == "pending":
            pending.append(case)
            continue
        absent = [key for key in ("human_reviewer", "human_reason") if not row.get(key, "").strip()]
        if absent:
            errors.append({"type": "missing_review_metadata", "case_id": case, "fields": absent})
            continue
        completed.append(case)
        counts[label] += 1
        strata[reference[case]["audit_stratum"]][label] += 1
    return {
        "status": "INVALID_REVIEW" if errors else ("INCOMPLETE" if pending else "COMPLETE_SAMPLE_REVIEW"),
        "expected_samples": len(source_rows), "input_rows": len(rows),
        "source_videos": len({r["original_video"] for r in source_rows}),
        "completed_count": len(completed), "pending_count": len(pending),
        "completed_case_ids": completed, "pending_case_ids": pending,
        "label_counts": {label: counts[label] for label in LABELS},
        "by_stratum": {stratum: {label: count[label] for label in LABELS} for stratum, count in sorted(strata.items())},
        "errors": errors, "label_scope": SCOPE,
        "label_provenance": "human-declared; helper checks completeness, not reviewer identity",
        "research_decision": "INCONCLUSIVE_NO_ONLINE_FILTER_VALIDATION",
        "sampling": "purposive diagnostic samples; no population prevalence, AUC, training or trajectory labels",
    }


def _machine(source: Path, expected_ids: set[str]) -> dict[str, dict[str, str]]:
    path = source / "B/E3_machine_assisted_review.csv"
    if not path.exists():
        return {}
    fields, rows = _rows(path)
    if not {"case_id", "proposed_visual_label", "reviewer_kind", "visual_reason"}.issubset(fields):
        raise ValueError("Malformed machine-assisted review CSV")
    result = {}
    for row in rows:
        key = row["case_id"]
        if key in result or key not in expected_ids:
            raise ValueError(f"Duplicate or unknown machine-assisted case ID: {key}")
        result[key] = row
    return result


def _signal_key(row: dict[str, str]) -> tuple[str, int, int]:
    return row["sequence"], int(row["frame"]), int(row["track_id"])


def _join_signals(path: Path, reviews: list[dict[str, str]]) -> tuple[dict, dict]:
    """Join only the selected center frames, never every row sharing track_id."""
    wanted = {_signal_key(row): row for row in reviews}
    found = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        if not {"sequence", "frame", "track_id", "original_video"}.issubset(fields):
            raise ValueError("Signal CSV needs sequence/frame/track_id/original_video")
        selected = [name for name in SIGNALS if name in fields]
        if not selected:
            raise ValueError("Signal CSV has no supported current/past signal columns")
        for row in reader:
            key = _signal_key(row)
            if key not in wanted:
                continue
            if key in found:
                raise ValueError(f"Multiple causal rows match one review center: {key}")
            if row["original_video"] != wanted[key]["original_video"]:
                raise ValueError(f"Original-video mismatch in causal join: {key}")
            found[key] = {"signal_" + name: row.get(name, "") for name in selected}
    return found, {
        "source": str(path.resolve()), "sha256": _sha(path), "join_key": ["sequence", "frame", "track_id"],
        "scope": "exact reviewed center frames only", "matched_samples": len(found),
        "missing_case_ids": [r["case_id"] for r in reviews if _signal_key(r) not in found],
        "columns": ["signal_" + name for name in selected],
    }


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _comparison(rows: list[dict[str, str]], machine: dict, signals: dict) -> list[dict]:
    result = []
    for row in rows:
        item = {key: row[key] for key in IDENTITY}
        item.update({"audit_stratum": row["audit_stratum"], "label_scope": SCOPE,
                     **{key: row[key].strip() for key in EDITABLE}})
        suggestion = machine.get(row["case_id"], {})
        proposed = suggestion.get("proposed_visual_label", "")
        mapped = MACHINE_MAPPING.get(proposed)
        human = item["human_label"]
        comparison = ("pending" if human == "pending" else "machine_missing" if not suggestion
                      else "unmapped_machine_label" if mapped is None else "both_uncertain"
                      if human == mapped == "uncertain" else "same_category" if human == mapped
                      else "different_category")
        item.update({"machine_reviewer_kind": suggestion.get("reviewer_kind", ""),
                     "machine_proposed_label": proposed, "machine_mapped_category": mapped or "",
                     "machine_visual_reason": suggestion.get("visual_reason", ""),
                     "comparison": comparison, "causal_join": "not_joined"})
        key = _signal_key(row)
        if key in signals:
            item.update(signals[key])
            item["causal_join"] = "exact_center_match"
        result.append(item)
    return result


def _report_markdown(summary: dict, rows: list[dict]) -> str:
    lines = ["# E3 人工样例复核收尾", "", f"状态：**{summary['status']}**。", "",
             f"冻结样例 {summary['expected_samples']} 个，来自 {summary['source_videos']} 个原始视频。"
             f"已完成 {summary['completed_count']} 个，待完成 {summary['pending_count']} 个。", "",
             "本报告仅核对人工声明与样例身份、对照机器预筛，并展示可选的中心帧信号。"
             "人工复核完成不等于在线过滤机制通过验证；脚本不能核实复核者的现实身份。", "",
             "## 标签与适用范围", "",
             "标签只针对清单中中心帧的被审阅目标，相邻帧用作判断上下文。"
             "不传播至整条轨迹，不外推至所有无 GT 支持输出。"
             "uncertain 属于已完成的无法确定判断；possible_unlabelled_object 与 uncertain 均不作为背景负例。", "",
             "| 人工标签 | 样例数 |", "|---|---:|"]
    lines += [f"| {label} | {summary['label_counts'][label]} |" for label in LABELS]
    lines += ["", "## 逐样例对照", "",
              "机器建议是预筛意见，不作为真值。background_like 与 object_like 仅为展示对照，"
              "分别映射到 confirmed_background 与 true_object；映射不增加其证据强度。", "",
              "| 样例 | 序列 / 中心帧 / ID | 人工标签 | 机器建议 | 对照 | 人工依据 |",
              "|---|---|---|---|---|---|"]
    lines += [f"| {_md(r['case_id'])} | {_md(r['sequence'])} / {r['frame']} / {r['track_id']} | "
              f"{r['human_label']} | {_md(r['machine_proposed_label'])} | {r['comparison']} | "
              f"{_md(r['human_reason'])} |" for r in rows]
    lines += ["", "## 研究决定", "",
              "当前输出不自动批准训练、门限搜索或在线过滤。应结合人工依据检查：", "",
              "- 无 GT 支持样例中，哪些确为背景，哪些可能漏标，哪些无法确定。",
              "- 简单的低分、小尺寸、近静止规则是否会误伤真实目标。",
              "- 是否存在跨原始视频一致、当帧可获得的新区分证据。", "",
              "没有稳定新证据时，保留诊断结论并暂停过滤方向。出现新证据时，另行冻结一个受限的可分性实验，"
              "不把本批定向样例当作训练集、独立测试集或总体背景比例估计。", "",
              "## 与旧脚本的关系", "",
              "旧 analyze_signals.py 使用整轨无 GT 支持代理标签，未读取人工标签，manual_label_count 固定为 0。"
              "本工具独立读取人工表，不调用依赖本地 audit_native_fp 导入的旧脚本。"
              "本报告没有复写历史 E3-C，也没有将其代理 AUC 改称人工标签 AUC。", ""]
    if summary.get("causal_signals"):
        join = summary["causal_signals"]
        lines += ["## 可选信号联结", "",
                  f"仅匹配中心帧 sequence/frame/track_id：{join['matched_samples']} 个样例。"
                  f"缺失 {len(join['missing_case_ids'])} 个，按缺失保留。"
                  "信号附于 CSV 中，仅用于样例解释，不输出分类准确率或 AUC。", ""]
    return "\n".join(lines)


def run(mode: str, input_path: str | Path | None = None,
        output_dir: str | Path | None = None, source_root: str | Path | None = None,
        causal_signals: str | Path | None = None) -> dict:
    """Prepare, check or report the human review. All output files are exclusive."""
    if mode not in {"prepare", "check", "report"}:
        raise ValueError(f"Unknown review mode: {mode}")
    source = Path(source_root or DEFAULT_SOURCE).expanduser().resolve()
    fields, frozen, contract = _reference(source)
    source_csv = source / "B/E3_manual_review.csv"
    if mode == "prepare":
        if input_path is not None or causal_signals is not None:
            raise ValueError("prepare copies the frozen review; --input/--causal-signals are not used")
        if output_dir is None:
            raise ValueError("prepare requires --output-dir")
        out = Path(output_dir).expanduser().resolve()
        names = ["E3_manual_review.csv", "review_index.csv", "PREPARE.json", "READ_ME.md"]
        _output(out, source, names)
        _write_csv(out / names[0], fields, frozen)
        index = []
        for row in frozen:
            card = (source / "B" / row["card_png"]).resolve()
            if source not in card.parents:
                raise ValueError("Review card escaped the source archive")
            index.append({**{k: row[k] for k in IDENTITY}, "card_path": str(card),
                          "card_available": card.is_file(), "label_scope": SCOPE})
        _write_csv(out / names[1], index[0].keys(), index)
        result = {"status": "PREPARED", "samples": len(frozen), "source_videos": contract["source_videos"],
                  "source_root": str(source), "source_sha256": _sha(source_csv),
                  "editable_csv": str(out / names[0]), "prepared_sha256": _sha(out / names[0]),
                  "missing_cards": [r["case_id"] for r in index if not r["card_available"]],
                  "label_scope": SCOPE}
        _write_json(out / names[2], result)
        with (out / names[3]).open("x", encoding="utf-8") as stream:
            stream.write("# E3 人工复核\n\n仅编辑 E3_manual_review.csv 的 human_label、human_reviewer、human_reason 三列。\n\n"
                         "请先独立查看 review_index.csv 指向的原图、上下文与邻帧，再填写标签。"
                         "若原图缺失，先恢复本地 E3 证据包，不凭机器建议补填。\n\n"
                         "允许标签：confirmed_background、possible_unlabelled_object、true_object、uncertain。"
                         "尚未审阅保持 pending。uncertain 需说明为什么无法判断。\n\n"
                         "标签范围固定为该样例中心帧目标；相邻帧为上下文，不将标签扩展到整条轨迹。\n\n"
                         "填写后运行 check，再用新的输出目录运行 report。check/report 返回码 3 表示复核尚未完成，"
                         "2 表示表格或输入有误，0 表示该命令完成。所有输出均拒绝覆盖已有文件。\n")
        return result
    if input_path is None:
        raise ValueError(f"{mode} requires --input")
    path = Path(input_path).expanduser().resolve()
    review_fields, reviews = _rows(path)
    result = validate_review(review_fields, reviews, fields, frozen)
    result.update({"input": str(path), "input_sha256": _sha(path),
                   "source_root": str(source), "frozen_review_sha256": _sha(source_csv)})
    if mode == "check":
        if causal_signals is not None:
            raise ValueError("--causal-signals is only supported by report")
        if output_dir is not None:
            out = Path(output_dir).expanduser().resolve()
            _output(out, source, ["review_validation.json"])
            _write_json(out / "review_validation.json", result)
        return result
    if output_dir is None:
        raise ValueError("report requires --output-dir")
    out = Path(output_dir).expanduser().resolve()
    names = ["review_validation.json", "human_machine_comparison.csv", "E3_REVIEW_REPORT.md"]
    _output(out, source, names)
    if result["errors"]:
        _write_json(out / names[0], result)
        return result
    machine = _machine(source, {r["case_id"] for r in frozen})
    signals = {}
    if causal_signals is not None:
        signals, result["causal_signals"] = _join_signals(Path(causal_signals), reviews)
    comparison = _comparison(reviews, machine, signals)
    result["machine_comparison_counts"] = dict(Counter(row["comparison"] for row in comparison))
    result["machine_label_mapping"] = MACHINE_MAPPING
    machine_path = source / "B/E3_machine_assisted_review.csv"
    result["machine_review_sha256"] = _sha(machine_path) if machine_path.exists() else None
    columns = list(comparison[0])
    for row in comparison:
        for key in row:
            if key not in columns:
                columns.append(key)
    _write_csv(out / names[1], columns, comparison)
    with (out / names[2]).open("x", encoding="utf-8") as stream:
        stream.write(_report_markdown(result, comparison))
    _write_json(out / names[0], result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "check", "report"))
    parser.add_argument("--input", dest="input_path")
    parser.add_argument("--output-dir")
    parser.add_argument("--source-root", help="Frozen E3 directory containing B/E3_manual_review.csv")
    parser.add_argument("--causal-signals", help="Optional original causal-signal CSV or CSV.GZ, report only")
    args = parser.parse_args(argv)
    try:
        result = run(**vars(args))
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result["status"] == "INVALID_REVIEW" else 3 if result["status"] == "INCOMPLETE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
