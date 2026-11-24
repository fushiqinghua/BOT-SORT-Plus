"""Verify causal replay outputs, evaluate with the frozen evaluator, and compare.

Run after applying the overlay to the uploaded baseline repository. This is a
separate process from prediction: only this program reads ground truth.
Validation and comparison helpers are importable without importing TrackEval,
OpenCV, PyTorch or BoxMOT.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any


METRICS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "FP", "FN", "IDSW")
COUNTS = {"FP", "FN", "IDSW"}
PROTOCOL_SCHEMA = "botsort_plus_run_v1"
_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_SHA = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _plain_name(name: str) -> None:
    if not isinstance(name, str) or _NAME.fullmatch(name) is None:
        raise ValueError(f"Unsafe or invalid tracker/sequence name: {name!r}")


def _check_mot(path: Path, frame_count: int) -> dict:
    """Validate output format, monotonic frame order and per-frame ID ownership."""
    rows = 0
    last_frame = 0
    seen_ids: set[int] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.strip().split(",")
            if len(fields) != 10:
                raise ValueError(f"{path}:{line_number}: expected 10 MOT columns")
            try:
                values = [float(value) for value in fields]
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: nonnumeric MOT value") from error
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}:{line_number}: nonfinite MOT value")
            frame, identity, _, _, width, height, score, *_ = values
            if frame != int(frame) or not 1 <= frame <= frame_count:
                raise ValueError(f"{path}:{line_number}: frame outside declared video length")
            if identity != int(identity) or identity < 1:
                raise ValueError(f"{path}:{line_number}: track ID must be a positive integer")
            if width <= 0 or height <= 0 or not 0 <= score <= 1:
                raise ValueError(f"{path}:{line_number}: invalid box size or confidence")
            if frame < last_frame:
                raise ValueError(f"{path}:{line_number}: output frame order is not causal")
            if frame != last_frame:
                seen_ids.clear()
                last_frame = int(frame)
            if int(identity) in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate ID in one frame")
            seen_ids.add(int(identity))
            rows += 1
    return {"rows": rows, "last_output_frame": last_frame}


def validate_runs(run_dir: Path | str, trackers: list[str]) -> dict[str, Any]:
    """Reject incomplete, mixed-input, stale or malformed experiment outputs.

    No ground truth is read by this helper. A declared frame count includes empty
    detection and empty-output frames; the last emitted MOT frame need not equal
    the video length.
    """
    run_dir = Path(run_dir).resolve()
    if not trackers or len(trackers) != len(set(trackers)):
        raise ValueError("Select at least one unique tracker name")
    protocols = {}
    files = {}
    common_sequences = None
    common_identity = None
    common_subset = None
    common_fingerprints = None
    common_contracts = {}
    for tracker in trackers:
        _plain_name(tracker)
        protocol_path = run_dir / "protocols" / f"{tracker}.json"
        protocol = _read_json(protocol_path)
        if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("status") != "COMPLETE":
            raise ValueError(f"{tracker}: expected a COMPLETE {PROTOCOL_SCHEMA} protocol")
        if not isinstance(protocol.get("config"), dict) or protocol["config"].get("name") != tracker:
            raise ValueError(f"{tracker}: protocol config.name does not match requested tracker")
        if not isinstance(protocol.get("native_config"), dict):
            raise ValueError(f"{tracker}: missing native tracker configuration")
        sequences = protocol.get("sequences")
        if not isinstance(sequences, dict) or not sequences:
            raise ValueError(f"{tracker}: missing sequence frame counts")
        for sequence, count in sequences.items():
            _plain_name(sequence)
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError(f"{tracker}/{sequence}: frame count must be a positive integer")
        if not isinstance(protocol.get("subset"), bool):
            raise ValueError(f"{tracker}: subset must be an explicit boolean")
        subset = protocol["subset"]
        if not subset and len(sequences) != 30:
            raise ValueError(f"{tracker}: full VT-Tiny-MOT test evaluation requires all 30 sequences")
        identity = {}
        for key in ("detector_sha256", "rgb_encoder_sha256", "source_protocol_sha256"):
            value = protocol.get(key)
            if not isinstance(value, str) or _SHA.fullmatch(value) is None:
                raise ValueError(f"{tracker}: missing or invalid {key}")
            identity[key] = value
        if common_sequences is None:
            common_sequences = dict(sequences)
            common_identity = identity
            common_subset = subset
        elif sequences != common_sequences:
            raise ValueError(f"{tracker}: sequence sets or video frame counts differ across runs")
        elif identity != common_identity:
            changed = [key for key in identity if identity[key] != common_identity[key]]
            raise ValueError(f"{tracker}: mixed input identity: {', '.join(changed)}")
        elif subset != common_subset:
            raise ValueError(f"{tracker}: inconsistent subset declarations")
        fingerprints = protocol.get("detection_fingerprints")
        if not isinstance(fingerprints, dict) or set(fingerprints) != set(sequences):
            raise ValueError(f"{tracker}: detection fingerprints must cover exactly the declared sequences")
        if any(not isinstance(value, str) or _SHA.fullmatch(value) is None for value in fingerprints.values()):
            raise ValueError(f"{tracker}: invalid detection content fingerprint")
        if common_fingerprints is None:
            common_fingerprints = dict(fingerprints)
        elif fingerprints != common_fingerprints:
            raise ValueError(f"{tracker}: mixed detection content/NMS despite identical detector weights")
        feature_hashes = protocol.get("feature_hashes")
        if not isinstance(feature_hashes, dict) or set(feature_hashes) != set(sequences):
            raise ValueError(f"{tracker}: feature hashes must cover exactly the declared sequences")
        for sequence, hashes in feature_hashes.items():
            if not isinstance(hashes, dict) or not {"features", "metadata"} <= set(hashes):
                raise ValueError(f"{tracker}/{sequence}: missing feature array or metadata hash")
            for key in ("features", "metadata"):
                if not isinstance(hashes[key], str) or _SHA.fullmatch(hashes[key]) is None:
                    raise ValueError(f"{tracker}/{sequence}: invalid {key} hash")
        contracts = protocol.get("encoder_contracts")
        if not isinstance(contracts, dict) or "rgb" not in contracts or set(contracts) - {"rgb", "thermal"}:
            raise ValueError(f"{tracker}: explicit RGB/optional thermal encoder contracts are required")
        for modality, contract in contracts.items():
            if not isinstance(contract, dict):
                raise ValueError(f"{tracker}: malformed {modality} encoder contract")
            digest = contract.get("sha256")
            if not isinstance(digest, str) or _SHA.fullmatch(digest) is None:
                raise ValueError(f"{tracker}: {modality} encoder contract needs a valid sha256")
            if modality == "rgb" and digest != identity["rgb_encoder_sha256"]:
                raise ValueError(f"{tracker}: RGB contract disagrees with rgb_encoder_sha256")
            if modality in common_contracts and contract != common_contracts[modality]:
                raise ValueError(f"{tracker}: mixed {modality} encoder/preprocessing contract")
            common_contracts[modality] = contract
        # RGB-only baseline caches and RGB+thermal C caches may differ in bytes;
        # shared modalities must retain the exact same encoder/preprocessing
        # contract. Array/metadata hashes remain recorded per run, not equated.
        appearance = protocol["config"].get("appearance", {})
        if (isinstance(appearance, dict) and appearance.get("enabled", False)
                and appearance.get("weight_thermal", 1.0) > 0 and "thermal" not in contracts):
            raise ValueError(f"{tracker}: C requests thermal appearance without a thermal encoder contract")
        predictions = protocol.get("output_prediction_hashes")
        if not isinstance(predictions, dict) or set(predictions) != set(sequences):
            raise ValueError(f"{tracker}: prediction hashes must cover exactly the declared sequences")
        data_dir = run_dir / "trackers" / tracker / "data"
        actual_names = {path.stem for path in data_dir.glob("*.txt")}
        if actual_names != set(sequences):
            raise ValueError(f"{tracker}: missing or extra MOT prediction files")
        files[tracker] = {}
        for sequence in sorted(sequences):
            path = data_dir / f"{sequence}.txt"
            expected = predictions[sequence]
            if not isinstance(expected, str) or _SHA.fullmatch(expected) is None or sha256(path) != expected:
                raise ValueError(f"{tracker}/{sequence}: prediction SHA256 mismatch")
            files[tracker][sequence] = {"sha256": expected,
                                       **_check_mot(path, sequences[sequence])}
        protocols[tracker] = {**protocol, "validated_protocol_sha256": sha256(protocol_path)}
    return {"run_dir": str(run_dir), "trackers": list(trackers),
            "sequences": common_sequences, "subset": common_subset,
            "detection_fingerprints": common_fingerprints, "encoder_contracts": common_contracts,
            **common_identity, "protocols": protocols, "prediction_files": files}


def validate_gt(gt_root: Path | str, sequences: dict[str, int]) -> dict:
    """Verify supplied RGB GT files are present and record their exact bytes.

    File presence/hash is not a claim that arbitrary caller-supplied files are
    official. Use the original unmodified RGB GT directory from the baseline.
    Empty GT files are valid and are deliberately retained.
    """
    root = Path(gt_root).resolve()
    hashes = {}
    for sequence in sorted(sequences):
        path = root / sequence / "gt" / "gt.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing RGB ground truth: {path}")
        hashes[sequence] = sha256(path)
    return {"root": str(root), "sha256": hashes,
            "provenance": "Caller-supplied official RGB GT; original bytes hashed, never modified"}


def verify_trackeval(trackeval_root: Path | str, repository_root: Path) -> dict:
    root = Path(trackeval_root).resolve()
    if not (root / "trackeval" / "__init__.py").is_file():
        raise FileNotFoundError(f"TrackEval package not found under {root}")
    manifest = repository_root / "provenance" / "trackeval.json"
    if not manifest.is_file():
        raise FileNotFoundError("Missing baseline provenance/trackeval.json; apply the overlay to the baseline repository")
    expected = _read_json(manifest)
    hashes = expected.get("package_sha256", {})
    if not hashes:
        raise ValueError("Baseline TrackEval provenance has no source hashes")
    verified = {}
    for relative, digest in hashes.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"TrackEval differs from the frozen evaluator: {relative}")
        data = path.read_bytes()
        raw = hashlib.sha256(data).hexdigest()
        match = "raw"
        if raw != digest:
            # The uploaded provenance was recorded from a CRLF checkout. GitHub
            # source ZIPs of the SAME commit normally use LF. Permit ONLY this
            # mechanical line-ending transformation; never strip whitespace,
            # alter source tokens, rewrite files or bypass source verification.
            lf = data.replace(b"\r\n", b"\n")
            if hashlib.sha256(lf).hexdigest() == digest:
                match = "lf_normalized"
            elif hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest() == digest:
                match = "crlf_normalized"
            else:
                raise ValueError(f"TrackEval differs from the frozen evaluator: {relative}")
        verified[relative] = {"raw_sha256": raw, "expected_sha256": digest, "match": match}
    return {"root": str(root), "commit": expected.get("commit"),
            "manifest_sha256": sha256(manifest), "verified_source_files": len(hashes),
            "eol_normalized_matched": any(row["match"] != "raw" for row in verified.values()),
            "source_files": verified}


def _metric_row(row: dict) -> dict:
    result = {}
    for name in METRICS:
        value = row.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Missing/nonfinite evaluation metric {name}")
        if name in COUNTS:
            if value != int(value) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            value = int(value)
        result[name] = value
    return result


def _summary_rows(summary: dict, tracker: str) -> tuple[dict, set[str]]:
    if summary.get("status") != "PASS" or summary.get("modality_reference") != "rgb" or summary.get("split") != "test":
        raise ValueError("Expected a PASS RGB official-test evaluator summary")
    rows = [row for row in summary.get("results", []) if row.get("tracker") == tracker]
    names = [row.get("sequence") for row in rows]
    if len(names) != len(set(names)) or names.count("COMBINED_SEQ") != 1:
        raise ValueError(f"{tracker}: summary needs unique sequence rows and one COMBINED_SEQ")
    combined = next(row for row in rows if row["sequence"] == "COMBINED_SEQ")
    return _metric_row(combined), set(names) - {"COMBINED_SEQ"}


def build_comparison(summary: dict, validation: dict, reference_tracker: str = "baseline",
                     reference_summary: Path | str | None = None,
                     reference_context: str | None = None,
                     reference_protocol: Path | str | None = None) -> dict:
    """Compare pooled metrics only under explicit and verified evaluation scope."""
    rows = []
    sequences = set(validation["sequences"])
    for tracker in validation["trackers"]:
        metrics, evaluated = _summary_rows(summary, tracker)
        if evaluated != sequences:
            raise ValueError(f"{tracker}: evaluator sequence set does not match validated predictions")
        rows.append({"tracker": tracker, **metrics})
    comparison = {
        "schema": "botsort_plus_comparison_v1", "status": "COMPLETE",
        "scope": "subset" if validation["subset"] else "full_official_test_development",
        "sequences": validation["sequences"], "frames": sum(validation["sequences"].values()),
        "aggregation": "TrackEval COMBINED_SEQ; never the arithmetic mean of sequence percentages",
        "selection_note": "历史官方测试集曾用于检测/ReID选模及门限校准；这些结果按开发实验解释，不能替代独立未见测试。",
        "delta_units": "HOTA/DetA/AssA/IDF1/MOTA: percentage points; FP/FN/IDSW: counts",
        "reference": {"tracker": reference_tracker, "status": "unavailable", "reason": None},
        "results": rows,
    }
    reference = comparison["reference"]
    reference_metrics = None
    if reference_summary is None:
        matched = [row for row in rows if row["tracker"] == reference_tracker]
        if matched:
            reference_metrics = _metric_row(matched[0])
            reference.update(status="comparable", kind="same_run_replay",
                             context="相同缓存、序列与评估器下重新运行的参考方法")
        else:
            reference["reason"] = "参考方法未包含在本次重放评估中；仅报告绝对指标。"
    else:
        path = Path(reference_summary).resolve()
        historical = _read_json(path)
        historical_metrics, historical_sequences = _summary_rows(historical, reference_tracker)
        reference.update(kind="historical_runtime", summary_path=str(path), summary_sha256=sha256(path),
                         context=reference_context)
        if validation["subset"] or len(sequences) != 30:
            reference["reason"] = "当前是子集；禁止与历史完整测试集 COMBINED_SEQ 计算差值。"
        elif historical_sequences != sequences or len(historical_sequences) != 30:
            reference["reason"] = "历史参考和当前结果不是同一组完整 30 序列。"
        elif not isinstance(reference_context, str) or not reference_context.strip():
            reference["reason"] = "历史参考需要 --reference-context 明确记录原运行环境/实验背景。"
        else:
            proto_path = (Path(reference_protocol).resolve() if reference_protocol is not None
                          else path.parent.parent / "protocol.json")
            if not proto_path.is_file():
                reference["reason"] = "缺少历史 protocol.json，无法确认检测、ReID和视频帧数一致；未计算差值。"
            else:
                proto = _read_json(proto_path)
                historical_counts = {}
                for sequence, records in proto.get("sequences", {}).items():
                    historical_counts[sequence] = len(records) if isinstance(records, list) else records
                same_inputs = (
                    historical_counts == validation["sequences"]
                    and proto.get("weight_sha256", proto.get("detector_sha256")) == validation["detector_sha256"]
                    and proto.get("reid_sha256", proto.get("rgb_encoder_sha256")) == validation["rgb_encoder_sha256"]
                    and sha256(proto_path) == validation["source_protocol_sha256"]
                )
                if not same_inputs:
                    reference["reason"] = "历史协议与当前检测、RGB ReID、帧数或来源协议 SHA256 不一致；未计算差值。"
                else:
                    reference_metrics = historical_metrics
                    reference.update(status="comparable", protocol_path=str(proto_path),
                                     protocol_sha256=sha256(proto_path),
                                     limitation="历史运行环境可能不同；优先同时评估本次重放 baseline。")
    if reference_metrics is not None:
        reference["metrics"] = reference_metrics
        for row in rows:
            row["delta"] = {name: row[name] - reference_metrics[name] for name in METRICS}
    return comparison


def write_comparison(output: Path | str, comparison: dict) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    has_delta = comparison["reference"]["status"] == "comparable"
    columns = ["tracker", *METRICS] + ([f"delta_{name}" for name in METRICS] if has_delta else [])
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in comparison["results"]:
            flat = {key: row[key] for key in ("tracker", *METRICS)}
            if has_delta:
                flat.update({f"delta_{name}": row["delta"][name] for name in METRICS})
            writer.writerow(flat)
    scope = "序列子集" if comparison["scope"] == "subset" else "完整官方测试集开发实验"
    lines = ["# BoT-SORT A/B/C 实验比较", "",
             f"评估范围：{scope}；{len(comparison['sequences'])} 个序列，{comparison['frames']} 帧。", "",
             "结果按 TrackEval COMBINED_SEQ 联合汇总，不对逐序列百分比取算术平均。", "",
             comparison["selection_note"], "",
             "| 方法 | HOTA↑ | DetA↑ | AssA↑ | IDF1↑ | MOTA↑ | FP↓ | FN↓ | IDSW↓ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in comparison["results"]:
        values = [str(row[name]) if name in COUNTS else f"{row[name]:.4f}" for name in METRICS]
        lines.append("| " + " | ".join([row["tracker"], *values]) + " |")
    lines += ["", "## 参考与差值", ""]
    ref = comparison["reference"]
    if has_delta:
        lines += [f"参考：`{ref['tracker']}`；{ref['context']}。", "",
                  "百分比指标的差值单位为百分点；计数指标为数量差。负的 FP/FN/IDSW 差值表示数量减少。", "",
                  "| 方法 | ΔHOTA | ΔDetA | ΔAssA | ΔIDF1 | ΔMOTA | ΔFP | ΔFN | ΔIDSW |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in comparison["results"]:
            values = [f"{row['delta'][name]:+d}" if name in COUNTS else f"{row['delta'][name]:+.4f}" for name in METRICS]
            lines.append("| " + " | ".join([row["tracker"], *values]) + " |")
        if ref.get("limitation"):
            lines += ["", ref["limitation"]]
    else:
        lines += [ref["reason"]]
    lines += ["", "各项改进是否有收益，应结合此表、逐序列结果、错误恢复/重复起轨日志和消融共同判断；本报告不预设收益。", ""]
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--trackers", nargs="+", required=True)
    parser.add_argument("--gt-root", type=Path, required=True, help="Original RGB GT root: ROOT/<seq>/gt/gt.txt")
    parser.add_argument("--trackeval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference-tracker", default="baseline")
    parser.add_argument("--reference-summary", type=Path)
    parser.add_argument("--reference-protocol", type=Path)
    parser.add_argument("--reference-context")
    args = parser.parse_args(argv)
    try:
        validation = validate_runs(args.run_dir, args.trackers)
        gt = validate_gt(args.gt_root, validation["sequences"])
        repository_root = Path(__file__).resolve().parents[1]
        evaluator = verify_trackeval(args.trackeval_root, repository_root)
        output = (args.output_dir or args.run_dir / "metrics").resolve()
        output.mkdir(parents=True, exist_ok=True)
        # These environment variables MUST precede import of the baseline's
        # vttiny_tracking_common/evaluate_vttiny modules.
        os.environ["TRACKEVAL_ROOT"] = str(args.trackeval_root.resolve())
        common = importlib.import_module("vttiny_tracking_common")
        if common.TRACK_EVAL.resolve() != args.trackeval_root.resolve():
            raise ValueError("Evaluator environment was imported earlier with a different TRACKEVAL_ROOT; run a fresh process")
        baseline = importlib.import_module("evaluate_vttiny")
        # Explicit SEQ_INFO makes the evaluator independent of a missing local
        # processed_v1 seqmap. All selected videos, including empty GT, remain.
        summary = baseline.evaluate("rgb", "test", args.run_dir.resolve() / "trackers",
                                    args.trackers, output, seqs=validation["sequences"],
                                    gt_folder=args.gt_root.resolve())
        comparison = build_comparison(summary, validation, args.reference_tracker,
                                      args.reference_summary, args.reference_context, args.reference_protocol)
        write_comparison(output, comparison)
        evidence = {"schema": "botsort_plus_evaluation_audit_v1", "status": "COMPLETE",
                    "validation": validation, "ground_truth": gt, "evaluator": evaluator,
                    "summary_sha256": sha256(output / "summary.json")}
        (output / "evaluation_audit.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "COMPLETE", "output_dir": str(output),
                          "scope": comparison["scope"], "reference": comparison["reference"]["status"]}, ensure_ascii=False))
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
