"""D1/D2/D3: offline attribution of a strictly observational tracker replay.

GT labels are conservative independent-frame IoU audit labels, not TrackEval
metrics. No parameters are fitted and no algorithm recommendation is asserted
automatically. Unknown identity, absent annotation and right censoring remain
explicit instead of being silently counted as tracking failures.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
from scipy.stats import spearmanr

from botsort_plus.io import sha256
from tools.evaluate_botsort_plus import validate_runs, validate_gt
from .common import (box_iou_matrix, csv_writer, detection_tier,
                     diagnostic_assignment, iter_trace, load_gt, read_json,
                     size_bin, squared_mahalanobis, statistics, write_json, xywh)


DETECTION_FIELDS = [
    "sequence", "frame", "det_index", "gt_id", "gt_match_status", "gt_class",
    "gt_visibility", "x1", "y1", "x2", "y2", "score", "rgb_quality", "rgb_valid",
    "size_bin", "det_center_error_px", "det_center_error_norm", "det_width_error_px",
    "det_height_error_px", "detection_tier", "accepted_track_id", "accepted_stage",
]
TRACK_FIELDS = [
    "sequence", "frame", "track_id", "state_before", "last_seen", "gap_frames",
    "anchor_gt_id", "anchor_status", "gt_status", "gt_size_bin", "pred_center_error_px",
    "pred_center_error_norm", "pre_gmc_center_error_px", "gmc_error_change_px",
    "pred_width_error_px", "pred_height_error_px", "post_center_error_px",
    "det_available", "usable_det_available", "ambiguous_det_available", "best_det_iou",
    "gt_paired_det_index", "det_center_error_px", "det_center_error_norm",
    "accepted_det_index", "accepted_stage", "accepted_gt_id", "accepted_identity",
    "state_position_standardized_sq", "state_position_95_covered", "gt_paired_nis4",
    "gt_paired_nis4_95_covered", "output_status", "state_after",
    "paired_prediction_minus_detection_error_px",
]
FRAME_FIELDS = [
    "sequence", "frame", "gt_count", "detection_count", "output_count",
    "anchored_predictions", "pred_error_mean_px", "pred_error_median_px",
    "pre_gmc_error_mean_px", "gmc_error_change_mean_px", "common_residual_x_px",
    "common_residual_y_px", "common_residual_norm_px", "gmc_status", "quality_source",
    "feature_count", "tracked_count", "inlier_count_proxy", "inlier_ratio_proxy",
    "reprojection_median_px", "reprojection_p90_px", "spatial_coverage", "translation_px",
    "rotation_deg", "scale_factor", "fallback_reason",
]
RECOVERY_FIELDS = [
    "sequence", "frame", "track_id", "candidate_id", "det_index", "age", "hits",
    "previous_seen_frame", "previous_gt_id", "current_gt_id", "current_gt_match_status",
    "identity_status", "cost", "geometry_cost", "appearance_cost",
]
WINDOW_FIELDS = [
    "sequence", "recovery_frame", "track_id", "gt_id", "identity_status", "horizon",
    "available_followup_frames", "right_censored", "annotated_followup_frames",
    "unannotated_followup_frames", "same_id_correct_frames", "different_id_frames",
    "wrong_identity_frames", "ambiguous_output_frames", "missing_with_usable_detection",
    "missing_without_usable_detection", "missing_with_ambiguous_detection",
    "complete_observable_window", "stable_all_frames", "first_failure_offset",
    "first_failure_reason", "stable_annotated_fraction",
]
CHECK_FIELDS = [
    "sequence", "frame", "track_id", "candidate_id", "reason", "old_gt_id", "age",
    "hits", "cost", "geometry_cost", "appearance_cost", "appearance_hits", "ready",
    "two_hit_evidence", "two_hit_max_geometry", "two_hit_max_appearance",
]
ERROR_KEYS = (
    "pred_center_error_px", "pred_center_error_norm", "pre_gmc_center_error_px",
    "post_center_error_px", "det_center_error_px", "det_center_error_norm",
    "state_position_standardized_sq", "gt_paired_nis4",
    "paired_prediction_minus_detection_error_px",
)
COVERAGE_KEYS = ("state_position_95_covered", "gt_paired_nis4_95_covered")


def _value_summary(rows, key):
    return statistics(row.get(key) for row in rows)


def _accepted_updates(events):
    result = {}
    used = set()
    for event in events:
        if event["event"] not in ("match", "recovered", "native_birth", "birth_committed"):
            continue
        tid, det = int(event["track_id"]), int(event["det_index"])
        if tid in result or det in used:
            raise ValueError("More than one accepted measurement per track/detection in trace")
        result[tid] = {"det_index": det, "stage": event.get("stage", event["event"])}
        used.add(det)
    return result


def _read_mot(path, frames):
    result = {frame: [] for frame in range(1, frames + 1)}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            if row:
                values = [float(v) for v in row]
                result[int(values[0])].append(values[:7])
    return result


def _verify_trace_output(frame, output, mot):
    rows = [[frame, int(row[4]), row[0], row[1], row[2] - row[0], row[3] - row[1], row[5]]
            for row in output]
    a, b = np.asarray(rows).reshape(-1, 7), np.asarray(mot).reshape(-1, 7)
    if a.shape != b.shape or not np.allclose(a, b, rtol=0., atol=1e-9):
        raise ValueError(f"Trace output disagrees with verified MOT file at frame {frame}")


def _validate_reference(run_root, tracker, reference_root, reference_tracker):
    if reference_root is None:
        raise ValueError("An independent diagnostics-OFF reference run is required")
    observed = validate_runs(run_root, [tracker])
    reference = validate_runs(reference_root, [reference_tracker or tracker])
    name = reference_tracker or tracker
    a, b = observed["protocols"][tracker], reference["protocols"][name]
    if Path(run_root).resolve() == Path(reference_root).resolve() and tracker == name:
        raise ValueError("Observed run cannot serve as its own reference")
    keys = ["native_config", "sequences", "subset", "detection_fingerprints", "feature_hashes",
            "detector_sha256", "rgb_encoder_sha256", "source_protocol_sha256",
            "encoder_contracts", "implementation_sha256", "boxmot_source_hashes",
            "decoded_image_hashes"]
    for key in keys:
        if a.get(key) != b.get(key):
            raise ValueError(f"Reference differs at {key}; diagnosis requires identical inputs/code")
    images = a.get("decoded_image_hashes", {})
    if (set(images) != set(a["sequences"]) or any(not isinstance(v, str) or len(v) != 64
            or any(c not in "0123456789abcdef" for c in v) for v in images.values())):
        raise ValueError("Verified decoded image hashes must cover exactly the replay sequences")
    for key in ("implementation_sha256", "boxmot_source_hashes"):
        if not isinstance(a.get(key), dict) or not a[key]:
            raise ValueError(f"Missing replay source provenance: {key}")
    cfg_a, cfg_b = dict(a["config"]), dict(b["config"])
    cfg_a.pop("name", None); cfg_b.pop("name", None)
    if cfg_a != cfg_b:
        raise ValueError("Reference and diagnostic tracker configurations differ")
    if a["output_prediction_hashes"] != b["output_prediction_hashes"]:
        raise ValueError("Diagnostics changed MOT output bytes; abort attribution")
    for sequence in a["sequences"]:
        left = Path(run_root) / "events" / tracker / f"{sequence}.jsonl"
        right = Path(reference_root) / "events" / name / f"{sequence}.jsonl"
        if not left.is_file() or not right.is_file():
            raise ValueError("Independent --trace original event streams are required in both runs")
        if sha256(left) != sha256(right):
            raise ValueError(f"Diagnostics changed original tracking events for {sequence}")
    if b.get("diagnostics", {}).get("enabled"):
        raise ValueError("Reference must have diagnostics disabled")
    diagnostics = a.get("diagnostics", {})
    if (not diagnostics.get("enabled") or diagnostics.get("status") != "COMPLETE"
            or diagnostics.get("schema") != "botsort_motion_diagnostics_v1"):
        raise ValueError("Observed run requires COMPLETE diagnostics")
    if set(diagnostics.get("traces", {})) != set(a["sequences"]):
        raise ValueError("Diagnostic protocol trace set differs from replay sequences")
    return observed, reference


def _frame_state(frame, gt, detections, det_labels, det_status, output, output_labels,
                 output_status, snapshots, native, threshold):
    det_boxes = [row["xyxy"] for row in detections]
    ious = box_iou_matrix(det_boxes, [row["xyxy"] for row in gt])
    availability = {}
    for j, truth in enumerate(gt):
        overlaps = ious[:, j] >= threshold - np.finfo(float).eps
        usable = np.asarray([detection_tier(d["score"], native) != "outside_native_score_bands"
                             for d in detections], dtype=bool)
        eligible = overlaps & usable
        # A rejected duplicate does not erase a uniquely labelled usable detection.
        reliable = any(eligible[i] and det_labels.get(i) == truth["gt_id"]
                       for i in range(len(detections)))
        ambiguous = (not reliable and any(eligible[i] and det_status[i] == "ambiguous"
                                          for i in range(len(detections))))
        availability[truth["gt_id"]] = {
            "det_available": bool(overlaps.any()), "usable_det_available": bool(eligible.any()),
            "ambiguous_det_available": ambiguous,
            "best_det_iou": float(ious[:, j].max()) if len(detections) else 0.,
        }
    return {"frame": frame, "gt": {row["gt_id"]: row for row in gt},
            "availability": availability,
            "output_labels": {int(output[i][4]): gid for i, gid in output_labels.items()},
            "gt_output": {gid: int(output[i][4]) for i, gid in output_labels.items()},
            "output_status": {int(row[4]): output_status[i] for i, row in enumerate(output)},
            "states": {s["track_id"]: s["state"] for s in snapshots}}


def _output_status(state, tid, gid):
    if gid not in state["gt"]:
        return "gt_unannotated"
    label = state["output_labels"].get(tid)
    if label == gid:
        return "same_id_correct"
    if state["output_status"].get(tid) == "ambiguous":
        return "ambiguous_output"
    if label is not None:
        return "wrong_identity"
    if gid in state["gt_output"]:
        return "different_id"
    available = state["availability"][gid]
    if available["ambiguous_det_available"]:
        return "missing_with_ambiguous_detection"
    if available["usable_det_available"]:
        return "missing_with_usable_detection"
    return "missing_without_usable_detection"


def recovery_windows(sequence, recovery, states, horizons):
    """Follow-up includes t+1 through t+h; no absence is converted to a failure."""
    result = []
    t, tid, gid = recovery["frame"], recovery["track_id"], recovery["previous_gt_id"]
    for horizon in horizons:
        actual = [states[f] for f in range(t + 1, t + horizon + 1) if f in states]
        c = Counter()
        first_offset, first_reason = None, None
        for state in actual:
            status = (_output_status(state, tid, gid) if recovery["identity_status"] == "same_gt"
                      else "recovery_identity_unverified")
            c[status] += 1
            if status not in ("same_id_correct", "gt_unannotated", "ambiguous_output",
                              "recovery_identity_unverified") and first_offset is None:
                first_offset, first_reason = state["frame"] - t, status
        annotated = sum(gid in state["gt"] for state in actual) if gid is not None else 0
        complete = (recovery["identity_status"] == "same_gt" and len(actual) == horizon
                    and annotated == horizon and not c["ambiguous_output"])
        result.append({
            "sequence": sequence, "recovery_frame": t, "track_id": tid, "gt_id": gid,
            "identity_status": recovery["identity_status"], "horizon": horizon,
            "available_followup_frames": len(actual), "right_censored": len(actual) < horizon,
            "annotated_followup_frames": annotated,
            "unannotated_followup_frames": len(actual) - annotated,
            "same_id_correct_frames": c["same_id_correct"], "different_id_frames": c["different_id"],
            "wrong_identity_frames": c["wrong_identity"], "ambiguous_output_frames": c["ambiguous_output"],
            "missing_with_usable_detection": c["missing_with_usable_detection"],
            "missing_without_usable_detection": c["missing_without_usable_detection"],
            "missing_with_ambiguous_detection": c["missing_with_ambiguous_detection"],
            "complete_observable_window": complete,
            "stable_all_frames": c["same_id_correct"] == horizon if complete else None,
            "first_failure_offset": first_offset, "first_failure_reason": first_reason,
            "stable_annotated_fraction": c["same_id_correct"] / annotated
                if recovery["identity_status"] == "same_gt" and annotated else None,
        })
    return result


def _correlation(rows, x, y):
    pairs = [(r.get(x), r.get(y)) for r in rows if r.get(x) is not None and r.get(y) is not None]
    if len(pairs) < 3:
        return {"n": len(pairs), "spearman": None}
    a, b = np.asarray(pairs, dtype=float).T
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite correlation input")
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return {"n": len(pairs), "spearman": None}
    return {"n": len(pairs), "spearman": float(spearmanr(a, b).statistic)}


def analyze_run(run_root, tracker_name, gt_root, output_dir, *, iou_threshold=.5,
                ambiguity_margin=.05, horizons=(3, 5, 10), reference_run_root=None,
                reference_tracker_name=None):
    """Write the three GT-based diagnostic groups after verifying replay identity.

    Input run trees follow ``tools/run_botsort_plus.py``. No default GT path,
    fitted thresholds, future-assisted identity anchors or forced conclusions
    are introduced. ``reference_run_root`` is mandatory and independent.
    """
    horizons = tuple(horizons)
    if (not horizons or len(set(horizons)) != len(horizons)
            or any(isinstance(h, bool) or not isinstance(h, int) or h < 1 for h in horizons)):
        raise ValueError("Horizons must be unique positive frame counts")
    if not 0 < iou_threshold <= 1 or not 0 <= ambiguity_margin < 1:
        raise ValueError("Invalid diagnostic overlap thresholds")
    run_root, output_dir = Path(run_root).resolve(), Path(output_dir).resolve()
    observed, reference = _validate_reference(run_root, tracker_name, reference_run_root,
                                               reference_tracker_name)
    protocol = observed["protocols"][tracker_name]
    ground_truth = validate_gt(gt_root, protocol["sequences"])
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema": "botsort_diagnostic_analysis_v1", "status": "RUNNING",
        "tracker": tracker_name, "run_root": str(run_root),
        "run_protocol_sha256": sha256(run_root / "protocols" / f"{tracker_name}.json"),
        "reference_run_root": str(Path(reference_run_root).resolve()),
        "reference_tracker": reference_tracker_name or tracker_name,
        "reference_protocol_sha256": reference["protocols"][reference_tracker_name or tracker_name]
            ["validated_protocol_sha256"],
        "identical_prediction_bytes": True, "identical_original_event_bytes": True,
        "original_events_sha256": {s: sha256(run_root / "events" / tracker_name / f"{s}.jsonl")
                                    for s in protocol["sequences"]},
        "analysis_source_sha256": {str(p.relative_to(Path(__file__).resolve().parents[2])): sha256(p)
            for p in (Path(__file__).resolve(), Path(__file__).resolve().with_name("common.py"),
                      Path(__file__).resolve().parents[2] / "tools/evaluate_botsort_plus.py")},
        "ground_truth": ground_truth,
        "sequences": protocol["sequences"], "iou_threshold": iou_threshold,
        "ambiguity_margin": ambiguity_margin, "horizons": list(horizons),
        "metric_scope": "diagnostic independent-frame IoU labels; NOT TrackEval HOTA/IDSW/FP/FN",
        "identity_anchor": "previous accepted real detection at snapshot.last_seen only; never current/future GT",
        "gt_policy": "all classes pooled; zero-marked rows removed; absent annotation remains unknown",
        "limitations": [
            "GMC statistics are deterministic optical-flow proxies, not the native RANSAC inlier mask.",
            "GMC before/after residual is same-state diagnostic, not an independent no-GMC tracker ablation.",
            "GMC correlations do not establish causality; common residual includes common target motion.",
            "State covariance/innovation coverage is diagnostic; GT has annotation noise and anchors are selected.",
            "NIS uses GT-paired available detections including unused detections, not only accepted matches.",
            "B endpoint identity is local evidence, not a guarantee of pure appearance or the whole trajectory.",
            "Missing GT, ambiguity and end-of-video censoring are excluded from fully observable stability windows.",
            "Historical official test was used in design; these are development diagnostics, not blind generalization.",
        ],
    }
    write_json(output_dir / "summary.json", summary)
    # Store numeric columns for exact quantiles, not every large track dictionary.
    d1_samples, d1_sizes, d1_coverage = defaultdict(list), defaultdict(list), defaultdict(Counter)
    all_detection_errors = defaultdict(list)
    anchor_counts, accepted_counts = Counter(), Counter()
    all_frames, all_recoveries, all_windows = [], [], []
    detection_counts, sequence_counts = Counter(), {}
    try:
        with ExitStack() as stack:
            det_writer = csv_writer(output_dir / "detection_audit.csv", DETECTION_FIELDS, stack)
            track_writer = csv_writer(output_dir / "track_error.csv", TRACK_FIELDS, stack)
            frame_writer = csv_writer(output_dir / "frame_error.csv", FRAME_FIELDS, stack)
            recovery_writer = csv_writer(output_dir / "recovery_events.csv", RECOVERY_FIELDS, stack)
            window_writer = csv_writer(output_dir / "recovery_windows.csv", WINDOW_FIELDS, stack)
            check_writer = csv_writer(output_dir / "recovery_checks.csv", CHECK_FIELDS, stack)
            for sequence, count in sorted(protocol["sequences"].items()):
                trace_path = run_root / "diagnostics" / tracker_name / f"{sequence}.jsonl.gz"
                manifest_path = trace_path.with_suffix("").with_suffix(".json")
                declaration = protocol["diagnostics"]["traces"][sequence]
                if declaration.get("status") != "COMPLETE" or declaration.get("records") != count:
                    raise ValueError(f"{sequence}: incomplete trace declaration")
                if declaration.get("manifest_sha256") != sha256(manifest_path):
                    raise ValueError(f"{sequence}: diagnostic manifest differs from run protocol")
                if declaration.get("data_sha256") != sha256(trace_path):
                    raise ValueError(f"{sequence}: diagnostic trace differs from run protocol")
                manifest = read_json(manifest_path)
                metadata = manifest.get("metadata", {})
                if (metadata.get("detection_fingerprint") != protocol["detection_fingerprints"][sequence]
                        or metadata.get("feature_hashes") != protocol["feature_hashes"][sequence]
                        or metadata.get("native_config") != protocol["native_config"]
                        or any(metadata.get(k) != protocol.get(k) for k in
                               ("source_protocol_sha256", "boxmot_source_hashes", "implementation_sha256"))):
                    raise ValueError(f"{sequence}: trace input lineage differs from replay")
                truth, ignored = load_gt(Path(gt_root) / sequence / "gt" / "gt.txt", count)
                mot = _read_mot(run_root / "trackers" / tracker_name / "data" / f"{sequence}.txt", count)
                native = protocol["native_config"]
                accepted_gt, states, sequence_recoveries = {}, {}, []
                image_digest = hashlib.sha256()
                seq_counts = Counter(ignored_zero_marked_gt_rows=ignored)
                for record in iter_trace(trace_path, sequence, count):
                    frame = record["frame"]
                    image_digest.update((json.dumps({"frame": frame,
                        "shape": record["image_shape"], "dtype": record["image_dtype"],
                        "sha256": record["image_sha256"]}, sort_keys=True,
                        separators=(",", ":")) + "\n").encode("utf-8"))
                    gt, detections = truth[frame], record["detections"]
                    gt_by_id = {g["gt_id"]: g for g in gt}
                    det_labels, det_status, det_iou, det_pairs = diagnostic_assignment(
                        [d["xyxy"] for d in detections], gt, iou_threshold, ambiguity_margin)
                    output = np.asarray(record["output"], dtype=float).reshape(-1, 8)
                    _verify_trace_output(frame, output, mot[frame])
                    out_labels, out_status, _, _ = diagnostic_assignment(
                        output[:, :4], gt, iou_threshold, ambiguity_margin)
                    accepted = _accepted_updates(record["events"])
                    if any(u["det_index"] < 0 or u["det_index"] >= len(detections)
                           for u in accepted.values()):
                        raise ValueError("Accepted event detection index outside trace")
                    used = {u["det_index"]: (tid, u["stage"]) for tid, u in accepted.items()}
                    post = {s["track_id"]: s for s in record["after_update"]}
                    pre = {s["track_id"]: s for s in record["after_prediction"]}
                    before = {s["track_id"]: s for s in record["before_prediction"]}
                    state = _frame_state(frame, gt, detections, det_labels, det_status,
                                         output, out_labels, out_status, record["after_update"], native,
                                         iou_threshold)
                    states[frame] = state
                    det_for_gt = {gid: index for index, gid in det_labels.items()}
                    for i, detection in enumerate(detections):
                        gid = det_labels.get(i)
                        target = gt_by_id.get(gid)
                        box = detection["xyxy"]
                        error = xywh(box) - xywh(target["xyxy"]) if target else None
                        norm = float(min(xywh(target["xyxy"])[2:])) if target else None
                        tid, stage = used.get(i, (None, None))
                        det_writer.writerow({
                            "sequence": sequence, "frame": frame, "det_index": i, "gt_id": gid,
                            "gt_match_status": det_status[i],
                            "gt_class": target["gt_class"] if target else None,
                            "gt_visibility": target["gt_visibility"] if target else None,
                            **dict(zip(("x1", "y1", "x2", "y2"), box)),
                            "score": detection["score"], "rgb_quality": detection.get("rgb_quality"),
                            "rgb_valid": detection.get("rgb_valid"), "size_bin": size_bin(box),
                            "det_center_error_px": float(np.linalg.norm(error[:2])) if target else None,
                            "det_center_error_norm": float(np.linalg.norm(error[:2]) / norm) if target else None,
                            "det_width_error_px": float(error[2]) if target else None,
                            "det_height_error_px": float(error[3]) if target else None,
                            "detection_tier": detection_tier(detection["score"], native),
                            "accepted_track_id": tid, "accepted_stage": stage,
                        })
                        detection_counts[det_status[i]] += 1
                        seq_counts["detections_" + det_status[i]] += 1
                        if target:
                            all_detection_errors["center_error_px"].append(float(np.linalg.norm(error[:2])))
                            all_detection_errors["center_error_norm"].append(float(np.linalg.norm(error[:2]) / norm))
                    frame_tracks, residuals = [], []
                    for prediction in record["after_gmc"]:
                        tid, last_seen = prediction["track_id"], prediction["last_seen"]
                        if last_seen >= frame:
                            raise ValueError("Prediction history contains current/future measurement")
                        gid = accepted_gt.get((last_seen, tid))
                        target = gt_by_id.get(gid)
                        item = {key: None for key in TRACK_FIELDS}
                        item.update(sequence=sequence, frame=frame, track_id=tid,
                                    state_before=before.get(tid, prediction)["state"], last_seen=last_seen,
                                    gap_frames=frame - last_seen, anchor_gt_id=gid,
                                    anchor_status="verified_previous_observation" if gid is not None else "unknown",
                                    gt_status="present" if target else "unannotated" if gid is not None else "unknown_identity",
                                    state_after=post.get(tid, {}).get("state"))
                        update = accepted.get(tid)
                        if update:
                            current_gt = det_labels.get(update["det_index"])
                            item.update(accepted_det_index=update["det_index"], accepted_stage=update["stage"],
                                        accepted_gt_id=current_gt,
                                        accepted_identity="unknown" if gid is None or current_gt is None
                                        else "same" if gid == current_gt else "different")
                        if target is not None:
                            measurement = xywh(target["xyxy"])
                            difference = xywh(prediction["xyxy"]) - measurement
                            error = float(np.linalg.norm(difference[:2]))
                            norm = float(min(measurement[2:]))
                            pre_error = float(np.linalg.norm(xywh(pre[tid]["xyxy"])[:2] - measurement[:2]))
                            residuals.append(difference[:2])
                            cov = np.asarray(prediction["covariance"], dtype=float)
                            state_d2 = squared_mahalanobis(difference[:2], cov[:2, :2])
                            item.update(gt_size_bin=size_bin(target["xyxy"]), pred_center_error_px=error,
                                        pred_center_error_norm=error / norm, pre_gmc_center_error_px=pre_error,
                                        gmc_error_change_px=error - pre_error,
                                        pred_width_error_px=float(difference[2]),
                                        pred_height_error_px=float(difference[3]),
                                        state_position_standardized_sq=state_d2,
                                        state_position_95_covered=state_d2 <= 5.991464547 if state_d2 is not None else None,
                                        output_status=_output_status(state, tid, gid), **state["availability"][gid])
                            if tid in post:
                                item["post_center_error_px"] = float(np.linalg.norm(
                                    xywh(post[tid]["xyxy"])[:2] - measurement[:2]))
                            di = det_for_gt.get(gid)
                            if di is not None:
                                detection = detections[di]
                                det_error = xywh(detection["xyxy"]) - measurement
                                innovation = xywh(detection["xyxy"]) - np.asarray(prediction["projected_mean"])
                                # projected covariance already includes R. Do not add R again.
                                nis = squared_mahalanobis(innovation, prediction["projected_covariance"])
                                item.update(gt_paired_det_index=di,
                                            det_center_error_px=float(np.linalg.norm(det_error[:2])),
                                            det_center_error_norm=float(np.linalg.norm(det_error[:2]) / norm),
                                            paired_prediction_minus_detection_error_px=error - float(np.linalg.norm(det_error[:2])),
                                            gt_paired_nis4=nis,
                                            gt_paired_nis4_95_covered=nis <= 9.487729037 if nis is not None else None)
                        track_writer.writerow(item)
                        frame_tracks.append(item)
                        for key in ERROR_KEYS:
                            if item[key] is not None:
                                d1_samples[key].append(item[key])
                        for key in COVERAGE_KEYS:
                            if item[key] is not None:
                                d1_coverage[key][bool(item[key])] += 1
                        if item["gt_size_bin"] is not None:
                            d1_sizes[item["gt_size_bin"]].append(item["pred_center_error_norm"])
                        anchor_counts[item["gt_status"]] += 1
                        accepted_counts[item["accepted_identity"] or "no_update"] += 1
                        seq_counts["prediction_" + item["gt_status"]] += 1
                    gmc = record["gmc"]
                    common = np.median(np.asarray(residuals), axis=0) if residuals else None
                    frow = {key: None for key in FRAME_FIELDS}
                    frow.update(sequence=sequence, frame=frame, gt_count=len(gt), detection_count=len(detections),
                                output_count=len(output), anchored_predictions=len(residuals),
                                pred_error_mean_px=_value_summary(frame_tracks, "pred_center_error_px")["mean"],
                                pred_error_median_px=_value_summary(frame_tracks, "pred_center_error_px")["median"],
                                pre_gmc_error_mean_px=_value_summary(frame_tracks, "pre_gmc_center_error_px")["mean"],
                                gmc_error_change_mean_px=_value_summary(frame_tracks, "gmc_error_change_px")["mean"],
                                common_residual_x_px=float(common[0]) if common is not None else None,
                                common_residual_y_px=float(common[1]) if common is not None else None,
                                common_residual_norm_px=float(np.linalg.norm(common)) if common is not None else None,
                                gmc_status=gmc.get("status"))
                    for key in FRAME_FIELDS[13:]:
                        if key != "gmc_status":
                            frow[key] = gmc.get(key)
                    frow["quality_source"] = gmc.get("quality_source", gmc.get("quality_method"))
                    frame_writer.writerow(frow); all_frames.append(frow)
                    for event in record["events"]:
                        if event["event"] == "recovered":
                            tid, di = int(event["track_id"]), int(event["det_index"])
                            previous_frame = frame - int(event["age"])
                            old_gid, current_gid = accepted_gt.get((previous_frame, tid)), det_labels.get(di)
                            identity = ("unknown" if old_gid is None or current_gid is None
                                        else "same_gt" if old_gid == current_gid else "different_gt")
                            rec = {key: event.get(key) for key in RECOVERY_FIELDS}
                            rec.update(sequence=sequence, frame=frame, previous_seen_frame=previous_frame,
                                       previous_gt_id=old_gid, current_gt_id=current_gid,
                                       current_gt_match_status=det_status[di], identity_status=identity)
                            recovery_writer.writerow(rec)
                            sequence_recoveries.append(rec); all_recoveries.append(rec)
                        elif event["event"] == "recovery_check":
                            check = {key: event.get(key) for key in CHECK_FIELDS}
                            tid = event.get("track_id")
                            snap = before.get(tid)
                            check.update(sequence=sequence, frame=frame,
                                         old_gt_id=accepted_gt.get((snap["last_seen"], tid)) if snap else None)
                            check_writer.writerow(check)
                    # Current/future GT never influences the anchor used above.
                    for tid, update in accepted.items():
                        accepted_gt[frame, tid] = det_labels.get(update["det_index"])
                if image_digest.hexdigest() != protocol.get("decoded_image_hashes", {}).get(sequence):
                    raise ValueError(f"{sequence}: trace decoded image identities differ from replay")
                for rec in sequence_recoveries:
                    windows = recovery_windows(sequence, rec, states, horizons)
                    window_writer.writerows(windows); all_windows.extend(windows)
                sequence_counts[sequence] = dict(seq_counts)
        d1 = {key: statistics(d1_samples[key]) for key in ERROR_KEYS}
        for key in COVERAGE_KEYS:
            counts = d1_coverage[key]; n = sum(counts.values())
            d1[key] = {"n": n, "fraction": counts[True] / n if n else None}
        d1["all_matched_detection_errors"] = {key: statistics(all_detection_errors[key])
                                                for key in ("center_error_px", "center_error_norm")}
        d1["by_gt_size"] = {size: statistics(d1_sizes[size])
                               for size in ("lt8", "8to16", "16to32", "ge32")}
        d1["anchor_status_counts"] = dict(anchor_counts)
        d1["accepted_identity_counts"] = dict(accepted_counts)
        d2 = {"status_counts": dict(Counter(r["gmc_status"] for r in all_frames)),
              "same_state_gmc_error_change_px": _value_summary(all_frames, "gmc_error_change_mean_px"),
              "correlations": {x: {y: _correlation(all_frames, x, y) for y in
                                    ("pred_error_mean_px", "common_residual_norm_px")}
                               for x in ("reprojection_median_px", "inlier_ratio_proxy", "spatial_coverage")}}
        d3 = {"recovery_identity_counts": dict(Counter(r["identity_status"] for r in all_recoveries)),
              "windows": {}}
        for horizon in horizons:
            rows = [r for r in all_windows if r["horizon"] == horizon]
            complete = [r for r in rows if r["complete_observable_window"]]
            d3["windows"][str(horizon)] = {
                "events": len(rows), "complete_observable_same_gt_events": len(complete),
                "stable_all_frames_count": sum(r["stable_all_frames"] for r in complete),
                "stable_all_frames_fraction": sum(r["stable_all_frames"] for r in complete) / len(complete)
                    if complete else None,
                "right_censored_events": sum(r["right_censored"] for r in rows),
                "unknown_recovery_identity_events": sum(r["identity_status"] == "unknown" for r in rows),
                "observed_first_failure_reasons_including_censored": dict(Counter(
                    r["first_failure_reason"] for r in rows if r["identity_status"] == "same_gt"
                    and r["first_failure_reason"])),
                "complete_window_first_failure_reasons": dict(Counter(
                    r["first_failure_reason"] for r in complete if r["first_failure_reason"])),
            }
        summary.update(status="COMPLETE", D1=d1, D2=d2, D3=d3,
                       detection_label_counts=dict(detection_counts), per_sequence=sequence_counts)
        summary["files_sha256"] = {p.name: sha256(p) for p in sorted(output_dir.glob("*.csv"))}
        write_json(output_dir / "summary.json", summary)
        (output_dir / "REPORT.md").write_text(_report(summary), encoding="utf-8")
        return summary
    except Exception:
        summary.update(status="FAILED", error=traceback.format_exc())
        write_json(output_dir / "summary.json", summary)
        raise


def _report(summary):
    def fmt(value):
        return "暂无可判定样本" if value is None else f"{value:.4f}"
    d1, d3 = summary["D1"], summary["D3"]
    rows = ["# 运动与身份恢复诊断报告", "", "本报告只描述本次离线诊断证据，未训练或修改跟踪模块。",
            "", "## 输入与完整性", "",
            f"- 配置：`{summary['tracker']}`。",
            "- 诊断开启与关闭的独立回放：预测文件逐字节一致。",
            "- GT：原始 RGB MOT 格式，所有类别合并，零标记行排除。",
            "- 诊断标签来自逐帧一对一 IoU 配对与歧义排除，不是 TrackEval 的官方指标。",
            "", "## D1：预测和检测误差", "",
            f"有历史身份锚点且当帧 GT 存在的预测数：{d1['pred_center_error_px']['n']}。",
            f"预测中心误差中位数：{fmt(d1['pred_center_error_px']['median'])} 像素。",
            f"对应可唯一配对检测的中心误差中位数：{fmt(d1['det_center_error_px']['median'])} 像素。",
            "", "两项误差的样本集合可能不同；请用 track_error.csv 中两项均存在的行进行配对比较。",
            "协方差覆盖只用于检查校准，不能忽略 GT 噪声、历史锚点选择与歧义排除造成的偏差。",
            "", "## D2：GMC 可靠性", "",
            "查看 frame_error.csv 的补偿前后同一状态误差变化、共同残差和光流质量代理。",
            "质量相关性并不证明因果；公共残差也可能含目标共同运动。此处没有运行独立的无 GMC 对照。",
            "", "## D3：B 恢复后稳定性", "",
            f"端点身份判定：`{d3['recovery_identity_counts']}`。", "",
            "| 后续帧数 | 完整可观测且同 GT 的事件 | 全程保持原 ID 的比例 |", "|---|---:|---:|"]
    for horizon, value in d3["windows"].items():
        rows.append(f"| {horizon} | {value['complete_observable_same_gt_events']} | {fmt(value['stable_all_frames_fraction'])} |")
    rows += ["", "视频结束、GT 未标注或输出身份歧义的窗口不计为完整失败窗口。",
             "缺少可用检测、存在可用检测却未输出、错误身份和另一 ID 覆盖分别记录。",
             "", "## 下一步人工判断", "",
             "1. 控制目标尺度与检测分数，判断定位噪声是否超过原生不确定性表达。",
             "2. 对 GMC 低质量片段查看原图，区分相机补偿问题与真实目标运动。",
             "3. 对同 GT 恢复后的首次失败检查真实检测是否存在，再判断是否值得重建运动状态。",
             "4. 结合独立 D4 报告与人工自然模糊标注，避免将低纹理、小目标直接称为运动模糊。",
             "", "诊断本身不证明新模块有效；选择候选机制后仍需保持输入和参数预算一致的受控实验。", ""]
    return "\n".join(rows)
