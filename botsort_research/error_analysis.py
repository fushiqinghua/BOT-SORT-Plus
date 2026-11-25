# CLEAR matching and box IoU are adapted from Jonathon Luiten's TrackEval,
# commit 12c8791b303e0a0b50f753af204249e622d0281a, MIT License.
# Copyright (c) 2020 Jonathon Luiten. See provenance/TRACKEVAL_LICENSE.
"""Offline, verified CLEAR error events and geometric diagnostic cohorts.

These labels describe observed box alignment/competition, not established causal
failure reasons. The source cache, predictions, output trace and GT must match
the completed evaluated run before any result can be marked COMPLETE.
"""
from __future__ import annotations

import contextlib
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .diagnostics import gap_bin, size_bin
from .io import detection_files, dump_json, load_detections, load_json, sha256


TRACKEVAL_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"
THRESHOLD = 0.5
EPS = np.finfo(float).eps
RECOVERY_VARIANTS = {"reid_recovery_fixed", "reid_recovery"}
RECOVERY_CLASSIFICATIONS = ("correct_same_gt", "wrong_identity", "current_unmatched_competition",
                            "current_unlocalized", "unverifiable_no_history", "not_emitted")


def clear_iou(boxes_a, boxes_b):
    """XYXY IoU with the pinned TrackEval floating-point operations/order."""
    a = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=float).reshape(-1, 4)
    minimum = np.minimum(a[:, None, :], b[None, :, :])
    maximum = np.maximum(a[:, None, :], b[None, :, :])
    intersection = np.maximum(minimum[..., 2] - maximum[..., 0], 0) * np.maximum(minimum[..., 3] - maximum[..., 1], 0)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    intersection[area_a <= EPS, :] = 0
    intersection[:, area_b <= EPS] = 0
    intersection[union <= EPS] = 0
    union[union <= EPS] = 1
    return intersection / union


class ClearMatcher:
    """Pinned CLEAR matching, returning row/column matches and IDSW flags.

    Unlike a maximum-cardinality diagnostic, this first rewards previous
    timestep identity continuity by 1000 and then IoU. Empty GT or prediction
    frames intentionally preserve previous-timestep state, as upstream does.
    """
    def __init__(self):
        self.previous_tracker = {}
        self.previous_timestep = {}

    def update(self, gt_ids, tracker_ids, similarity):
        gt_ids = np.asarray(gt_ids, dtype=np.int64)
        tracker_ids = np.asarray(tracker_ids, dtype=np.int64)
        similarity = np.asarray(similarity, dtype=float)
        if similarity.shape != (len(gt_ids), len(tracker_ids)):
            raise ValueError("CLEAR similarity dimensions disagree with IDs")
        if not len(gt_ids) or not len(tracker_ids):
            return np.empty((0, 2), dtype=int), np.empty(0, dtype=bool)
        previous = np.array([self.previous_timestep.get(int(g), np.nan) for g in gt_ids])
        score = 1000 * (tracker_ids[None, :] == previous[:, None]) + similarity
        score[similarity < THRESHOLD - EPS] = 0
        rows, columns = linear_sum_assignment(-score)
        keep = score[rows, columns] > EPS
        rows, columns = rows[keep], columns[keep]
        switches = np.array([int(gt_ids[i]) in self.previous_tracker and
                             self.previous_tracker[int(gt_ids[i])] != int(tracker_ids[j])
                             for i, j in zip(rows, columns)], dtype=bool)
        self.previous_timestep = {int(gt_ids[i]): int(tracker_ids[j]) for i, j in zip(rows, columns)}
        self.previous_tracker.update(self.previous_timestep)
        return np.column_stack((rows, columns)), switches


def _mot_frames(path, frame_count, *, ground_truth=False):
    """Preserve file row order, pool classes, and use TrackEval's nonzero mark."""
    result, seen = defaultdict(list), defaultdict(set)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for line in csv.reader(stream):
            if not line or not line[0].strip():
                continue
            row = np.asarray([float(v) for v in line], dtype=float)
            if len(row) < (8 if ground_truth else 7) or not np.isfinite(row).all():
                raise ValueError(f"Invalid MOT row in {path}")
            frame, identity, x, y, width, height = row[:6]
            if frame != int(frame) or not 1 <= frame <= frame_count or identity != int(identity) or identity < 0:
                raise ValueError(f"Invalid frame/identity in {path}")
            frame, identity = int(frame), int(identity)
            if identity in seen[frame]:
                raise ValueError(f"Duplicate identity at {path}:{frame}")
            seen[frame].add(identity)
            # TrackEval casts marks to int before checking nonzero; DO_PREPROC
            # is false, so it retains all object classes and negative marks.
            if ground_truth and int(row[6]) == 0:
                continue
            if width <= 0 or height <= 0:
                raise ValueError(f"Nonpositive MOT box in {path}")
            if not ground_truth and len(row) >= 8 and int(row[7]) > 1:
                raise ValueError(f"Prediction classes violate pooled TrackEval protocol in {path}")
            result[frame].append([x, y, x + width, y + height, identity, row[6]])
    return {f: np.asarray(rows, dtype=float).reshape(-1, 6) for f, rows in result.items()}


def _checked_hash(path, expected, label):
    if not expected or not Path(path).is_file() or sha256(path) != expected:
        raise ValueError(f"Missing or changed {label}: {path}")


def _validate_inputs(run_dir, gt_root):
    experiment_path = run_dir / "experiment.json"
    record = load_json(experiment_path)
    metrics_path = run_dir / "metrics/summary.json"
    metrics = load_json(metrics_path)
    if record.get("status") != "COMPLETE" or metrics.get("status") != "PASS":
        raise ValueError("Error analysis requires a complete, evaluated run")
    if not record.get("output_trace"):
        raise ValueError("Error analysis requires run --output-trace")
    _checked_hash(experiment_path, metrics.get("experiment_manifest_sha256"), "evaluated experiment manifest")
    if metrics.get("trackeval_provenance", {}).get("commit") != TRACKEVAL_COMMIT:
        raise ValueError("Metrics must come from the pinned TrackEval commit")
    names, variants = record.get("sequences", []), record.get("variants", [])
    if not names or not variants or len(set(names)) != len(names) or len(set(variants)) != len(variants):
        raise ValueError("Expected unique nonempty sequence/variant lists")
    files = detection_files(record["cache_dir"])
    expected_pairs = {(variant, seq) for variant in variants for seq in names}
    timing = {(r["variant"], r["sequence"]): r for r in record.get("timing", [])}
    if set(timing) != expected_pairs or len(timing) != len(record.get("timing", [])):
        raise ValueError("Timing/provenance entries do not cover each variant and sequence exactly once")
    official = {(r["tracker"], r["sequence"]): r for r in metrics.get("results", [])}
    expected_metrics = expected_pairs | {(variant, "COMBINED_SEQ") for variant in variants}
    if set(official) != expected_metrics or len(official) != len(metrics.get("results", [])):
        raise ValueError("Official metrics must contain each sequence and COMBINED_SEQ exactly once")
    for seq in names:
        if seq not in files:
            raise ValueError(f"Missing detection cache for {seq}")
        _checked_hash(files[seq], record.get("cache_sha256", {}).get(seq), "detection cache")
        _checked_hash(gt_root / seq / "gt/gt.txt", metrics.get("gt_sha256", {}).get(seq), "GT")
        frame_count = record.get("frames", {}).get(seq)
        if not isinstance(frame_count, int) or frame_count < 1:
            raise ValueError(f"Invalid frame count for {seq}")
        for variant in variants:
            entry = timing[variant, seq]
            if entry.get("frames") != frame_count:
                raise ValueError(f"Timing frame count differs for {variant}/{seq}")
            _checked_hash(run_dir / "trackers" / variant / "data" / f"{seq}.txt",
                          entry.get("predictions_sha256"), "prediction")
            _checked_hash(run_dir / "outputs" / variant / f"{seq}.jsonl",
                          entry.get("output_trace_sha256"), "output trace")
    return record, metrics, files, official


def _box(value, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (4,) or not np.isfinite(result).all() or np.any(result[2:] <= result[:2]):
        raise ValueError(f"Invalid {label} box in output trace")
    return result


def _verified_trace(path, cache_frames, prediction_frames, policy):
    """Validate every trace row against both the cache and the MOT artifact."""
    if policy not in ("kalman", "detection"):
        raise ValueError("Missing/invalid output policy in the experiment manifest")
    with Path(path).open(encoding="utf-8") as stream:
        for frame in cache_frames:
            line = stream.readline()
            if not line:
                raise ValueError(f"Output trace ended early: {path}")
            event = json.loads(line)
            if event.get("frame") != frame["frame"] or event.get("policy") != policy:
                raise ValueError(f"Output trace frame/policy mismatch: {path}")
            dets = np.asarray(frame["detections"], dtype=np.float32).reshape(-1, 6)
            predicted = prediction_frames.get(frame["frame"], np.empty((0, 6)))
            predicted_by_id = {int(row[4]): row for row in predicted}
            observations = event.get("observations")
            if not isinstance(observations, list) or len(observations) != len(predicted):
                raise ValueError(f"Output trace count differs from predictions: {path}")
            by_id, used_detection = {}, set()
            for item in observations:
                tid, index = item.get("track_id"), item.get("det_index")
                if not isinstance(tid, int) or tid <= 0 or tid in by_id or tid not in predicted_by_id:
                    raise ValueError(f"Invalid/duplicate output trace track ID: {path}")
                if not isinstance(index, int) or not 0 <= index < len(dets) or index in used_detection:
                    raise ValueError(f"Invalid/duplicate current detection index in output trace: {path}")
                raw = _box(item["detection_box"], "detection")
                kalman = _box(item["kalman_box"], "Kalman")
                output = _box(item["output_box"], "output")
                score = item.get("score")
                if not isinstance(score, (float, int)) or not np.isfinite(score):
                    raise ValueError(f"Invalid score in output trace: {path}")
                prediction = predicted_by_id[tid]
                if (not np.array_equal(raw, dets[index, :4]) or
                        not np.isclose(score, float(dets[index, 4]), rtol=0, atol=1e-7) or
                        not np.isclose(score, prediction[5], rtol=0, atol=1e-7) or
                        not np.allclose(output, prediction[:4], rtol=0, atol=1e-8) or
                        not np.array_equal(output, raw if policy == "detection" else kalman)):
                    raise ValueError(f"Output trace does not agree with cache, policy or predictions: {path}")
                by_id[tid] = item
                used_detection.add(index)
            yield by_id
        if stream.read().strip():
            raise ValueError(f"Output trace has extra frames: {path}")


def _csv_writer(stack, path, fields, compressed=False):
    opener = gzip.open if compressed else open
    stream = stack.enter_context(opener(path, "wt", encoding="utf-8", newline=""))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    return writer


def _score_bin(score):
    return "high" if score > .5 else "low" if .1 < score < .5 else "outside_association"


def _max_iou(values):
    return float(np.max(values)) if len(values) else 0.0


def _verify_counts(actual, expected, label):
    mismatches = {key: {"diagnostic": int(actual[key]), "official": expected.get(key)}
                  for key in ("FP", "FN", "IDSW") if actual[key] != expected.get(key)}
    if mismatches:
        raise ValueError(f"CLEAR attribution differs from official metrics for {label}: {mismatches}")


def _recovery_events(run_dir, variant, sequence, frames, timing_entry):
    """Read accepted recovery decisions only from a fingerprinted stage trace.

    Missing instrumentation is a skipped diagnostic, never a claim of zero
    recoveries. Once a trace fingerprint was recorded, missing/changed bytes
    or missing per-frame recovery events are integrity errors.
    """
    path = run_dir / "traces" / variant / f"{sequence}.jsonl"
    expected = timing_entry.get("trace_sha256")
    report = {"variant": variant, "sequence": sequence, "trace_sha256": expected,
              "status": "COMPLETE", "accepted": 0}
    if not expected:
        report.update(status="SKIPPED", accepted=None, reason="no_stage_trace_fingerprint")
        return {}, report
    _checked_hash(path, expected, "recovery stage trace")
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("stage") != "appearance_recovery":
                continue
            number = event.get("frame")
            if (not isinstance(number, int) or isinstance(number, bool) or
                    not 1 <= number <= len(frames) or number in result or event.get("variant") != variant):
                raise ValueError(f"Invalid/duplicate recovery stage frame or variant: {path}")
            matches = event.get("matches")
            if not isinstance(matches, list):
                raise ValueError(f"Recovery stage requires an explicit matches list: {path}")
            dets = np.asarray(frames[number-1]["detections"], np.float32).reshape(-1, 6)
            track_ids, detection_indices = set(), set()
            for match in matches:
                tid, index = match.get("track_id"), match.get("det_index")
                gap, age = match.get("feature_gap"), match.get("lost_age")
                distance, threshold = match.get("raw_cosine_distance"), match.get("appearance_threshold")
                if (not isinstance(tid, int) or isinstance(tid, bool) or tid <= 0 or tid in track_ids or
                        not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(dets) or index in detection_indices):
                    raise ValueError(f"Invalid/duplicate accepted recovery identity or detection: {path}:{number}")
                if (not isinstance(gap, int) or isinstance(gap, bool) or gap < 0 or
                        not isinstance(age, int) or isinstance(age, bool) or age < 1):
                    raise ValueError(f"Invalid accepted recovery feature gap/age: {path}:{number}")
                if (not isinstance(distance, (float, int)) or not isinstance(threshold, (float, int)) or
                        not np.isfinite([distance, threshold]).all() or not 0 <= distance <= threshold <= 2 or
                        not float(dets[index, 4]) > .5):
                    raise ValueError(f"Invalid accepted recovery appearance evidence/high-score index: {path}:{number}")
                track_ids.add(tid)
                detection_indices.add(index)
            result[number] = matches
            report["accepted"] += len(matches)
    if set(result) != set(range(1, len(frames)+1)):
        raise ValueError(f"Recovery stage trace must cover every frame, including empty matches: {path}")
    return result, report


def classify_recovery(match, number, observed, current_gt_by_track, previous_gt_by_track,
                      max_iou_by_track, current_switch_by_track=None):
    """Join one accepted event to current output and strictly earlier CLEAR GT.

    The previous identity is the most recent CLEAR assignment of THIS tracker
    ID, not the first identity, a majority vote, or the current frame's match.
    GT ID zero is a valid known identity. Unmatched/unlocalized observations and
    absent historical evidence are never relabeled as an identity switch.
    """
    tid, index = match["track_id"], match["det_index"]
    previous = previous_gt_by_track.get(tid)
    if previous is not None and previous[0] >= number:
        raise ValueError("Recovery attribution history must strictly precede the recovery frame")
    item = observed.get(tid)
    emitted = item is not None
    if emitted and item["det_index"] != index:
        raise ValueError("Accepted recovery does not match the emitted track/detection pair")
    current_gt = current_gt_by_track.get(tid) if emitted else None
    maximum = max_iou_by_track.get(tid) if emitted else None
    if emitted and (maximum is None or not np.isfinite(maximum)):
        raise ValueError("Emitted recovery needs its current output IoU evidence")
    if not emitted:
        current_status = "not_emitted"
    elif tid in current_gt_by_track:
        current_status = "matched"
    elif maximum >= THRESHOLD-EPS:
        current_status = "unmatched_competition"
    else:
        current_status = "unlocalized"
    if not emitted:
        classification = "not_emitted"
    elif previous is None:
        classification = "unverifiable_no_history"
    elif current_status == "unmatched_competition":
        classification = "current_unmatched_competition"
    elif current_status == "unlocalized":
        classification = "current_unlocalized"
    else:
        classification = "correct_same_gt" if previous[1] == current_gt else "wrong_identity"
    return {"frame": number, "track_id": tid, "det_index": index,
            "feature_gap": match["feature_gap"], "gap": gap_bin(match["feature_gap"]),
            "lost_age": match["lost_age"], "raw_cosine_distance": match["raw_cosine_distance"],
            "appearance_threshold": match["appearance_threshold"], "emitted": emitted,
            "history_available": previous is not None,
            "previous_gt_frame": previous[0] if previous is not None else None,
            "previous_gt_id": previous[1] if previous is not None else None,
            "frames_since_previous_gt": number-previous[0] if previous is not None else None,
            "current_gt_id": current_gt, "current_status": current_status,
            "current_clear_idsw": bool((current_switch_by_track or {}).get(tid, False)) if emitted else None,
            "output_max_iou": maximum, "score": item["score"] if emitted else None,
            "classification": classification}


def _recovery_summary_row(variant, sequence, gap, counts):
    accepted = int(counts["accepted"])
    known = int(counts["correct_same_gt"] + counts["wrong_identity"])
    return {"variant": variant, "sequence": sequence, "gap": gap, "accepted": accepted,
            **{key: int(counts[key]) for key in RECOVERY_CLASSIFICATIONS},
            "identity_verifiable": known,
            "conditional_identity_precision": counts["correct_same_gt"]/known if known else None,
            "correct_fraction_all_acceptances": counts["correct_same_gt"]/accepted if accepted else None}


def analyze_run(run_dir, gt_root, output):
    """Produce only verified offline error events for one completed run."""
    run_dir, gt_root, output = Path(run_dir).resolve(), Path(gt_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new error analysis directory: {output}")
    record, metrics, files, official = _validate_inputs(run_dir, gt_root)
    output.mkdir(parents=True)
    summary = {"schema_version": 1, "status": "RUNNING", "interpretation": "retrospective_development_only",
               "clear_threshold": THRESHOLD, "trackeval_commit": TRACKEVAL_COMMIT,
               "source": {"run_dir": str(run_dir), "experiment_sha256": sha256(run_dir / "experiment.json"),
                          "metrics_sha256": sha256(run_dir / "metrics/summary.json"),
                          "gt_sha256": metrics["gt_sha256"], "cache_sha256": record["cache_sha256"],
                          "diagnostic_source_sha256": sha256(Path(__file__)),
                          "outputs": [{k: entry[k] for k in ("variant", "sequence", "predictions_sha256", "output_trace_sha256")}
                                      for entry in record["timing"]]},
               "case_sampling": "First 3 FP events encountered per (variant, sequence, cohort), chronological order; not worst cases or a representative random sample.",
               "notes": ["FP labels are geometric cohorts, not established causal explanations.",
                         "CLEAR continuity matching differs from maximum-cardinality detection coverage.",
                         "Empty GT/prediction frames preserve previous-timestep state, matching pinned TrackEval.",
                         "Gap denotes time since a matched output, not known physical occlusion.",
                         "Only GT with nonzero integer marks is retained; object classes are pooled."],
               "sequences": [], "combined": []}
    summary["recovery_attribution"] = {
        "status": "RUNNING" if set(record["variants"]) & RECOVERY_VARIANTS else "NOT_APPLICABLE",
        "traces": [], "groups": [],
        "definition": "Compare the same tracker ID's latest strictly earlier CLEAR GT assignment with its actual emitted recovery output's current CLEAR GT assignment.",
        "precision_definition": "correct_same_gt / (correct_same_gt + wrong_identity), conditional on both historical and current GT assignments; other outcomes are separate, not counted as wrong joins.",
        "limitations": ["A same-GT event does not prove lifelong identity correctness; earlier identity corruption may already be present.",
                        "The history is the latest CLEAR-matched output, not necessarily the trusted RGB gallery frame.",
                        "Unlocalized/competing output, no historical assignment and not-emitted decisions are reported separately.",
                        "No stage trace fingerprint means SKIPPED, not zero accepted recoveries."]}
    dump_json(output / "summary.json", summary)
    sizes, gaps, categories, combined = defaultdict(Counter), defaultdict(Counter), Counter(), defaultdict(Counter)
    case_counts = Counter()
    recovery_counts = defaultdict(Counter)
    timing_lookup = {(value["variant"], value["sequence"]): value for value in record["timing"]}
    try:
        with contextlib.ExitStack() as stack:
            frame_writer = _csv_writer(stack, output / "frame_errors.csv", ["variant", "sequence", "frame", "GT", "predictions", "TP", "FP", "FN", "IDSW"])
            fp_fields = ["variant", "sequence", "frame", "track_id", "det_index", "score", "score_bin", "policy", "cohort", "detection_max_iou", "output_max_iou", "kalman_max_iou", "detection_best_gt_id", "output_best_gt_id", "output_iou_to_detection_best_gt", "center_shift_px", "width_delta_px", "height_delta_px"]
            fp_writer = _csv_writer(stack, output / "fp_cases.csv.gz", fp_fields, compressed=True)
            case_writer = _csv_writer(stack, output / "case_samples.csv", fp_fields)
            gt_writer = _csv_writer(stack, output / "gt_frames.csv.gz", ["variant", "sequence", "frame", "gt_id", "size_bin", "matched", "track_id", "det_index", "score", "output_iou", "detection_iou", "best_cache_iou", "has_high_candidate", "has_low_candidate", "last_match_frame", "last_track_id", "frames_since_last_match", "missing_frames_before_match", "id_changed"], compressed=True)
            recovery_fields = ["variant", "sequence", "frame", "track_id", "det_index", "feature_gap", "gap", "lost_age",
                "raw_cosine_distance", "appearance_threshold", "emitted", "history_available", "previous_gt_frame",
                "previous_gt_id", "frames_since_previous_gt", "current_gt_id", "current_status", "current_clear_idsw",
                "output_max_iou", "score", "classification"]
            recovery_writer = _csv_writer(stack, output / "recovery_cases.csv", recovery_fields)
            for seq in record["sequences"]:
                payload = load_detections(files[seq])
                frames = payload["frames"]
                if len(frames) != record["frames"][seq]:
                    raise ValueError(f"Cache and run frame counts differ for {seq}")
                gt = _mot_frames(gt_root / seq / "gt/gt.txt", len(frames), ground_truth=True)
                for variant in record["variants"]:
                    predicted = _mot_frames(run_dir / "trackers" / variant / "data" / f"{seq}.txt", len(frames))
                    policy = record.get("effective_variants", {}).get(variant, {}).get("output_policy")
                    events = _verified_trace(run_dir / "outputs" / variant / f"{seq}.jsonl", frames, predicted, policy)
                    recovery_events = {}
                    if variant in RECOVERY_VARIANTS:
                        recovery_events, recovery_report = _recovery_events(run_dir, variant, seq, frames, timing_lookup[variant, seq])
                        summary["recovery_attribution"]["traces"].append(recovery_report)
                        if recovery_report["status"] == "COMPLETE":
                            for scope in (seq, "COMBINED_SEQ"):
                                recovery_counts[variant, scope, "ALL"].update(accepted=0)
                    matcher, last_match, totals, previous_gt_by_track = ClearMatcher(), {}, Counter(), {}
                    for frame, observed in zip(frames, events):
                        number = frame["frame"]
                        g = gt.get(number, np.empty((0, 6)))
                        p = predicted.get(number, np.empty((0, 6)))
                        similarity = clear_iou(g[:, :4], p[:, :4])
                        matches, switches = matcher.update(g[:, 4], p[:, 4], similarity)
                        by_gt = {int(i): (int(j), bool(switch)) for (i, j), switch in zip(matches, switches)}
                        matched_columns = set(matches[:, 1])
                        current_gt_by_track = {int(p[j, 4]): int(g[i, 4]) for i, j in matches}
                        current_switch_by_track = {int(p[j, 4]): bool(switch) for (_, j), switch in zip(matches, switches)}
                        max_iou_by_track = {int(row[4]): _max_iou(similarity[:, j]) for j, row in enumerate(p)}
                        for accepted in recovery_events.get(number, []):
                            recovery_row = {"variant": variant, "sequence": seq, **classify_recovery(
                                accepted, number, observed, current_gt_by_track, previous_gt_by_track,
                                max_iou_by_track, current_switch_by_track)}
                            recovery_writer.writerow(recovery_row)
                            for scope in (seq, "COMBINED_SEQ"):
                                for group in (recovery_row["gap"], "ALL"):
                                    recovery_counts[variant, scope, group].update({"accepted": 1, recovery_row["classification"]: 1})
                        # Update only AFTER accepted events have been attributed;
                        # using current matches as history would make every
                        # recovery with a GT match spuriously look correct.
                        previous_gt_by_track.update({tid: (number, gid) for tid, gid in current_gt_by_track.items()})
                        counts = {"GT": len(g), "predictions": len(p), "TP": len(matches),
                                  "FP": len(p) - len(matches), "FN": len(g) - len(matches), "IDSW": int(switches.sum())}
                        totals.update(counts)
                        frame_writer.writerow({"variant": variant, "sequence": seq, "frame": number, **counts})
                        det = np.asarray(frame["detections"], dtype=np.float32).reshape(-1, 6)
                        cache_iou = clear_iou(g[:, :4], det[:, :4])
                        # Native _split_detections appends an int64 index,
                        # promoting the already-quantized float32 score to
                        # float64 before comparing against Python thresholds.
                        # In particular np.float32(.1) is slightly above .1.
                        scores = det[:, 4].astype(float)
                        for j, prediction in enumerate(p):
                            if j in matched_columns:
                                continue
                            item = observed[int(prediction[4])]
                            raw, filtered, final = (np.asarray(item[k], dtype=float) for k in ("detection_box", "kalman_box", "output_box"))
                            overlaps = clear_iou(g[:, :4], [raw, filtered, final])
                            raw_max, filtered_max, output_max = (_max_iou(overlaps[:, k]) for k in range(3))
                            if not len(g):
                                cohort = "empty_gt_frame"
                            elif output_max >= THRESHOLD - EPS:
                                cohort = "competition_or_continuity"
                            elif raw_max >= THRESHOLD - EPS:
                                cohort = "localization_drift_candidate"
                            else:
                                cohort = "raw_and_output_unaligned"
                            categories[variant, cohort] += 1
                            best_raw = int(np.argmax(overlaps[:, 0])) if len(g) else None
                            best_output = int(np.argmax(overlaps[:, 2])) if len(g) else None
                            fp_row = {"variant": variant, "sequence": seq, "frame": number,
                                "track_id": int(prediction[4]), "det_index": item["det_index"], "score": item["score"],
                                "score_bin": _score_bin(item["score"]), "policy": policy, "cohort": cohort,
                                "detection_max_iou": raw_max, "output_max_iou": output_max, "kalman_max_iou": filtered_max,
                                "detection_best_gt_id": int(g[best_raw, 4]) if best_raw is not None else None,
                                "output_best_gt_id": int(g[best_output, 4]) if best_output is not None else None,
                                "output_iou_to_detection_best_gt": float(overlaps[best_raw, 2]) if best_raw is not None else None,
                                "center_shift_px": float(np.linalg.norm((final[:2] + final[2:] - raw[:2] - raw[2:]) / 2)),
                                "width_delta_px": float((final[2] - final[0]) - (raw[2] - raw[0])),
                                "height_delta_px": float((final[3] - final[1]) - (raw[3] - raw[1]))}
                            fp_writer.writerow(fp_row)
                            sample_key = variant, seq, cohort
                            if case_counts[sample_key] < 3:
                                case_writer.writerow(fp_row)
                                case_counts[sample_key] += 1
                        for i, truth in enumerate(g):
                            gid, size = int(truth[4]), size_bin(truth[:4])
                            pair = by_gt.get(i)
                            previous = last_match.get(gid)
                            row = {"variant": variant, "sequence": seq, "frame": number, "gt_id": gid,
                                "size_bin": size, "matched": int(pair is not None), "track_id": None, "det_index": None,
                                "score": None, "output_iou": None, "detection_iou": None,
                                "best_cache_iou": _max_iou(cache_iou[i]),
                                "has_high_candidate": bool(np.any(cache_iou[i, scores > .5] >= THRESHOLD - EPS)),
                                "has_low_candidate": bool(np.any(cache_iou[i, (scores > .1) & (scores < .5)] >= THRESHOLD - EPS)),
                                "last_match_frame": previous[0] if previous else None,
                                "last_track_id": previous[1] if previous else None,
                                "frames_since_last_match": number - previous[0] if previous else None,
                                "missing_frames_before_match": None, "id_changed": 0}
                            is_switch = False
                            if pair is not None:
                                j, is_switch = pair
                                tid = int(p[j, 4])
                                item = observed[tid]
                                row.update(track_id=tid, det_index=item["det_index"], score=item["score"],
                                           output_iou=float(similarity[i, j]),
                                           detection_iou=float(clear_iou([truth[:4]], [item["detection_box"]])[0, 0]),
                                           id_changed=int(is_switch))
                                if previous:
                                    missing = number - previous[0] - 1
                                    row["missing_frames_before_match"] = missing
                                    for scope in (seq, "COMBINED_SEQ"):
                                        gaps[variant, scope, gap_bin(missing)].update(transitions=1, id_changed=int(is_switch), same_id=int(not is_switch))
                                last_match[gid] = number, tid
                            for scope in (seq, "COMBINED_SEQ"):
                                sizes[variant, scope, size].update(GT=1, TP=int(pair is not None), FN=int(pair is None), IDSW=int(is_switch))
                            gt_writer.writerow(row)
                    # Exhaust the trace iterator to trigger its trailing-frame
                    # check: zip stops at the frame iterable's end first.
                    next(events, None)
                    _verify_counts(totals, official[variant, seq], f"{variant}/{seq}")
                    summary["sequences"].append({"variant": variant, "sequence": seq, "verified": True, **dict(totals)})
                    combined[variant].update(totals)
                    print(f"ATTRIBUTION {variant} {seq} FP={totals['FP']} FN={totals['FN']} IDSW={totals['IDSW']} verified", flush=True)
            for variant in record["variants"]:
                _verify_counts(combined[variant], official[variant, "COMBINED_SEQ"], f"{variant}/COMBINED_SEQ")
                summary["combined"].append({"variant": variant, "verified": True, **dict(combined[variant])})
            size_writer = _csv_writer(stack, output / "size_summary.csv", ["variant", "sequence", "size_bin", "GT", "TP", "FN", "IDSW", "recall"])
            for (variant, seq, size), counts in sorted(sizes.items()):
                size_writer.writerow({"variant": variant, "sequence": seq, "size_bin": size, **dict(counts), "recall": counts["TP"] / counts["GT"]})
            gap_writer = _csv_writer(stack, output / "gap_summary.csv", ["variant", "sequence", "gap", "transitions", "id_changed", "same_id"])
            for (variant, seq, gap), counts in sorted(gaps.items()):
                gap_writer.writerow({"variant": variant, "sequence": seq, "gap": gap, **dict(counts)})
            check_writer = _csv_writer(stack, output / "clear_verification.csv", ["variant", "sequence", "verified", "GT", "predictions", "TP", "FP", "FN", "IDSW"])
            check_writer.writerows(summary["sequences"])
            recovery_rows = [_recovery_summary_row(variant, seq, gap, counts)
                             for (variant, seq, gap), counts in sorted(recovery_counts.items())]
            recovery_columns = ["variant", "sequence", "gap", "accepted", *RECOVERY_CLASSIFICATIONS,
                                "identity_verifiable", "conditional_identity_precision", "correct_fraction_all_acceptances"]
            recovery_summary_writer = _csv_writer(stack, output / "recovery_summary.csv", recovery_columns)
            recovery_summary_writer.writerows(recovery_rows)
            recovery_info = summary["recovery_attribution"]
            recovery_info["groups"] = recovery_rows
            if recovery_info["status"] != "NOT_APPLICABLE":
                statuses = [value["status"] for value in recovery_info["traces"]]
                recovery_info["status"] = ("COMPLETE" if all(status == "COMPLETE" for status in statuses)
                    else "SKIPPED" if all(status == "SKIPPED" for status in statuses) else "PARTIAL")
        summary["fp_cohorts"] = [{"variant": v, "cohort": c, "count": count} for (v, c), count in sorted(categories.items())]
        summary["status"] = "COMPLETE"
        summary["artifacts_sha256"] = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.name != "summary.json"}
        dump_json(output / "summary.json", summary)
        return summary
    except Exception as exc:
        summary["status"] = "FAILED"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        dump_json(output / "summary.json", summary)
        raise
