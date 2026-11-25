"""Classify output-only supplements against official GT using frozen native track history."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import median
import sys
from tempfile import TemporaryDirectory

import numpy as np


def report_file(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def database_row(frame, track, det, detections, mot, clear, gt_by_frame, box_iou_matrix):
    box = np.asarray(detections[frame - 1]["detections"][det])
    labels = gt_by_frame[frame]
    poses = [box[:4]]
    if (frame, track) in mot:
        poses.append(mot[frame, track][0])
    if labels:
        overlaps = box_iou_matrix(poses, [g["xyxy"] for g in labels])
        det_index = int(np.argmax(overlaps[0]))
        det_max = float(overlaps[0, det_index])
        out_max = float(np.max(overlaps[1])) if len(poses) == 2 else None
        best_det_gt = int(labels[det_index]["gt_id"]) if det_max > 0 else None
    else:
        det_max, best_det_gt = 0.0, None
        out_max = 0.0 if len(poses) == 2 else None
        overlaps = np.empty((len(poses), 0))
    official_assignment = clear.get((frame, track), {}).get("gt_id")
    if official_assignment is not None:
        assert out_max is not None and out_max >= .5 - 1e-6, (frame, track, official_assignment)
    return {
        "frame": frame, "track_id": track, "det_index": det,
        "det_box": [float(x) for x in box[:4]], "score": float(box[4]),
        "best_det_iou": det_max, "best_det_gt": best_det_gt,
        "best_output_iou": out_max, "clear_gt": official_assignment,
        "by_gt_det": {int(g["gt_id"]): float(v) for g, v in zip(labels, overlaps[0])},
    }


def compact_box(box):
    return json.dumps(box, ensure_ascii=False, separators=(",", ":"))


def track_quality(history):
    if not history:
        raise ValueError("Missing native measurements")
    recent = history[-3:]
    scores = [h["score"] for h in recent]
    history_gt = Counter(h["clear_gt"] for h in history if h["clear_gt"] is not None)
    return {
        "history_n": len(history), "history_output_n": sum(h["best_output_iou"] is not None for h in history),
        "history_native_match_n": sum(h["kind"] == "match" for h in history),
        "history_b_recover_n": sum(h["kind"] == "recovered" for h in history),
        "history_det_best_iou": max(h["best_det_iou"] for h in history),
        "history_output_best_iou": max((h["best_output_iou"] or 0.0) for h in history),
        "history_det_gte05_n": sum(h["best_det_iou"] >= 0.50 for h in history),
        "history_output_gte05_n": sum((h["best_output_iou"] or 0.0) >= 0.50 for h in history),
        "history_gt_match_n": sum(history_gt.values()), "history_unique_clear_gt_n": len(history_gt),
        "history_top_clear_gt": history_gt.most_common(1)[0][0] if history_gt else None,
        "last3_measurement_median_conf": median(scores),
        "all_measurement_median_conf": median(h["score"] for h in history),
    }


CATEGORIES = (
    "fp_competing_assignment",
    "fp_persistent_background",
    "fp_localization_same_anchor",
    "fp_displaced_or_different_gt",
    "fp_weak_unconfirmed_overlap",
    "fp_unconfirmed_no_overlap",
)


def explain(row, current, history, clear_by_gt):
    current_iou = row["best_gt_iou"]
    label = row["clear_gt"]
    known_gt = next((h["clear_gt"] for h in reversed(history) if h["clear_gt"] is not None), None)
    known_det_iou = current["by_gt_det"].get(known_gt, 0.0) if known_gt is not None else None
    if label is not None:
        if row["prior_det_gt"] == row["prior_output_gt"] == label:
            kind = "tp_same_id_double_witness"
        elif known_gt is not None and label != known_gt:
            kind = "tp_different_from_recent_clear_gt"
        elif known_gt == label:
            kind = "tp_same_gt_longer_history_only"
        else:
            kind = "tp_prior_identity_unverified"
    elif current_iou >= .50 - 1e-12:
        kind = "fp_competing_assignment"
    elif (row["history_n"] >= 3 and current_iou < .10 and row["history_det_best_iou"] < .10
          and row["history_output_best_iou"] < .10):
        kind = "fp_persistent_background"
    elif known_gt is not None and known_det_iou >= .10 and current["best_det_gt"] == known_gt:
        kind = "fp_localization_same_anchor"
    elif known_gt is not None:
        kind = "fp_displaced_or_different_gt"
    elif current_iou >= .10:
        kind = "fp_weak_unconfirmed_overlap"
    else:
        kind = "fp_unconfirmed_no_overlap"
    neighbors = [gt for gt, iou in current["by_gt_det"].items() if iou >= .50 - 1e-12]
    return {
        "diagnostic_category": kind, "most_recent_prior_clear_gt": known_gt,
        "iou_to_recent_prior_clear_gt": known_det_iou,
        "same_iou_plus_gt_claims": compact_box({gt: clear_by_gt[gt] for gt in neighbors if gt in clear_by_gt}),
    }


def classify_sequence(sequence, rows, clears, gt_by_frame, cache, boxes, events, box_iou_matrix):
    tracks = {row["track_id"] for row in rows}
    histories = defaultdict(list)
    current_limits = {
        track: max(row["frame"] for row in rows if row["track_id"] == track)
        for track in tracks
    }
    supplements = defaultdict(list)
    for row in rows:
        supplements[row["frame"]].append(row)
    clear_by_time = defaultdict(dict)
    for (frame, track), label in clears.items():
        truth = label["gt_id"]
        if truth is not None:
            clear_by_time[frame][truth] = track
    for _, event in events:
        if event["event"] not in {"match", "recovered", "native_birth"}:
            continue
        track = int(event["track_id"])
        frame = int(event["frame"])
        if track not in tracks or frame >= current_limits[track]:
            continue
        det = int(event["det_index"])
        native = database_row(frame, track, det, cache, boxes, clears, gt_by_frame, box_iou_matrix)
        histories[track].append({**native, "kind": event["event"], "stage": event.get("stage"),
                                 "event_cost": event.get("cost"), "event_age": event.get("age"),
                                 "event_hits": event.get("hits"), "event_appearance_cost": event.get("appearance_cost"),
                                 "event_geometry_cost": event.get("geometry_cost")})
    combined = []
    for frame, records in sorted(supplements.items()):
        for row in records:
            track, det = row["track_id"], row["det_index"]
            history = [h for h in histories[track] if h["frame"] < frame]
            assert history[-1]["frame"] == frame - 1
            assert len({h["frame"] for h in history}) == len(history), (sequence, frame, track, "multiple commits")
            earlier = history[-1]
            anchor = next((h for h in reversed(history) if h["kind"] == "recovered"), None)
            assert anchor is not None and anchor["frame"] == row["last_recovery_frame"]
            assert earlier["clear_gt"] == row["prior_output_gt"]
            current = database_row(frame, track, det, cache, boxes, clears, gt_by_frame, box_iou_matrix)
            assert math.isclose(current["best_det_iou"], row["best_gt_iou"], abs_tol=1e-6), (sequence, row)
            result = {
                **row, **track_quality(history),
                "last_real_detection_score": earlier["score"], "last_real_detection_kind": earlier["kind"],
                "last_real_detection_stage": earlier["stage"], "last_real_detection_best_gt_iou": earlier["best_det_iou"],
                "last_real_output_best_gt_iou": earlier["best_output_iou"],
                "last_real_detection_box": compact_box(earlier["det_box"]),
                "b_recovery_detection_score": anchor["score"], "b_recovery_geometry_cost": anchor["event_geometry_cost"],
                "b_recovery_appearance_cost": anchor["event_appearance_cost"], "b_recovery_event_hits": anchor["event_hits"],
                "b_recovery_event_age": anchor["event_age"], "b_recovery_age_now": frame - anchor["frame"],
                "b_recovery_detection_best_gt_iou": anchor["best_det_iou"],
                "b_recovery_output_best_gt_iou": anchor["best_output_iou"],
                "b_recovery_clear_gt": anchor["clear_gt"], "supplement_best_gt": current["best_det_gt"],
                "supplement_detection_box": compact_box(current["det_box"]),
            }
            if row["clear_gt"] is not None:
                result["history_match_same_current_gt_n"] = sum(h["clear_gt"] == row["clear_gt"] for h in history)
            else:
                result["history_match_same_current_gt_n"] = None
            result.update(explain(result, current, history, clear_by_time[frame]))
            combined.append(result)
    return combined


def summarize(rows):
    fp = [r for r in rows if r["clear_gt"] is None]
    tp = [r for r in rows if r["clear_gt"] is not None]
    assert (len(rows), len(fp), len(tp)) == (191, 119, 72)
    background = [r for r in fp if r["diagnostic_category"] == "fp_persistent_background"]
    split = defaultdict(Counter)
    for row in rows:
        split[row["sequence"]][row["diagnostic_category"]] += 1
    sensitivities = []
    for min_observations in (1, 3, 5):
        for threshold in (.000001, .05, .10, .20):
            eligible = [r for r in fp if r["history_n"] >= min_observations and r["best_gt_iou"] < threshold
                        and r["history_det_best_iou"] < threshold and r["history_output_best_iou"] < threshold]
            sensitivities.append({"history_minimum": min_observations, "iou_strict_upper": threshold,
                                  "eligible_fp": len(eligible)})
    return {
        "total": len(rows), "clear_fp": len(fp), "clear_tp": len(tp),
        "categories": Counter(row["diagnostic_category"] for row in rows),
        "sequences": {key: dict(value) for key, value in split.items()},
        "fp_by_current_best_iou": {
            "zero": sum(r["best_gt_iou"] == 0.0 for r in fp),
            "between_0_and_0p1": sum(0.0 < r["best_gt_iou"] < .10 for r in fp),
            "between_0p1_and_0p3": sum(.10 <= r["best_gt_iou"] < .30 for r in fp),
            "between_0p3_and_0p5": sum(.30 <= r["best_gt_iou"] < .50 for r in fp),
            "at_least_0p5": sum(r["best_gt_iou"] >= .50 for r in fp),
        },
        "past_background": {"schema": "all native committed detector/output boxes of the same numeric ID strictly "
                                       "before supplemental f; minimum 3 measured frames and current/prior maximum "
                                       "GT IoU strictly below .10; diagnostic only",
                            "rows": len(background), "tracks": len({(r["sequence"], r["track_id"]) for r in background}),
                            "sequences": Counter(r["sequence"] for r in background),
                            "sensitivities": sensitivities},
        "trigger_fp_have_prior_clear_gt": sum(r["history_gt_match_n"] > 0 for r in fp),
        "native_past_all_unmatched": {
            "clear_fp": sum(r["history_gt_match_n"] == 0 for r in fp),
            "clear_tp": sum(r["history_gt_match_n"] == 0 for r in tp),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("repo", "parity_source", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    project = args.repo.parents[2]
    sys.path[:0] = [str(args.repo), str(project / "external/TrackEval"), str(args.parity_source.parent)]
    from check_output_only import (BASE, clear_row_assignment, config_paths, digest, event_pairs, mot_box,
                                   protocol, read_json, read_mot)
    from botsort_plus.diagnostics.common import box_iou_matrix, load_gt
    from botsort_plus.io import read_detection_cache
    for name, value in (("int", int), ("float", float), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, value)
    import trackeval
    from trackeval.metrics.clear import linear_sum_assignment

    gt_root = project / "data/official_gt/GT/GT_00/test"
    frozen = args.repo / "outputs/diagnostics_v0.4_full30_round1/observed"
    round_root = args.repo.parent / "round1"
    prior_eval = round_root / "d3_output_only_comparison_v1/full/trackeval/evaluation_audit.json"
    gt_reference = read_json(prior_eval)["ground_truth"]["sha256"]
    native = protocol(frozen, BASE)
    sequences = native["sequences"]
    supplement_path = args.parity_source / "supplemental_rows.csv"
    rows_by_seq = defaultdict(list)
    with supplement_path.open(encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            for key in ("frame", "track_id", "det_index", "last_recovery_frame", "last_native_observation_frame",
                        "clear_gt", "prior_det_gt", "prior_output_gt"):
                row[key] = int(row[key]) if row[key] else None
            for key in ("detector_score", "prediction_iou", "best_gt_iou"):
                row[key] = float(row[key])
            rows_by_seq[row["sequence"]].append(row)
    result = []
    clear = trackeval.metrics.CLEAR({"PRINT_CONFIG": False})
    with TemporaryDirectory(prefix="assign_", dir=args.output) as scratch:
        (Path(scratch) / BASE).symlink_to(frozen / "trackers" / BASE)
        selected = {s: n for s, n in sequences.items() if s in rows_by_seq}
        dataset = trackeval.datasets.MotChallenge2DBox({
            "GT_FOLDER": str(gt_root), "TRACKERS_FOLDER": scratch, "TRACKERS_TO_EVAL": [BASE],
            "BENCHMARK": "VT-Tiny-MOT", "SPLIT_TO_EVAL": "test", "SKIP_SPLIT_FOL": True,
            "SEQ_INFO": selected, "DO_PREPROC": False, "PRINT_CONFIG": False,
            "CLASSES_TO_EVAL": ["pedestrian"],
        })
        dataset.valid_classes = dataset.class_list = ["object"]
        dataset.class_name_to_class_id["object"] = 1
        for seq, length in selected.items():
            gt_file = gt_root / seq / "gt/gt.txt"
            assert digest(gt_file) == gt_reference[seq]
            event_file, mot_file = config_paths(frozen, BASE, seq)
            assert digest(mot_file) == native["output_prediction_hashes"][seq]
            _, native_rows = read_mot(mot_file)
            native_boxes = {key: mot_box(row) for key, row in native_rows.items()}
            cached = args.repo / "results/vttiny/yolo11m_rgb_rgbt/rgbt/detections" / f"{seq}.json.gz"
            detections = read_detection_cache(cached)["frames"]
            truth, _ = load_gt(gt_file, length)
            clears, _ = clear_row_assignment(dataset, BASE, seq, clear, linear_sum_assignment)
            result.extend(classify_sequence(seq, rows_by_seq[seq], clears, truth, detections,
                                            native_boxes, event_pairs(event_file), box_iou_matrix))
            print(seq, len(rows_by_seq[seq]), "done", flush=True)
    report_file(args.output / "lineage_rows.csv", result)
    summary = summarize(result)
    with (args.output / "lineage_summary.json").open("x", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "sequences"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
