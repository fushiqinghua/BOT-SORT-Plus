"""Score detached E2 output with official TrackEval and original D3 windows."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from e2_boxes import AUDIT_PATH, BASE, KF, RAW, TARGET, output_csv, output_json, read_json

PREFIX = "results/vttiny/iterations/v0.4-motion-blur/d3-follow-up"
EVENT_HELPERS = PREFIX + "/02-continuation/event_audit"
RAW_NOMINATIONS = PREFIX + "/03-output-only/parity/full/supplemental_rows.csv"
RAW_WINDOWS = PREFIX + "/03-output-only/identity/full/paired_events.csv"
LINEAGE = PREFIX + "/04-background-filter/lineage/lineage_rows.csv"
METRICS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "FP", "FN", "IDSW")
TRACKERS = (BASE, RAW, KF)
SCORER_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def keyed(rows, fields):
    result = {tuple(row[key] for key in fields): row for row in rows}
    assert len(rows) == len(result)
    return result


def strict_match(old, new, columns):
    for key in columns:
        assert math.isclose(float(old[key]), float(new[key]), rel_tol=0, abs_tol=1e-8), (key, old, new)


def score(args, integrity, selected):
    from tools.evaluate_botsort_plus import sha256

    history = read_json(args.history / "evaluation_audit.json")
    assert history["status"] == "COMPLETE"
    historical_summary = args.history / "summary.json"
    assert history["summary_sha256"] == sha256(historical_summary)
    assert history["evaluator"]["commit"] == SCORER_COMMIT
    output = args.run / "trackeval"
    subprocess.run([sys.executable, "-B", "-u", str(args.repo / "tools/evaluate_botsort_plus.py"),
                    "--run-dir", str(args.run / "inputs"), "--trackers", *TRACKERS,
                    "--gt-root", str(args.gt), "--trackeval-root", str(args.trackeval),
                    "--output-dir", str(output), "--reference-tracker", BASE],
                   cwd=args.repo, check=True)
    current_audit = read_json(output / "evaluation_audit.json")
    assert current_audit["status"] == "COMPLETE"
    assert current_audit["evaluator"]["commit"] == SCORER_COMMIT
    assert current_audit["summary_sha256"] == sha256(output / "summary.json")
    assert current_audit["ground_truth"]["sha256"] == {
        seq: history["ground_truth"]["sha256"][seq] for seq in selected
    }
    historical = keyed(read_json(historical_summary)["results"], ("tracker", "sequence"))
    current = keyed(read_json(output / "summary.json")["results"], ("tracker", "sequence"))
    for seq in selected if integrity["smoke"] else (*selected, "COMBINED_SEQ"):
        for name in (BASE, RAW):
            strict_match(historical[name, seq], current[name, seq], METRICS)
    metrics = []
    for seq in ("COMBINED_SEQ", *selected):
        for name in TRACKERS:
            row = dict(current[name, seq])
            for ref, suffix in ((BASE, "B"), (RAW, "raw")):
                for metric in METRICS:
                    row[f"delta_{metric}_vs_{suffix}"] = row[metric] - current[ref, seq][metric]
            metrics.append(row)
    output_csv(args.run / "E2_metrics.csv", metrics)
    return current


def truth_audit(args, selected, metrics, boxes):
    for name, value in (("int", int), ("float", float), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, value)
    import trackeval
    from trackeval.metrics.clear import linear_sum_assignment
    from botsort_plus.diagnostics.common import box_iou_matrix, load_gt
    from check_output_only import clear_row_assignment, mot_box, read_mot

    dataset = trackeval.datasets.MotChallenge2DBox({
        "GT_FOLDER": str(args.gt), "TRACKERS_FOLDER": str(args.run / "inputs/trackers"),
        "TRACKERS_TO_EVAL": list(TRACKERS), "BENCHMARK": "VT-Tiny-MOT", "SPLIT_TO_EVAL": "test",
        "SKIP_SPLIT_FOL": True, "SEQ_INFO": selected, "DO_PREPROC": False, "PRINT_CONFIG": False,
        "CLASSES_TO_EVAL": ["pedestrian"],
    })
    dataset.valid_classes = dataset.class_list = ["object"]
    dataset.class_name_to_class_id["object"] = 1
    clear = trackeval.metrics.CLEAR()
    nominations = keyed(csv_rows(args.repo / RAW_NOMINATIONS), ("sequence", "frame", "track_id", "det_index"))
    lineage = keyed(csv_rows(args.repo / LINEAGE), ("sequence", "frame", "track_id", "det_index"))
    assert len(nominations) == 191 and nominations.keys() == lineage.keys()
    assert boxes.keys() == {key for key in nominations if key[0] in selected}
    rows, native_changes, native_baseline_to_raw = [], [], 0
    for seq, frames in selected.items():
        positions = {}
        matching = {}
        for name in TRACKERS:
            path = args.run / "inputs/trackers" / name / "data" / f"{seq}.txt"
            matching[name], totals = clear_row_assignment(dataset, name, seq, clear, linear_sum_assignment)
            positions[name] = read_mot(path)[1]
            for expected, actual in (("FP", "CLR_FP"), ("FN", "CLR_FN"), ("IDSW", "IDSW")):
                assert metrics[name, seq][expected] == totals[actual], (seq, name, totals)
        labels_base, labels_raw, labels_kf = (matching[name] for name in TRACKERS)
        assert labels_raw.keys() == labels_kf.keys()
        for key, before in labels_base.items():
            native_baseline_to_raw += before["gt_id"] != labels_raw[key]["gt_id"]
            if labels_raw[key]["gt_id"] != labels_kf[key]["gt_id"]:
                native_changes.append(dict(sequence=seq, frame=key[0], track_id=key[1],
                                           B_clear_gt=before["gt_id"], raw_clear_gt=labels_raw[key]["gt_id"],
                                           kf_clear_gt=labels_kf[key]["gt_id"]))
        truth, _ = load_gt(args.gt / seq / "gt/gt.txt", frames)
        for key, row in boxes.items():
            if key[0] != seq:
                continue
            frame, track, index = map(int, key[1:])
            original = nominations[key]
            context = lineage[key]
            z = frame, track
            before, after = labels_raw[z], labels_kf[z]
            assert (int(original["clear_gt"]) if original["clear_gt"] else None) == before["gt_id"]
            assert math.isclose(float(original["best_gt_iou"]), before["max_gt_iou"], rel_tol=0, abs_tol=1e-8)
            raw_box, _ = mot_box(positions[RAW][z])
            new_box, _ = mot_box(positions[KF][z])
            anchor = context["most_recent_prior_clear_gt"]
            anchors_here = [item["xyxy"] for item in truth[frame] if anchor and item["gt_id"] == int(anchor)]
            iou = [float(box_iou_matrix([box], anchors_here)[0, 0])
                   for box in (raw_box, new_box)] if anchors_here else [None, None]
            rows.append(dict(sequence=seq, frame=frame, track_id=track, det_index=index,
                             prior_gt=anchor, lineage_category=context["diagnostic_category"],
                             raw_clear_gt=before["gt_id"], kf_clear_gt=after["gt_id"],
                             raw_best_gt_iou=before["max_gt_iou"], kf_best_gt_iou=after["max_gt_iou"],
                             raw_iou_prior_gt=iou[0], kf_iou_prior_gt=iou[1],
                             raw_TP=before["gt_id"] is not None, kf_TP=after["gt_id"] is not None))
    assert len(rows) == len(boxes)
    output_csv(args.run / "E2_clear_deltas.csv", rows)
    if native_changes:
        output_csv(args.run / "E2_native_clear_changes.csv", native_changes)
    buckets = {}
    keys = sorted({row["lineage_category"] for row in rows})
    for category, group in [("all", rows), *((x, [y for y in rows if y["lineage_category"] == x]) for x in keys)]:
        buckets[category] = dict(rows=len(group),
                                raw_tp=sum(row["raw_TP"] for row in group),
                                kf_tp=sum(row["kf_TP"] for row in group),
                                fp_to_tp=sum(not row["raw_TP"] and row["kf_TP"] for row in group),
                                tp_to_fp=sum(row["raw_TP"] and not row["kf_TP"] for row in group),
                                tp_gt_changed=sum(row["raw_TP"] and row["kf_TP"] and
                                                  row["raw_clear_gt"] != row["kf_clear_gt"] for row in group))
    result = {"by_lineage": buckets, "native_CLEAR_changes_B_to_raw": native_baseline_to_raw,
              "native_CLEAR_changes_raw_to_KF": len(native_changes),
              "target": next(row for row in rows if tuple(row[key] for key in
                                          ("sequence", "frame", "track_id", "det_index")) == TARGET)}
    output_json(args.run / "E2_clear_summary.json", result)
    return result


def fixed_windows(args, selected):
    import compare_events as old
    from botsort_plus.diagnostics.analysis import _frame_state, _output_status, _read_mot
    from botsort_plus.diagnostics.common import diagnostic_assignment, load_gt
    from botsort_plus.io import detection_fingerprint, read_detection_cache, sha256
    for item in (sha256, diagnostic_assignment, _frame_state, _output_status):
        setattr(old, item.__name__, item)
    analysis = read_json(args.frozen / "analysis/summary.json")
    cases, fingerprint = old.read_cases(args.frozen, analysis)
    history = keyed(csv_rows(args.repo / RAW_WINDOWS),
                    ("sequence", "recovery_frame", "baseline_track_id", "run"))
    assert sum(map(len, cases.values())) == 287
    visible_cases = sum(len(cases.get(seq, ())) for seq in selected)
    fields = ("first_interruption_offset", "first_interruption_reason", "restoration_status",
              "original_failure_verified_anchor_correct", "stable_3", "stable_5", "stable_10")
    rows, followups = [], []
    for seq, frames in selected.items():
        if not cases.get(seq):
            continue
        path = args.gt / seq / "gt/gt.txt"
        assert sha256(path) == analysis["ground_truth"]["sha256"][seq]
        truth, _ = load_gt(path, frames)
        cache = read_detection_cache(args.detections / f"{seq}.json.gz")
        protocol = read_json(args.frozen / "observed/protocols" / f"{BASE}.json")
        assert detection_fingerprint(cache) == protocol["detection_fingerprints"][seq]
        dets = {entry["frame"]: [{"xyxy": row[:4], "score": float(row[4])}
                                  for row in np.array(entry["detections"], dtype=np.float32).reshape(-1, 6)]
                for entry in cache["frames"]}
        base_events = args.frozen / "observed/events" / BASE / f"{seq}.jsonl"
        assert sha256(base_events) == analysis["original_events_sha256"][seq]
        updates = old.accepted_updates(base_events, seq, frames)
        for name in TRACKERS:
            mot = _read_mot(args.run / "inputs/trackers" / name / "data" / f"{seq}.txt", frames)
            auditor = old.SequenceAudit(truth, dets, mot, updates, protocol["native_config"],
                                        analysis["iou_threshold"], analysis["ambiguity_margin"])
            for case in cases[seq]:
                row, per_frame = auditor.compare_case(case, name)
                if name != KF:
                    expected = history[seq, str(row["recovery_frame"]), str(row["baseline_track_id"]), name]
                    for key in fields:
                        actual = "" if row[key] is None else str(row[key])
                        assert actual == expected[key], (seq, name, row["recovery_frame"], key, actual, expected[key])
                rows.append(row)
                followups.extend(per_frame)
    assert len(rows) == len(TRACKERS) * visible_cases
    assert len(followups) == 10 * len(rows)
    output_csv(args.run / "E2_fixed_windows.csv", rows)
    output_csv(args.run / "E2_fixed_window_frames.csv", followups)
    summary = {}
    for name in TRACKERS:
        group = [row for row in rows if row["run"] == name]
        summary[name] = dict(windows=len(group), original_failure_same_id=sum(
            row["original_failure_verified_anchor_correct"] is True for row in group),
            stable={str(horizon): sum(row[f"stable_{horizon}"] for row in group) for horizon in old.HORIZONS})
    output_json(args.run / "E2_fixed_window_summary.json", {"input_sha256": fingerprint, "trackers": summary})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "frozen", "run", "gt", "trackeval", "detections", "history"):
        parser.add_argument(name, type=lambda value: Path(value).resolve())
    args = parser.parse_args()
    os.environ["TRACKEVAL_ROOT"] = str(args.trackeval)
    sys.path[:0] = [str(args.repo), str(args.trackeval), str(args.repo / AUDIT_PATH), str(args.repo / EVENT_HELPERS)]
    integrity = read_json(args.run / "E2_row_integrity.json")
    assert integrity["status"] == "PASS"
    protocol = read_json(args.run / "inputs/protocols" / f"{KF}.json")
    selected = protocol["sequences"]
    boxes = keyed(csv_rows(args.run / "E2_box_pairs.csv"), ("sequence", "frame", "track_id", "det_index"))
    assert len(boxes) == integrity["nomination_count"]
    metrics = score(args, integrity, selected)
    clear = truth_audit(args, selected, metrics, boxes)
    windows = fixed_windows(args, selected)
    output_json(args.run / "E2_evaluation_summary.json", {"status": "COMPLETE", "smoke": integrity["smoke"],
                "reproduced_historical_B_and_raw": True, "clear": clear, "fixed_windows": windows,
                "combined": {name: metrics[name, "COMBINED_SEQ"] for name in TRACKERS}})
    print("E2_EVALUATION_COMPLETE", args.run, flush=True)


if __name__ == "__main__":
    main()
