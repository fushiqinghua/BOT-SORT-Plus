"""D3 score-band interruptions, using the archived run and its original RGB GT."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import median
import sys

import numpy as np


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def count_distribution(values):
    return dict(sorted(Counter(values).items()))


def number(value):
    return float(value) if value else None


def describe(values):
    selected = [value for value in values if value is not None]
    if not selected:
        return {"n": 0}
    return {"n": len(selected), "min": min(selected), "median": median(selected),
            "max": max(selected)}


def prefix(rows, condition):
    for index, row in enumerate(rows):
        if not condition(row):
            return index
    return len(rows)


def event_key(row):
    return (row["sequence"], int(row["recovery_frame"]),
            int(row["first_failure_frame"]), int(row["track_id"]), int(row["gt_id"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    sys.path.insert(0, str(args.repository))
    from botsort_plus.diagnostics.common import diagnostic_assignment, detection_tier, load_gt

    run = args.run
    protocol_path = run / "observed/protocols/B_early_moderate.json"
    analysis_path = run / "analysis/summary.json"
    split_path = run / "round1_findings/D3_complete_first_failure_candidates.csv"
    det_audit_path = run / "analysis/detection_audit.csv"
    quality_path = run / "blur/observation_quality.csv"
    tracks_path = run / "analysis/track_error.csv"
    protocol = read(protocol_path)
    analysis = read(analysis_path)
    blur = read(run / "blur/summary.json")
    assert analysis["status"] == blur["status"] == "COMPLETE"
    assert analysis["run_protocol_sha256"] == digest(protocol_path)
    for path in (det_audit_path, tracks_path):
        assert digest(path) == analysis["files_sha256"][path.name]
    assert digest(quality_path) == blur["derived_files_sha256"][quality_path.name]
    low = protocol["native_config"]["track_low_thresh"]
    high = protocol["native_config"]["track_high_thresh"]
    threshold, margin = analysis["iou_threshold"], analysis["ambiguity_margin"]
    split = list(csv_rows(split_path))
    cases = {event_key(row): row for row in split
             if row["candidate_type"] == "aligned_but_outside_score"}
    assert len(split) == 210 and len(cases) == 89
    assert sum(row["candidate_type"] == "aligned_but_outside_score" for row in split) == len(cases)
    assert all(row["first_failure_reason"] == "missing_without_usable_detection" for row in cases.values())
    wanted = {(seq, frame) for seq, start, _, _, _ in cases for frame in range(start, start+11)}
    det_rows = defaultdict(list)
    quality = {}
    tracks = {}
    for row in csv_rows(det_audit_path):
        key = (row["sequence"], int(row["frame"]))
        if key in wanted:
            det_rows[key].append(row)
    for row in csv_rows(quality_path):
        key = (row["sequence"], int(row["frame"]), int(row["det_index"]))
        if key[:2] in wanted:
            quality[key] = row
    for row in csv_rows(tracks_path):
        key = (row["sequence"], int(row["frame"]), int(row["track_id"]))
        if key[:2] in wanted:
            tracks[key] = row

    truth = {}
    for seq in sorted({case[0] for case in cases}):
        path = Path(analysis["ground_truth"]["root"]) / seq / "gt/gt.txt"
        assert digest(path) == analysis["ground_truth"]["sha256"][seq]
        truth[seq], _ = load_gt(path, protocol["sequences"][seq])
    frames = {}
    for seq, frame in sorted(wanted):
        rows = det_rows[seq, frame]
        assert [int(row["det_index"]) for row in rows] == list(range(len(rows)))
        boxes = [[float(row[k]) for k in ("x1", "y1", "x2", "y2")] for row in rows]
        labels, status, iou, _ = diagnostic_assignment(boxes, truth[seq][frame], threshold, margin)
        for ix, row in enumerate(rows):
            assert (row["gt_match_status"], row["gt_id"]) == (status[ix], str(labels.get(ix, "")))
        frames[seq, frame] = labels, status, iou

    summaries, window_rows, aligned_rows = [], [], []
    for case_id, (key, case) in enumerate(sorted(cases.items()), 1):
        seq, start, failure, tid, gid = key
        window = []
        for frame in range(start, start+11):
            labels, status, iou = frames[seq, frame]
            gt_ids = [row["gt_id"] for row in truth[seq][frame]]
            gt_index = gt_ids.index(gid)
            target = truth[seq][frame][gt_index]
            gt_box = target["xyxy"]
            ok = np.flatnonzero(iou[:, gt_index] >= threshold - np.finfo(float).eps)
            candidates = []
            track_row = tracks.get((seq, frame, tid), {})
            for ix in ok:
                det = det_rows[seq, frame][ix]
                meta = quality[seq,frame,ix]
                assert (det["gt_id"], det["score"], det["rgb_valid"]) == (
                    meta["gt_id"], meta["score"], meta["rgb_valid"])
                score = float(det["score"])
                tier = detection_tier(score, protocol["native_config"])
                assert tier == det["detection_tier"]
                rival = np.delete(iou[ix], gt_index)
                other_iou = float(rival.max()) if rival.size else 0.
                iou_value = float(iou[ix, gt_index])
                row = dict(case_id=case_id, sequence=seq, original_track_id=tid,
                           target_gt_id=gid, frame=frame, offset_from_failure=frame-failure,
                           det_index=int(ix), score=score,
                           iou_to_target=iou_value, max_other_gt_iou=other_iou,
                           target_iou_margin=iou_value-other_iou, assignment_status=status[ix],
                           assigned_gt_id=labels.get(ix),
                           same_gt=labels.get(ix) == gid,
                           tier=tier, eligible=tier != "outside_native_score_bands",
                           original_accepted_track=det["accepted_track_id"],
                           original_accepted_stage=det["accepted_stage"],
                           rgb_valid=meta["rgb_valid"] == "True",
                           rgb_quality=number(meta["rgb_quality"]),
                           same_gt_cosine_adjacent=number(meta["same_gt_cosine_adjacent"]) if labels.get(ix) == gid else None,
                           appearance_margin=number(meta["appearance_margin"]) if labels.get(ix) == gid else None,
                           short_side_px=number(meta["short_side_px"]),
                           det_center_error_px=number(det["det_center_error_px"]) if labels.get(ix) == gid else None)
                candidates.append(row)
                aligned_rows.append(row)
            same = [r for r in candidates if r["same_gt"]]
            below = [r for r in same if r["score"] <= low and not r["eligible"]]
            eligible = [r for r in candidates if r["eligible"]]
            rep = max(candidates, key=lambda r: (r["same_gt"], r["iou_to_target"], r["score"]), default={})
            score = rep.get("score")
            window.append(dict(case_id=case_id, sequence=seq,recovery_frame=start,
                               first_failure_frame=failure, frame=frame, offset_from_failure=frame-failure,
                               original_track_id=tid, target_gt_id=gid, aligned=len(candidates),
                               score_eligible_aligned=len(eligible), same_gt_aligned=len(same),
                               same_gt_at_or_below_low=len(below),
                               same_gt_eligible_aligned=sum(r["eligible"] for r in same),
                               ambiguous_aligned=sum(r["assignment_status"] == "ambiguous" for r in candidates),
                               other_gt_overlap_aligned=sum(r["max_other_gt_iou"] >= threshold for r in candidates),
                               target_short_side_px=min(gt_box[2]-gt_box[0],gt_box[3]-gt_box[1]),
                               representative_det=rep.get("det_index"), representative_score=score,
                               representative_iou=rep.get("iou_to_target"),
                               representative_best_rival_iou=rep.get("max_other_gt_iou"),
                               representative_assignment_status=rep.get("assignment_status"),
                               representative_gt=rep.get("assigned_gt_id"),
                               representative_rgb_valid=rep.get("rgb_valid"),
                               representative_rgb_quality=rep.get("rgb_quality"),
                               appearance_margin=rep.get("appearance_margin"),
                               adjacent_identity_cosine=rep.get("same_gt_cosine_adjacent"),
                               original_track_state_after=track_row.get("state_after"),
                               original_track_output_status=track_row.get("output_status"),
                               original_track_pred_center_error_px=number(track_row.get("pred_center_error_px")),
                               original_track_gap_frames=track_row.get("gap_frames")))
            if frame == failure:
                assert len(candidates) == int(case["aligned_detections"]) and not eligible
                assert abs(max(iou[:, gt_index]) - float(case["best_iou"])) < 1e-6
                assert candidates
                first = window[-1]
                failure_candidates = candidates
        forward = [r for r in window if r["frame"] >= failure]
        loose_n = prefix(forward, lambda row: bool(row["aligned"]) and not row["score_eligible_aligned"])
        strict_n = prefix(forward, lambda row: bool(row["same_gt_at_or_below_low"]) and not row["score_eligible_aligned"])
        next_usable = next((r["offset_from_failure"] for r in forward if r["same_gt_eligible_aligned"]), None)
        bridge = 0 < strict_n < len(forward) and bool(forward[strict_n]["same_gt_eligible_aligned"])
        summary = dict(first, max_aligned_score=max(r["score"] for r in failure_candidates),
                       min_aligned_score=min(r["score"] for r in failure_candidates),
                       aligned_at_or_below_low=sum(r["score"] <= low for r in failure_candidates),
                       aligned_at_high_boundary=sum(r["score"] == high for r in failure_candidates),
                       followup_available_frames=len(forward),
                       aligned_outside_band_prefix=loose_n,
                       same_gt_low_only_prefix=strict_n,
                       same_gt_low_prefix_right_censored=strict_n == len(forward),
                       next_same_gt_eligible_offset=next_usable,
                       immediate_usable_after_same_gt_low_prefix=bridge)
        summaries.append(summary)
        window_rows.extend(window)

    selected_candidates = [r for r in aligned_rows if r["offset_from_failure"] == 0]
    paired_appearance = [r for r in summaries if r["appearance_margin"] is not None]
    same_gt_first = [r for r in summaries if r["same_gt_aligned"] > 0]
    prefix_bins = {}
    for n in (1, 2, 3, 5):
        prefix_bins[str(n)] = {
            "exposed": sum(row["followup_available_frames"] >= n for row in summaries),
            "aligned_outside_band": sum(row["aligned_outside_band_prefix"] >= n for row in summaries),
            "same_gt_low_only": sum(row["same_gt_low_only_prefix"] >= n for row in summaries)}
    supportive = [r for r in summaries if r["immediate_usable_after_same_gt_low_prefix"] and r["same_gt_low_only_prefix"] >= 2]
    weak = [r for r in summaries if not r["same_gt_aligned"]]
    examples = {}
    if supportive:
        examples["same_gt_low_prefix_then_eligible"] = max(supportive, key=lambda r: r["same_gt_low_only_prefix"])
    if weak:
        examples["no_unique_same_gt"] = max(weak, key=lambda r: (r["other_gt_overlap_aligned"], r["ambiguous_aligned"]))
    if not weak:
        examples["isolated"] = next((r for r in summaries if r["same_gt_low_only_prefix"] == 1 and not r["immediate_usable_after_same_gt_low_prefix"]), {})
    report = {
        "scope": "Only aligned_but_outside_score at the first interruption in complete same-GT recovery+1..+10 windows.",
        "events": len(summaries), "sequences": count_distribution(r["sequence"] for r in summaries),
        "source_hashes": {str(path.relative_to(run)): digest(path) for path in
                          (protocol_path, analysis_path, split_path, det_audit_path, quality_path, tracks_path)},
        "thresholds": {"low": low, "high": high, "diagnostic_iou": threshold, "diagnostic_ambiguity_margin": margin},
        "first_failure_offset_after_recovery": count_distribution(r["first_failure_frame"]-r["recovery_frame"] for r in summaries),
        "failure_target_short_side_lt_8": sum(row["target_short_side_px"] < 8 for row in summaries),
        "all_aligned_scores_at_failure": describe(r["score"] for r in selected_candidates),
        "max_aligned_score_per_event": describe(r["max_aligned_score"] for r in summaries),
        "candidate_counts": {
            "aligned": len(selected_candidates),
            "at_or_below_low": sum(r["score"] <= low for r in selected_candidates),
            "equal_high_boundary": sum(r["score"] == high for r in selected_candidates),
            "same_gt_assigned": sum(r["same_gt"] for r in selected_candidates),
            "ambiguous": sum(r["assignment_status"] == "ambiguous" for r in selected_candidates),
            "overlaps_another_gt_above_iou_threshold": sum(r["max_other_gt_iou"] >= threshold for r in selected_candidates)
        },
        "events_with_same_gt_at_first_failure": len(same_gt_first),
        "events_with_multiple_aligned_candidates": sum(r["aligned"] > 1 for r in summaries),
        "events_with_other_gt_overlap_at_failure": sum(r["other_gt_overlap_aligned"] > 0 for r in summaries),
        "representative_same_gt_with_valid_rgb": sum(r["representative_rgb_valid"] for r in same_gt_first),
        "same_gt_representative_rgb_quality": describe(r["representative_rgb_quality"] for r in same_gt_first),
        "same_gt_representative_adjacent_cosine": describe(r["adjacent_identity_cosine"] for r in same_gt_first),
        "same_gt_representative_appearance_margin": describe(r["appearance_margin"] for r in paired_appearance),
        "positive_appearance_margin": sum(r["appearance_margin"] > 0 for r in paired_appearance),
        "continuity_prefix": prefix_bins,
        "strict_prefix_right_censored": sum(r["same_gt_low_prefix_right_censored"] for r in summaries),
        "next_same_gt_eligible_within_original_window": sum(r["next_same_gt_eligible_offset"] is not None for r in summaries),
        "bridge_opportunities_by_low_prefix_length": count_distribution(r["same_gt_low_only_prefix"] for r in summaries if r["immediate_usable_after_same_gt_low_prefix"]),
        "sensitivity_ge_0.05_and_same_gt_at_failure": sum((r["representative_score"] >= .05) for r in same_gt_first),
        "sensitivity_ge_0.075_and_same_gt_at_failure": sum((r["representative_score"] >= .075) for r in same_gt_first),
        "example_cases": examples,
        "interpretation": "Oracle GT used only for offline classification. Score sensitivity and prefix/bridge do not measure online eligibility, association success, false positives, or benefit. Prefixes stop at original recovery+10; error target is fixed to the recovery identity."
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "events_89.csv", summaries)
    write_csv(args.output / "event_frames.csv", window_rows)
    write_csv(args.output / "event_aligned_candidates.csv", aligned_rows)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
