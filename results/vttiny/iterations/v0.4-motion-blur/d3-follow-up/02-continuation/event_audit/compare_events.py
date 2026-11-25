"""Compare the original D3 recovery windows against independent completed replays."""

import argparse
from collections import Counter, defaultdict
import csv
from functools import cache
import json
from pathlib import Path
import sys
import time

import numpy as np


BASE = "B_early_moderate"
HORIZONS = (3, 5, 10)
ACCEPTED = {"match", "recovered", "native_birth", "birth_committed"}
INPUT_KEYS = (
    "native_config", "sequences", "subset", "detection_fingerprints", "feature_hashes",
    "detector_sha256", "rgb_encoder_sha256", "source_protocol_sha256",
    "encoder_contracts", "boxmot_source_hashes", "decoded_image_hashes",
)


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_cases(frozen, analysis):
    window_path = frozen / "analysis/recovery_windows.csv"
    event_path = frozen / "analysis/recovery_events.csv"
    candidates_path = frozen / "round1_findings/D3_complete_first_failure_candidates.csv"
    for path in (window_path, event_path):
        if sha256(path) != analysis["files_sha256"][path.name]:
            raise ValueError(f"Frozen analysis hash mismatch for {path}")
    recovery_events = {
        (row["sequence"], int(row["frame"]), int(row["track_id"])): row
        for row in csv_rows(event_path)
    }
    candidates = {
        (row["sequence"], int(row["recovery_frame"]), int(row["track_id"])): row
        for row in csv_rows(candidates_path)
    }
    cases = defaultdict(list)
    counts = Counter()
    candidate_failures = set()
    for row in csv_rows(window_path):
        if (row["horizon"], row["identity_status"], row["complete_observable_window"]) != (
                "10", "same_gt", "True"):
            continue
        seq = row["sequence"]
        t, tid, gid = (int(row[col]) for col in ("recovery_frame", "track_id", "gt_id"))
        key = (seq, t, tid)
        event = recovery_events[key]
        if {int(event["previous_gt_id"]), int(event["current_gt_id"])} != {gid}:
            raise ValueError(f"Frozen event history disagrees with window {key}")
        previous = int(event["previous_seen_frame"])
        offset = int(row["first_failure_offset"]) if row["first_failure_offset"] else None
        failure = t + offset if offset else None
        if offset is None:
            group = "originally_stable"
        else:
            candidate = candidates[key]
            if int(candidate["first_failure_frame"]) != failure or int(candidate["gt_id"]) != gid:
                raise ValueError(f"Frozen candidate disagrees with window {key}")
            kind = candidate["candidate_type"]
            group = kind if kind in ("aligned_but_outside_score", "aligned_and_eligible") else "other_original_failure"
            candidate_failures.add(key)
        counts[group] += 1
        cases[seq].append(dict(sequence=seq, recovery_frame=t, baseline_track_id=tid,
                               gt_id=gid, previous_baseline_observation_frame=previous,
                               original_failure_frame=failure,
                               original_failure_reason=row["first_failure_reason"], group=group))
    expected = Counter(originally_stable=77, aligned_but_outside_score=89,
                       aligned_and_eligible=22, other_original_failure=99)
    if counts != expected or candidate_failures != candidates.keys():
        raise ValueError(f"Frozen cohort mismatch: {counts}")
    return cases, {path.name: sha256(path) for path in (window_path, event_path, candidates_path)}


def load_completed(root, name, deadline):
    path = root / "protocols" / f"{name}.json"
    while time.monotonic() < deadline:
        if path.is_file():
            protocol = read_json(path)
            if protocol.get("status") == "COMPLETE":
                return protocol, sha256(path)
            if protocol.get("status") == "FAILED":
                raise RuntimeError(f"Replay failed: {path}")
        time.sleep(3)
    raise TimeoutError(f"Replay did not complete: {path}")


def accepted_updates(path, sequence, last_frame):
    by_frame = defaultdict(dict)
    det_used = defaultdict(set)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event["event"] not in ACCEPTED:
                continue
            frame, tid, det = (int(event[key]) for key in ("frame", "track_id", "det_index"))
            if event.get("sequence", sequence) != sequence or not 1 <= frame <= last_frame:
                raise ValueError(f"Invalid frame or sequence in {path}")
            if tid in by_frame[frame] or det in det_used[frame]:
                raise ValueError(f"Multiple accepted updates for one track or detection: {sequence}:{frame}")
            by_frame[frame][tid] = event
            det_used[frame].add(det)
    return by_frame


class SequenceAudit:
    def __init__(self, truth, detections, mot, events, native, threshold, ambiguity):
        self.truth = truth
        self.detections = detections
        self.mot = mot
        self.events = events
        self.native = native
        self.threshold = threshold
        self.ambiguity = ambiguity
        self.detection_labels = cache(self.detection_labels)
        self.output_state = cache(self.output_state)

    def detection_labels(self, frame):
        return diagnostic_assignment(
            [row["xyxy"] for row in self.detections[frame]], self.truth[frame],
            self.threshold, self.ambiguity)[:2]

    def output_state(self, frame):
        mot = self.mot[frame]
        output = np.asarray([
            (x, y, x + w, y + h, tid, score) for _, tid, x, y, w, h, score in mot
        ], dtype=float).reshape(-1, 6)
        labels, status, _, _ = diagnostic_assignment(
            output[:, :4], self.truth[frame], self.threshold, self.ambiguity)
        detections, det_status = self.detection_labels(frame)
        return _frame_state(frame, self.truth[frame], self.detections[frame], detections,
                            det_status, output, labels, status, [], self.native, self.threshold)

    def accepted_labels(self, frame):
        labels, statuses = self.detection_labels(frame)
        result = {}
        for tid, event in self.events.get(frame, {}).items():
            det = int(event["det_index"])
            if not 0 <= det < len(statuses):
                raise ValueError(f"Accepted detection index out of bounds at frame {frame}")
            result[tid] = labels.get(det)
        return result

    def restoration(self, previous, t, gid):
        old = next((tid for tid, label in self.accepted_labels(previous).items() if label == gid), None)
        labels = self.accepted_labels(t)
        out = self.output_state(t)
        current = next((tid for tid, label in labels.items() if label == gid), None)
        output_tid = out["gt_output"].get(gid)
        if old is None:
            status = "prior_identity_unverified"
        elif labels.get(old) == gid:
            event = self.events[t][old]
            if event["event"] == "recovered":
                actual_previous = t - int(event["age"])
                status = ("same_id_recovered" if self.accepted_labels(actual_previous).get(old) == gid
                          else "recovered_with_unverified_prior_identity")
            else:
                status = "same_id_continued"
        elif labels.get(old) is not None or out["output_labels"].get(old) not in (None, gid):
            status = "prior_id_wrong_identity"
        elif output_tid == old and old not in labels:
            status = "same_id_output_without_accepted_update"
        elif current is not None or output_tid is not None:
            status = "other_id_at_restoration"
        elif old in labels or out["output_status"].get(old) == "ambiguous":
            status = "restoration_identity_unverified"
        else:
            status = "target_missing_at_restoration"
        anchor = old if status in (
            "same_id_recovered", "same_id_continued", "same_id_output_without_accepted_update"
        ) else None
        return old, anchor, current, output_tid, status

    def compare_case(self, case, name):
        gid = case["gt_id"]
        t = case["recovery_frame"]
        failure = case["original_failure_frame"]
        old, anchor, current, output_tid, restoration_status = self.restoration(
            case["previous_baseline_observation_frame"], t, gid)
        row = dict(case, run=name, run_prior_track_id=old, run_verified_anchor_id=anchor,
                   restoration_accepted_track_id=current, restoration_output_track_id=output_tid,
                   restoration_status=restoration_status)
        followups = []
        for offset in range(1, 11):
            frame = t + offset
            state = self.output_state(frame)
            event = self.events.get(frame, {}).get(old, {})
            target_tid = state["gt_output"].get(gid)
            prior_output = (_output_status(state, old, gid) if old is not None
                            else "prior_identity_unverified")
            followups.append(dict(sequence=case["sequence"], recovery_frame=t,
                                  baseline_track_id=case["baseline_track_id"], gt_id=gid,
                                  run=name, frame=frame, offset=offset,
                                  run_prior_track_id=old, target_output_track_id=target_tid,
                                  prior_output_status=prior_output,
                                  verified_anchor_output_correct=anchor is not None and target_tid == anchor,
                                  prior_acceptance=event.get("stage", event.get("event")),
                                  matches_original_failure=(frame == failure)))
        failed = next((f for f in followups if not f["verified_anchor_output_correct"]), None)
        at_failure = next((f for f in followups if f["matches_original_failure"]), None)
        row.update(first_interruption_offset=(failed["offset"] if failed else None),
                   first_interruption_reason=(failed["prior_output_status"] if failed and anchor is not None
                                              else restoration_status if failed else None),
                   original_failure_old_id_correct=(at_failure["prior_output_status"] == "same_id_correct"
                                                    if at_failure else None),
                   original_failure_verified_anchor_correct=(at_failure["verified_anchor_output_correct"]
                                                            if at_failure else None),
                   original_failure_any_id_correct=(at_failure["target_output_track_id"] is not None
                                                    if at_failure else None),
                   original_failure_prior_output_status=(at_failure["prior_output_status"]
                                                         if at_failure else None),
                   original_failure_prior_acceptance=(at_failure["prior_acceptance"] if at_failure else None))
        for horizon in HORIZONS:
            row[f"stable_{horizon}"] = all(f["verified_anchor_output_correct"] for f in followups[:horizon])
        return row, followups


def compare_run(name, root, protocol, frozen, analysis, cases, detection_paths):
    evidence = {}
    rows, frames = [], []
    for seq, length in protocol["sequences"].items():
        gt_path = Path(analysis["ground_truth"]["root"]) / seq / "gt/gt.txt"
        mot_path = root / "trackers" / name / "data" / f"{seq}.txt"
        event_path = root / "events" / name / f"{seq}.jsonl"
        hashes = {"gt": sha256(gt_path), "mot": sha256(mot_path), "events": sha256(event_path)}
        if (hashes["gt"] != analysis["ground_truth"]["sha256"][seq]
                or hashes["mot"] != protocol["output_prediction_hashes"][seq]):
            raise ValueError(f"GT or MOT hash mismatch in {name}/{seq}")
        if name == BASE and hashes["events"] != analysis["original_events_sha256"][seq]:
            raise ValueError(f"Original event hash mismatch in {seq}")
        evidence[seq] = hashes
        if seq not in cases:
            continue
        raw = read_detection_cache(detection_paths[seq])
        if (len(raw["frames"]) != length or
                detection_fingerprint(raw) != protocol["detection_fingerprints"][seq]):
            raise ValueError(f"Detection cache mismatch: {seq}")
        detections = {
            r["frame"]: [{"xyxy": d[:4], "score": float(d[4])} for d in
                         np.asarray(r["detections"], dtype=np.float32).reshape(-1, 6)]
            for r in raw["frames"]
        }
        truth, _ = load_gt(gt_path, length)
        audit = SequenceAudit(truth, detections, _read_mot(mot_path, length),
                              accepted_updates(event_path, seq, length), protocol["native_config"],
                              analysis["iou_threshold"], analysis["ambiguity_margin"])
        for case in cases[seq]:
            row, followups = audit.compare_case(case, name)
            if name == BASE:
                expected_offset = (case["original_failure_frame"] - case["recovery_frame"]
                                   if case["original_failure_frame"] is not None else None)
                if (row["restoration_status"] != "same_id_recovered"
                        or row["run_prior_track_id"] != case["baseline_track_id"]
                        or row["first_interruption_offset"] != expected_offset
                        or (row["first_interruption_reason"] or "") != case["original_failure_reason"]):
                    raise ValueError(f"Original window parity failure: {row}")
            rows.append(row)
            frames.extend(followups)
    return rows, frames, evidence


def group_summary(rows, baseline):
    original = [r for r in rows if r["original_failure_frame"] is not None]
    rescue = [r for r in original if r["original_failure_verified_anchor_correct"]]
    horizons = {}
    for horizon in HORIZONS:
        key = f"stable_{horizon}"
        pairs = [(baseline[identity_key(row)][key], row[key]) for row in rows]
        horizons[str(horizon)] = {
            "continuous": sum(after for _, after in pairs),
            "improved": sum(after and not before for before, after in pairs),
            "worsened": sum(before and not after for before, after in pairs),
        }
    return {
        "cases": len(rows), "restoration_status": dict(Counter(r["restoration_status"] for r in rows)),
        "horizons": horizons, "original_first_failures": len(original),
        "at_original_failure_verified_same_id": len(rescue),
        "at_original_failure_prior_id_regardless_of_restoration": sum(
            r["original_failure_old_id_correct"] for r in original),
        "at_original_failure_any_correct_output": sum(r["original_failure_any_id_correct"] for r in original),
        "at_original_failure_prior_acceptance_among_verified_rescues": dict(Counter(
            r["original_failure_prior_acceptance"] or "no_accepted_update" for r in rescue)),
        "at_original_failure_prior_output_status": dict(Counter(
            r["original_failure_prior_output_status"] for r in original)),
    }


def identity_key(row):
    return (row["sequence"], row["recovery_frame"], row["baseline_track_id"], row["gt_id"])


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run", action="append", required=True, metavar="TRACKER=RUN_DIR")
    parser.add_argument("--wait-seconds", type=int, default=1200)
    args = parser.parse_args()
    repo = args.frozen.resolve().parents[1]
    sys.path.insert(0, str(repo))
    global _frame_state, _output_status, _read_mot, diagnostic_assignment, load_gt, read_json
    global detection_files, detection_fingerprint, read_detection_cache, sequence_name, sha256, write_json
    from botsort_plus.diagnostics.analysis import _frame_state, _output_status, _read_mot
    from botsort_plus.diagnostics.common import diagnostic_assignment, load_gt, read_json
    from botsort_plus.io import (
        detection_files, detection_fingerprint, read_detection_cache, sequence_name, sha256, write_json,
    )

    args.output.mkdir(parents=True, exist_ok=False)
    analysis = read_json(args.frozen / "analysis/summary.json")
    cases, frozen_hashes = read_cases(args.frozen, analysis)
    frozen_hashes["analysis_summary.json"] = sha256(args.frozen / "analysis/summary.json")
    cache_dir = repo / "results/vttiny/yolo11m_rgb_rgbt/rgbt/detections"
    detection_paths = {sequence_name(path): path for path in detection_files(cache_dir)}
    runs = [(BASE, args.frozen / "observed")]
    runs.extend((name, Path(path)) for name, path in (value.split("=", 1) for value in args.run))
    names = [name for name, _ in runs]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate tracker name")
    evidence = {}
    comparison = {}
    all_rows, all_frames = [], []
    deadline = time.monotonic() + args.wait_seconds
    for name, root in runs:
        protocol, protocol_sha = load_completed(root, name, deadline)
        if name == BASE:
            original_protocol = protocol
            if protocol_sha != analysis["run_protocol_sha256"]:
                raise ValueError("Observed baseline protocol differs from the frozen analysis")
        for key in INPUT_KEYS:
            if protocol[key] != original_protocol[key]:
                raise ValueError(f"Replay input mismatch: {name}/{key}")
        rows, frames, hashes = compare_run(name, root, protocol, args.frozen, analysis, cases, detection_paths)
        if name == BASE:
            baseline = {identity_key(row): row for row in rows}
        if len(rows) != 287 or len(frames) != 2870 or len({identity_key(row) for row in rows}) != 287:
            raise ValueError(f"Cohort coverage failure in {name}")
        groups = {group: [r for r in rows if r["group"] == group] for group in sorted({r["group"] for r in rows})}
        comparison[name] = {group: group_summary(records, baseline) for group, records in groups.items()}
        comparison[name]["all_287"] = group_summary(rows, baseline)
        evidence[name] = {"root": str(root), "protocol_sha256": protocol_sha, "file_sha256": hashes}
        write_json(args.output / f"{name}.json", comparison[name])
        print(json.dumps({"run": name, "all_287": comparison[name]["all_287"]}), flush=True)
        all_rows.extend(rows)
        all_frames.extend(frames)
    write_csv(args.output / "paired_events.csv", all_rows)
    write_csv(args.output / "paired_frames.csv", all_frames)
    write_json(args.output / "summary.json", {
        "status": "COMPLETE", "analysis_sha256": frozen_hashes, "input_protocol_fields_verified": INPUT_KEYS,
        "iou_threshold": analysis["iou_threshold"], "ambiguity_margin": analysis["ambiguity_margin"],
        "original_case_parity": {"verified_original_restoration_and_first_failure": len(baseline)},
        "cohorts": comparison, "run_evidence": evidence,
        "files_sha256": {p.name: sha256(p) for p in sorted(args.output.glob("*.csv"))},
    })


if __name__ == "__main__":
    main()
