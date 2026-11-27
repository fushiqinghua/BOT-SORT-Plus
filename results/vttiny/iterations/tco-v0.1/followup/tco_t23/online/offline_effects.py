"""Attribute TCO measurement and trajectory changes with the frozen CLEAR evaluator."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from replay import sha256


def jsonl(path):
    with Path(path).open("r", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def csv_writer(path, fields):
    handle = path.open("x", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    return handle, writer


def boxes_iou(xyxy, xywh):
    if xyxy is None or xywh is None:
        return None
    left, top, width, height = xywh
    gt = [left, top, left + width, top + height]
    overlap_w = max(0., min(xyxy[2], gt[2]) - max(xyxy[0], gt[0]))
    overlap_h = max(0., min(xyxy[3], gt[3]) - max(xyxy[1], gt[1]))
    intersection = overlap_w * overlap_h
    return intersection / (width * height + (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]) - intersection)


def emitted_indices(run, method, sequence, official_frames):
    path = run / "decisions" / method / f"{sequence}.jsonl"
    emitted = {}
    for event in jsonl(path):
        if event["event"] != "output_index":
            continue
        frame = int(event["frame"])
        rows = {int(row["track_id"]): int(row["det_index"]) for row in event["rows"]}
        if (frame in emitted or len(rows) != len(event["rows"]) or len(set(rows.values())) != len(rows)
                or set(rows) != official_frames[frame - 1]["outputs"].keys()):
            raise ValueError(f"Output row join mismatch: {method}/{sequence}/{frame}")
        emitted[frame] = rows
    if set(emitted) != set(range(1, len(official_frames) + 1)):
        raise ValueError(f"Incomplete output detector-index stream: {method}/{sequence}")
    return emitted


def anchor_gt(record, frames, emitted):
    """Require at least two exact emitted history rows with one CLEAR identity."""
    identity = []
    frame, tid = int(record["frame"]), int(record["track_id"])
    for observation in record["history"]:
        previous, index = int(observation["frame"]), int(observation["det_index"])
        if not 1 <= previous < frame:
            raise ValueError("A measurement history contains current or future observations")
        if emitted[previous].get(tid) == index:
            match = frames[previous - 1]["outputs"][tid]["matched_gt"]
            if match is not None:
                identity.append(match)
    return identity[0] if len(identity) >= 2 and len(set(identity)) == 1 else None


def reference_events(path, changed_tracks):
    counts = defaultdict(Counter)
    for event in jsonl(path):
        tid = event.get("track_id")
        if tid not in changed_tracks:
            continue
        frame = int(event["frame"])
        kind = event["event"]
        if kind == "recovered":
            counts[tid][frame, "B_recovery"] += 1
        elif kind == "match":
            counts[tid][frame, event["stage"]] += 1
    return counts


def write_changes(output, gt, trackeval_path, lengths, runs, roles, summary):
    from experiments.vttiny_gmc.round3_joint import analysis
    from experiments.vttiny_gmc.round3_joint.common import write_json

    trackeval, audit = analysis.load_trackeval(trackeval_path)
    lookup = {(row["tracker"], row["sequence"]): row for row in summary["results"]}
    coverage_fields = ("reference", "sequence", "frame", "key_type", "key", "change")
    local_fields = ("reference", "sequence", "frame", "track_id", "det_index",
                    "known_history_gt_id", "current_gt_present", "raw_iou", "used_iou",
                    "updated_kf_iou", "output_iou", "output_status", "output_gt_id", "gt_covering_track_id")
    future_fields = ("reference", "sequence", "frame", "track_id", "det_index", "gt_id",
                     "requested_frames", "observed_frames", "right_censored", "track_output_frames", "track_FP",
                     "anchor_present_frames", "anchor_same_id_TP", "anchor_other_id_TP", "anchor_FN",
                     "next_stage", "high_matches", "low_matches", "unconfirmed_matches", "B_recoveries")
    out = [(output / name, fields) for name, fields in (("changed_events.csv", coverage_fields),
            ("refinement_effects.csv", local_fields), ("followup_5_10.csv", future_fields))]
    stats = defaultdict(Counter)
    from contextlib import ExitStack
    with ExitStack() as stack:
        writers = []
        for path, fields in out:
            handle, writer = csv_writer(path, fields)
            stack.enter_context(handle)
            writers.append(writer)
        coverage_writer, measurement_writer, after_writer = writers
        for method in roles:
            suffix = method + "_tco"
            off, on = runs[method][0], runs[suffix][0]
            for seq, length in lengths.items():
                results = [analysis.evaluate_sequence(trackeval, gt, output / "evaluation_input" / "trackers",
                           alias, seq, length, output / "metrics") for alias in (method, suffix)]
                for alias, result in zip((method, suffix), results, strict=True):
                    if any(value != lookup[alias, seq][key] for key, value in result["counts"].items() if key != "TP"):
                        raise ValueError(f"An attribution ledger disagrees with official CLEAR: {alias}/{seq}")
                off_frames, on_frames = (value["frames"] for value in results)
                off_indices = emitted_indices(off, method, seq, off_frames)
                on_indices = emitted_indices(on, method, seq, on_frames)
                off_sw, on_sw = ({(row["frame"], row["gt_id"]) for row in value["switches"]} for value in results)
                settings = {"reference": method, "sequence": seq}

                def record_difference(left, right, frame, key_type, gain, loss):
                    for keys, change in ((right - left, gain), (left - right, loss)):
                        for key in sorted(keys):
                            coverage_writer.writerow(dict(**settings, frame=frame, key_type=key_type, key=key, change=change))
                        stats[method][change] += len(keys)

                for frame, (left, right) in enumerate(zip(off_frames, on_frames, strict=True), 1):
                    record_difference(set(left["coverage"]), set(right["coverage"]), frame,
                                      "gt_id", "GT_coverage_gained", "GT_coverage_lost")
                    left_fp, right_fp = (
                        {indices[frame][tid] for tid, row in values["outputs"].items() if row["status"] == "FP"}
                        for values, indices in ((left, off_indices), (right, on_indices)))
                    record_difference(left_fp, right_fp, frame, "raw_det_index", "FP_added", "FP_removed")
                for keys, name in ((on_sw - off_sw, "IDSW_added"), (off_sw - on_sw, "IDSW_removed")):
                    for frame, gid in sorted(keys):
                        coverage_writer.writerow(dict(**settings, frame=frame, key_type="gt_id", key=gid, change=name))
                    stats[method][name] += len(keys)

                source = on / "tco_measurements" / method / f"{seq}.jsonl"
                active = [row for row in jsonl(source) if row["applied"]]
                tracker_events = reference_events(on / "events" / method / f"{seq}.jsonl",
                                                  {row["track_id"] for row in active})
                for row in active:
                    frame, tid, did = row["frame"], row["track_id"], row["det_index"]
                    frame_state = on_frames[frame - 1]
                    anchor = anchor_gt(row, on_frames, on_indices)
                    box = frame_state["gt_boxes"].get(anchor)
                    output_row = frame_state["outputs"].get(tid)
                    if output_row is not None and on_indices[frame][tid] != did:
                        raise ValueError("TCO output held a different detection index after measurement")
                    measurement_writer.writerow(dict(**settings, frame=frame, track_id=tid, det_index=did,
                        known_history_gt_id=anchor, current_gt_present=bool(box) if anchor is not None else None,
                        raw_iou=boxes_iou(row["raw_xyxy"], box), used_iou=boxes_iou(row["used_xyxy"], box),
                        updated_kf_iou=boxes_iou(row["kf_after"]["xyxy"], box),
                        output_iou=boxes_iou(row["output_xyxy"], box), output_status=output_row["status"] if output_row else "not_emitted",
                        output_gt_id=output_row["matched_gt"] if output_row else None,
                        gt_covering_track_id=frame_state["coverage"].get(anchor)))
                    stats[method]["adopted_measurements"] += 1
                    stats[method]["adopted_known_anchor"] += anchor is not None
                    events = tracker_events[tid]
                    for horizon in (5, 10):
                        followup = analysis.followup(on_frames, frame, tid, anchor, horizon)
                        first = [stage for (f, stage), _ in events.items() if f == frame + 1]
                        usage = {stage: sum(events[past, stage] for past in range(frame + 1, frame + 1 + horizon))
                                 for stage in ("high", "low", "unconfirmed", "B_recovery")}
                        after_writer.writerow(dict(**settings, frame=frame, track_id=tid, det_index=did, gt_id=anchor,
                            **followup, next_stage=";".join(sorted(first)), high_matches=usage["high"],
                            low_matches=usage["low"], unconfirmed_matches=usage["unconfirmed"], B_recoveries=usage["B_recovery"]))
            before, after = lookup[method, "COMBINED_SEQ"], lookup[suffix, "COMBINED_SEQ"]
            diff = stats[method]
            for field, observed in (("FN", diff["GT_coverage_lost"] - diff["GT_coverage_gained"]),
                                    ("FP", diff["FP_added"] - diff["FP_removed"]),
                                    ("IDSW", diff["IDSW_added"] - diff["IDSW_removed"])):
                if observed != after[field] - before[field]:
                    raise ValueError(f"Signed {field} change differs from official pooled evaluation: {method}")
    write_json(output / "effect_summary.json", {
        "status": "PASS", "trackeval_commit": audit["commit"], "counts": stats,
        "files_sha256": {path.name: sha256(path) for path, _ in out},
        "fp_event_unit": "frame and original detector index; one consumer per run",
        "gt_event_unit": "frame and official GT identity",
        "identity_causality": "No cross-run comparison of numeric track IDs",
    })
