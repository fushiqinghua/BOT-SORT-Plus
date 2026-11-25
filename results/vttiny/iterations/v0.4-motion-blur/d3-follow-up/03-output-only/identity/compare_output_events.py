"""Compare fixed D3 identity events and label native versus patched frame output."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np

import compare_events as prior


NEW = "B_postlow_output_v1"
STATEFUL = "B_postlow_v1"
OUTPUT_EVENT = "output_only_low"
SEQUENCE_INPUTS = {"sequences", "detection_fingerprints", "feature_hashes", "decoded_image_hashes"}


def events_and_native_output(name, replay, sequence, length, emitted):
    path = replay / "events" / name / f"{sequence}.jsonl"
    updates = prior.accepted_updates(path, sequence, length)
    extras = defaultdict(dict)
    if name == NEW:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                if event["event"] != OUTPUT_EVENT:
                    continue
                frame, track = int(event["frame"]), int(event["track_id"])
                if event["stage"] != "post_B_weak_output" or not 1 <= frame <= length:
                    raise ValueError(f"Invalid output-only event in {sequence}:{frame}:{track}")
                if track in extras[frame] or track in updates.get(frame, {}):
                    raise ValueError(f"Output-only trace collides with accepted update in {sequence}:{frame}:{track}")
                rows = [row for row in emitted[frame] if row[1] == track]
                if len(rows) != 1:
                    raise ValueError(f"Output-only trace has {len(rows)} MOT rows in {sequence}:{frame}:{track}")
                _, _, x, y, w, h, _ = rows[0]
                if not np.allclose(event["xyxy"], [x, y, x + w, y + h], atol=2e-3, rtol=0.):
                    raise ValueError(f"Output-only trace geometry differs from MOT row in {sequence}:{frame}:{track}")
                extras[frame][track] = event
    native = {f: [row for row in emitted[f] if row[1] not in extras.get(f, {})]
              for f in range(1, length + 1)} if extras else emitted
    return updates, extras, native


def bind_diagnostic_imports(repo):
    sys.path.insert(0, str(repo))
    from botsort_plus.diagnostics.analysis import _frame_state, _output_status, _read_mot
    from botsort_plus.diagnostics.common import diagnostic_assignment, load_gt, read_json
    from botsort_plus.io import (
        detection_files, detection_fingerprint, read_detection_cache, sequence_name, sha256, write_json,
    )

    for function in (_frame_state, _output_status, _read_mot, diagnostic_assignment, load_gt, read_json,
                     detection_files, detection_fingerprint, read_detection_cache, sequence_name, sha256,
                     write_json):
        setattr(prior, function.__name__, function)


def comparison_run(name, root, protocol, audit, cases, det_paths, selected):
    rows, frames, evidence, overlay_truth = [], [], {}, []
    for seq in selected:
        length = protocol["sequences"][seq]
        gt = Path(audit["ground_truth"]["root"]) / seq / "gt/gt.txt"
        mot = root / "trackers" / name / "data" / f"{seq}.txt"
        events = root / "events" / name / f"{seq}.jsonl"
        hashes = {"gt": prior.sha256(gt), "mot": prior.sha256(mot), "events": prior.sha256(events)}
        if hashes["gt"] != audit["ground_truth"]["sha256"][seq] or hashes["mot"] != protocol["output_prediction_hashes"][seq]:
            raise ValueError(f"GT or MOT hash mismatch in {name}/{seq}")
        if name == prior.BASE and hashes["events"] != audit["original_events_sha256"][seq]:
            raise ValueError(f"Baseline event bytes differ for {seq}")
        evidence[seq] = hashes
        if seq not in cases and name != NEW:
            continue

        cache = prior.read_detection_cache(det_paths[seq])
        if len(cache["frames"]) != length or prior.detection_fingerprint(cache) != protocol["detection_fingerprints"][seq]:
            raise ValueError(f"Detection cache mismatch in {seq}")
        dets = {
            entry["frame"]: [{"xyxy": box[:4], "score": float(box[4])}
                             for box in np.array(entry["detections"], dtype=np.float32).reshape(-1, 6)]
            for entry in cache["frames"]
        }
        truth, _ = prior.load_gt(gt, length)
        mot_rows = prior._read_mot(mot, length)
        updates, extras, native = events_and_native_output(name, root, seq, length, mot_rows)
        args = (updates, protocol["native_config"], audit["iou_threshold"], audit["ambiguity_margin"])
        emitted_audit = prior.SequenceAudit(truth, dets, mot_rows, *args)
        native_audit = prior.SequenceAudit(truth, dets, native, *args) if extras else emitted_audit

        for frame, labels in extras.items():
            output = emitted_audit.output_state(frame)
            for track, event in labels.items():
                recovery = int(event["b_recovery_frame"])
                age = frame - recovery
                tag_gt = output["output_labels"].get(track)
                recovery_gt = emitted_audit.accepted_labels(recovery).get(track)
                if not 1 <= age <= 3 or int(event["last_native_observation_frame"]) != frame - 1:
                    raise ValueError(f"Patched frame violates recovery age or immediately previous observation: {seq}/{frame}/{track}")
                if updates.get(recovery, {}).get(track, {}).get("event") != "recovered" or track not in updates.get(frame - 1, {}):
                    raise ValueError(f"Patched frame lacks existing real recovery or observation: {seq}/{frame}/{track}")
                det_index = int(event["det_index"])
                det = dets[frame][det_index]
                if not np.allclose(event["xyxy"], det["xyxy"], atol=2e-3, rtol=0.) or not np.isclose(event["score"], det["score"]):
                    raise ValueError(f"Patched frame trace does not use the cached detection: {seq}/{frame}/{track}")
                previous_gt = emitted_audit.accepted_labels(frame - 1).get(track)
                overlay_truth.append({
                    "sequence": seq, "frame": frame, "track_id": track, "det_index": event["det_index"],
                    "recovery_frame": recovery, "score": event["score"], "predicted_iou": 1. - float(event["cost"]),
                    "recovery_gt_id": recovery_gt, "previous_gt_id": previous_gt, "emitted_gt_id": tag_gt,
                    "emitted_diagnostic_status": output["output_status"].get(track),
                    "labelled_original_id": tag_gt is not None and tag_gt == recovery_gt,
                    "agrees_with_previous_observation": tag_gt is not None and tag_gt == previous_gt,
                })

        for case in cases.get(seq, ()):
            row, followups = emitted_audit.compare_case(case, name)
            native_row, native_followups = native_audit.compare_case(case, name)
            if name == prior.BASE:
                old_offset = (case["original_failure_frame"] - case["recovery_frame"]
                              if case["original_failure_frame"] is not None else None)
                if (row["restoration_status"] != "same_id_recovered"
                        or row["run_prior_track_id"] != case["baseline_track_id"]
                        or row["first_interruption_offset"] != old_offset
                        or (row["first_interruption_reason"] or "") != case["original_failure_reason"]):
                    raise ValueError(f"Frozen D3 case did not reproduce {case}")
            at_failure = None
            for f, native_f in zip(followups, native_followups, strict=True):
                export = extras.get(f["frame"], {}).get(f["run_prior_track_id"])
                f["output_only_on_prior_id"] = export is not None
                f["output_only_det_index"] = export.get("det_index") if export else None
                f["native_verified_anchor_output_correct"] = native_f["verified_anchor_output_correct"]
                f["native_prior_output_status"] = native_f["prior_output_status"]
                if f["matches_original_failure"]:
                    at_failure = f
            row["original_failure_native_verified_anchor_correct"] = native_row["original_failure_verified_anchor_correct"]
            row["original_failure_true_output_only_rescue"] = (
                at_failure["verified_anchor_output_correct"] and at_failure["output_only_on_prior_id"]
                and not at_failure["native_verified_anchor_output_correct"] if at_failure else None
            )
            for horizon in prior.HORIZONS:
                row[f"native_stable_{horizon}"] = native_row[f"stable_{horizon}"]
            rows.append(row)
            frames.extend(followups)
    return rows, frames, evidence, overlay_truth


def summarize(rows, baseline):
    result = prior.group_summary(rows, baseline)
    result.update({
        "output_only_verified_same_id_rescues": sum(r["original_failure_true_output_only_rescue"] is True
                                                   for r in rows),
        "native_verified_same_id_rescues": sum(r["original_failure_native_verified_anchor_correct"] is True
                                               for r in rows),
        "native_continuity": {str(horizon): sum(r[f"native_stable_{horizon}"] for r in rows)
                              for horizon in prior.HORIZONS},
    })
    return result


def compare_protocol(name, current, original, selected):
    if name == NEW and set(current["sequences"]) != set(selected):
        raise ValueError(f"Unexpected sequence set in new replay {current['sequences']}")
    if current["subset"] != (name == NEW and len(selected) != len(original["sequences"])):
        raise ValueError(f"Unexpected subset contract in {name}")
    for key in prior.INPUT_KEYS:
        if key == "subset":
            continue
        expected, actual = original[key], current[key]
        if key in SEQUENCE_INPUTS:
            expected = {seq: expected[seq] for seq in selected}
            actual = {seq: actual[seq] for seq in selected}
        if actual != expected:
            raise ValueError(f"Frozen input mismatch {name}/{key}")
    if {**current["config"], "name": prior.BASE} != original["config"]:
        raise ValueError(f"Effective config mismatch {name}")
    if not current["causal"] or current["interpolation"] or current["backfilled_frames"]:
        raise ValueError(f"Noncausal output in {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("frozen", "stateful", "new", "output", "prior_comparison"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--wait-seconds", type=int, default=1200)
    args = parser.parse_args()
    repo = args.frozen.resolve().parents[1]
    bind_diagnostic_imports(repo)
    args.output.mkdir(parents=True, exist_ok=False)
    previous = prior.read_json(args.prior_comparison)
    frozen = prior.read_json(args.frozen / "analysis/summary.json")
    cases, frozen_hashes = prior.read_cases(args.frozen, frozen)
    frozen_hashes["analysis_summary.json"] = prior.sha256(args.frozen / "analysis/summary.json")
    if previous["analysis_sha256"] != frozen_hashes:
        raise ValueError("Frozen inputs have changed since the previous D3 comparison")
    det_paths = {prior.sequence_name(path): path for path in prior.detection_files(
        repo / "results/vttiny/yolo11m_rgb_rgbt/rgbt/detections")}
    deadline = time.monotonic() + args.wait_seconds
    original, base_sha = prior.load_completed(args.frozen / "observed", prior.BASE, deadline)
    if base_sha != frozen["run_protocol_sha256"]:
        raise ValueError("Frozen run protocol changed")
    selected = tuple(args.sequences or original["sequences"])
    if set(selected) - original["sequences"].keys() or len(selected) != len(set(selected)):
        raise ValueError(f"Invalid scope: {selected}")
    denominator = sum(len(cases.get(seq, ())) for seq in selected)
    case_counts = Counter(case["group"] for seq in selected for case in cases.get(seq, ()))
    scope = "full" if len(selected) == len(original["sequences"]) else "subset"

    all_rows, all_frames = [], []
    evidence, cohorts = {}, {}
    overlays = []
    for name, replay in ((prior.BASE, args.frozen / "observed"), (STATEFUL, args.stateful), (NEW, args.new)):
        protocol, sha = prior.load_completed(replay, name, deadline)
        compare_protocol(name, protocol, original, selected)
        if name != NEW and sha != previous["run_evidence"][name]["protocol_sha256"]:
            raise ValueError(f"Previous replay protocol changed for {name}")
        rows, frames, hashes, overlays_for_run = comparison_run(name, replay, protocol, frozen, cases,
                                                                det_paths, selected)
        if name != NEW and any(previous["run_evidence"][name]["file_sha256"][seq] != h
                               for seq, h in hashes.items()):
            raise ValueError(f"Previous replay outputs changed for {name}")
        overlays.extend(overlays_for_run)
        keyed = {prior.identity_key(row): row for row in rows}
        if len(rows) != denominator or len(keyed) != denominator or len(frames) != denominator * 10:
            raise ValueError(f"Invalid D3 window coverage for {name}")
        if name == prior.BASE:
            baseline = keyed
        if set(keyed) != set(baseline):
            raise ValueError("D3 fixed-window keys disagree across replays")
        groups = {group: [r for r in rows if r["group"] == group] for group in sorted(case_counts)}
        cohorts[name] = {group: summarize(records, baseline) for group, records in groups.items()}
        cohorts[name]["all"] = summarize(rows, baseline)
        if name != NEW and scope == "full":
            for group, records in {**groups, "all_287": rows}.items():
                if prior.group_summary(records, baseline) != previous["cohorts"][name][group]:
                    raise ValueError(f"Previous D3 result mismatch in {name}/{group}")
        if name == NEW and any(row[f"native_stable_{h}"] != baseline[key][f"stable_{h}"]
                               for key, row in keyed.items() for h in prior.HORIZONS):
            raise ValueError("Native-only D3 continuity differs from baseline")
        evidence[name] = {"root": str(replay), "protocol_sha256": sha, "file_sha256": hashes}
        all_rows.extend(rows)
        all_frames.extend(frames)
        print(json.dumps({"run": name, **cohorts[name]["all"]}), flush=True)
    prior.write_csv(args.output / "paired_events.csv", all_rows)
    prior.write_csv(args.output / "paired_frames.csv", all_frames)
    if overlays:
        prior.write_csv(args.output / "output_only_truth.csv", overlays)
    overlay_by_seq = {
        seq: dict(total=sum(r["sequence"] == seq for r in overlays),
                  original_gt=sum(r["sequence"] == seq and r["labelled_original_id"] for r in overlays))
        for seq in selected
    }
    prior.write_json(args.output / "summary.json", {
        "status": "COMPLETE", "scope": scope, "sequences": selected, "fixed_event_count": denominator,
        "fixed_event_counts_by_sequence": {seq: len(cases.get(seq, ())) for seq in selected},
        "analysis_sha256": frozen_hashes, "legacy_comparison_sha256": prior.sha256(args.prior_comparison),
        "input_protocol_fields_verified": prior.INPUT_KEYS,
        "original_case_parity": {"verified_original_restoration_and_first_failure": len(baseline)},
        "iou_threshold": frozen["iou_threshold"], "ambiguity_margin": frozen["ambiguity_margin"],
        "cohorts": cohorts, "all_output_only_emits_by_sequence": overlay_by_seq, "run_evidence": evidence,
        "files_sha256": {p.name: prior.sha256(p) for p in sorted(args.output.glob("*.csv"))},
    })


if __name__ == "__main__":
    main()
