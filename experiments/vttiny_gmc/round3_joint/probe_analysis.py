"""Offline J3 interpretation of causal probe traces, with pinned CLEAR labels.

This module runs after prediction.  It does not fit thresholds, modify a
tracker, or call a conditional counterfactual a complete alternative method.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import statistics

try:
    from . import analysis
except ImportError:  # Direct script invocation.
    import analysis


SCHEMA = "v04_joint_probe_v1"


def probe_frames(path, expected_frames):
    """Stream complete frame transactions; reject truncated or reordered logs."""
    opener = gzip.open if Path(path).suffix == ".gz" else open
    next_frame, events, contract_seen = 1, [], False
    with opener(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            record = json.loads(line)
            if record.get("event") == "contract":
                if (contract_seen or number != 1 or record.get("schema") != SCHEMA or
                        record.get("reads_gt") is not False or record.get("affects_reference") is not False):
                    raise ValueError("Invalid J3 causal trace contract")
                contract_seen = True
                continue
            if not contract_seen or record.get("frame") != next_frame:
                raise ValueError(f"J3 frame order/truncation at {path}:{number}")
            if record.get("event") == "parity_failure":
                raise ValueError(f"Failed probe parity: {record.get('check')}")
            events.append(record)
            if record.get("event") == "frame_end":
                yield next_frame, events
                events = []
                next_frame += 1
    if not contract_seen or events or next_frame - 1 != expected_frames:
        raise ValueError(f"Incomplete J3 trace: {next_frame-1}/{expected_frames} complete frames")


def _iou_xywh_xyxy(a, b):
    ax, ay, aw, ah = map(float, a)
    bx, by, br, bb = map(float, b)
    intersection = max(0., min(ax + aw, br) - max(ax, bx)) * max(0., min(ay + ah, bb) - max(ay, by))
    union = aw * ah + (br - bx) * (bb - by) - intersection
    return intersection / union if union > 0 else 0.


def raw_detection_label(frame, dets, det_index, anchor_gt):
    """Unique current-frame GT support only; never physical-background truth."""
    missing = {"raw_detection_status": "unknown_missing_exact_detection", "raw_matched_gt": None,
               "raw_anchor_iou": None, "raw_gt_already_covered": None}
    if dets is None or det_index is None:
        return missing
    index = int(det_index)
    if index != det_index or index < 0 or index >= len(dets):
        raise ValueError("Probe detector index is outside the verified ordered cache")
    similarities = {gid: _iou_xywh_xyxy(box, dets[index][:4]) for gid, box in frame.get("gt_boxes", {}).items()}
    passing = [gid for gid, value in similarities.items() if value >= .5]
    match = passing[0] if len(passing) == 1 else None
    if not similarities:
        status = "unknown_no_valid_GT"
    elif len(passing) > 1:
        status = "unknown_multiple_GT_support"
    elif not passing:
        status = "below_CLEAR_IoU_threshold"
    elif anchor_gt is None:
        status = "unique_GT_support_prior_identity_unknown"
    elif match == anchor_gt:
        status = "supports_prior_GT"
    else:
        status = "supports_different_GT"
    return {"raw_detection_status": status, "raw_matched_gt": match,
            "raw_anchor_iou": similarities.get(anchor_gt) if anchor_gt is not None else None,
            "raw_gt_already_covered": int(match in frame["coverage"]) if match is not None else None}


def actual_label(frame, tid, anchor):
    gt = anchor[0] if anchor is not None else None
    output = frame["outputs"].get(tid)
    if anchor is None:
        label = "unknown_prior_identity"
    elif output is None:
        label = "not_emitted"
    elif output["matched_gt"] == gt:
        label = "same_GT_identity"
    elif output["matched_gt"] is not None:
        label = "different_GT_identity"
    elif gt not in frame["gt_ids"]:
        label = "FP_anchor_GT_absent"
    elif output["gt_ious"].get(gt, 0.) >= .5:
        label = "FP_competition"
    else:
        label = "FP_localization_or_wrong_identity_unresolved"
    return {"prior_anchor_gt": gt, "prior_anchor_frame": anchor[1] if anchor is not None else None,
            "actual_outcome": label, "current_matched_gt": output["matched_gt"] if output else None}


def _pairs(record, key="pairs"):
    # Candidate IDs can differ across private C pools. The observation index
    # and old identity are common at this frame; compare those, not private IDs.
    return {(int(pair[0]), int(pair[2])) for pair in record.get(key, [])}


def _pair_change(base, alternate):
    return {"selected_pairs": len(base), "native_pairs": len(alternate),
            "selected_only_pairs": len(base - alternate), "native_only_pairs": len(alternate - base),
            "changed": int(base != alternate)}


def route_rows(events):
    """Count conditional P/N/C changes; never sum routes into a causal total."""
    result = []
    grouped = defaultdict(dict)
    for record in events:
        if "branch" in record:
            key, branch = record["event"], record["branch"]
            if branch in grouped[key]:
                raise ValueError(f"Duplicate branch record: {key}/{branch}")
            grouped[key][branch] = record
    for event, route, pair_key in (("prediction_recovery", "P_B_assignment", "pairs"),
                                   ("candidate_recovery", "C_B_assignment", "pairs"),
                                   ("native_counterfactual", "N_B_assignment", "recovery_pairs")):
        branches = grouped.get(event, {})
        if {"selected", "native"} <= branches.keys():
            result.append({"route": route, "available": 1, "reason": "conditional_same_reference_history",
                           **_pair_change(_pairs(branches["selected"], pair_key), _pairs(branches["native"], pair_key))})
        else:
            result.append({"route": route, "available": 0, "reason": "not_recorded_in_this_probe_mode_or_gate",
                           **dict.fromkeys(("selected_pairs", "native_pairs", "selected_only_pairs", "native_only_pairs", "changed"))})
    native = grouped.get("native_counterfactual", {})
    if {"selected", "native"} <= native.keys():
        for stage in ("high", "low", "unconfirmed"):
            sets = {}
            for branch in ("selected", "native"):
                matching = [row for row in native[branch]["stages"] if row["stage"] == stage]
                if len(matching) != 1:
                    raise ValueError("Missing or duplicate native stage in probe")
                sets[branch] = {tuple(map(int, pair)) for pair in matching[0]["pairs"]}
            result.append({"route": "N_" + stage, "available": 1, "reason": "single_frame_same_prior",
                           **_pair_change(sets["selected"], sets["native"])})
        for key in ("residual_indices", "eligible_lost_ids"):
            # Reuse the generic count fields; the route explicitly names the unit.
            result.append({"route": "N_" + key, "available": 1, "reason": "single_frame_same_prior",
                           **_pair_change(set(native["selected"][key]), set(native["native"][key]))})
    candidate = grouped.get("candidate_counterfactual", {})
    if {"selected", "native"} <= candidate.keys():
        groups = {branch: {tuple(tuple(map(int, key)) for key in history)
                           for history in candidate[branch]["candidates"].values()}
                  for branch in ("selected", "native")}
        result.append({"route": "C_observation_members", "available": 1, "reason": "fixed_residual_and_actual_consumption_schedule",
                       **_pair_change(groups["selected"], groups["native"])})
    return result


def analyze_frame(events, frame, anchors, dets=None):
    """Detached offline joins; the caller advances anchors after this returns."""
    motion, proposals, competition = [], [], []
    actual = [event for event in events if event["event"] == "actual_recovery"]
    if len(actual) != 1:
        raise ValueError("Expected one actual_recovery record per frame")
    actual = actual[0]
    accepted = {(int(pair[0]), int(pair[1])): int(pair[2]) for pair in actual["pairs"]}
    end = events[-1]
    if end["event"] != "frame_end":
        raise ValueError("J3 frame transaction lacks frame_end")
    emitted = {int(row[4]): row for row in end["outputs"]}
    if len(emitted) != len(end["outputs"]) or set(emitted) != set(frame["outputs"]):
        raise ValueError("Trace output identities differ from validated MOT")
    for tid, row in emitted.items():
        x, y, w, h = frame["outputs"][tid]["box_xywh"]
        if max(abs(float(a) - float(b)) for a, b in zip(row[:4], (x, y, x+w, y+h))) > 1e-7:
            raise ValueError("Trace boxes differ from validated MOT")
    detection_by_candidate = {int(pair[1]): int(pair[2]) for pair in actual["pairs"]}
    for cid, history in actual.get("current_candidates", {}).items():
        if history and int(history[-1][0]) == frame["frame"]:
            detection_by_candidate[int(cid)] = int(history[-1][1])
    for event in events:
        if event["event"] == "identity_competition":
            detection_by_candidate[int(event["candidate_id"])] = int(event["current_det_index"])
        elif event["event"] == "candidate_counterfactual" and event["branch"] == "selected":
            for cid, history in event["candidates"].items():
                if history and int(history[-1][0]) == frame["frame"]:
                    detection_by_candidate[int(cid)] = int(history[-1][1])
    decision_map = {(int(row["track_id"]), int(row["candidate_id"])): row for row in actual["diagnostics"]}
    if len(decision_map) != len(actual["diagnostics"]):
        raise ValueError("Duplicate identity/candidate decision in actual recovery")
    motion_by_id = {}
    native_matches = {int(event["track_id"]): event for event in end["reference_events"]
                      if event.get("event") == "match" and "track_id" in event}
    for event in events:
        if event["event"] != "prediction_state":
            continue
        tid = int(event["track_id"])
        fields = {key: event.get(key) for key in ("anchor_frame", "last_seen", "state", "center_pixels",
                  "center_normalized", "size_normalized", "normalization_scale")}
        b_accepted = any(identity == tid for identity, _ in accepted)
        native = native_matches.get(tid, {})
        row = {"frame": frame["frame"], "track_id": tid, **fields,
               "accepted_B": int(b_accepted), "accepted_native_stage": native.get("stage"),
               "native_reactivation_from_recorded_Lost": int(native.get("stage") == "high" and event.get("state") == 2),
               **actual_label(frame, tid, anchors.get(tid))}
        motion.append(row)
        motion_by_id[tid] = row
    for (tid, cid), event in decision_map.items():
        label = actual_label(frame, tid, anchors.get(tid))
        did = detection_by_candidate.get(cid)
        diff = motion_by_id.get(tid, {})
        proposals.append({"frame": frame["frame"], "track_id": tid, "candidate_id": cid,
                          "det_index": did, "accepted": int((tid, cid) in accepted),
                          "reason": event.get("reason"), "age": event.get("age"), "hits": event.get("hits"),
                          "cost": event.get("cost"), "geometry_cost": event.get("geometry_cost"),
                          "appearance_cost": event.get("appearance_cost"), "appearance_hits": event.get("appearance_hits"),
                          "center_normalized": diff.get("center_normalized"), "size_normalized": diff.get("size_normalized"),
                          **label, **raw_detection_label(frame, dets, did, label["prior_anchor_gt"])})
    for event in events:
        if event["event"] != "identity_competition":
            continue
        active = [score for score in event["scores"] if score["role"] == "active"]
        available = [score for score in active if score.get("available") and score.get("score") is not None]
        best = min(available, key=lambda score: (score["score"], score["track_id"])) if available else None
        cid, did = int(event["candidate_id"]), int(event["current_det_index"])
        for score in event["scores"]:
            if score["role"] != "lost":
                continue
            tid = int(score["track_id"])
            label = actual_label(frame, tid, anchors.get(tid))
            decision = decision_map.get((tid, cid), {})
            lost_score = score.get("score") if score.get("available") else None
            raw_difference = float(lost_score - best["score"]) if lost_score is not None and best else None
            comparable = (best is not None and lost_score is not None and
                          all(score.get(key) is not None and score[key] == best.get(key)
                              for key in ("appearance_hits", "history_hits")))
            difference = raw_difference if comparable else None
            active_output = frame["outputs"].get(int(best["track_id"])) if best else None
            competition.append({"frame": frame["frame"], "track_id": tid, "candidate_id": cid, "det_index": did,
                                "accepted": int((tid, cid) in accepted), "reason": decision.get("reason"),
                                "lost_support": lost_score, "lost_support_reason": score.get("reason"),
                                "lost_appearance_hits": score.get("appearance_hits"),
                                "lost_history_hits": score.get("history_hits"),
                                "active_identities": len(active), "active_available": len(available),
                                "best_active_id": best["track_id"] if best else None,
                                "best_active_support": best["score"] if best else None,
                                "best_active_appearance_hits": best.get("appearance_hits") if best else None,
                                "best_active_history_hits": best.get("history_hits") if best else None,
                                "best_active_current_gt": active_output["matched_gt"] if active_output else None,
                                "support_coverage_status": "matched" if comparable else (
                                    "coverage_mismatch_or_unknown" if raw_difference is not None else "score_unavailable"),
                                "unadjusted_lost_minus_best_active": raw_difference,
                                "lost_minus_best_active": difference,
                                "active_stronger": None if difference is None else int(difference > 0),
                                **label, **raw_detection_label(frame, dets, did, label["prior_anchor_gt"])})
    return route_rows(events), motion, proposals, competition


def _detector_frames(settings, sequence, protocol):
    if not settings.get("baseline_root"):
        return None, {"status": "MISSING", "reason": "baseline_root not configured"}
    folder = Path(settings["baseline_root"]) / "detections"
    paths = [path for path in (folder/f"{sequence}.json.gz", folder/f"{sequence}.json") if path.is_file()]
    if not paths:
        return None, {"status": "MISSING", "reason": "ordered detection cache unavailable"}
    if len(paths) != 1:
        raise ValueError("Ambiguous compressed/uncompressed detector caches")
    from botsort_plus.io import read_detection_cache, detection_fingerprint
    cache = read_detection_cache(paths[0])
    fingerprint = detection_fingerprint(cache)
    if fingerprint != protocol["detection_fingerprints"][sequence]:
        raise ValueError("Detector rows differ from the probe's frozen prediction inputs")
    if len(cache["frames"]) != protocol["sequences"][sequence]:
        raise ValueError("Detector cache/probe frame count mismatch")
    return [row["detections"] for row in cache["frames"]], {
        "status": "VERIFIED", "path": str(paths[0]), "sha256": analysis.digest(paths[0]), "fingerprint": fingerprint}


def _write_table(path, rows, empty_fields):
    analysis.csv_rows(path, rows, list(rows[0]) if rows else empty_fields)


def load_decisions(run_dir, tracker, sequence, length, protocol):
    path = Path(run_dir)/"decisions"/tracker/f"{sequence}.jsonl"
    if not path.is_file():
        return {}, {}, {"status": "UNKNOWN_MISSING_DECISIONS", "path": str(path)}
    checksum = analysis.digest(path)
    declared = protocol.get("joint_batch", {}).get("decision_manifests", {}).get(sequence)
    if declared is not None:
        if declared.get("data_sha256") != checksum or declared.get("frames") != length:
            raise ValueError("Decision log differs from the run protocol")
    by_frame, guarded = defaultdict(list), defaultdict(list)
    last, output_frames = 0, set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            event = json.loads(line)
            frame = event.get("frame")
            if (isinstance(frame, bool) or not isinstance(frame, int) or
                    not 1 <= frame <= length or frame < last or event.get("sequence") != sequence):
                raise ValueError(f"Invalid decision-log frame at {path}:{number}")
            last = frame
            by_frame[frame].append(event)
            if event["event"] == "output_index":
                if frame in output_frames:
                    raise ValueError("Duplicate output_index transaction")
                output_frames.add(frame)
            elif event["event"] == "dual_low_rejected_edge":
                guarded[(frame, int(event["det_index"]))].append(event)
    if output_frames != set(range(1, length+1)):
        raise ValueError("Incomplete decision output-index frame coverage")
    return by_frame, guarded, {"status": "COMPLETE", "path": str(path), "sha256": checksum,
                              "bound_to_protocol_hash": declared is not None,
                              "guarded_edges": sum(map(len, guarded.values()))}


def decision_joins(events, frame, anchors, decisions, guards):
    """Join only recorded decisions and exact observation memberships.

    guards is indexed by (frame, detector row); no scan of future frames is
    needed and the join never equates a cost decrease with fresh evidence.
    """
    native_rows, guard_rows, indices = [], [], []
    actual = next(event for event in events if event["event"] == "actual_recovery")
    accepted = {(int(pair[0]), int(pair[1])) for pair in actual["pairs"]}
    motion = {int(event["track_id"]): event for event in events if event["event"] == "prediction_state"}
    current = actual.get("current_candidates", {})
    if not current:
        selected = next((event for event in events if event["event"] == "candidate_counterfactual" and
                         event.get("branch") == "selected"), {})
        current = selected.get("candidates", {})
    proposals = defaultdict(list)
    for diagnostic in actual["diagnostics"]:
        proposals[int(diagnostic["candidate_id"])].append(diagnostic)
    expected_indices = {int(row[4]): int(row[7]) for row in events[-1]["outputs"]}
    for event in decisions:
        if event["event"] == "output_index":
            recorded = {int(row["track_id"]): int(row["det_index"]) for row in event["rows"]}
            if len(recorded) != len(event["rows"]) or recorded != expected_indices:
                raise ValueError("Decision output_index differs from the probe's actual outputs")
            indices.extend({"frame": frame["frame"], "track_id": tid, "det_index": did,
                            "current_matched_gt": frame["outputs"][tid]["matched_gt"]}
                           for tid, did in recorded.items())
        elif event["event"] == "native_accepted" and event.get("stage") == "high":
            state, gap = event.get("state_before"), event.get("gap")
            if state == 1 and gap is not None and gap <= 1:
                continue
            tid = int(event["track_id"])
            signal = motion.get(tid, {})
            native_rows.append({"frame": frame["frame"], "stage": "high", "track_id": tid, "det_index": event["det_index"],
                                "state_before": state, "actual_Lost_state2": None if state is None else int(state == 2),
                                "last_seen": event.get("last_seen"), "gap": gap, "cost": event.get("cost"),
                                "center_normalized": signal.get("center_normalized"),
                                "size_normalized": signal.get("size_normalized"),
                                "prediction_anchor_frame": signal.get("anchor_frame"),
                                **actual_label(frame, tid, anchors.get(tid))})
    for cid_text, history in current.items():
        cid = int(cid_text)
        members = [tuple(map(int, key)) for key in history]
        if not members:
            continue
        # Each candidate contains actual observation keys, with at most one
        # observation per frame. Keep the recorded history, not reconstructed
        # boxes or scores, as the membership authority.
        if len(members) != len(set(members)) or any(key[0] > frame["frame"] for key in members):
            raise ValueError("Invalid actual candidate observation history")
        for member in members:
            for guard in guards.get(member, []):
                for proposal in proposals.get(cid, [None]):
                    tid = int(proposal["track_id"]) if proposal is not None else None
                    is_accepted = tid is not None and (tid, cid) in accepted
                    guard_rows.append({"frame": frame["frame"], "candidate_id": cid,
                        "guard_frame": member[0], "guard_track_id": int(guard["track_id"]), "guard_det_index": member[1],
                        "selected_iou": guard["selected_iou"], "backup_iou": guard["backup_iou"],
                        "current_det_index": members[-1][1] if members[-1][0] == frame["frame"] else None,
                        "proposal_track_id": tid, "recovered_track_id": tid if is_accepted else None,
                        "same_identity_as_guard": None if tid is None else int(tid == int(guard["track_id"])),
                        "B_accepted": int(is_accepted),
                        "B_reason": proposal.get("reason") if proposal is not None else "no_recorded_Lost_pair",
                        "history_observations": len(members),
                        "later_observations_after_guard": sum(key[0] > member[0] for key in members),
                        "appearance_hits": proposal.get("appearance_hits") if proposal is not None else None,
                        "appearance_cost": proposal.get("appearance_cost") if proposal is not None else None,
                        "two_hit_evidence": proposal.get("two_hit_evidence") if proposal is not None else None,
                        "evidence_interpretation": "recorded_observations_and_appearance_only;no_sufficiency_claim",
                        **actual_label(frame, tid, anchors.get(tid))})
    return native_rows, guard_rows, indices


def summarize(settings, run_dir, tracker, output, sequences=None):
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Probe summary output must be empty: {output}")
    helpers = analysis._evaluation_helpers()
    validated = helpers.validate_runs(run_dir, [tracker])
    protocol = validated["protocols"][tracker]
    names = list(sequences if sequences is not None else protocol["sequences"])
    if not names or len(names) != len(set(names)) or set(names)-set(protocol["sequences"]):
        raise ValueError("Invalid probe-summary sequence selection")
    grouping = settings.get("original_video_groups", {})
    if any(not isinstance(grouping.get(seq), str) or not grouping[seq] for seq in names):
        raise ValueError("Explicit original_video_groups required for every selected sequence")
    trackeval, evaluator_audit = analysis.load_trackeval(settings["trackeval_root"])
    lengths = {seq: protocol["sequences"][seq] for seq in names}
    gt_audit = helpers.validate_gt(settings["gt_root"], lengths)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "v04_joint_probe_analysis_v1", "status": "RUNNING", "tracker": tracker,
                "sequences": lengths, "source_protocol_sha256": protocol["validated_protocol_sha256"],
                "evaluator": evaluator_audit, "ground_truth": gt_audit, "probe_sources": {}, "detection_sources": {},
                "decision_sources": {},
                "scope": "conditional_shadow_diagnostics_on_development_data;not_closed_loop_gains",
                "thresholds_trained": False, "label_use": "GT offline only; all probe signals were produced causally"}
    analysis.dump(output/"manifest.json", manifest)
    all_routes, all_motion, all_proposals, all_competition, parity_rows = [], [], [], [], []
    all_native, all_guard, all_indices = [], [], []
    try:
        for sequence, length in lengths.items():
            path = run_dir/"probes"/tracker/f"{sequence}.jsonl.gz"
            summary_path = path.with_name(path.name + ".summary.json")
            summary = analysis.read_json(summary_path)
            expected = protocol.get("joint_batch", {}).get("probe_manifests", {}).get(sequence)
            if summary.get("schema") != SCHEMA or summary.get("frames") != length or summary.get("status") != "PASS":
                raise ValueError(f"Incomplete/failed probe summary: {sequence}")
            if expected is None or summary != expected:
                raise ValueError("Probe summary differs from the validated run protocol")
            checksum = analysis.digest(path)
            declared = summary.get("data_sha256")
            if declared is not None and checksum != declared:
                raise ValueError("Probe data checksum mismatch")
            manifest["probe_sources"][sequence] = {"path": str(path), "sha256": checksum,
                "summary_sha256": analysis.digest(summary_path), "bound_to_protocol_hash": declared is not None}
            parity_rows.append({"sequence": sequence, "original_video": grouping[sequence],
                "mode": summary["mode"], "internal_parity": summary["status"],
                "parity_checks": summary.get("counts", {}).get("parity_checks", 0),
                "parity_failures": summary.get("counts", {}).get("parity_failures", 0),
                "trace_hash_bound_to_protocol": int(declared is not None),
                "MOT_bytes_parity": "check_separate_batch_parity_artifact;not_inferred_here"})
            clear = analysis.evaluate_sequence(trackeval, settings["gt_root"], run_dir/"trackers", tracker,
                                                sequence, length, output/"evaluator")
            dets, source = _detector_frames(settings, sequence, protocol)
            manifest["detection_sources"][sequence] = source
            decisions, guards, decision_source = load_decisions(run_dir, tracker, sequence, length, protocol)
            manifest["decision_sources"][sequence] = decision_source
            anchors, counts = {}, Counter()
            for number, records in probe_frames(path, length):
                counts.update(record["event"] for record in records)
                frame = clear["frames"][number-1]
                groups = analyze_frame(records, frame, anchors, dets[number-1] if dets is not None else None)
                prefix = {"sequence": sequence, "original_video": grouping[sequence], "tracker": tracker}
                for target, rows in zip((all_routes, all_motion, all_proposals, all_competition), groups):
                    target.extend({**prefix, **({"frame": number} if "frame" not in row else {}), **row} for row in rows)
                joins = decision_joins(records, frame, anchors, decisions.get(number, []), guards)
                for target, rows in zip((all_native, all_guard, all_indices), joins):
                    target.extend({**prefix, **row} for row in rows)
                for tid, row in frame["outputs"].items():
                    if row["matched_gt"] is not None:
                        anchors[tid] = (row["matched_gt"], number)
            for kind, count in counts.items():
                if summary.get("counts", {}).get(kind) != count:
                    raise ValueError(f"Probe event count mismatch: {sequence}/{kind}")
        _write_table(output/"route_changes.csv", all_routes, ["sequence", "frame", "route", "available", "changed"])
        _write_table(output/"motion_disagreement.csv.gz", all_motion, ["sequence", "frame", "track_id", "center_normalized", "actual_outcome"])
        _write_table(output/"recovery_signals.csv.gz", all_proposals, ["sequence", "frame", "track_id", "candidate_id", "accepted", "reason"])
        _write_table(output/"identity_competition.csv.gz", all_competition, ["sequence", "frame", "track_id", "candidate_id", "lost_minus_best_active", "actual_outcome"])
        _write_table(output/"parity.csv", parity_rows, ["sequence", "internal_parity"])
        _write_table(output/"native_high_reconnections.csv", all_native,
                     ["sequence", "frame", "track_id", "actual_Lost_state2", "actual_outcome", "center_normalized"])
        _write_table(output/"guard_to_B.csv", all_guard,
                     ["sequence", "frame", "guard_frame", "guard_track_id", "guard_det_index", "candidate_id", "proposal_track_id", "B_accepted"])
        _write_table(output/"output_indices.csv", all_indices, ["sequence", "frame", "track_id", "det_index", "current_matched_gt"])
        guard_coverage = []
        for sequence in names:
            source = manifest["decision_sources"][sequence]
            members = {(row["guard_frame"], row["guard_track_id"], row["guard_det_index"])
                       for row in all_guard if row["sequence"] == sequence}
            accepted_members = {(row["guard_frame"], row["guard_track_id"], row["guard_det_index"])
                                for row in all_guard if row["sequence"] == sequence and row["B_accepted"]}
            total = source.get("guarded_edges")
            guard_coverage.append({"sequence": sequence, "original_video": grouping[sequence],
                                   "decision_status": source["status"], "guarded_edges": total,
                                   "seen_in_recorded_B_history": len(members) if total is not None else None,
                                   "with_recorded_B_acceptance": len(accepted_members) if total is not None else None,
                                   "not_seen_in_recorded_B_history": total-len(members) if total is not None else None})
        _write_table(output/"guard_membership_summary.csv", guard_coverage, ["sequence", "decision_status"])
        route_counts = defaultdict(Counter)
        for row in all_routes:
            for group in ("COMBINED", "video:"+row["original_video"]):
                key = (group, row["route"])
                route_counts[key]["frames"] += 1
                route_counts[key]["available_frames"] += row["available"]
                route_counts[key]["changed_frames"] += row["changed"] or 0
                route_counts[key]["selected_only_pairs"] += row["selected_only_pairs"] or 0
                route_counts[key]["native_only_pairs"] += row["native_only_pairs"] or 0
        route_summary = [{"group": group, "route": route, **counts} for (group, route), counts in sorted(route_counts.items())]
        _write_table(output/"route_summary.csv", route_summary, ["group", "route", "frames", "available_frames", "changed_frames"])
        distributions = defaultdict(list)
        for row in all_proposals:
            if row["accepted"] and row["center_normalized"] is not None:
                distributions[(row["original_video"], row["actual_outcome"])].append(row["center_normalized"])
        signal_summary = [{"original_video": video, "actual_outcome": label, "accepted_events": len(values),
                           "center_normalized_min": min(values), "center_normalized_median": statistics.median(values),
                           "center_normalized_max": max(values)} for (video, label), values in sorted(distributions.items())]
        _write_table(output/"accepted_signal_summary.csv", signal_summary,
                     ["original_video", "actual_outcome", "accepted_events", "center_normalized_min", "center_normalized_median", "center_normalized_max"])
        manifest.update(status="COMPLETE", route_counts=route_summary,
                        actual_B_outcomes=dict(Counter(row["actual_outcome"] for row in all_proposals if row["accepted"])),
                        raw_detection_status=dict(Counter(row["raw_detection_status"] for row in all_proposals)),
                        common_identity_score_unknown=sum(row["lost_minus_best_active"] is None for row in all_competition),
                        active_stronger_accepted=sum(row["accepted"] and row["active_stronger"] == 1 for row in all_competition),
                        native_high_reconnection_records=len(all_native),
                        actual_Lost_high_reconnections=sum(row["actual_Lost_state2"] == 1 for row in all_native),
                        guard_to_B_membership_records=len(all_guard),
                        guarded_edges_in_B_histories=len({(row["sequence"], row["guard_frame"], row["guard_track_id"], row["guard_det_index"]) for row in all_guard}),
                        guarded_edges_with_B_acceptance=len({(row["sequence"], row["guard_frame"], row["guard_track_id"], row["guard_det_index"]) for row in all_guard if row["B_accepted"]}),
                        missing_decision_sequences=sum(source["status"] == "UNKNOWN_MISSING_DECISIONS" for source in manifest["decision_sources"].values()),
                        gate_decision="DIAGNOSTIC_REVIEW_REQUIRED;no_threshold_selected")
        lines = ["# J3 只读诊断解读", "",
                 f"方法 `{tracker}`，{len(names)} 段、{sum(lengths.values())} 帧。GT 仅在本离线分析中读取。", "",
                 "## 条件路径变化", "", "| 路径 | 可比较帧 | 分配/成员变化帧 |", "|---|---:|---:|"]
        for row in route_summary:
            if row["group"] == "COMBINED":
                lines.append(f"| {row['route']} | {row['available_frames']} | {row['changed_frames']} |")
        lines += ["", "P/N/C 是同一参考历史上的条件诊断，不是三个独立全序列算法；变化数不能相加为净收益。", "",
                  "## 需要查看的证据", "",
                  "- recovery_signals：全部实际 B 提名/拒绝及接受结果，含过去 GT 锚点和精确检测行的离线支持。", 
                  "- motion_disagreement：逐身份的相机假设分歧；正确和错误恢复都应检查。",
                  "- identity_competition：Lost 与活跃身份的诊断支持。仅外观覆盖次数与历史长度相同时报告 lost_minus_best_active；正值表示该活跃身份得分更低。coverage_mismatch_or_unknown 不作强弱判断，未校正差值另列。",
                  "- accepted_signal_summary：按原视频和结果分组描述分歧，不训练阈值，不宣称统计显著。", "",
                  "- native_high_reconnections：依据实际 state_before 区分 Lost=2 与其他状态，并联结此前 GT 锚点和相机分歧。",
                  "- guard_to_B：按精确 (frame, det_index) 检查被 dual_low 拒绝的边是否进入后来 B 候选历史；保留原 guard 身份、提名/接回身份、后续实测数量和外观支持。",
                  "- output_indices：正式输出对应的原始检测行；有 decisions 日志时与 probe 输出逐帧核对。", "",
                  "## 标签与验收边界", "",
                  "same_GT_identity / different_GT_identity 依赖恢复前已确认的 GT 锚点。无旧锚点、未输出及定位/身份混合误差单列。",
                  "raw_detection_status 是当前原始检测对官方 GT 的唯一 IoU 支持，不能替代实际分配或物理背景判断。",
                  "guard_to_B 只描述已记录历史成员关系；代价下降不代表新增身份支持，未出现成员关系也不等于永远没有后续影响。缺失旧版 decisions 日志明确记 UNKNOWN。",
                  "parity.csv 记录内部影子一致性和 trace/MOT 对应；开关前后 MOT 字节一致性仍以 batch parity 产物为准。",
                  "进入 J4 前需核对正例保留、错误解释和主要设计视频外的重复证据。本报告不自动选择门限或宣称新模块有效。", ""]
        (output/"report.md").write_text("\n".join(lines), encoding="utf-8")
        manifest["outputs_sha256"] = {path.name: analysis.digest(path) for path in sorted(output.iterdir())
                                      if path.is_file() and path.name != "manifest.json"}
        analysis.dump(output/"manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        analysis.dump(output/"manifest.json", manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--tracker", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sequences", nargs="+")
    args = parser.parse_args(argv)
    try:
        from .common import load_settings
    except ImportError:
        from common import load_settings
    report = summarize(load_settings(args.settings), args.run_dir, args.tracker, args.output_dir, args.sequences)
    print(json.dumps({"status": report["status"], "sequences": len(report["sequences"]),
                      "actual_B_outcomes": report["actual_B_outcomes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
