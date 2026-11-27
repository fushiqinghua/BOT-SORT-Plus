"""J1: offline, official-CLEAR-aligned four-arm evidence ledger.

This program is an evaluator, never an online tracker.  It reads GT only after
prediction.  Numeric tracker IDs are local to each run; cross-run units are GT
frames and GT-anchored switch events.  A MOT score is never used to guess the
detector row.  The pinned CLEAR implementation owns all matching decisions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[3]
ROLES = ("native", "B", "G", "BG")
ALIASES = {"native": "native", "B": "B_early_moderate", "G": "dual_low",
           "BG": "B_early_moderate_dual_low"}
INPUT_KEYS = ("native_config", "detector_sha256", "rgb_encoder_sha256", "source_protocol_sha256",
              "detection_fingerprints", "feature_hashes", "decoded_image_hashes", "encoder_contracts")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")


def csv_rows(path, rows, fields):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _evaluation_helpers():
    path = REPO / "tools/evaluate_botsort_plus.py"
    spec = importlib.util.spec_from_file_location("_j1_eval_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_trackeval(trackeval_root):
    """Verify all source files, allowing only the baseline's EOL normalization."""
    root = Path(trackeval_root).resolve()
    audit = _evaluation_helpers().verify_trackeval(root, REPO)
    loaded = sys.modules.get("trackeval")
    if loaded and Path(loaded.__file__).resolve().parent != root / "trackeval":
        raise RuntimeError("A different TrackEval is already imported; run J1 in a new process")
    sys.path.insert(0, str(root))
    import numpy as np
    for name, value in (("int", int), ("float", float), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, value)
    import trackeval
    return trackeval, audit


def capture_clear(data, metric=None):
    """Observe the official solver without changing its inputs or assignment.

    The source is verified by load_trackeval before the public analysis runs.
    This wrapper is process-local and sequential; it is not a concurrent API.
    """
    import numpy as np
    from trackeval.metrics import CLEAR
    clear_module = importlib.import_module("trackeval.metrics.clear")
    metric = metric or CLEAR({"PRINT_CONFIG": False})
    expected = iter(t for t in range(data["num_timesteps"])
                    if len(data["gt_ids"][t]) and len(data["tracker_ids"][t]))
    original = clear_module.linear_sum_assignment
    captured = {}

    def observe(cost):
        timestep = next(expected)
        rows, columns = original(cost)
        keep = -cost[rows, columns] > np.finfo(float).eps
        captured[timestep] = {int(j): int(i) for i, j in zip(rows[keep], columns[keep])}
        return rows, columns

    clear_module.linear_sum_assignment = observe
    try:
        official = metric.eval_sequence(data)
    finally:
        clear_module.linear_sum_assignment = original
    if next(expected, None) is not None:
        raise RuntimeError("Pinned CLEAR solver call schedule changed")
    return official, captured


def evaluate_sequence(trackeval, gt_root, trackers_folder, tracker, sequence, length, output):
    """Same class-agnostic RGB loader as tools/evaluate_vttiny.py, without its I/O side effects."""
    import numpy as np

    class VTTinyObjects(trackeval.datasets.MotChallenge2DBox):
        def __init__(self, config):
            super().__init__(dict(config, CLASSES_TO_EVAL=["pedestrian"]))
            self.valid_classes = ["object"]
            self.class_list = ["object"]
            self.class_name_to_class_id["object"] = 1
            self.config["CLASSES_TO_EVAL"] = ["object"]

    dataset = VTTinyObjects(dict(
        GT_FOLDER=str(gt_root), TRACKERS_FOLDER=str(trackers_folder),
        OUTPUT_FOLDER=str(output), TRACKERS_TO_EVAL=[tracker], BENCHMARK="VT-Tiny-MOT",
        SPLIT_TO_EVAL="test", SKIP_SPLIT_FOL=True, SEQ_INFO={sequence: length},
        DO_PREPROC=False, PRINT_CONFIG=False))
    raw = dataset.get_raw_seq_data(tracker, sequence)
    data = dataset.get_preprocessed_seq_data(raw, "object")
    official, assignments = capture_clear(data)
    frames, switches, last_gt_track = [], [], {}
    totals = Counter(TP=0, FP=0, FN=0, IDSW=0)
    for t in range(length):
        mask = np.asarray(raw["gt_extras"][t]["zero_marked"]) != 0
        gt_ids = raw["gt_ids"][t][mask]
        tracker_ids = raw["tracker_ids"][t]
        # Avoid treating remapped evaluator IDs as dataset IDs.  The verified
        # loader preserves row order while remapping identity values.
        if not np.array_equal(data["gt_dets"][t], raw["gt_dets"][t][mask]):
            raise RuntimeError("GT preprocessing changed row order unexpectedly")
        if not np.array_equal(data["tracker_dets"][t], raw["tracker_dets"][t]):
            raise RuntimeError("Tracker preprocessing changed row order unexpectedly")
        matches = assignments.get(t, {})
        outputs, coverage = {}, {}
        nsw = 0
        for j, raw_tid in enumerate(tracker_ids):
            tid = int(raw_tid)
            matched = matches.get(j)
            gid = int(gt_ids[matched]) if matched is not None else None
            sims = data["similarity_scores"][t][:, j]
            best = int(np.argmax(sims)) if len(sims) and float(sims.max()) > 0 else None
            row = {"track_id": tid, "matched_gt": gid, "status": "TP" if gid is not None else "FP",
                   "box_xywh": data["tracker_dets"][t][j].astype(float).tolist(),
                   "nearest_gt": int(gt_ids[best]) if best is not None else None,
                   "max_iou": float(sims.max()) if len(sims) else 0.,
                   "gt_ious": {int(g): float(v) for g, v in zip(gt_ids, sims)}}
            outputs[tid] = row
            if gid is not None:
                previous = last_gt_track.get(gid)
                if previous is not None and previous[0] != tid:
                    switches.append({"frame": t + 1, "gt_id": gid,
                                     "previous_frame": previous[1], "previous_track_id": previous[0],
                                     "current_track_id": tid})
                    nsw += 1
                last_gt_track[gid] = (tid, t + 1)
                coverage[gid] = tid
        counts = {"TP": len(matches), "FP": len(tracker_ids) - len(matches),
                  "FN": len(gt_ids) - len(matches), "IDSW": nsw}
        totals.update(counts)
        frames.append({"frame": t + 1, "gt_ids": tuple(map(int, gt_ids)),
                       "gt_boxes": {int(gid): box.astype(float).tolist()
                                    for gid, box in zip(gt_ids, data["gt_dets"][t])},
                       "coverage": coverage, "outputs": outputs, "counts": counts})
    for key, field in (("TP", "CLR_TP"), ("FP", "CLR_FP"), ("FN", "CLR_FN"), ("IDSW", "IDSW")):
        if totals[key] != int(official[field]):
            raise RuntimeError(f"CLEAR ledger mismatch {tracker}/{sequence}/{key}: {totals[key]} != {official[field]}")
    return {"frames": frames, "switches": switches, "counts": dict(totals)}


def coverage_ledger(sequence, arms, source_video):
    """Partition the native-missed universe; also retain native-covered regressions."""
    lengths = {len(arms[role]["frames"]) for role in ROLES}
    if len(lengths) != 1:
        raise ValueError("Four arms must contain the same frame count")
    rows = []
    for offset in range(next(iter(lengths))):
        frames = {role: arms[role]["frames"][offset] for role in ROLES}
        gt_ids = set(frames["native"]["gt_ids"])
        if any(set(value["gt_ids"]) != gt_ids for value in frames.values()):
            raise ValueError("Four arms disagree on valid official GT IDs")
        for gid in sorted(gt_ids):
            hit = {role: gid in frames[role]["coverage"] for role in ROLES}
            n, b, g, bg = (hit[role] for role in ROLES)
            rows.append({"sequence": sequence, "original_video": source_video,
                         "frame": offset + 1, "gt_id": gid,
                         **{f"{role}_hit": int(hit[role]) for role in ROLES},
                         "B_repairs_native_miss": int(not n and b),
                         "G_repairs_native_miss": int(not n and g),
                         "both_repair_native_miss": int(not n and b and g),
                         "union_repair_preserved": int(not n and (b or g) and bg),
                         "union_repair_lost": int(not n and (b or g) and not bg),
                         "joint_only_repair": int(not n and not b and not g and bg),
                         "native_coverage_lost_by_BG": int(n and not bg),
                         "B_coverage_lost_by_BG": int(b and not bg),
                         "G_coverage_lost_by_BG": int(g and not bg)})
    return rows


def followup(frames, frame, tid, anchor_gt, horizon):
    """Future-window evidence only: never an online feature, never padded as success."""
    window = frames[frame:min(len(frames), frame + horizon)]
    result = {"requested_frames": horizon, "observed_frames": len(window),
              "right_censored": len(window) < horizon,
              "track_output_frames": 0, "track_FP": 0,
              "anchor_present_frames": None if anchor_gt is None else 0,
              "anchor_same_id_TP": None if anchor_gt is None else 0,
              "anchor_other_id_TP": None if anchor_gt is None else 0,
              "anchor_FN": None if anchor_gt is None else 0}
    for value in window:
        output = value["outputs"].get(tid)
        result["track_output_frames"] += int(output is not None)
        result["track_FP"] += int(output is not None and output["status"] == "FP")
        if anchor_gt is not None and anchor_gt in value["gt_ids"]:
            result["anchor_present_frames"] += 1
            matched = value["coverage"].get(anchor_gt)
            if matched == tid:
                result["anchor_same_id_TP"] += 1
            elif matched is None:
                result["anchor_FN"] += 1
            else:
                result["anchor_other_id_TP"] += 1
    return result


def load_events(entry, sequence):
    folder = Path(entry.get("events_dir", Path(entry["run_dir"]) / "events" / entry["tracker"]))
    paths = [folder / f"{sequence}.jsonl", folder / f"{sequence}.jsonl.gz"]
    found = [path for path in paths if path.is_file()]
    if len(found) > 1:
        raise ValueError(f"Ambiguous event files: {found}")
    if not found:
        return [], {"status": "MISSING", "reason": "No event log; stages/detector indices remain unknown"}
    path = found[0]
    opener = gzip.open if path.suffix == ".gz" else open
    events = []
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("sequence", sequence) != sequence:
                raise ValueError(f"Wrong sequence at {path}:{line_number}")
            if "frame" not in event:
                # Final aggregate events do not identify an online decision.
                continue
            frame = event["frame"]
            if isinstance(frame, bool) or not isinstance(frame, int) or frame < 1:
                raise ValueError(f"Invalid event frame at {path}:{line_number}")
            events.append(event)
    return events, {"status": "PRESENT", "path": str(path), "sha256": digest(path), "records": len(events)}


def event_ledger(sequence, source_video, role, result, events, other_arms):
    """Accepted-event consequences and all logged B proposals/rejections.

    Native high events with an output gap are observational gap reconnections,
    not a claim about the unrecorded internal Lost state.  Rejection outcomes
    are not reconstructed from final MOT rows.
    """
    by_frame = defaultdict(list)
    for event in events:
        if event["frame"] > len(result["frames"]):
            # Finish-only summaries may use the terminal next-frame index.
            if event.get("event") in ("match", "recovered", "recovery_check"):
                raise ValueError("Association event is outside sequence bounds")
            continue
        by_frame[event["frame"]].append(event)
    accepted, proposals, output_rows = [], [], []
    last_output, last_anchor, chain_root = {}, {}, {}
    for value in result["frames"]:
        frame = value["frame"]
        event_map = {}
        current_events = by_frame.get(frame, [])
        for event in current_events:
            if event.get("event") == "recovery_check":
                proposals.append({"sequence": sequence, "original_video": source_video,
                                  "role": role, "frame": frame,
                                  "track_id": event.get("track_id"),
                                  "candidate_id": event.get("candidate_id"),
                                  "reason": event.get("reason", "unknown"),
                                  "det_index": event.get("det_index"),
                                  "detail_json": json.dumps(event, ensure_ascii=False, sort_keys=True)})
            if event.get("event") not in ("match", "recovered"):
                continue
            if "track_id" not in event:
                raise ValueError("Accepted match/recovery lacks track_id")
            tid = int(event["track_id"])
            if tid in event_map:
                raise ValueError(f"Multiple accepted associations for {sequence}/{frame}/{tid}")
            event_map[tid] = event
            stage = "B" if event["event"] == "recovered" else "native_" + event.get("stage", "unknown")
            anchor = last_anchor.get(tid)
            anchor_gt = anchor[0] if anchor is not None else None
            output = value["outputs"].get(tid)
            matched_gt = output["matched_gt"] if output is not None else None
            previous = last_output.get(tid)
            gap = frame - previous if previous is not None else None
            if output is None:
                verdict = "not_emitted"
            elif anchor_gt is None:
                verdict = "unknown_prior_identity"
            elif matched_gt == anchor_gt:
                verdict = "same_gt_identity"
            elif matched_gt is not None:
                verdict = "different_gt_identity"
            elif anchor_gt not in value["gt_ids"]:
                verdict = "FP_anchor_gt_absent"
            elif output["gt_ious"].get(anchor_gt, 0.) >= .5:
                verdict = "FP_assignment_competition"
            else:
                verdict = "FP_anchor_iou_below_threshold"
            is_reconnection = stage == "B" or (stage == "native_high" and gap is not None and gap > 1)
            if is_reconnection:
                chain_root[tid] = (frame, stage)
            row = {"sequence": sequence, "original_video": source_video, "role": role,
                   "frame": frame, "track_id": tid, "stage": stage,
                   "det_index": event.get("det_index"), "candidate_id": event.get("candidate_id"),
                   "age_reported": event.get("age"), "previous_output_gap": gap,
                   "is_reconnection": int(is_reconnection),
                   "reconnection_evidence": "explicit_B" if stage == "B" else
                       ("output_gap_only" if is_reconnection else "ordinary_accepted_match"),
                   "prior_anchor_gt": anchor_gt, "prior_anchor_frame": anchor[1] if anchor else None,
                   "current_matched_gt": matched_gt, "verdict": verdict,
                   "current_status": output["status"] if output else "absent",
                   "current_anchor_iou": output["gt_ious"].get(anchor_gt) if output and anchor_gt is not None else None}
            for other in ROLES:
                other_frame = other_arms[other]["frames"][frame - 1]
                row[f"{other}_anchor_covered"] = None if anchor_gt is None else int(anchor_gt in other_frame["coverage"])
            for horizon in (3, 5, 10):
                row.update({f"next{horizon}_{key}": item
                            for key, item in followup(result["frames"], frame, tid, anchor_gt, horizon).items()})
            accepted.append(row)
        for tid, output in value["outputs"].items():
            event = event_map.get(tid, {})
            stage = "B" if event.get("event") == "recovered" else (
                "native_" + event.get("stage", "unknown") if event else "unknown")
            chain = chain_root.get(tid)
            output_rows.append({"sequence": sequence, "original_video": source_video, "role": role,
                                "frame": frame, "track_id": tid, "matched_gt": output["matched_gt"],
                                "status": output["status"], "nearest_gt": output["nearest_gt"],
                                "max_iou": output["max_iou"], "stage": stage,
                                "det_index": event.get("det_index"),
                                "last_observed_reconnect_frame": chain[0] if chain else None,
                                "last_observed_reconnect_stage": chain[1] if chain else None,
                                "lineage_scope": "descriptive_same_run_ID;not_causal_attribution"})
            last_output[tid] = frame
            if output["matched_gt"] is not None:
                last_anchor[tid] = (output["matched_gt"], frame)
    return accepted, proposals, output_rows


def _reference_contracts(settings, selected):
    references = settings["references"]
    helpers = _evaluation_helpers()
    entries, protocols, inputs, files = {}, {}, None, {}
    for role in ROLES:
        key = role if role in references else ALIASES[role]
        if key not in references:
            raise ValueError(f"Missing J1 reference {role} ({ALIASES[role]})")
        supplied = references[key]
        entry = dict(supplied) if isinstance(supplied, dict) else {"run_dir": supplied}
        entry.setdefault("tracker", ALIASES[role])
        entry["run_dir"] = str(Path(entry["run_dir"]).resolve())
        validated = helpers.validate_runs(entry["run_dir"], [entry["tracker"]])
        protocol = validated["protocols"][entry["tracker"]]
        this_inputs = {key: protocol.get(key) for key in INPUT_KEYS}
        if any(value is None for value in this_inputs.values()):
            raise ValueError(f"{role}: missing frozen input fields")
        if inputs is None:
            inputs = this_inputs
            lengths = protocol["sequences"]
        elif this_inputs != inputs or protocol["sequences"] != lengths:
            raise ValueError(f"{role}: mixed input caches, native config or frame scope")
        entries[role], protocols[role] = entry, protocol
        files[role] = validated["prediction_files"][entry["tracker"]]
    if settings.get("sequence_lengths") is not None and settings["sequence_lengths"] != lengths:
        raise ValueError("Configured sequence_lengths differs from reference protocols")
    names = sorted(selected if selected is not None else lengths)
    if not names or len(names) != len(set(names)) or set(names) - set(lengths):
        raise ValueError("Select unique nonempty sequences present in all four references")
    return entries, protocols, {key: lengths[key] for key in names}, files


def _check_metrics(entry, sequence, counts):
    path = Path(entry.get("metrics_file", Path(entry["run_dir"]) / "metrics/summary.json"))
    if not path.is_file():
        if entry.get("metrics_file"):
            raise FileNotFoundError(path)
        return {"status": "UNAVAILABLE", "path": str(path)}
    metrics = read_json(path)
    rows = [row for row in metrics.get("results", [])
            if row.get("tracker") == entry["tracker"] and row.get("sequence") == sequence]
    if len(rows) != 1:
        raise ValueError(f"Expected one archived metric row: {entry['tracker']}/{sequence}")
    for name in ("FP", "FN", "IDSW"):
        if int(rows[0][name]) != counts[name]:
            raise ValueError(f"Archived {name} mismatch: {entry['tracker']}/{sequence}")
    return {"status": "PASS", "path": str(path), "sha256": digest(path)}


def analyze(settings, output, sequences=None):
    """Create a complete J1 ledger.  Existing output folders are never overwritten."""
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"J1 output must be empty: {output}")
    entries, protocols, lengths, prediction_files = _reference_contracts(settings, sequences)
    grouping = settings.get("original_video_groups", {})
    if not isinstance(grouping, dict) or any(not isinstance(grouping.get(seq), str) or not grouping[seq] for seq in lengths):
        raise ValueError("original_video_groups must explicitly map every selected sequence to its source video")
    trackeval, evaluator_audit = load_trackeval(settings["trackeval_root"])
    gt_audit = _evaluation_helpers().validate_gt(settings["gt_root"], lengths)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "v04_joint_J1_v1", "status": "RUNNING",
               "sequences": lengths, "frames": sum(lengths.values()),
               "scope": "existing_development_sequences", "original_video_groups": {seq: grouping[seq] for seq in lengths},
               "reference_protocol_sha256": {role: protocols[role]["validated_protocol_sha256"] for role in ROLES},
               "prediction_files": prediction_files, "evaluator": evaluator_audit, "ground_truth": gt_audit,
               "events": {}, "archived_metrics": {}, "counts": {},
               "semantics": {"G": "adaptive GMC plus dual_low; not pure warp",
                             "coverage": "official CLEAR GT-frame detection coverage; not HOTA/IDF1 decomposition",
                             "switches": "each arm's CLEAR switch flag on the common GT-frame; numeric track IDs never matched across arms",
                             "event_windows": "future frames are offline outcomes, not online features; incomplete tails are censored",
                             "FP_lineage": "observational same-run ID lineage; causality requires controlled replay",
                             "missing_events": "unknown stages and detection indices; never reconstructed from scores or nearest boxes"}}
    dump(output / "manifest.json", summary)
    coverage, accepted, proposals, outputs, metrics_rows, switch_rows, switch_presence = [], [], [], [], [], [], []
    try:
        for sequence, length in lengths.items():
            arms = {}
            for role, entry in entries.items():
                arms[role] = evaluate_sequence(trackeval, settings["gt_root"], Path(entry["run_dir"]) / "trackers",
                                               entry["tracker"], sequence, length, output / "evaluator")
                summary["archived_metrics"][f"{role}/{sequence}"] = _check_metrics(entry, sequence, arms[role]["counts"])
                summary["counts"][f"{role}/{sequence}"] = arms[role]["counts"]
            coverage.extend(coverage_ledger(sequence, arms, grouping[sequence]))
            switch_sets = {role: {(row["frame"], row["gt_id"]) for row in arms[role]["switches"]} for role in ROLES}
            for frame, gid in sorted(set().union(*switch_sets.values())):
                switch_presence.append({"sequence": sequence, "original_video": grouping[sequence],
                                        "frame": frame, "gt_id": gid,
                                        **{f"{role}_IDSW": int((frame, gid) in switch_sets[role]) for role in ROLES}})
            for role, entry in entries.items():
                events, metadata = load_events(entry, sequence)
                summary["events"][f"{role}/{sequence}"] = metadata
                a, p, o = event_ledger(sequence, grouping[sequence], role, arms[role], events, arms)
                accepted.extend(a); proposals.extend(p); outputs.extend(o)
                switch_rows.extend({"sequence": sequence, "original_video": grouping[sequence], "role": role, **row}
                                   for row in arms[role]["switches"])
                metrics_rows.extend({"sequence": sequence, "original_video": grouping[sequence], "role": role,
                                     "frame": value["frame"], **value["counts"]} for value in arms[role]["frames"])
        coverage_fields = list(coverage[0]) if coverage else ["sequence", "original_video", "frame", "gt_id"]
        csv_rows(output / "coverage_by_gt_frame.csv.gz", coverage, coverage_fields)
        counters = {key: Counter() for key in ("COMBINED",)}
        numeric = coverage_fields[4:]
        for row in coverage:
            for group in ("COMBINED", "sequence:" + row["sequence"], "video:" + row["original_video"]):
                counters.setdefault(group, Counter()).update({key: row[key] for key in numeric})
                counters[group]["gt_frames"] += 1
        aggregates = [{"group": key, "gt_frames": value["gt_frames"], **{field: value[field] for field in numeric}}
                      for key, value in sorted(counters.items())]
        csv_rows(output / "coverage_overlap.csv", aggregates, ["group", "gt_frames", *numeric])
        first_divergences = {}
        for row in coverage:
            if row["B_hit"] != row["BG_hit"]:
                key = (row["sequence"], row["gt_id"])
                first_divergences.setdefault(key, {
                    "sequence": row["sequence"], "original_video": row["original_video"],
                    "gt_id": row["gt_id"], "first_B_BG_coverage_divergence_frame": row["frame"],
                    "B_hit": row["B_hit"], "BG_hit": row["BG_hit"],
                    "scope": "first_observed_CLEAR_coverage_difference;not_first_state_cause"})
        csv_rows(output / "first_coverage_divergence.csv", list(first_divergences.values()),
                 ["sequence", "original_video", "gt_id", "first_B_BG_coverage_divergence_frame", "B_hit", "BG_hit", "scope"])
        csv_rows(output / "frame_metrics.csv", metrics_rows, ["sequence", "original_video", "role", "frame", "TP", "FP", "FN", "IDSW"])
        csv_rows(output / "identity_switches.csv", switch_rows,
                 ["sequence", "original_video", "role", "frame", "gt_id", "previous_frame", "previous_track_id", "current_track_id"])
        csv_rows(output / "switch_presence_by_gt_frame.csv", switch_presence,
                 ["sequence", "original_video", "frame", "gt_id", *[f"{role}_IDSW" for role in ROLES]])
        csv_rows(output / "association_events.csv.gz", accepted, list(accepted[0]) if accepted else
                 ["sequence", "original_video", "role", "frame", "track_id", "stage", "verdict"])
        csv_rows(output / "recovery_proposals.csv.gz", proposals,
                 ["sequence", "original_video", "role", "frame", "track_id", "candidate_id", "reason", "det_index", "detail_json"])
        csv_rows(output / "output_rows_CLEAR.csv.gz", outputs, list(outputs[0]) if outputs else
                 ["sequence", "original_video", "role", "frame", "track_id", "matched_gt", "status", "stage", "det_index"])
        reason_counts = Counter((row["role"], row["reason"]) for row in proposals)
        summary.update(status="COMPLETE", coverage_overlap=counters["COMBINED"],
                       accepted_event_records=len(accepted), B_proposal_records=len(proposals),
                       proposal_reasons=[{"role": role, "reason": reason, "count": count}
                                         for (role, reason), count in sorted(reason_counts.items())],
                       missing_event_sequences=sum(value["status"] == "MISSING" for value in summary["events"].values()),
                       outputs_sha256={path.name: digest(path) for path in sorted(output.glob("*.csv*"))})
        summary["event_analysis_status"] = "PARTIAL_MISSING_EVENT_LOGS" if summary["missing_event_sequences"] else "COMPLETE_RECORDED_EVENTS_ONLY"
        lines = ["# J1 四组收益交集与事件账本", "",
                 f"覆盖 {len(lengths)} 个序列、{sum(lengths.values())} 帧。匹配来自固定 TrackEval CLEAR；逐帧计数与 CLEAR 返回值一致。", "",
                 "G 代表自适应 GMC 与 dual_low 的现有组合。覆盖交集只解释 GT 帧命中，不分解 HOTA 或 IDF1。", "",
                 "## 原始基线漏检集合中的覆盖变化", "",
                 "| 项目 | GT 帧数 |", "|---|---:|"]
        for key in ("B_repairs_native_miss", "G_repairs_native_miss", "both_repair_native_miss",
                    "union_repair_preserved", "union_repair_lost", "joint_only_repair"):
            lines.append(f"| {key} | {counters['COMBINED'][key]} |")
        lines += ["", "## 证据范围", "",
                  f"缺少事件文件的运行/序列数：{summary['missing_event_sequences']}。这些位置不重建关联阶段或检测索引。", "",
                  "association_events 含已记录的原生关联与 B 恢复及后 3/5/10 帧结果；recovery_proposals 保留全部已记录的 B 候选判定。", "",
                  "原生高分的 output_gap_only 只说明输出中断后重连，不能替代内部 Lost 状态证据。", "",
                  "output_rows_CLEAR 的重连来源只是同一运行内的身份沿袭，不能据此宣称某事件因果导致全部后续误报。", "",
                  "后续窗口以事件之后的帧计数；序列末端不补齐，GT 不在场与 FN 分开。未知身份不强制归为错误。", "",
                  "按 original_video_groups 显式分组；现有数据仍属于开发性评估。下一步由 J2/J3 受控回放确定路径。", ""]
        (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
        summary["outputs_sha256"]["report.md"] = digest(output / "report.md")
        dump(output / "manifest.json", summary)
        return summary
    except Exception as error:
        summary.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        dump(output / "manifest.json", summary)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sequences", nargs="+")
    args = parser.parse_args(argv)
    result = analyze(read_json(args.settings), args.output, args.sequences)
    print(json.dumps({"status": result["status"], "sequences": len(result["sequences"]),
                      "frames": result["frames"], "missing_event_sequences": result["missing_event_sequences"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
