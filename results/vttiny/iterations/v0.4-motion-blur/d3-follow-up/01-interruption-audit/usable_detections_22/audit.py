"""Read-only, parity-checked decision audit of the frozen D3 usable cohort."""
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import sys


TRACKER = "B_early_moderate"
SOURCE = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
REPO = SOURCE.parents[1]
sys.path.insert(0, str(REPO))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import numpy as np

from botsort_plus.diagnostics.analysis import _accepted_updates, _frame_state, _output_status
from botsort_plus.diagnostics.common import (
    box_iou_matrix,
    detection_tier,
    diagnostic_assignment,
    iter_trace,
    load_gt,
    read_json,
    write_json,
    xywh,
)
from botsort_plus.io import sha256


def read_cohort(analysis):
    path = SOURCE / "round1_findings/D3_complete_first_failure_candidates.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    assert len(all_rows) == 210
    cohort = [r for r in all_rows if r["candidate_type"] == "aligned_and_eligible"]
    assert len(cohort) == 22
    assert {r["first_failure_reason"] for r in cohort} == {"missing_with_usable_detection"}
    windows = SOURCE / "analysis/recovery_windows.csv"
    assert sha256(windows) == analysis["files_sha256"][windows.name]
    with windows.open(encoding="utf-8-sig", newline="") as stream:
        expected = {(r["sequence"], int(r["recovery_frame"]),
                     int(r["recovery_frame"]) + int(r["first_failure_offset"]),
                     int(r["track_id"]), int(r["gt_id"]))
                    for r in csv.DictReader(stream)
                    if r["horizon"] == "10" and r["complete_observable_window"] == "True"
                    and r["first_failure_reason"] == "missing_with_usable_detection"}
    actual = {(r["sequence"], int(r["recovery_frame"]), int(r["first_failure_frame"]),
               int(r["track_id"]), int(r["gt_id"])) for r in cohort}
    assert len(actual) == len(expected) == 22 and actual == expected
    return cohort


def by_track(snapshots, track_id):
    return next((s for s in snapshots if s["track_id"] == track_id), None)


def geometry(box, gt_box):
    if box is None:
        return {"iou": None, "center_px": None, "dx": None, "dy": None,
                "width_ratio": None, "height_ratio": None}
    gt_center, other = xywh(gt_box), xywh(box)
    offset = other[:2] - gt_center[:2]
    return {
        "iou": float(box_iou_matrix([box], [gt_box])[0, 0]),
        "center_px": float(np.linalg.norm(offset)),
        "dx": float(offset[0]),
        "dy": float(offset[1]),
        "width_ratio": float(other[2] / gt_center[2]),
        "height_ratio": float(other[3] / gt_center[3]),
    }


def offline_cases(cohort, settings, protocol, analysis):
    grouped = defaultdict(list)
    for row in cohort:
        grouped[row["sequence"]].append(row)
    threshold, ambiguity = analysis["iou_threshold"], analysis["ambiguity_margin"]
    for sequence, targets in sorted(grouped.items()):
        frames = protocol["sequences"][sequence]
        gt_path = Path(settings["gt_root"]) / sequence / "gt/gt.txt"
        assert sha256(gt_path) == analysis["ground_truth"]["sha256"][sequence]
        trace = SOURCE / "observed/diagnostics" / TRACKER / (sequence + ".jsonl.gz")
        declaration = protocol["diagnostics"]["traces"][sequence]
        assert declaration["data_sha256"] == sha256(trace)
        assert declaration["manifest_sha256"] == sha256(trace.with_suffix("").with_suffix(".json"))
        truth, _ = load_gt(gt_path, frames)
        wanted = {int(t["first_failure_frame"]) for t in targets}
        records = {row["frame"]: row for row in iter_trace(trace, sequence, frames)
                   if row["frame"] in wanted}
        for t in targets:
            frame = int(t["first_failure_frame"])
            tid, gid = int(t["track_id"]), int(t["gt_id"])
            record = records[frame]
            gt = truth[frame]
            gi = next(i for i, r in enumerate(gt) if r["gt_id"] == gid)
            gt_box = gt[gi]["xyxy"]
            det = record["detections"]
            output = record["output"]
            labels, statuses, iou, _ = diagnostic_assignment(
                [d["xyxy"] for d in det], gt, threshold, ambiguity)
            out_labels, out_statuses, _, _ = diagnostic_assignment(
                [o[:4] for o in output], gt, threshold, ambiguity)
            updates = _accepted_updates(record["events"])
            used = {u["det_index"]: {"track": track, "stage": u["stage"]}
                    for track, u in updates.items()}
            state = _frame_state(frame, gt, det, labels, statuses,
                                 np.asarray(output, dtype=float).reshape(-1, 8),
                                 out_labels, out_statuses, record["after_update"],
                                 protocol["native_config"], threshold)
            assert _output_status(state, tid, gid) == t["first_failure_reason"]
            stages = {k: by_track(record[k], tid) for k in
                      ("before_prediction", "after_prediction", "after_gmc", "after_update")}
            post = stages["after_update"]
            own = next((o for o in output if int(o[4]) == tid), None)
            eligible = []
            for i, item in enumerate(det):
                tier = detection_tier(item["score"], protocol["native_config"])
                if iou[i, gi] >= threshold - np.finfo(float).eps and tier in ("high", "low"):
                    eligible.append({
                        "index": i, "tier": tier, "score": item["score"],
                        "gt_iou": float(iou[i, gi]), "label": labels.get(i),
                        "gt_status": statuses[i], "accepted_by": used.get(i),
                    })
            assert len(eligible) == int(t["eligible_aligned"])
            assert eligible
            eligible_ids = {d["index"] for d in eligible}
            candidate_events = [e for e in record["events"] if e.get("det_index") in eligible_ids]
            candidate_ids = {e["candidate_id"] for e in candidate_events if "candidate_id" in e}
            rec = {
                "sequence": sequence, "recovery_frame": int(t["recovery_frame"]),
                "frame": frame, "delay": frame - int(t["recovery_frame"]),
                "tid": tid, "gid": gid, "gt_size": min(xywh(gt_box)[2:]),
                "eligible": eligible, "pre_state": stages["after_gmc"]["state"] if stages["after_gmc"] else None,
                "post_state": None if post is None else post["state"],
                "gap": frame - stages["after_gmc"]["last_seen"] if stages["after_gmc"] else None,
                **{k: geometry(s["xyxy"] if s else None, gt_box) for k, s in stages.items()},
                "output": geometry(own[:4] if own else None, gt_box),
                "update": updates.get(tid),
                "accepted_gt": labels.get(updates[tid]["det_index"]) if tid in updates else None,
                "accepted_det_iou": float(iou[updates[tid]["det_index"], gi]) if tid in updates else None,
                "candidate_events": candidate_events,
                "recovery_checks": [e for e in record["events"] if e["event"] == "recovery_check"
                                    and e["track_id"] == tid and e["candidate_id"] in candidate_ids],
            }
            yield rec


def replay_cost(case_rows, settings, protocol):
    # Hook the native delegation only inside this read-only diagnostic process.
    # Call original cost methods unchanged. Cases are interpreted only after parity.
    sys.path.insert(0, str(Path(settings["runtime_root"]) / "tracking_runtime"))
    from scipy.spatial.distance import cdist
    from boxmot.utils.matching import fuse_score, iou_distance
    from botsort_plus import adapter_v2
    from botsort_plus.native_adapter import BoTSORTPlus
    from tools import run_botsort_plus

    native_cost = BoTSORTPlus._cost
    native_update = BoTSORTPlus.update
    adapter_iou = adapter_v2.iou_distance
    captured = {}
    current_frames = {}
    current_frame = None

    def remember(stage, tracks, detections, distance, base, cost):
        for c in current_frames[current_frame]:
            ids = [int(t.id) for t in tracks]
            indices = [int(d.det_ind) for d in detections]
            per = captured.setdefault((c["sequence"], c["frame"], c["tid"]), {})
            audit = {
                "tid_in_stage": c["tid"] in ids,
                "available_detection_indices": indices,
                "pair_costs": [],
            }
            per[stage] = audit
            if c["tid"] not in ids:
                continue
            i = ids.index(c["tid"])
            track = tracks[i]
            for eligible in c["eligible"]:
                if eligible["index"] not in indices:
                    continue
                j = indices.index(eligible["index"])
                det = detections[j]
                proximity_allows_rgb = float(distance[i, j]) <= protocol["native_config"]["proximity_thresh"]
                a, b = track.smooth_feat, det.curr_feat
                rgb = None
                if stage != "low" and a is not None and b is not None and np.linalg.norm(a) and np.linalg.norm(b):
                    rgb = max(0., float(cdist(np.asarray(a).reshape(1, -1), np.asarray(b).reshape(1, -1), "cosine")[0, 0]))
                audit["pair_costs"].append({
                    "index": eligible["index"], "geometry_iou": float(1 - distance[i, j]),
                    "base_cost": float(base[i, j]), "effective_cost": float(cost[i, j]),
                    "rgb_cosine": rgb,
                    "proximity_allows_rgb": proximity_allows_rgb if stage != "low" else None,
                })

    def audited_update(self, *args, **kwargs):
        nonlocal current_frame
        current_frame = self.frame_count + 1
        return native_update(self, *args, **kwargs)

    def audited_cost(self, tracks, detections, unconfirmed=False):
        costs = native_cost(self, tracks, detections, unconfirmed=unconfirmed)
        if current_frame in current_frames:
            d = iou_distance(tracks, detections)
            base = fuse_score(d, detections) if unconfirmed or self.fuse_first_associate else d
            remember("unconfirmed" if unconfirmed else "high", tracks, detections, d, base, costs)
        return costs

    def audited_low(tracks, detections):
        costs = adapter_iou(tracks, detections)
        if current_frame in current_frames:
            remember("low", tracks, detections, costs, costs, costs)
        return costs

    BoTSORTPlus._cost = audited_cost
    BoTSORTPlus.update = audited_update
    adapter_v2.iou_distance = audited_low
    checks = {}
    allseq = sorted({row["sequence"] for row in case_rows})
    for seq in allseq:
        current_frames = defaultdict(list)
        for row in case_rows:
            if row["sequence"] == seq:
                current_frames[row["frame"]].append(row)
        dest = OUT / "probe" / seq
        args = run_botsort_plus.parser().parse_args([
            "--baseline-root", settings["baseline_root"],
            "--features-dir", settings["features_dir"],
            "--images-root", settings["images_root"],
            "--runtime-root", settings["runtime_root"],
            "--config", str(REPO / "configs/botsort_plus/B_early_moderate.json"),
            "--output-dir", str(dest),
            "--sequences", seq, "--trace",
        ])
        assert run_botsort_plus.run(args) == 0
        pred, event = f"trackers/{TRACKER}/data/{seq}.txt", f"events/{TRACKER}/{seq}.jsonl"
        observed, probe = SOURCE / "observed", dest
        pred_sha, event_sha = sha256(probe / pred), sha256(probe / event)
        assert pred_sha == protocol["output_prediction_hashes"][seq] == sha256(observed / pred), seq
        assert event_sha == sha256(observed / event), seq
        result = read_json(probe / "protocols" / f"{TRACKER}.json")
        assert result["decoded_image_hashes"][seq] == protocol["decoded_image_hashes"][seq], seq
        for k in ("config", "native_config", "detector_sha256", "rgb_encoder_sha256",
                  "source_protocol_sha256", "boxmot_source_hashes", "implementation_sha256"):
            assert result[k] == protocol[k], (k, seq)
        for k in ("detection_fingerprints", "feature_hashes"):
            assert result[k][seq] == protocol[k][seq], (k, seq)
        checks[seq] = {"mot_sha256": pred_sha, "events_sha256": event_sha,
                       "pixels_sha256": result["decoded_image_hashes"][seq]}
        print("Verified original MOT and events:", seq, flush=True)
    return captured, checks


def classify(row, probe, config, threshold):
    update = row["update"]
    if update is not None:
        if row["accepted_gt"] == row["gid"]:
            if row["output"]["iou"] is None:
                return "matched_same_gt_no_output"
            if row["output"]["iou"] < threshold:
                return "matched_same_gt_output_iou_below_threshold"
            return "matched_same_gt_output_diagnostic_assignment"
        return "accepted_other_or_unlabelled_detection"
    costs = []
    for stage in ("high", "low"):
        gate = config["match_thresh"] if stage == "high" else .5
        for pair in probe[stage]["pair_costs"]:
            costs.append((stage, pair["effective_cost"], gate))
    if costs and all(cost > gate + 1e-7 for _, cost, gate in costs):
        return "no_match_all_visible_pairs_above_cost_limit"
    if costs:
        return "no_match_cost_feasible_pair_lost_to_assignment"
    return "no_match_eligible_detection_outside_reachable_stage"


def main():
    validation = read_json(SOURCE / "verification.json")
    experiment = read_json(SOURCE / "experiment.json")
    analysis = read_json(SOURCE / "analysis/summary.json")
    assert experiment["status"] == "COMPLETE"
    assert analysis["status"] == "COMPLETE"
    assert validation["status"] == "PASS"
    protocol_path = SOURCE / "observed/protocols" / f"{TRACKER}.json"
    assert sha256(protocol_path) == analysis["run_protocol_sha256"]
    protocol = read_json(protocol_path)
    cohort = read_cohort(analysis)
    cases = list(offline_cases(cohort, experiment["settings"], protocol, analysis))
    OUT.mkdir(exist_ok=False)
    write_json(OUT / "observed_cases.json", cases)
    capture, parity = replay_cost(cases, experiment["settings"], protocol)
    write_json(OUT / "reprobe_parity.json", parity)
    for row in cases:
        costs = capture[row["sequence"], row["frame"], row["tid"]]
        row["cost_probe"] = costs
        row["classification"] = classify(row, costs, protocol["native_config"], analysis["iou_threshold"])
    write_json(OUT / "diagnosis.json", cases)
    flat = []
    for row in cases:
        reason = row["classification"]
        pairs = {s: {p["index"]: p for p in row["cost_probe"][s]["pair_costs"]} for s in ("high", "low")}
        for e in row["eligible"]:
            assigned = e["accepted_by"] or {}
            pair = pairs[e["tier"]].get(e["index"], {})
            flat.append({"sequence": row["sequence"], "recovery_frame": row["recovery_frame"],
                         "first_failure_frame": row["frame"], "track_id": row["tid"], "gt_id": row["gid"],
                         "failure_offset": row["delay"], "track_gap": row["gap"],
                         "classification": reason, "gt_short_side": row["gt_size"],
                         "pred_gt_iou_before_gmc": row["after_prediction"]["iou"],
                         "pred_gt_iou_after_gmc": row["after_gmc"]["iou"],
                         "pred_gt_center_error_px": row["after_gmc"]["center_px"],
                         "after_update_gt_iou": row["after_update"]["iou"],
                         "output_gt_iou": row["output"]["iou"],
                         "own_accepted_det": (row["update"] or {}).get("det_index"),
                         "own_accepted_gt": row["accepted_gt"],
                         "own_accepted_gt_iou": row["accepted_det_iou"],
                         "det_index": e["index"], "det_score": e["score"], "det_gt_iou": e["gt_iou"],
                         "det_gt_label": e["label"], "det_score_tier": e["tier"],
                         "det_assigned_track": assigned.get("track"), "det_assigned_stage": assigned.get("stage"),
                         "pred_det_iou": pair.get("geometry_iou"), "effective_cost": pair.get("effective_cost"),
                         "rgb_cosine": pair.get("rgb_cosine"),
                         "rgb_proximity_gate_pass": pair.get("proximity_allows_rgb")})
    with (OUT / "candidate_decisions.csv").open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(Counter(r["classification"] for r in cases))
    print(Counter((r["update"] or {}).get("stage", "none") for r in cases))


if __name__ == "__main__":
    main()
