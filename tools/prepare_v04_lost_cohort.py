#!/usr/bin/env python3
"""Offline E1-A cohort extraction; no tracker, GT injection, or new matching."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

TRACKER = "B_early_moderate"
ARCHIVE = Path(__file__).resolve().parents[1] / "results/vttiny/iterations/v0.4-motion-blur/historical_B_comparison.json"

def require(condition, message):
    if not condition:
        raise ValueError(message)

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def rows(path, required):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(set(required) <= set(reader.fieldnames or []), f"Missing fields: {path}")
        yield from reader

def number(value):
    if value is None or value.strip() == "":
        return None
    return int(value)

def bucket(row, det):
    status = row["output_status"]
    if status != "missing_with_usable_detection":
        return status or "unknown_output_status"
    if det is None:
        return "usable_without_unique_detection"
    tier = det["detection_tier"]
    owner = number(det["accepted_track_id"])
    if tier not in ("high", "low"):
        return "usable_but_paired_outside_bands"
    if owner is None:
        return "unconsumed_" + tier
    return ("same_track_consumed_" if owner == number(row["track_id"]) else "other_track_consumed_") + tier

def stats(items):
    return {
        "track_frames": len(items),
        "gt_frames": len({(x["sequence"], x["frame"], x["anchor_gt_id"]) for x in items if x["anchor_gt_id"] != ""}),
        "gt_tracks": len({(x["sequence"], x["anchor_gt_id"]) for x in items if x["anchor_gt_id"] != ""}),
        "lost_episodes": len({(x["sequence"], x["track_id"], x["last_seen"]) for x in items}),
        "sequences": len({x["sequence"] for x in items}),
    }

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    root, out = args.run_root.resolve(), args.output.resolve()
    require(not out.exists(), "Output exists; choose a new directory")
    require(not out.is_relative_to(root), "Keep new outputs outside the frozen run")
    analysis = root / "analysis"
    verification = read_json(root / "verification.json")
    summary = read_json(analysis / "summary.json")
    hist = read_json(ARCHIVE)
    require(verification["status"] == "PASS" and verification["tracker"] == TRACKER, "Reference/observed parity required")
    require(summary["status"] == "COMPLETE" and summary["tracker"] == TRACKER, "Complete D1-D3 analysis required")
    require(summary["identical_prediction_bytes"] and summary["identical_original_event_bytes"], "Analysis parity not established")
    seqs = summary["sequences"]
    require(len(seqs) == 30 and sum(seqs.values()) == 12590, "Expected full 30 sequences and 12590 frames")
    require(set(seqs) == set(verification["sequences"]) == set(hist["prediction_sha256"]), "Sequence identities differ")
    for name, key, skey in (("observed", "observed_protocol_sha256", "run_protocol_sha256"),
                            ("reference", "reference_protocol_sha256", "reference_protocol_sha256")):
        path = root / name / "protocols" / (TRACKER + ".json")
        require(digest(path) == verification[key] == summary[skey], f"{name} protocol differs")
        protocol = read_json(path)
        require(protocol["status"] == "COMPLETE" and protocol["name"] == TRACKER and protocol["sequences"] == seqs, f"{name} protocol scope differs")
        for seq in seqs:
            v = verification["sequences"][seq]
            require(v["byte_identical"] and v["original_events_byte_identical"] and v["frames"] == seqs[seq], f"Parity failed: {seq}")
            pred = root / name / "trackers" / TRACKER / "data" / (seq + ".txt")
            event = root / name / "events" / TRACKER / (seq + ".jsonl")
            require(digest(pred) == v["sha256"] == hist["prediction_sha256"][seq], f"Frozen prediction differs: {seq}")
            require(digest(event) == v["original_events_sha256"], f"Frozen events differ: {seq}")
    for file in ("track_error.csv", "detection_audit.csv", "recovery_checks.csv"):
        require(digest(analysis / file) == summary["files_sha256"][file], f"Analysis file differs: {file}")
    dfields = ("sequence","frame","det_index","gt_id","gt_match_status","detection_tier","accepted_track_id","accepted_stage","score")
    dets = {}
    for d in rows(analysis / "detection_audit.csv", dfields):
        key = (d["sequence"], number(d["frame"]), number(d["det_index"]))
        require(key not in dets, f"Duplicate detection: {key}")
        dets[key] = d
    fields = ("sequence","frame","track_id","state_before","last_seen","gap_frames","anchor_gt_id","anchor_status","gt_status",
              "gt_paired_det_index","output_status","usable_det_available","pred_center_error_px","det_center_error_px",
              "gmc_error_change_px","accepted_stage","accepted_identity")
    selected, seen, groups, boundary = [], set(), defaultdict(list), defaultdict(int)
    for row in rows(analysis / "track_error.csv", fields):
        seq, frame, tid = row["sequence"], number(row["frame"]), number(row["track_id"])
        key = (seq, frame, tid)
        require(key not in seen, f"Duplicate prediction: {key}")
        seen.add(key)
        state, gap = number(row["state_before"]), number(row["gap_frames"])
        if state != 2:
            if state in (3, 4):
                boundary["state_" + str(state)] += 1
            continue
        if gap is None or not 2 <= gap <= 30:
            boundary["lost_gap_outside_2_30"] += 1
            if gap == 31:
                boundary["lost_gap31"] += 1
            continue
        require(frame - number(row["last_seen"]) == gap, f"Gap mismatch: {key}")
        if row["gt_status"] != "present" or row["anchor_gt_id"] == "":
            boundary["lost_unknown_or_unannotated"] += 1
            continue
        di = number(row["gt_paired_det_index"])
        det = dets.get((seq, frame, di)) if di is not None else None
        if di is not None:
            require(det is not None and det["gt_match_status"] == "matched" and det["gt_id"] == row["anchor_gt_id"], f"Detection GT pairing differs: {key}")
        item = {k: row[k] for k in fields}
        item.update(cohort=bucket(row, det), detection_tier=det["detection_tier"] if det else "",
                    detection_owner=det["accepted_track_id"] if det else "",
                    detection_owner_stage=det["accepted_stage"] if det else "",
                    detector_score=det["score"] if det else "")
        selected.append(item)
        groups[item["cohort"]].append(item)
    out.mkdir(parents=True)
    with (out / "lost_cohort.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*fields,"cohort","detection_tier","detection_owner","detection_owner_stage","detector_score"])
        writer.writeheader()
        writer.writerows(selected)
    missing = [x for x in selected if x["output_status"] == "missing_with_usable_detection"]
    result = {"status":"COMPLETE", "scope":"offline diagnostic cohort, not official FP/FN or recovered matches",
              "run_root":str(root), "tracker":TRACKER, "input_hashes":summary["files_sha256"],
              "all":stats(selected),"missing_with_usable_detection":stats(missing),
              "groups":{k:stats(v) for k,v in sorted(groups.items())},
              "boundary_counts":dict(boundary),
              "notes":["Groups share GT frames; do not sum their deduplicated counts.",
                       "The paired detection is a conservative diagnostic association, not all possible overlapping candidates.",
                       "No counterfactual prediction or assignment was executed."]}
    (out / "cohort_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__ == "__main__":
    main()
