"""Audit the six paired DJI_0061_2 CLEAR results and the output-only diagnostic."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path


ARMS = ("native_off", "native_W0", "B_off", "B_W0", "BG_off", "BG_W0", "native_emit_only")
PAIRS = (("native_off", "native_W0"), ("B_off", "B_W0"), ("BG_off", "BG_W0"), ("native_off", "native_emit_only"))
FOCUS = ((47, 75, 94), (126, 79, 97), (138, 91, 103), (119, 171, 196), (147, 221, 240),
         (99, 167, 176), (95, 167, 176), (139, 216, 223))


def read(path):
    return json.loads(path.read_bytes())


def jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def load_arm(path):
    clear = read(path / "clear.json")
    frames = clear["frames"]
    switches = {(event["frame"], str(event["gt_id"])): event for event in clear["switches"]}
    if len(switches) != clear["counts"]["IDSW"]:
        raise ValueError(f"IDs do not match official CLEAR totals: {path.name}")
    weak = {(row["frame"], str(row["track_id"])): row for row in jsonl(path / "weak_events.jsonl")
            if row["event"] == "committed"}
    ordinary = {}
    births = {}
    for row in jsonl(path / "native_events.jsonl"):
        kind = row["event"]
        if kind in ("match", "recovered", "native_birth"):
            key = row["frame"], str(row["track_id"])
            stage = row["stage"] if kind == "match" else "B_recovery" if kind == "recovered" else "native_birth"
            ordinary[key] = (stage, row["det_index"])
            if stage == "native_birth":
                births[key[1]] = (key[0], row["det_index"])
    previous = {}
    observed = {}
    for frame in frames:
        time = frame["frame"]
        for gid, tid in frame["coverage"].items():
            before = previous.get(gid)
            stage, index = source(weak, ordinary, time, tid)
            tag = "switch" if (time, gid) in switches else "continue" if before else "first_match"
            output = frame["outputs"][str(tid)]
            observed[time, gid] = dict(tag=tag, track=tid, stage=stage, det_index=index, matched_iou=output["gt_ious"][gid],
                previous_frame=before[0] if before else None, previous_track=before[1] if before else None)
            previous[gid] = time, tid
    return dict(frames=frames, counts=clear["counts"], switches=switches, weak=weak, ordinary=ordinary,
                observed=observed, births=births)


def source(weak, ordinary, time, track):
    key = time, str(track)
    if key in weak:
        row = weak[key]
        stage = "weak_emit_only" if row.get("state_committed") is False else "weak_commit"
        return stage, row["det_index"]
    return ordinary.get(key, ("unlogged", None))


def save(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def frame_row(arm, time, gt_id, data, detections):
    hit = data["observed"].get((time, gt_id), {})
    idx = hit.get("det_index")
    score = detections[str(time)][idx][4] if idx is not None else None
    tid = hit.get("track")
    old_id = hit.get("previous_track")
    birth_frame, birth_det = data["births"].get(str(tid), (None, None))
    last_updates = [(frame, int(track)) for frame, track in data["weak"]
                    if time - 10 <= frame < time and track in {str(tid), str(old_id)}]
    return dict(arm=arm, frame=time, gt=gt_id, status=hit.get("tag", "miss"), track=tid,
                stage=hit.get("stage"), det_index=idx, raw_score=score, matched_iou=hit.get("matched_iou"),
                previous_frame=hit.get("previous_frame"), previous_track=old_id,
                birth_frame=birth_frame, birth_det_index=birth_det,
                prior_weak_on_previous_or_current_id_10f=json.dumps(sorted(last_updates), separators=(",", ":")))


def stable_history(frames, frame, tid):
    history = [f["outputs"].get(tid, {}).get("matched_gt") for f in frames[max(0, frame-6):frame-1]]
    valid = [gid for gid in history if gid is not None]
    return valid[0] if len(valid) >= 2 and len(set(valid)) == 1 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Private CLEAR and event snapshot directory")
    parser.add_argument("--output", type=Path, required=True, help="Directory for derived comparison tables")
    args = parser.parse_args()
    inputs = args.inputs.resolve()
    all_data = {name: load_arm(inputs / name) for name in ARMS}
    detector = read(inputs / "detections.json")
    gt_frames = read(inputs / "gt_frames.json")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    emit = read(inputs / "native_emit_only/receipt.json")
    if (not emit["canonical_equal_to_native_off"] or emit["counts"]["weak_state_updates"] != 0
            or (inputs / "native_emit_only/native_events.jsonl").read_bytes() !=
            (inputs / "native_off/native_events.jsonl").read_bytes()):
        raise ValueError("Output-only diagnostic did not retain the original tracking state and events")
    comparison = {}
    all_changes = set()
    for off, on in PAIRS:
        baseline, variant = all_data[off], all_data[on]
        added = set(variant["switches"]) - set(baseline["switches"])
        lost = set(baseline["switches"]) - set(variant["switches"])
        both = set(baseline["switches"]) & set(variant["switches"])
        if len(added)-len(lost) != variant["counts"]["IDSW"]-baseline["counts"]["IDSW"]:
            raise ValueError(f"Switch pair totals differ: {on}")
        all_changes |= added | lost
        stages = Counter(variant["observed"][key]["stage"] for key in added)
        comparison[on] = dict(reference=off, added=len(added), lost=len(lost), shared=len(both), stages=dict(stages),
                              added_keys=sorted(added), lost_keys=sorted(lost))

    details = []
    for frame, gt_id in sorted(all_changes):
        if gt_id not in gt_frames[frame-1]:
            raise ValueError(f"Switch outside official ground truth: {frame}/{gt_id}")
        for arm, data in all_data.items():
            details.append(frame_row(arm, frame, gt_id, data, detector))
    save(out / "switch_comparison.csv", details)

    weak_rows = []
    for arm, data in all_data.items():
        for (frame, tid), event in data["weak"].items():
            row = data["frames"][frame - 1]["outputs"][tid]
            observed = row["matched_gt"]
            anchor = stable_history(data["frames"], frame, tid)
            outcome = ("FP" if observed is None else "identity_unconfirmed" if anchor is None
                       else "same_stable_GT" if observed == anchor else "different_stable_GT")
            weak_rows.append(dict(arm=arm, frame=frame, track=tid, det_index=event["det_index"], matched_gt=observed,
                 stable_previous_gt=anchor, outcome=outcome, max_iou=row["max_iou"], raw_score=event["raw_score"],
                 state_committed=event.get("state_committed", True),
                 official_same_frame_switch=(frame, str(observed)) in data["switches"]))
    save(out / "weak_observations.csv", weak_rows)
    focus = [frame_row(arm, time, str(gid), data, detector)
             for gid, start, end in FOCUS for time in range(start, end+1) if str(gid) in gt_frames[time-1]
             for arm, data in all_data.items()]
    save(out / "focus_timelines.csv", focus)

    native_added = {tuple(key) for key in comparison["native_W0"]["added_keys"]}
    emit_switches = set(all_data["native_emit_only"]["switches"])
    observed = {arm: Counter(data["observed"].get(key, {}).get("tag", "miss") for key in native_added)
                for arm, data in all_data.items()}
    native_changes = {name: [] for name in ("off_miss", "same_detection_different_birth", "same_detection_same_birth",
                                           "different_detection")}
    baseline, variant = all_data["native_off"], all_data["native_W0"]
    for key in sorted(native_added):
        before, after = baseline["observed"].get(key), variant["observed"][key]
        if before is None:
            group = "off_miss"
        elif before["det_index"] != after["det_index"]:
            group = "different_detection"
        else:
            old_birth = baseline["births"][str(before["track"])]
            new_birth = variant["births"][str(after["track"])]
            group = "same_detection_same_birth" if old_birth == new_birth else "same_detection_different_birth"
        native_changes[group].append(key)
    summary = dict(counts={arm: data["counts"] for arm, data in all_data.items()}, pairs=comparison,
        same_native_added_frames={arm: dict(counts) for arm, counts in observed.items()},
        native_added_also_switch_in_emit_only=sorted(native_added & emit_switches),
        native_added_current_lineage=native_changes,
        weak_outcomes={arm: dict(Counter(row["outcome"] for row in weak_rows if row["arm"] == arm)) for arm in ARMS})
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(f"{len(all_changes)} changed GT/frame keys, {len(details)} aligned rows, {len(weak_rows)} weak output rows")


if __name__ == "__main__":
    main()
