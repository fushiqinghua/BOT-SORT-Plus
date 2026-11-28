"""Export per-frame W3 vs W2 evidence from the local CLEAR ledgers."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


SEQUENCES = ("DJI_0073_3", "DJI_0327_3")
POLICIES = ("off", "W2", "W3")
METRICS = ("TP", "FP", "FN", "IDSW")
TRACE_WINDOWS = {
    "DJI_0073_3": ((29, 35, 96, 98), (15, 30, 116, 121), (26, 33, 118, 122),
                   (47, 58, 207, 210), (18, None, 229, 360), (97, None, 317, 360)),
    "DJI_0327_3": ((28, 81, 79, 84), (36, 77, 87, 94), (47, 168, 126, 130),
                   (44, 169, 160, 164), (55, 16, 227, 238), (79, 9, 355, 360),
                   (72, 5, 388, 390)),
}


def jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def index_events(events, names):
    by_track = defaultdict(dict)
    for event in events:
        if event["event"] in names:
            key = event["frame"], int(event["track_id"])
            by_track[key][event["event"]] = event
    return by_track


def arm_inputs(path):
    result = json.loads((path / "clear.json").read_bytes())
    events = index_events(jsonl(path / "native_events.jsonl"), ("match", "recovered", "native_birth", "motion_output"))
    commits = index_events(jsonl(path / "weak_events.jsonl"), ("committed",))
    for frame in result["frames"]:
        for tid, output in frame["outputs"].items():
            key = frame["frame"], int(tid)
            native = events.get(key, {})
            weak = commits.get(key, {})
            if len(native) + len(weak) > 1:
                raise ValueError(f"Ambiguous emitted state update in {path.name}: {key}, {native}, {weak}")
            if weak:
                update = weak["committed"]
                stage = "W_commit"
            elif "match" in native:
                update = native["match"]
                stage = "native_" + update["stage"]
            elif native:
                stage, update = next(iter(native.items()))
                if stage == "recovered":
                    stage = "B_recovered"
            else:
                stage, update = "unlogged", {}
            output.update(stage=stage, det_index=update.get("det_index"))
    return result, events, commits


def save(path, rows, omit_boxes):
    excluded = {"gt_xywh", "gt_box_xywh", "output_box_xywh"} if omit_boxes else set()
    fields = [field for field in rows[0] if field not in excluded]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def match_at(frame, track):
    value = frame["outputs"].get(str(track), {})
    return {key: value.get(key) for key in ("status", "matched_gt", "nearest_gt", "max_iou", "stage", "det_index")}


def align_false_positives(key, frames):
    def index(frame):
        rows = {o["det_index"]: o for o in frame["outputs"].values() if o["status"] == "FP"}
        if None in rows or len(rows) != frame["counts"]["FP"]:
            raise ValueError(f"Cannot align FP by detector row for {key}/{frame['frame']}")
        return rows

    per_policy = {p: index(frame) for p, frame in frames.items()}
    rows = []
    for det in sorted(set(per_policy["W2"]) ^ set(per_policy["W3"])):
        values = {policy: fps.get(det, {}) for policy, fps in per_policy.items()}
        rows.append(dict(**key, frame=frames["off"]["frame"], det_index=det,
            W3_minus_W2=int(bool(values["W3"])) - int(bool(values["W2"])),
            **{f"{p}_{field}": value.get(field) for p, value in values.items()
               for field in ("track_id", "stage", "nearest_gt", "max_iou")}))
    return rows


def audit_case(directory, sequence, role, gt, by_policy, count_rows, gt_rows, fp_rows, gate_rows, recovery_rows,
               fp_difference_rows, timeline_rows):
    key = {"sequence": sequence, "reference": role}
    counts = {policy: Counter() for policy in POLICIES}
    for f_off, f_w2, f_w3 in zip(*(by_policy[policy][0]["frames"] for policy in POLICIES), strict=True):
        frame = f_off["frame"]
        frames = dict(zip(POLICIES, (f_off, f_w2, f_w3), strict=True))
        count_rows.append(dict(**key, frame=frame,
            **{f"{p}_{metric}": v["counts"][metric] for p, v in frames.items() for metric in METRICS},
            **{f"W3_minus_W2_{metric}": f_w3["counts"][metric] - f_w2["counts"][metric] for metric in METRICS}))
        for policy, v in frames.items():
            counts[policy].update(o["stage"] for o in v["outputs"].values() if o["status"] == "FP")
        fp_difference_rows.extend(align_false_positives(key, frames))
        for gid, box in gt[frame - 1].items():
            hits = {p: v["coverage"].get(gid) for p, v in frames.items()}
            if all(tid is None for tid in hits.values()) or len(set(hits.values())) == 1:
                continue
            after = {p: v["outputs"].get(str(hits[p]), {}) for p, v in frames.items()}
            gt_rows.append(dict(**key, frame=frame, gt=gid, gt_xywh=json.dumps(box),
                **{f"{p}_track": tid for p, tid in hits.items()},
                **{f"{p}_stage": o.get("stage") for p, o in after.items()},
                **{f"{p}_det_index": o.get("det_index") for p, o in after.items()},
                W3_vs_W2="lost" if hits["W2"] is not None and hits["W3"] is None else
                "gained" if hits["W2"] is None and hits["W3"] is not None else "same_coverage"))
    for policy in POLICIES:
        if sum(counts[policy].values()) != by_policy[policy][0]["counts"]["FP"]:
            raise ValueError(f"FP stages do not reconcile for {sequence}/{role}_{policy}")
    for stage in sorted(set().union(*counts.values())):
        fp_rows.append(dict(**key, stage=stage, **{p: counts[p][stage] for p in POLICIES},
                            W3_minus_W2=counts["W3"][stage] - counts["W2"][stage]))
    records = jsonl(directory / f"{role}_W3" / "wem_decisions.jsonl")
    shadow = {(r["frame"], r["track_id"], r["det_index"]): r for r in records if r["method"] == "W2" and r["selected"]}
    alternatives = {(r["frame"], r["track_id"]): r["det_index"] for r in records if r["method"] == "W3" and r["selected"]}
    for r in records:
        frame, tid, det = r["frame"], r["track_id"], r["det_index"]
        if r["method"] != "W3" or (frame, tid, det) not in shadow or not r["reasons"]:
            continue
        gate_rows.append(dict(**key, frame=frame, track=tid, det_index=det, reason="|".join(r["reasons"]),
            row_margin=r["row_margin"], column_margin=r["column_margin"], cosine=r["cosine"],
            geometric_iou=r["native_iou"], motion=r["motion"], W3_selected_alternative=alternatives.get((frame, tid)),
            W2_actual_same_commit=int(by_policy["W2"][2].get((frame, tid), {}).get("committed", {}).get("det_index") == det),
            **{f"{p}_{k}": v for p in POLICIES for k, v in match_at(by_policy[p][0]["frames"][frame-1], tid).items()}))
    for policy in POLICIES:
        item = by_policy[policy]
        for (frame, tid), events in item[1].items():
            if "recovered" not in events:
                continue
            r = events["recovered"]
            recovery_rows.append(dict(**key, arm=policy, frame=frame, track=tid, candidate=r["candidate_id"],
                det_index=r["det_index"], lost_age=r["age"], geometry_cost=r["geometry_cost"],
                appearance_cost=r["appearance_cost"], **{f"output_{k}": v for k, v in match_at(item[0]["frames"][frame-1], tid).items()}))
        for tid, gid, first, last in TRACE_WINDOWS[sequence]:
            for frame in range(first, last + 1):
                record = item[0]["frames"][frame - 1]
                output = record["outputs"].get(str(tid), {})
                box = output.get("box_xywh")
                target_box = gt[frame - 1].get(str(gid))
                timeline_rows.append(dict(**key, arm=policy, frame=frame, track=tid, focus_gt=gid,
                    gt_covered_by=record["coverage"].get(str(gid)), gt_box_xywh=json.dumps(target_box) if target_box else "",
                    output_box_xywh=json.dumps(box) if box else "", **match_at(record, tid)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path, help="Private ledgers produced by fetch.py")
    parser.add_argument("--output", required=True, type=Path, help="Audit table directory")
    parser.add_argument("--omit-boxes", action="store_true", help="Exclude ground-truth and tracker box coordinates")
    args = parser.parse_args()
    tables = {name: [] for name in ("frame_counts", "gt_differences", "fp_provenance", "w2_w3_competition",
                                    "B_recoveries", "fp_differences", "key_track_timeline")}
    for sequence in SEQUENCES:
        directory = args.inputs / sequence
        gt = json.loads((directory / "gt_frames.json").read_bytes())
        for role in ("B", "BG"):
            by_policy = {policy: arm_inputs(directory / f"{role}_{policy}") for policy in POLICIES}
            audit_case(directory, sequence, role, gt, by_policy, *tables.values())
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        save(args.output / f"{name}.csv", rows, args.omit_boxes)
        print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
