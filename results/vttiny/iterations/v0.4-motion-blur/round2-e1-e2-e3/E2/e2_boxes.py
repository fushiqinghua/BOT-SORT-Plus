"""Create E2 writer-only boxes from the verified after-GMC state, with no GT."""
from __future__ import annotations

import argparse
import csv
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import sys

import numpy as np

BASE = "B_early_moderate"
RAW = "B_postlow_output_v1"
KF = "E2_kf"
TARGET = ("DJI_0303_2", 58, 2, 5)
SMOKE = ("DJI_0303_2", "DJI_0327_3", "DJI_0339_2", "DJI_0141_2")
AUDIT_PATH = "results/vttiny/iterations/v0.4-motion-blur/d3-follow-up/03-output-only/parity"


def output_json(path, payload):
    with path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, allow_nan=False, indent=2)
        file.write("\n")


def output_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_bytes())


def translated_line(line, box):
    fields = line.split(b",")
    if len(fields) != 10:
        raise ValueError("Expected a canonical ten-field MOT row")
    fields[2:6] = [format(float(value), ".17g").encode("ascii") for value in box]
    return b",".join(fields)


def xywh(box):
    x1, y1, x2, y2 = map(float, box)
    return np.array([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dtype=np.float64)


def kf_box(update, snapshot, observation):
    prior = np.array(snapshot["mean"], dtype=np.float64, copy=True)
    covariance = np.array(snapshot["covariance"], dtype=np.float64, copy=True)
    measurement = xywh(observation)
    if prior.shape != (8,) or covariance.shape != (8, 8):
        raise ValueError("Unexpected state dimensions")
    if np.any(measurement[2:] <= 0) or not all(np.isfinite(x).all() for x in (prior, covariance, measurement)):
        raise ValueError("Invalid prior, covariance or measurement")
    posterior, posterior_covariance = update(prior, covariance, measurement)
    if posterior.shape != (8,) or posterior_covariance.shape != (8, 8):
        raise ValueError("Unexpected updated state dimensions")
    if not np.isfinite(posterior).all() or not np.isfinite(posterior_covariance).all() \
            or np.any(posterior[2:4] <= 0):
        raise ValueError("Invalid posterior box/covariance; no output substitution")
    x, y, w, h = map(float, posterior[:4])
    return (x-w/2, y-h/2, w, h)


def sequence_coordinates(frame_stream, nominated, raw_lines, row_parser, update):
    complete = {}
    for trace in frame_stream:
        frame = trace["frame"]
        rows = nominated.get(frame, {})
        if not rows:
            continue
        snapshots = {item["track_id"]: item for item in trace["after_gmc"]}
        for track, event in rows.items():
            eid = (frame, track)
            index = int(event["det_index"])
            detection = trace["detections"][index]
            raw_xyxy, conf = row_parser(raw_lines[eid])
            assert track in snapshots, (trace["sequence"], eid, "after-GMC state missing")
            assert detection["det_index"] == index
            assert conf == float(detection["score"]) == float(event["score"])
            np.testing.assert_allclose(raw_xyxy, detection["xyxy"], atol=1e-5, rtol=0)
            np.testing.assert_allclose(event["xyxy"], detection["xyxy"], atol=1e-5, rtol=0)
            box = kf_box(update, snapshots[track], detection["xyxy"])
            complete[eid] = (box, dict(sequence=trace["sequence"], frame=frame, track_id=track, det_index=index,
                                        confidence=conf, pri_state=snapshots[track]["state"],
                                        pri_last_seen=snapshots[track]["last_seen"],
                                        raw_left=raw_xyxy[0], raw_top=raw_xyxy[1],
                                        raw_width=raw_xyxy[2]-raw_xyxy[0], raw_height=raw_xyxy[3]-raw_xyxy[1],
                                        kf_left=box[0], kf_top=box[1], kf_width=box[2], kf_height=box[3]))
    assert len(complete) == sum(map(len, nominated.values())), "Missing after-GMC trace/state"
    return complete


def verify_rows(original, rewritten, added, row_key):
    assert len(original) == len(rewritten)
    for old, new in zip(original, rewritten, strict=True):
        if row_key(old) in added:
            first = old.split(b",")
            second = new.split(b",")
            assert first[:2] == second[:2] and first[6:] == second[6:], "Writer altered frozen ID/conf/order"
        else:
            assert new == old, "Writer altered native bytes"


def link_tracker(output, root, name, seqs):
    target = output / "trackers" / name / "data"
    target.mkdir(parents=True)
    for seq in seqs:
        path = root / "trackers" / name / "data" / f"{seq}.txt"
        (target / path.name).symlink_to(path.resolve())


def protocol_subset(protocol, selected):
    result = {**protocol, "subset": len(selected) != 30}
    for key in ("sequences", "detection_fingerprints", "feature_hashes",
                "decoded_image_hashes", "output_prediction_hashes"):
        result[key] = {seq: protocol[key][seq] for seq in selected}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "frozen", "raw", "p0", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / AUDIT_PATH))
    from boxmot.motion.kalman_filters.aabb.xywh_kf import KalmanFilterXYWH
    from botsort_plus.diagnostics.common import iter_trace
    from botsort_plus.native_adapter import verify_upstream
    from tools.evaluate_botsort_plus import sha256
    from check_output_only import check_protocol, check_sequence, protocol, frame_track, row_key, mot_box

    p0 = read_json(args.p0)
    if p0.get("status") != "PASS":
        raise RuntimeError("P0 must pass before E2")
    base_root = args.frozen / "observed"
    roots = {BASE: base_root, RAW: args.raw}
    protocols = {name: protocol(root, name) for name, root in roots.items()}
    check_protocol(roots, protocols)
    frozen = protocols[BASE]
    if version("boxmot") != "16.0.11":
        raise RuntimeError("Use pinned BoxMOT 16.0.11")
    source_hashes = verify_upstream()
    assert all(frozen["boxmot_source_hashes"][key] == value for key, value in source_hashes.items())
    update = KalmanFilterXYWH().update
    update_file = Path(inspect.getsourcefile(update))
    selected = SMOKE if args.smoke else tuple(frozen["sequences"])
    assert len(frozen["sequences"]) == 30 and sum(frozen["sequences"].values()) == 12590
    args.output.mkdir(exist_ok=False)
    output = args.output.resolve()
    assembly = output / "inputs"
    protocol_dir = assembly / "protocols"
    protocol_dir.mkdir(parents=True)
    target = assembly / "trackers" / KF / "data"
    target.mkdir(parents=True)
    for name, root in roots.items():
        link_tracker(assembly, root, name, selected)
        output_json(protocol_dir / f"{name}.json", protocol_subset(protocols[name], selected))
    candidates, evidence, new_hashes = [], {}, {}
    for seq in selected:
        count = frozen["sequences"][seq]
        additional, lines, details = check_sequence(roots, seq, count)
        nominated = {}
        for event in additional:
            frame, track = frame_track(event)
            nominated.setdefault(frame, {})[track] = event
        path = base_root / "diagnostics" / BASE / f"{seq}.jsonl.gz"
        if additional:
            complete = sequence_coordinates(iter_trace(path, seq, count), nominated, lines, mot_box, update)
        else:
            complete = {}
        rewritten = []
        for key, line in lines.items():
            if key in complete:
                geometry, row = complete[key]
                candidates.append(row)
                line = translated_line(line, geometry)
            rewritten.append(line)
        verify_rows(list(lines.values()), rewritten, complete.keys(), row_key)
        result = target / f"{seq}.txt"
        result.write_bytes(b"".join(rewritten))
        new_hashes[seq] = sha256(result)
        evidence[seq] = {**details, "kf_mot_sha256": new_hashes[seq],
                         "diagnostic_sha256": sha256(path), "native_byte_exact": True}
        print(seq, len(complete), "frozen candidates updated", flush=True)
    keys = {(r["sequence"], r["frame"], r["track_id"], r["det_index"]) for r in candidates}
    assert len(keys) == len(candidates) and TARGET in keys
    if not args.smoke:
        assert len(keys) == 191, keys
    output_csv(output / "E2_box_pairs.csv", candidates)
    protocol = protocol_subset(protocols[RAW], selected)
    protocol["config"] = {**protocol["config"], "name": KF}
    protocol["output_prediction_hashes"] = new_hashes
    protocol["status"] = "COMPLETE"
    protocol["e2"] = {"type": "fixed-nomination detached native Kalman XYWH update",
                      "source_protocol_sha256": sha256(args.raw / "protocols" / f"{RAW}.json"),
                      "gt_used": False}
    output_json(protocol_dir / f"{KF}.json", protocol)
    output_json(output / "E2_row_integrity.json", {"status": "PASS", "smoke": args.smoke,
                "p0_sha256": sha256(args.p0), "source_sha256": sha256(Path(__file__)), "boxmot": version("boxmot"),
                "update_source_file": str(update_file), "update_source_sha256": sha256(update_file),
                "nomination_count": len(keys), "native_outputs_byte_exact": True, "sequences": evidence})


if __name__ == "__main__":
    main()
