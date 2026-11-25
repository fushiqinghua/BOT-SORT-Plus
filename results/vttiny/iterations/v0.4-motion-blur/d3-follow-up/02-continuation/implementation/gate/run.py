"""Create a frozen-source local replica and replay the adjacent gate ablation."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

JOB = Path(__file__).resolve().parent
SUITE = JOB.parent
SOURCE = SUITE / "BOT-SORT-Plus"
ROUND = SUITE / "round1"
REPLICA = JOB / "repo"
ANCHOR = SOURCE / "outputs/diagnostics_v0.4_full30_round1"
B_CONFIG = "B_early_moderate"
VARIANT = "B_adjacent_gate"
SMOKE = ["DJI_0159_2", "DJI_0109_1"]

INSERTION = """
    if plus.name == "B_adjacent_gate":
        from .adjacent_gate import select_matches
        fallback = select_matches(tracker, frame, low_tracks, low_detections,
                                  second_cost, unmatched_low_tracks, unmatched_low, warp)
        kept_tracks, kept_detections = set(), set()
        for i, j, probe in fallback:
            track, detection = low_tracks[i], low_detections[j]
            track.update(detection, frame)
            tracker._write_memory(track, int(detection.det_ind))
            track._gate_uses += 1
            activated.append(track)
            kept_tracks.add(i)
            kept_detections.add(j)
            tracker.last_events.append({"event": "match", "stage": "low_adjacent_fallback",
                                        "track_id": int(track.id), "det_index": int(detection.det_ind),
                                        "cost": float(second_cost[i, j]),
                                        "recovery_frame": probe["recovery_frame"],
                                        "normalized_centroid_motion": probe["normalized_centroid_motion"]})
        unmatched_low_tracks = [i for i in unmatched_low_tracks if int(i) not in kept_tracks]
        unmatched_low = [j for j in unmatched_low if int(j) not in kept_detections]
"""
RECOVERY_MARKER = "            refound.append(track)\n            lost = [t for t in lost if t.id != track.id]\n"
RECOVERY_INSERT = """            if plus.name == "B_adjacent_gate":
                track._gate_recovery_frame = frame
                track._gate_uses = 0
"""
END_MARKER = "    if diagnostics is not None:\n        diagnostics.finish(tracker, output, removed)\n"
END_INSERT = """    if plus.name == "B_adjacent_gate":
        tracker._gate_previous_boxes = all_dets[:, :4].copy()
"""


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def patch(source, marker, addition):
    if source.count(marker) != 1:
        raise ValueError(f"Adapter marker changed: {marker!r}")
    return source.replace(marker, addition + marker)


def prepare():
    REPLICA.mkdir(exist_ok=False)
    for directory in ("botsort_plus", "tools", "configs"):
        shutil.copytree(SOURCE / directory, REPLICA / directory,
                        ignore=shutil.ignore_patterns("__pycache__"))
    adapter = REPLICA / "botsort_plus/adapter_v2.py"
    original = adapter.read_text(encoding="utf-8")
    modified = patch(original, "    for i in unmatched_low_tracks:\n", INSERTION)
    modified = patch(modified, RECOVERY_MARKER, RECOVERY_INSERT)
    modified = patch(modified, END_MARKER, END_INSERT)
    adapter.write_text(modified, encoding="utf-8")
    shutil.copyfile(JOB / "adjacent_gate.py", REPLICA / "botsort_plus/adjacent_gate.py")

    config = read(REPLICA / f"configs/botsort_plus/{B_CONFIG}.json")
    config["name"] = VARIANT
    write(REPLICA / f"configs/botsort_plus/{VARIANT}.json", config)
    boundary = {
        "variant": VARIANT, "recovery_window": 3, "consecutive_measurement": 1,
        "fallback_limit_per_restore": 1, "iou_range": "[0.40,0.50)",
        "prior_normalized_center_displacement_max": 0.50,
        "candidate_guard": "Only unassigned native low detections; one candidate in residual row at .4, no other low-track row at .4 for that candidate",
        "method_sha256": sha(JOB/"adjacent_gate.py"),
        "original_adapter_sha256": sha(SOURCE/"botsort_plus/adapter_v2.py"),
        "patched_adapter_sha256": sha(adapter),
    }
    write(JOB/"policy.json", boundary)


def replay(label, name, sequences=None):
    base = read(ANCHOR / "experiment.json")["settings"]
    dest = JOB / label
    args = [sys.executable, "-B", "-u", str(REPLICA/"tools/run_botsort_plus.py"),
            "--baseline-root", base["baseline_root"], "--features-dir", base["features_dir"],
            "--images-root", base["images_root"], "--runtime-root", base["runtime_root"],
            "--config", str(REPLICA/f"configs/botsort_plus/{name}.json"),
            "--output-dir", str(dest), "--trace"]
    if sequences:
        args.extend(["--sequences", *sequences])
    with (JOB / (label + ".log")).open("x", encoding="utf-8") as out:
        subprocess.run(args, cwd=REPLICA, stdout=out, stderr=subprocess.STDOUT, check=True)
    return dest


def compare(dest, name, inactive):
    ap = ANCHOR/"observed"
    before = read(ap/"protocols"/f"{B_CONFIG}.json")
    current = read(dest/"protocols"/f"{name}.json")
    assert current["status"] == "COMPLETE"
    for key in ("native_config", "detector_sha256", "rgb_encoder_sha256",
                "source_protocol_sha256", "boxmot_source_hashes"):
        assert before[key] == current[key], key
    assert current["config"] == before["config"] | {"name": name}
    changed = []
    for seq in current["sequences"]:
        for key in ("sequences", "feature_hashes", "detection_fingerprints", "decoded_image_hashes"):
            assert before[key][seq] == current[key][seq], (seq, key)
        old = ap/f"trackers/{B_CONFIG}/data/{seq}.txt"
        new = dest/f"trackers/{name}/data/{seq}.txt"
        assert sha(old) == before["output_prediction_hashes"][seq]
        assert sha(new) == current["output_prediction_hashes"][seq]
        if old.read_bytes() != new.read_bytes():
            changed.append(seq)
        if inactive:
            assert seq not in changed
            assert (ap/f"events/{B_CONFIG}/{seq}.jsonl").read_bytes() == (dest/f"events/{name}/{seq}.jsonl").read_bytes()
    reasons = Counter()
    accepted = []
    for seq in current["sequences"]:
        with (dest/f"events/{name}/{seq}.jsonl").open(encoding="utf-8") as stream:
            current_frame = None
            for line in stream:
                row = json.loads(line)
                if row.get("event") == "frame":
                    current_frame = row["frame"]
                if row.get("event") == "adjacent_gate_probe":
                    reasons[row["reason"]] += 1
                if row.get("event") == "match" and row.get("stage") == "low_adjacent_fallback":
                    accepted.append({"sequence": seq, "frame": current_frame, **row})
    if inactive:
        assert not (reasons or accepted)
    else:
        assert reasons["accepted"] == len(accepted)
    return {"variant": name, "num_sequences": len(current["sequences"]),
            "per_input_verification": "same detector, config (except name), features, pixels and runtime as frozen B",
            "changed_sequences": changed, "num_changed": len(changed),
            "probe_reasons": dict(reasons), "accepted": accepted}


def main():
    if not REPLICA.exists():
        prepare()
    off = replay("smoke_disabled", B_CONFIG, SMOKE)
    record = {"disabled":compare(off, B_CONFIG, inactive=True)}
    on = replay("smoke_enabled", VARIANT, SMOKE)
    record["enabled"] = compare(on, VARIANT, inactive=False)
    write(JOB/"smoke_summary.json", record)
    print("smoke", len(record["enabled"]["accepted"]), flush=True)
    full = replay("full", VARIANT)
    output = compare(full, VARIANT, inactive=False)
    assert output["num_sequences"] == 30
    write(JOB/"full_summary.json", output)
    print("full", output["num_changed"], len(output["accepted"]), flush=True)


if __name__ == "__main__":
    main()
