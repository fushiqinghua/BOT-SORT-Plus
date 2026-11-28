"""Verify the published native BoT-SORT W0 ablation using only the archive."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re


BASES = ("native", "B", "BG")
ARMS = tuple(f"{base}_{mode}" for base in BASES for mode in ("off", "W0"))
OFF, W0, B_OFF, B_W0, BG_OFF, BG_W0 = ARMS
EMIT = "native_emit_only"
CASE_ARMS = (*ARMS, EMIT)
CASE = "DJI_0061_2"
CASE_ROOT = "dji_0061/tables/"
POOLED = "COMBINED_SEQ"
COUNTS = ("FP", "FN", "IDSW")
METRICS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", *COUNTS)
PERCENT = ("HOTA", "IDF1", "MOTA")
REQUIRED = {
    "run_provenance.json", "trackeval/summary.json", "comparison.csv", "paired_per_sequence.csv", "summary.json",
    "receipts/plan.json", "receipts/complete.json", "receipts/smoke_verified.json", "receipts/full_verified.json",
    "dji_0061/emit_only_receipt.json",
} | {CASE_ROOT + name for name in (
    "summary.json", "camera_evidence.json", "switch_comparison.csv", "weak_observations.csv", "focus_timelines.csv",
)}
SWITCH_COUNT = {OFF: 158, W0: 174, B_OFF: 178, B_W0: 182, BG_OFF: 178, BG_W0: 182, EMIT: 158}
R4_1 = "3132e69fcca86ac5ce6d59271b2dff38454fc628"


def require(test, context):
    if not test:
        raise ValueError(context)


def read_json(path):
    return json.loads(path.read_bytes())


def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def hex_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def verify_manifest(root):
    manifest = read_json(root / "manifest.json")
    require(manifest["schema"] == "wem_native_w0_public_archive_v1", "Wrong archive schema")
    files = manifest["files"]
    require(REQUIRED <= set(files), f"Missing manifest entries: {REQUIRED - set(files)}")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path != root / "manifest.json"}
    require(set(files) == actual, f"Unlisted or missing archive files: {set(files) ^ actual}")
    for name, receipt in files.items():
        relative = PurePosixPath(name)
        path = root / relative
        require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in name and ":" not in name
                and path.resolve().is_relative_to(root), f"Invalid archive path: {name}")
        raw = path.read_bytes()
        require(len(raw) == receipt["bytes"] and hashlib.sha256(raw).hexdigest() == receipt["public_sha256"],
                f"Digest or length differs: {name}")
    return files


def index(rows, key, expected=None):
    result = {key(row): row for row in rows}
    require(len(result) == len(rows) and (expected is None or set(result) == expected), "Unexpected or duplicated rows")
    return result


def close(observed, expected):
    return math.isclose(float(observed), expected, rel_tol=0, abs_tol=1e-9)


def verify_metrics(root, lengths, summary):
    official = read_json(root / "trackeval/summary.json")
    require(official["status"] == "PASS" and official["modality_reference"] == "rgb" and official["split"] == "test",
            "Unexpected TrackEval status or scope")
    metrics = index(official["results"], lambda row: (row["tracker"], row["sequence"]),
                    {(arm, seq) for arm in ARMS for seq in (*lengths, POOLED)})
    comparisons = [index(rows, lambda row: row["arm"], set(ARMS)) for rows in (
        read_table(root / "comparison.csv"), summary["comparison"],
    )]
    paired = index(read_table(root / "paired_per_sequence.csv"), lambda row: (row["arm"], row["sequence"]),
                   {(arm, seq) for arm in ARMS for seq in lengths})
    directions = {base: {} for base in BASES}
    for arm in ARMS:
        base = arm.split("_")[0]
        off = f"{base}_off"
        for seq in (*lengths, POOLED):
            actual, control = metrics[arm, seq], metrics[off, seq]
            records = [table[arm] for table in comparisons] if seq == POOLED else [paired[arm, seq]]
            for row in records:
                require((row["base"], row["reference"], row["sequence"]) == (base, off, seq),
                        f"Wrong pairing: {arm}/{seq}")
                for name in METRICS:
                    a, b = actual[name], control[name]
                    require(type(a) in (int, float) and math.isfinite(a) and (name not in COUNTS or type(a) is int and a >= 0),
                            f"Invalid official metric: {arm}/{seq}/{name}")
                    require(float(row[name]) == a and close(row[f"delta_{name}"], a - b), f"Wrong metric or delta: {arm}/{seq}/{name}")
        for name in COUNTS:
            require(sum(metrics[arm, seq][name] for seq in lengths) == metrics[arm, POOLED][name], f"Wrong pooled count: {arm}/{name}")
        weak_counts = [int(paired[arm, seq]["weak_commits"]) for seq in lengths]
        expected = {W0: 524, B_W0: 730, BG_W0: 730}.get(arm, 0)
        require(all(0 <= count for count in weak_counts)
                and sum(weak_counts) == expected == int(comparisons[0][arm]["weak_commits"]) == comparisons[1][arm]["weak_commits"],
                f"Wrong weak output count: {arm}")
        if arm.endswith("_W0"):
            for name in (*PERCENT, *COUNTS):
                direction = 1 if name in PERCENT else -1
                moves = [direction * float(paired[arm, seq][f"delta_{name}"]) for seq in lengths]
                better, worse = sum(value > 1e-9 for value in moves), sum(value < -1e-9 for value in moves)
                directions[base][name] = dict(better=better, worse=worse, tied=len(moves) - better - worse)
    require(summary["sequence_direction"] == directions, "Wrong per-sequence direction counts")
    difference_pairs = {("B", "native"), ("BG", "B"), ("BG", "native")}
    contrasts = index(summary["incremental_differences"], lambda row: (row["base"], row["reference"]), difference_pairs)
    pooled = comparisons[1]
    for (base, reference), row in contrasts.items():
        for name in METRICS:
            delta = pooled[f"{base}_W0"][f"delta_{name}"] - pooled[f"{reference}_W0"][f"delta_{name}"]
            require(close(row[name], delta), f"Wrong between-base difference: {base}/{reference}/{name}")
    return metrics, paired


def recorded_sha(files, name):
    fields = files[name]
    return fields.get("original_sha256", fields["public_sha256"])


def verify_receipts(root, files, lengths, summary, paired):
    plan = read_json(root / "receipts/plan.json")
    complete = read_json(root / "receipts/complete.json")
    full = read_json(root / "receipts/full_verified.json")
    smoke = read_json(root / "receipts/smoke_verified.json")
    require(complete == summary["completion"] and complete["status"] == "COMPLETE"
            and complete["sequence_count"] == len(lengths) and complete["frames_per_arm"] == sum(lengths.values())
            and complete["prior_metric_parity"] == complete["native_trajectory_parity"] == "PASS"
            and complete["new_replays"] == list(ARMS[:2]) and complete["retained_references"] == list(ARMS[2:])
            and complete["metric_sha256"] == recorded_sha(files, "trackeval/summary.json")
            and complete["full_verification_sha256"] == recorded_sha(files, "receipts/full_verified.json"),
            "Frozen completion or original digests differ")
    require(plan["sequences"] == lengths and plan["toggles"] == ["off", "W0"]
            and plan["smoke"] == ["DJI_0030_2", "DJI_0061_2", "DJI_0327_3"]
            and all(hex_sha(plan[key]) for key in ("source_sha256", "prediction_bundle_sha256", "native_reference_protocol_sha256")),
            "Frozen W0 plan differs")
    for record, scope in ((smoke, {seq: lengths[seq] for seq in plan["smoke"]}), (full, lengths)):
        require(record["sequences"] == scope and set(record["results"]) == set(scope)
                and record["status"] == "PASS" and record["native_MOT_parity"] is True, "Native receipt scope or parity differs")
        for seq, rows in record["results"].items():
            off, on = rows[OFF], rows[W0]
            require(off["commits"] == 0 and off["first_commit"] is None and off["frames"] == on["frames"] == scope[seq]
                    and off["warp_sha256"] == on["warp_sha256"], f"Wrong native scope or SOF parity: {seq}")
            start = on["first_commit"]
            require((start is None and on["commits"] == 0 and off["mot"] == on["mot"]
                     and rows["prefix_verified"] == scope[seq]) or (type(start) is int and 1 <= start <= scope[seq]
                     and 0 < on["commits"] and rows["prefix_verified"] == start - 1), f"Wrong native state prefix: {seq}")
            if record is full:
                require(on["commits"] == int(paired[W0, seq]["weak_commits"]), f"Rerun weak tally differs: {seq}")
    same = sum(rows[OFF]["mot"] == rows[W0]["mot"] for rows in full["results"].values())
    require(same == summary["native_sequences_with_same_trajectory"], "Wrong same-trajectory count")
    return full["results"][CASE]


def frame_key(row):
    return int(row["frame"]), str(row["gt"])


def audit_switches(root, report):
    def read_frames(filename):
        values = read_table(root / (CASE_ROOT + filename))
        times = {frame_key(row) for row in values}
        require(all(1 <= frame <= 240 and int(gt) > 0 for frame, gt in times), f"Invalid frame or GT: {filename}")
        return index(values, lambda row: (row["arm"], *frame_key(row)),
                     {(arm, *key) for arm in CASE_ARMS for key in times})

    diff = read_frames("switch_comparison.csv")
    focus = read_frames("focus_timelines.csv")
    require(len(diff) == 665 and len(focus) == 861 and all(focus[key] == diff[key] for key in set(focus) & set(diff)),
            "Case tables are incomplete or disagree")
    switches = {arm: set() for arm in CASE_ARMS}
    for (arm, frame, gt), row in diff.items():
        if row["status"] == "switch":
            switches[arm].add((frame, gt))
    distinct = set()
    expected = {W0: (OFF, 34, 18, {"high": 25, "unconfirmed": 6, "low": 3}),
                B_W0: (B_OFF, 31, 27, {"B_recovery": 3, "high": 16, "unconfirmed": 5, "low": 7}),
                BG_W0: (BG_OFF, 31, 27, {"B_recovery": 3, "high": 16, "unconfirmed": 5, "low": 7}),
                EMIT: (OFF, 1, 1, {"high": 1})}
    require(set(report["pairs"]) == set(expected), "Unexpected switch comparisons")
    for variant, (baseline, added_count, lost_count, stages) in expected.items():
        added = switches[variant] - switches[baseline]
        lost = switches[baseline] - switches[variant]
        distinct.update(added | lost)
        t = report["pairs"][variant]
        require(len(added) == t["added"] == added_count and len(lost) == t["lost"] == lost_count
                and t["reference"] == baseline and t["shared"] == SWITCH_COUNT[baseline] - lost_count == SWITCH_COUNT[variant] - added_count
                and added == {tuple(k) for k in t["added_keys"]} and lost == {tuple(k) for k in t["lost_keys"]}
                and Counter(diff[variant, *key]["stage"] for key in added) == t["stages"] == stages,
                f"Wrong added, removed or originating switch stages: {variant}")
    require(distinct == {key[1:] for key in diff}, "Unaccounted switch events")
    native_added = switches[W0] - switches[OFF]
    by_arm = {arm: dict(Counter(diff[arm, *key]["status"] for key in native_added)) for arm in CASE_ARMS}
    require(by_arm == report["same_native_added_frames"] and by_arm[B_W0] == {"continue": 16, "switch": 12, "miss": 6},
            "Wrong treatment of native added switches by other arms")
    lineage = {key: [] for key in ("off_miss", "same_detection_different_birth", "same_detection_same_birth", "different_detection")}
    for key in sorted(native_added):
        old, new = diff[OFF, *key], diff[W0, *key]
        if old["status"] == "miss":
            group = "off_miss"
        elif old["det_index"] != new["det_index"]:
            group = "different_detection"
        else:
            origin = ("birth_frame", "birth_det_index")
            group = ("same_detection_same_birth" if all(old[field] == new[field] for field in origin)
                     else "same_detection_different_birth")
        lineage[group].append(list(key))
    require(lineage == report["native_added_current_lineage"] and tuple(map(len, lineage.values())) == (6, 23, 5, 0),
            "Wrong native lineage breakdown")
    require(switches[EMIT] - switches[OFF] == native_added & switches[EMIT] == {(221, "139")}
            and switches[OFF] - switches[EMIT] == {(220, "139")}
            and report["native_added_also_switch_in_emit_only"] == [[221, "139"]], "Wrong emit-only GT139 shift")
    for table in (diff, focus):
        for (arm, *key), row in table.items():
            if arm in (B_OFF, B_W0):
                other = table["BG" + arm[1:], *key]
                require({k: v for k, v in row.items() if k != "arm"} == {k: v for k, v in other.items() if k != "arm"},
                        f"B/BG frame rows differ: {arm}/{key}")
    require((focus[B_W0, 184, "119"]["stage"], focus[B_W0, 184, "119"]["status"],
             focus[W0, 224, "147"]["stage"], focus[W0, 224, "147"]["status"]) ==
            ("B_recovery", "continue", "weak_commit", "first_match"), "Wrong GT119 recovery or GT147 first match")


def audit_case(root, provenance, native, metrics, paired):
    report = read_json(root / (CASE_ROOT + "summary.json"))
    require(set(report["counts"]) == set(CASE_ARMS), "Incomplete case arm scope")
    for arm in CASE_ARMS:
        row = report["counts"][arm]
        require(row["IDSW"] == SWITCH_COUNT[arm], f"Unexpected case IDSW: {arm}")
        if arm == EMIT:
            require((row["FP"], row["FN"], row["IDSW"]) == (1942, 17480, 158), "Emit-only scores differ")
        else:
            require(all(row[name] == metrics[arm, CASE][name] for name in COUNTS), f"Case scores differ from TrackEval: {arm}")
    audit_switches(root, report)
    weak = read_table(root / (CASE_ROOT + "weak_observations.csv"))
    require(len(weak) == 543 and len({(row["arm"], row["frame"], row["track"]) for row in weak}) == len(weak),
            "Missing or duplicate weak rows")
    for row in weak:
        arm = row["arm"]
        require(arm in (W0, B_W0, BG_W0, EMIT) and 1 <= int(row["frame"]) <= 240
                and row["state_committed"] == ("False" if arm == EMIT else "True")
                and row["official_same_frame_switch"] == "False", f"Invalid weak output: {row}")
    tallies = {arm: dict(Counter(row["outcome"] for row in weak if row["arm"] == arm)) for arm in CASE_ARMS}
    require(tallies == report["weak_outcomes"] and all(sum(tallies[arm].values()) == int(paired[arm, CASE]["weak_commits"])
                                                       for arm in ARMS), "Weak case totals disagree")
    weak_b = {tuple(v for k, v in row.items() if k != "arm") for row in weak if row["arm"] == B_W0}
    weak_bg = {tuple(v for k, v in row.items() if k != "arm") for row in weak if row["arm"] == BG_W0}
    require(weak_b == weak_bg and len(weak_b) == 151, "B/BG weak observations differ")
    emit = read_json(root / "dji_0061/emit_only_receipt.json")
    require(emit["sequence"] == CASE and emit["canonical_equal_to_native_off"] is True
            and emit["counts"]["frames"] == 240 and emit["counts"]["weak_outputs_emitted"] == sum(tallies[EMIT].values()) == 117
            and emit["counts"]["weak_state_updates"] == 0 and emit["state_chain_sha256"] == native[OFF]["states"]
            and emit["published_mot_sha256"] != native[OFF]["mot"], "Emit-only state, publication or output receipt differs")
    digests = provenance["dji_0061_stream_sha256"]
    require(set(digests) == set(ARMS) and all(hex_sha(value) for streams in digests.values() for value in streams.values()),
            "Missing case SHA-256 stream receipts")
    for left, right in ((B_OFF, BG_OFF), (B_W0, BG_W0)):
        for field in ("mot.txt", "native_events.jsonl", "weak_events.jsonl", "states.sha256"):
            require(digests[left][field] == digests[right][field], f"B/BG parity differs: {left}/{field}")
    for arm in (OFF, W0):
        require(digests[arm]["mot.txt"] == native[arm]["mot"] and digests[arm]["states.sha256"] == native[arm]["states"]
                and digests[arm]["weak_events.jsonl"] == native[arm]["weak_events"], f"Native case stream receipts differ: {arm}")
    require(digests[OFF]["states.sha256"] == emit["state_chain_sha256"], "Native archived state hash differs")
    camera = read_json(root / (CASE_ROOT + "camera_evidence.json"))
    require(camera["sequence"] == CASE and camera["ref"] == R4_1 and camera["identical_selected_matrices"] is True
            and set(camera["arms"]) == {B_OFF, B_W0, BG_OFF, BG_W0}, "Wrong camera evidence scope or SOF parity")
    for arm, row in camera["arms"].items():
        require(row["frames"] == 240 and row["warp_choices"] == {"sof": 240}
                and all(row[name] == 0 for name in ("adaptive_triggers", "lightglue_calls", "dual_low_rejected_edges")),
                f"Unexpected adaptive or low-score camera intervention: {arm}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().archive_root.resolve()
    files = verify_manifest(root)
    provenance = read_json(root / "run_provenance.json")
    lengths = provenance["scope"]["sequence_lengths"]
    summary = read_json(root / "summary.json")
    require(lengths == summary["sequence_lengths"] and len(lengths) == 30 and sum(lengths.values()) == 12590
            and all(type(length) is int and length > 0 for length in lengths.values()) and lengths[CASE] == 240
            and summary["status"] == "VERIFIED" and tuple(summary["evaluated_arms"]) == ARMS, "Wrong 30-sequence scope")
    metrics, paired = verify_metrics(root, lengths, summary)
    native = verify_receipts(root, files, lengths, summary, paired)
    audit_case(root, provenance, native, metrics, paired)
    print(f"PASS: {len(files)} file hashes, six full arms, 180 paired rows, 95 switch keys, emit-only 158 IDSW/117 outputs/0 updates, B/BG parity")


if __name__ == "__main__":
    main()
