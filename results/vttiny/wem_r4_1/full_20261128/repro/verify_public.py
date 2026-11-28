"""Verify this archive using only the published files and Python's standard library."""

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "FP", "FN", "IDSW")
COUNTS = ("FP", "FN", "IDSW")
ROLES = ("B", "BG")
MODES = ("off", "W0", "W1", "W2", "W3")
CASES = ("DJI_0073_3", "DJI_0327_3")
CASE_TABLES = "frame_audit/tables/"


def require(condition, context):
    if not condition:
        raise ValueError(context)


def load(relative):
    return json.loads((ROOT / relative).read_bytes())


def read_csv(relative):
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(not {"gt_xywh", "gt_box_xywh", "output_box_xywh"}.intersection(reader.fieldnames), relative)
        return list(reader)


def verify_files():
    manifest = load("manifest.json")
    require(manifest["schema"] == "wem_r41_public_archive_v1", "Archive schema")
    for name, info in manifest["files"].items():
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts and "\\" not in name and ":" not in name, name)
        data = (ROOT / path).read_bytes()
        require(len(data) == info["bytes"] and hashlib.sha256(data).hexdigest() == info["public_sha256"], name)
    return len(manifest["files"])


def keyed(rows, fields, keys):
    indexed = {tuple(r[f] for f in fields): r for r in rows}
    require(len(indexed) == len(rows) and set(indexed) == keys, f"Missing, duplicated or extra entries for {fields}")
    return indexed


def check_numbers(row, official, reference, absolute):
    for name in METRICS:
        value = official[name]
        if absolute:
            require(float(row[name]) == value, (row, name))
        require(math.isclose(float(row["delta_" + name]), value - reference[name], abs_tol=1e-9, rel_tol=0), (row, name))


def verify_metrics(lengths):
    arms = tuple(f"{role}_{mode}" for role in ROLES for mode in MODES)
    total = "COMBINED_SEQ"
    all_locations = {(a, s) for a in arms for s in (*lengths, total)}
    official = keyed(load("trackeval/summary.json")["results"], ("tracker", "sequence"), all_locations)
    comparisons = keyed(read_csv("report/comparison.csv"), ("arm",), {(a,) for a in arms})
    paired = keyed(read_csv("report/paired_per_sequence.csv"), ("arm", "sequence"),
                   {(a, s) for a in arms for s in lengths})
    for arm in arms:
        ref = arm.split("_")[0] + "_off"
        for seq in (*lengths, total):
            row = comparisons[arm,] if seq == total else paired[arm, seq]
            require(row["reference"] == ref, row)
            check_numbers(row, official[arm, seq], official[ref, seq], absolute=True)
        for name in COUNTS:
            require(sum(official[arm, seq][name] for seq in lengths) == official[arm, total][name], (arm, name))
    components = {f"{role}_{after}_minus_{role}_{before}": (f"{role}_{after}", f"{role}_{before}")
                  for role in ROLES for before, after in zip(MODES[1:], MODES[2:])}
    component_rows = keyed(read_csv("report/component_per_sequence.csv"), ("comparison", "sequence"),
                           {(c, s) for c in components for s in lengths})
    aggregate = keyed(read_csv("report/component_deltas.csv"), ("comparison",), {(c,) for c in components})
    for key, (arm, ref) in components.items():
        for seq in (*lengths, total):
            row = aggregate[key,] if seq == total else component_rows[key, seq]
            require((row["arm"], row["reference"]) == (arm, ref), row)
            check_numbers(row, official[arm, seq], official[ref, seq], absolute=False)
    return official


def verify_cases(lengths, official):
    frames = read_csv(CASE_TABLES + "frame_counts.csv")
    aligned = keyed(frames, ("sequence", "reference", "frame"),
                    {(s, r, str(f)) for s in CASES for r in ROLES for f in range(1, lengths[s] + 1)})
    fps = read_csv(CASE_TABLES + "fp_differences.csv")
    gt = read_csv(CASE_TABLES + "gt_differences.csv")
    for seq in CASES:
        for role in ROLES:
            rows = [aligned[seq, role, str(f)] for f in range(1, lengths[seq] + 1)]
            for mode in ("off", "W2", "W3"):
                for name in COUNTS:
                    require(sum(int(row[f"{mode}_{name}"]) for row in rows) == official[f"{role}_{mode}", seq][name],
                            (seq, role, mode, name))
            plus_minus = Counter(r["W3_minus_W2"] for r in fps if (r["sequence"], r["reference"]) == (seq, role))
            coverage = Counter(r["W3_vs_W2"] for r in gt if (r["sequence"], r["reference"]) == (seq, role))
            require(plus_minus["1"] - plus_minus["-1"] == sum(int(r["W3_minus_W2_FP"]) for r in rows), (seq, role, "FP"))
            require(coverage["lost"] - coverage["gained"] == sum(int(r["W3_minus_W2_FN"]) for r in rows), (seq, role, "FN"))
            if seq == "DJI_0073_3":
                require((plus_minus["1"], plus_minus["-1"]) == (67, 3), (seq, role, plus_minus))
            else:
                require((coverage["lost"], coverage["gained"]) == (16, 0), (seq, role, coverage))


def main():
    n = verify_files()
    lengths = load("run_provenance.json")["scope"]["sequence_lengths"]
    require(len(lengths) == 30 and sum(lengths.values()) == 12590, "Wrong frozen sequence scope")
    verify_cases(lengths, verify_metrics(lengths))
    print(f"PASS: {n} file hashes, 10 full arms, 300 paired rows, 180 component rows, 1,500 frame rows")


if __name__ == "__main__":
    main()
