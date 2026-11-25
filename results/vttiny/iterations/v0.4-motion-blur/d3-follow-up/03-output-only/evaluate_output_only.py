"""Three-way evaluation of the stateless continuation ablation."""
from pathlib import Path
import argparse
import csv
import hashlib
import math
import os
import subprocess
import sys

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY.parent))
from run_round import FULL, GT, PROJECT, REPO, env, read, write

BASE = "B_early_moderate"
LOW = "B_postlow_v1"
OUTPUT_ONLY = "B_postlow_output_v1"
SMOKE_SEQUENCES = ("DJI_0339_2", "DJI_0327_3")
METRICS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "FP", "FN", "IDSW")
SCOPE_FIELDS = ("sequences", "detection_fingerprints", "feature_hashes", "decoded_image_hashes")
SHARED_FIELDS = (
    "native_config", "detector_sha256", "rgb_encoder_sha256", "source_protocol_sha256",
    "encoder_contracts", "boxmot_source_hashes", "runtime", "input_roots",
)


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def assemble(entries, directory, scope):
    assembly = directory / "input"
    for path in ("protocols", "trackers"):
        (assembly / path).mkdir(parents=True, exist_ok=False)
    base = read(FULL / "observed/protocols" / (BASE + ".json"))
    sequence_names = tuple(base["sequences"]) if scope == "full" else SMOKE_SEQUENCES
    assert len(base["sequences"]) == 30 and sum(base["sequences"].values()) == 12590
    evidence = {}
    for name, root in entries.items():
        path = root / "protocols" / (name + ".json")
        protocol = read(path)
        assert protocol["status"] == "COMPLETE" and protocol["causal"], name
        assert not protocol["interpolation"] and not protocol["backfilled_frames"], name
        is_subset = scope == "smoke" and name == OUTPUT_ONLY
        assert protocol["subset"] == is_subset, name
        for key in SHARED_FIELDS:
            assert protocol[key] == base[key], (name, key)
        allowed = sequence_names if is_subset else tuple(base["sequences"])
        for key in SCOPE_FIELDS:
            assert protocol[key] == {seq: base[key][seq] for seq in allowed}, (name, key)
        assert protocol["config"]["name"] == name
        assert dict(protocol["config"], name=BASE) == base["config"], name
        (assembly / "protocols" / path.name).symlink_to(path)
        tracker = root / "trackers" / name / "data"
        target = assembly / "trackers" / name / "data"
        target.mkdir(parents=True, exist_ok=False)
        prediction_hashes = {}
        for seq in sequence_names:
            prediction = tracker / (seq + ".txt")
            sha256 = digest(prediction)
            assert protocol["output_prediction_hashes"][seq] == sha256, (name, seq)
            (target / prediction.name).symlink_to(prediction)
            assert (root / "events" / name / (seq + ".jsonl")).is_file(), (name, seq)
            prediction_hashes[seq] = sha256
        evidence[name] = dict(root=str(root), protocol_sha256=digest(path),
                              prediction_sha256=prediction_hashes)
    write(directory / "same_input_verification.json",
          dict(sequences=sequence_names, run_evidence=evidence,
               shared_fields=SHARED_FIELDS, per_sequence_fields=SCOPE_FIELDS))
    return assembly, {seq: base["sequences"][seq] for seq in sequence_names}


def verify_previous(rows, previous, include_combined):
    old = {(row["tracker"], row["sequence"]): row for row in previous["results"]}
    for row in rows:
        tracker, seq = row["tracker"], row["sequence"]
        if tracker == OUTPUT_ONLY or (seq == "COMBINED_SEQ" and not include_combined):
            continue
        ref = old[tracker, seq]
        for key in METRICS:
            assert math.isclose(row[key], ref[key], rel_tol=0, abs_tol=1e-8), (tracker, seq, key)


def table(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_metrics(directory, rows, sequences):
    entries = {(row["tracker"], row["sequence"]): row for row in rows}
    assert len(entries) == 3 * (len(sequences) + 1)
    combined = []
    per_sequence = []
    for sequence in ("COMBINED_SEQ", *sequences):
        for method in (BASE, LOW, OUTPUT_ONLY):
            raw = entries[method, sequence]
            row = dict(sequence=sequence, method=method, **{key: raw[key] for key in METRICS})
            for reference, suffix in ((BASE, "base"), (LOW, "stateful")):
                for key in METRICS:
                    row[f"delta_{key}_vs_{suffix}"] = raw[key] - entries[reference, sequence][key]
            (combined if sequence == "COMBINED_SEQ" else per_sequence).append(row)
    table(directory / "combined_metrics.csv", combined)
    table(directory / "sequence_metrics.csv", per_sequence)
    for row in (per_sequence if len(sequences) == 2 else combined):
        print(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", choices=("smoke", "full"))
    parser.add_argument("stateful_root", type=Path)
    parser.add_argument("stateless_root", type=Path)
    parser.add_argument("prior_eval", type=Path)
    args = parser.parse_args()
    directory = STUDY / args.scope
    directory.mkdir(exist_ok=False)
    entries = {BASE: FULL / "observed", LOW: args.stateful_root, OUTPUT_ONLY: args.stateless_root}
    assembly, sequences = assemble(entries, directory, args.scope)
    captures = read(args.prior_eval / "evaluation_audit.json")
    assert captures["status"] == "COMPLETE"
    previous_file = args.prior_eval / "summary.json"
    assert digest(previous_file) == captures["summary_sha256"]
    for sequence in sequences:
        assert digest(GT / sequence / "gt/gt.txt") == captures["ground_truth"]["sha256"][sequence], sequence

    pinned = PROJECT / "external/TrackEval"
    result_path = directory / "trackeval"
    if args.scope == "full":
        run_environment = env()
        run_environment["TRACKEVAL_ROOT"] = str(pinned)
        subprocess.run(
            [sys.executable, "-B", "-u", str(REPO / "tools/evaluate_botsort_plus.py"),
             "--run-dir", str(assembly), "--trackers", *entries, "--gt-root", str(GT),
             "--trackeval-root", str(pinned), "--reference-tracker", BASE, "--output-dir", str(result_path)],
            env=run_environment, cwd=REPO, check=True,
        )
        audited = read(result_path / "evaluation_audit.json")
        assert audited["status"] == "COMPLETE"
        assert audited["evaluator"]["commit"] == captures["evaluator"]["commit"]
        assert audited["summary_sha256"] == digest(result_path / "summary.json")
        official = read(result_path / "summary.json")
    else:
        os.environ["TRACKEVAL_ROOT"] = str(pinned)
        sys.path[:0] = [str(REPO / "tools"), str(REPO)]
        from evaluate_vttiny import evaluate
        official = evaluate("rgb", "test", assembly / "trackers", list(entries),
                            result_path, seqs=sequences, gt_folder=GT)
    verify_previous(official["results"], read(previous_file), include_combined=args.scope == "full")
    write_metrics(directory, official["results"], sequences)
    print("EVALUATION_COMPLETE", args.scope)


if __name__ == "__main__":
    main()
