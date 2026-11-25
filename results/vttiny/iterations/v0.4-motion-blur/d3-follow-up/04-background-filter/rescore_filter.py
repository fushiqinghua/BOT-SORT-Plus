"""Apply causal masks only to marked writer additions and rescore frozen results."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY.parent))
from run_round import FULL, GT, PROJECT, REPO, read, write

BASE = "B_early_moderate"
RAW = "B_postlow_output_v1"
ADDED = "output_only_low"
PROBE = "post_B_weak_output_check"
KEY_METRICS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "FP", "FN", "IDSW")


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write_table(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_masks(path, required):
    masks = defaultdict(dict)
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            name = row["rule"]
            assert re.fullmatch("[A-Za-z0-9_]+", name), name
            assert name not in {BASE, RAW}
            key = row["sequence"], int(row["frame"]), int(row["track_id"])
            keep = row["keep"].strip().lower()
            assert keep in {"0", "1", "true", "false"}, (name, key)
            assert key not in masks[name], (name, key)
            masks[name][key] = keep in {"1", "true"}
    assert 1 <= len(masks) <= 4
    for name, mask in masks.items():
        assert set(mask) == required, (name, set(mask) ^ required)
    return masks


def prediction(root, tracker, sequence):
    return root / "trackers" / tracker / "data" / f"{sequence}.txt"


def trace(root, tracker, sequence):
    return root / "events" / tracker / f"{sequence}.jsonl"


def read_lines(path):
    return path.read_bytes().splitlines(keepends=True)


def row_key(line):
    return tuple(map(int, line.split(b",", 2)[:2]))


def additions(events):
    rows = [(int(event["frame"]), int(event["track_id"])) for _, event in events
            if event["event"] == ADDED]
    assert len(rows) == len(set(rows)), "Duplicate output-only key"
    return set(rows)


def without_filtered_events(events, rejected):
    for line, event in events:
        if event["event"] == ADDED and (int(event["frame"]), int(event["track_id"])) in rejected:
            continue
        yield line


def materialize(masks, full_run, evidence, destination):
    frozen = FULL / "observed"
    rules = {}
    for name in masks:
        root = destination / name / "filtered_run"
        (root / "protocols").mkdir(parents=True)
        (root / "trackers" / RAW / "data").mkdir(parents=True)
        (root / "events" / RAW).mkdir(parents=True)
        rules[name] = dict(root=root, kept=0, hashes={})
    proof = []
    for sequence in evidence["sequences"]:
        original_prediction = prediction(full_run, RAW, sequence)
        baseline_prediction = prediction(frozen, BASE, sequence)
        assert digest(original_prediction) == evidence["run_evidence"][RAW]["prediction_sha256"][sequence]
        assert digest(baseline_prediction) == evidence["run_evidence"][BASE]["prediction_sha256"][sequence]
        output = read_lines(original_prediction)
        original_trace = trace(full_run, RAW, sequence)
        events = [(line, json.loads(line)) for line in read_lines(original_trace)]
        extras = additions(events)
        native_mot = b"".join(line for line in output if row_key(line) not in extras)
        assert native_mot == baseline_prediction.read_bytes(), sequence
        native_trace = b"".join(line for line, event in events if event["event"] not in {ADDED, PROBE})
        assert native_trace == trace(frozen, BASE, sequence).read_bytes(), sequence
        for name, mask in masks.items():
            root = rules[name]["root"]
            rejected = {key for key in extras if not mask[(sequence, *key)]}
            selected = [line for line in output if row_key(line) not in rejected]
            assert len(output) - len(selected) == len(rejected), (name, sequence)
            retained = extras - rejected
            assert b"".join(line for line in selected if row_key(line) not in retained) == native_mot
            written = prediction(root, RAW, sequence)
            written.write_bytes(b"".join(selected))
            trace(root, RAW, sequence).write_bytes(b"".join(without_filtered_events(events, rejected)))
            rules[name]["kept"] += len(retained)
            rules[name]["hashes"][sequence] = digest(written)
            proof.append(dict(rule=name, sequence=sequence, proposed=len(extras), kept=len(retained),
                              rejected=len(rejected), native_bytes_identical=True))
    source_protocol = full_run / "protocols" / f"{RAW}.json"
    assert digest(source_protocol) == evidence["run_evidence"][RAW]["protocol_sha256"]
    for name, row in rules.items():
        protocol = read(source_protocol)
        protocol["output_prediction_hashes"] = row["hashes"]
        protocol["posthoc_writer_filter"] = dict(rule=name, replay="selected original marked outputs",
                                                 tracker_state_recomputed=False)
        write(row["root"] / "protocols" / f"{RAW}.json", protocol)
    write_table(destination / "line_integrity.csv", proof)
    return rules


def score(destination, source, rules, previous):
    from evaluate_vttiny import evaluate

    tracker_dir = destination / "scoring_input"
    tracker_dir.mkdir()
    choices = {
        BASE: FULL / "observed" / "trackers" / BASE,
        RAW: source / "trackers" / RAW,
        **{rule: meta["root"] / "trackers" / RAW for rule, meta in rules.items()},
    }
    for name, target in choices.items():
        (tracker_dir / name).symlink_to(target, target_is_directory=True)
    audited = read(previous / "full/trackeval/evaluation_audit.json")
    prior_summary = previous / "full/trackeval/summary.json"
    assert audited["status"] == "COMPLETE" and digest(prior_summary) == audited["summary_sha256"]
    frozen = read(FULL / "observed/protocols" / f"{BASE}.json")
    for seq in frozen["sequences"]:
        assert digest(GT / seq / "gt/gt.txt") == audited["ground_truth"]["sha256"][seq], seq
    previous_rows = {(row["tracker"], row["sequence"]): row for row in read(prior_summary)["results"]}
    summary = evaluate("rgb", "test", tracker_dir, list(choices), destination / "trackeval",
                       seqs=frozen["sequences"], gt_folder=GT)
    metrics = {(row["tracker"], row["sequence"]): row for row in summary["results"]}
    assert len(metrics) == len(choices) * 31
    for key, row in metrics.items():
        if key[0] in {BASE, RAW}:
            for metric in KEY_METRICS:
                assert math.isclose(row[metric], previous_rows[key][metric],
                                    abs_tol=1e-8, rel_tol=0), (key, metric)
    combined = []
    sequences = []
    for seq in ("COMBINED_SEQ", *frozen["sequences"]):
        for name in choices:
            row = dict(rule=name, sequence=seq, **{metric: metrics[name, seq][metric] for metric in KEY_METRICS})
            for reference, suffix in ((BASE, "baseline"), (RAW, "output_only")):
                for metric in KEY_METRICS:
                    row[f"delta_{metric}_vs_{suffix}"] = row[metric] - metrics[reference, seq][metric]
            (combined if seq == "COMBINED_SEQ" else sequences).append(row)
    write_table(destination / "combined_metrics.csv", combined)
    write_table(destination / "sequence_metrics.csv", sequences)
    print(json.dumps(combined, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mask", "source", "previous", "destination"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.destination.mkdir(exist_ok=False)
    evidence = read(args.previous / "full/same_input_verification.json")
    required = set()
    for sequence in evidence["sequences"]:
        path = trace(args.source, RAW, sequence)
        events = [(line, json.loads(line)) for line in read_lines(path)]
        required.update((sequence, *key) for key in additions(events))
    assert len(required) == 191 and len(evidence["sequences"]) == 30
    masks = load_masks(args.mask, required)
    rules = materialize(masks, args.source, evidence, args.destination)
    os.environ["TRACKEVAL_ROOT"] = str(PROJECT / "external/TrackEval")
    sys.path[:0] = [str(REPO / "tools"), str(REPO)]
    score(args.destination, args.source, rules, args.previous)
    print("FILTER_SCORE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
