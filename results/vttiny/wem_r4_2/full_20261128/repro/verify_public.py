"""Verify a published W R4.2 archive without access to the tracking assets."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath


OFF, W0, LOWER, RELAXED, EMIT = ARMS = (
    "BG_off", "BG_W0", "BG_lower_low", "BG_W0_relaxed_anchor", "BG_W0_emit_only",
)
CONTRASTS = {
    f"{arm}_minus_{reference}": (arm, reference, question)
    for arm, reference, question in (
        (W0, OFF, "complete_W0_effect"),
        (W0, LOWER, "constrained_vs_global_low_threshold"),
        (RELAXED, W0, "relaxed_anchor_support"),
        (W0, EMIT, "future_tracker_state_feedback"),
    )
}
METRICS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "FP", "FN", "IDSW")
COUNTS = ("FP", "FN", "IDSW")
WEAK_COUNTS = {OFF: (0, 0), W0: (730, 730), LOWER: (0, 0), RELAXED: (946, 946), EMIT: (701, 0)}
OUTCOMES = ("unmatched_FP", "TP_same_old_identity", "TP_other_identity", "TP_past_identity_unconfirmed")
CASES = {
    "lower_low_DJI_0339_2": ("DJI_0339_2", 47, 53, (W0, LOWER)),
    "relaxed_anchor_DJI_0339_2": ("DJI_0339_2", 95, 99, (W0, RELAXED)),
    "state_feedback_DJI_0327_3": ("DJI_0327_3", 227, 237, (OFF, W0, EMIT)),
}
REQUIRED = {
    "run_provenance.json", "trackeval/summary.json", "report/summary.json", "report/comparison.csv",
    "report/paired_per_sequence.csv", "report/component_deltas.csv", "report/component_per_sequence.csv",
    "report/attribution.csv", "attribution/committed_clear.csv", "attribution/per_sequence.csv",
    "attribution/changed_B_events.csv",
} | {f"cases/{case}.json" for case in CASES}
COMMIT = "eded63f9079734560966d3df59ba5c938c3e9ac4"


def require(condition, context):
    if not condition:
        raise ValueError(context)


def read_json(path):
    return json.loads(path.read_bytes())


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_files(root):
    manifest = read_json(root / "manifest.json")
    require(manifest["schema"] == "wem_r42_public_archive_v1", "Wrong archive schema")
    files = manifest["files"]
    require(REQUIRED <= set(files), f"Missing files from manifest: {REQUIRED - set(files)}")
    for name, receipt in files.items():
        relative = PurePosixPath(name)
        path = root / relative
        require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in name and ":" not in name
                and path.resolve().is_relative_to(root), f"Invalid archive path: {name}")
        data = path.read_bytes()
        require(len(data) == receipt["bytes"] and hashlib.sha256(data).hexdigest() == receipt["public_sha256"],
                f"Archive digest or length differs: {name}")
    return len(files)


def keyed(rows, fields, expected):
    index = {tuple(row[field] for field in fields): row for row in rows}
    require(len(index) == len(rows) and set(index) == expected, f"Wrong row scope or duplicate keys: {fields}")
    return index


def check_values(row, actual, baseline, context, *, absolute):
    for metric in METRICS:
        value, reference = actual[metric], baseline[metric]
        require(type(value) in (float, int) and math.isfinite(value) and (metric not in COUNTS or type(value) is int),
                f"Invalid official metric: {context}/{metric}")
        if absolute:
            require(float(row[metric]) == value, f"Official absolute metric differs: {context}/{metric}")
        require(math.isclose(float(row[f"delta_{metric}"]), value - reference, rel_tol=0, abs_tol=1e-9),
                f"Official metric difference does not match paired subtraction: {context}/{metric}")


def verify_metrics(root, lengths, report):
    official = read_json(root / "trackeval/summary.json")
    require(official["status"] == "PASS" and official["split"] == "test", "Incomplete official TrackEval results")
    pooled = "COMBINED_SEQ"
    metrics = keyed(official["results"], ("tracker", "sequence"),
                    {(arm, sequence) for arm in ARMS for sequence in (*lengths, pooled)})
    arm_keys = {(arm,) for arm in ARMS}
    comparisons = [keyed(rows, ("arm",), arm_keys) for rows in (
        read_rows(root / "report/comparison.csv"), report["comparison"],
    )]
    paired = keyed(read_rows(root / "report/paired_per_sequence.csv"), ("arm", "sequence"),
                   {(arm, sequence) for arm in ARMS for sequence in lengths})
    for arm in ARMS:
        for sequence in (*lengths, pooled):
            rows = [table[arm,] for table in comparisons] if sequence == pooled else [paired[arm, sequence]]
            for row in rows:
                require(row["reference"] == OFF, f"Incorrect baseline for {arm}/{sequence}")
                check_values(row, metrics[arm, sequence], metrics[OFF, sequence], f"{arm}/{sequence}", absolute=True)
        for name in COUNTS:
            require(sum(metrics[arm, sequence][name] for sequence in lengths) == metrics[arm, pooled][name],
                    f"Wrong COMBINED_SEQ count: {arm}/{name}")

    contrast_keys = {(name,) for name in CONTRASTS}
    comparisons = [keyed(rows, ("comparison",), contrast_keys) for rows in (
        read_rows(root / "report/component_deltas.csv"), report["component_deltas"],
    )]
    paired = keyed(read_rows(root / "report/component_per_sequence.csv"), ("comparison", "sequence"),
                   {(name, sequence) for name in CONTRASTS for sequence in lengths})
    for name, (arm, reference, question) in CONTRASTS.items():
        for sequence in (*lengths, pooled):
            rows = [table[name,] for table in comparisons] if sequence == pooled else [paired[name, sequence]]
            for row in rows:
                require((row["arm"], row["reference"], row["question"]) == (arm, reference, question),
                        f"Contrast direction differs: {name}/{sequence}")
                check_values(row, metrics[arm, sequence], metrics[reference, sequence],
                             f"{name}/{sequence}", absolute=False)
    return metrics


def verify_attribution(root, lengths, report, metrics):
    variant_keys = {(arm, sequence) for arm in ARMS if arm != OFF for sequence in lengths}
    paired = keyed(read_rows(root / "attribution/per_sequence.csv"), ("arm", "sequence"), variant_keys)
    aggregates = keyed(read_rows(root / "report/attribution.csv"), ("arm",), {(arm,) for arm in ARMS if arm != OFF})
    outputs, updates, outcomes, dispositions = Counter(), Counter(), Counter(), Counter()
    keys = set()
    for row in read_rows(root / "attribution/committed_clear.csv"):
        arm, sequence = pair = row["arm"], row["sequence"]
        key = pair + (row["frame"], row["track_id"], row["det_index"])
        require(key not in keys and arm in (W0, RELAXED, EMIT) and 1 <= int(row["frame"]) <= lengths[sequence],
                f"Invalid or duplicated weak output: {key}")
        keys.add(key)
        flag, kind = row["state_committed"], row["output_kind"]
        expected = ("0", "emitted_measurement_no_state_feedback") if arm == EMIT else ("1", "measurement")
        require((flag, kind) == expected and row["outcome"] in OUTCOMES and row["newly_covered"] in ("0", "1"),
                f"Invalid weak output semantics: {key}")
        outputs[pair] += 1
        updates[pair] += int(flag)
        outcomes[pair + (row["outcome"],)] += 1
        outcomes[pair + ("newly_covered",)] += int(row["newly_covered"])

    for row in read_rows(root / "attribution/changed_B_events.csv"):
        pair = row["arm"], row["sequence"]
        require(pair in variant_keys and pair[0] != EMIT and 1 <= int(row["frame"]) <= lengths[pair[1]],
                f"Unexpected changed B event: {pair}")
        dispositions[pair + (row["disposition"],)] += 1

    for arm in ARMS:
        activity = report["weak_activity"][arm]
        expected_outputs, expected_updates = WEAK_COUNTS[arm]
        require(activity["weak_outputs_emitted"] == sum(outputs[arm, seq] for seq in lengths) == expected_outputs
                and activity["weak_state_updates"] == sum(updates[arm, seq] for seq in lengths) == expected_updates,
                f"Weak output/state totals differ: {arm}")
        if arm == OFF:
            continue
        sums = Counter()
        for sequence in lengths:
            pair = arm, sequence
            row = {name: int(value) for name, value in paired[pair].items() if name not in ("arm", "sequence")}
            current, reference = metrics[pair], metrics[OFF, sequence]
            require(row["gained"] - row["lost"] == reference["FN"] - current["FN"]
                    and row["fp_delta"] == current["FP"] - reference["FP"]
                    and row["idsw_delta"] == current["IDSW"] - reference["IDSW"],
                    f"CLEAR coverage attribution differs: {pair}")
            for name in (*OUTCOMES, "newly_covered"):
                require(row[name] == outcomes[pair + (name,)], f"Weak-output outcome differs: {pair}/{name}")
            for name in ("reference_B_not_matched", "reference_B_coverage_lost", "same_id_covered_with_recent_W", "coverage_retained"):
                require(row[name] == dispositions[pair + (name,)], f"Changed B outcome differs: {pair}/{name}")
            sums.update(row)
        require(dict(sums) == report["attribution"][arm] and all(int(aggregates[arm,][key]) == value for key, value in sums.items())
                and int(aggregates[arm,]["weak_outputs_emitted"]) == expected_outputs
                and int(aggregates[arm,]["weak_state_updates"]) == expected_updates, f"Attribution totals differ: {arm}")
    require(all(report["checks"][name] == "PASS" for name in (
        "emit_only_native_events", "emit_only_canonical_native_state_chain", "weak_output_vs_state_counts",
    )), "Emit-only technical parity checks are incomplete")


def verify_cases(root, metrics):
    windows = {}
    for name, (sequence, start, end, arms) in CASES.items():
        row = read_json(root / "cases" / f"{name}.json")
        require((row["sequence"], row["start"], row["end"], set(row["arms"])) == (sequence, start, end, set(arms))
                and row["commit"] == COMMIT, f"Wrong case window or source commit: {name}")
        window = windows[name] = {}
        for arm, data in row["arms"].items():
            require(data["official_confirmed"] is True and all(
                data["sequence_counts"][metric] == metrics[arm, sequence][metric] for metric in COUNTS
            ), f"Case sequence counts differ from official metrics: {name}/{arm}")
            window[arm] = keyed(data["frames"], ("frame",), {(frame,) for frame in range(start, end + 1)})
        if name.startswith("state_feedback"):
            off, emit = row["arms"][OFF], row["arms"][EMIT]
            require(emit["weak_outputs_emitted"] == 165 and emit["weak_state_updates"] == 0
                    and all(emit[f"verified_{key}_sha256"] == off[f"verified_{key}_sha256"]
                            for key in ("state_chain", "native_event"))
                    and emit["verified_mot_sha256"] != off["verified_mot_sha256"], "GT16 case canonical/published evidence differs")

    def track(window, arm, frame, tid):
        return next(row for row in window[arm][frame,]["tracks"] if row["track_id"] == tid)

    lower = windows["lower_low_DJI_0339_2"]
    require(track(lower, W0, 48, 2)["det_index"] == track(lower, LOWER, 48, 2)["det_index"] == 7
            and track(lower, W0, 48, 2)["stage"] == "weak_measurement", "Lower-threshold frame 48 does not align")
    for frame in range(49, 54):
        value = track(lower, LOWER, frame, 2)
        require(track(lower, W0, frame, 2)["status"] == "no_output"
                and (value["status"], value["stage"], value["max_iou"]) == ("FP", "low", 0), f"Lower-threshold FP differs at {frame}")
    low_metrics, strict_metrics = metrics[LOWER, "DJI_0339_2"], metrics[W0, "DJI_0339_2"]
    require(low_metrics["FP"] - strict_metrics["FP"] == 335 and low_metrics["FN"] == strict_metrics["FN"],
            "Lower threshold case net errors differ")

    relaxed = windows["relaxed_anchor_DJI_0339_2"]
    weak = track(relaxed, RELAXED, 97, 2)
    require((weak["det_index"], weak["stage"], weak["status"]) == (4, "weak_measurement", "FP")
            and any(event["event"] == "committed" and event["anchor_not_ready"] is True
                    and event["previous_regular_observations"] == 1 for event in weak["weak_events"])
            and track(relaxed, W0, 97, 2)["status"] == "no_output"
            and track(relaxed, RELAXED, 98, 2)["stage"] == "low" and track(relaxed, W0, 99, 2)["stage"] == "B_recovery",
            "Relaxed-anchor case weak acceptance or subsequent source differs")
    relaxed_metrics = metrics[RELAXED, "DJI_0339_2"]
    require(relaxed_metrics["FP"] - strict_metrics["FP"] == 212 and relaxed_metrics["FN"] == strict_metrics["FN"],
            "Relaxed-anchor case net errors differ")

    feedback = windows["state_feedback_DJI_0327_3"]
    for arm, flag in ((W0, True), (EMIT, False)):
        weak = track(feedback, arm, 228, 55)
        require((weak["det_index"], weak["status"]) == (42, "FP") and any(
            event["event"] == "committed" and event["state_committed"] is flag for event in weak["weak_events"]
        ), f"GT16 first weak event disagrees with state semantics: {arm}")
    for frame in range(230, 237):
        actual = track(feedback, W0, frame, 55)
        require((actual["status"], actual["matched_gt"], actual["stage"]) == ("TP", 16, "low")
                and all(feedback[arm][frame,]["coverage"]["16"]["matched_track"] is None for arm in (OFF, EMIT)),
                f"GT16 coverage differs at frame {frame}")
    require(all(feedback[arm][237,]["coverage"]["16"]["matched_track"] == 55 for arm in (OFF, W0, EMIT)),
            "GT16 frame 237 native reacquisition differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().archive_root.resolve()
    count = verify_files(root)
    provenance = read_json(root / "run_provenance.json")
    lengths = provenance["scope"]["sequence_lengths"]
    report = read_json(root / "report/summary.json")
    require(len(lengths) == 30 and sum(lengths.values()) == report["frames"] == 12590
            and all(type(length) is int and length > 0 for length in lengths.values())
            and report["schema"] == "wem_r42_report" and report["status"] == "FULL_EVALUATION_COMPLETE"
            and report["sequence_count"] == 30 and set(report["sequences"]) == set(lengths) and tuple(report["arms"]) == ARMS,
            "Wrong frozen R4.2 experimental scope")
    hashed = provenance["reference_and_state_sha256"]
    require(set(hashed) == set(lengths) and provenance["source"]["github_commit"] == COMMIT
            and provenance["status"]["full"] == "FULL_BENCHMARK_COMPLETE", "Wrong run provenance or stream digest scope")
    for seq, arms in hashed.items():
        off, emitted = arms[OFF], arms[EMIT]
        for field in ("native_events", "states"):
            require(emitted[field] == off[field] and len(off[field]) == 64, f"Emit-only native {field} differs: {seq}")
    require(provenance["emit_only_weak_outputs"] == 701 and provenance["emit_only_weak_state_updates"] == 0,
            "Wrong published weak counts in run provenance")
    metrics = verify_metrics(root, lengths, report)
    verify_attribution(root, lengths, report, metrics)
    verify_cases(root, metrics)
    print(f"PASS: {count} file hashes, 5 complete arms, 150 paired/120 contrast rows, emit-only 701 outputs/0 updates, 3 frame-level cases")


if __name__ == "__main__":
    main()
