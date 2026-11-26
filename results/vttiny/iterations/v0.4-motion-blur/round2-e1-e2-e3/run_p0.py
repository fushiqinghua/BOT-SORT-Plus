"""Verify the frozen v0.4 run and extract the first follow-up cohort."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

TRACKER = "B_early_moderate"
HEAD = "c3e142e6c11df0f2a7e8a84bede8556c7dabec1e"
TRACK_EVAL = "12c8791b303e0a0b50f753af204249e622d0281a"
PINNED = ("name", "config", "native_config", "sequences", "subset", "detector_sha256",
          "rgb_encoder_sha256", "source_protocol_sha256", "detection_fingerprints",
          "encoder_contracts", "feature_hashes", "feature_coverage", "input_roots")
METRICS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "FP", "FN", "IDSW")


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def execute(repo, path, *args):
    command = [sys.executable, "-B", "-X", "utf8", "-u", str(repo / args[0]), *map(str, args[1:])]
    print("COMMAND", *command, flush=True)
    with path.open("x", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=repo, check=True)
    return path.read_text(encoding="utf-8")


def load_command_json(text):
    end = text.rfind("\n{")
    return json.loads(text[end+1:] if end >= 0 else text)


def decoded_hashes(settings, frozen):
    import cv2
    import numpy as np
    from botsort_plus.io import detection_files, read_detection_cache, resolve_image, sequence_name

    cv2.setNumThreads(1)
    baseline = Path(settings["baseline_root"])
    images = Path(settings["images_root"])
    found = {sequence_name(path): path for path in detection_files(baseline / "detections")}

    def one(sequence):
        cache = read_detection_cache(found[sequence])
        digest = hashlib.sha256()
        for row in cache["frames"]:
            image_path = resolve_image(row["image"], images)
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot decode {sequence} frame {row['frame']}")
            pixels = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
            payload = dict(frame=int(row["frame"]), shape=list(image.shape), dtype=str(image.dtype), sha256=pixels)
            digest.update((json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
        value = digest.hexdigest()
        if value != frozen["decoded_image_hashes"][sequence]:
            raise ValueError(f"Decoded image bytes differ in {sequence}")
        return sequence, value

    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for sequence, value in pool.map(one, frozen["sequences"]):
            results[sequence] = value
            print("PIXELS", len(results), "of", len(frozen["sequences"]), sequence, flush=True)
    return results


def run(suite, job):
    source = suite / "BOT-SORT-Plus"
    repo = suite / "BOT-SORT-Plus-followup-c3e142e6"
    full = source / "outputs/diagnostics_v0.4_full30_round1"
    project = suite.parents[1]
    prior_eval = suite / "round1/d3_output_only_comparison_v1/full/trackeval"
    p0 = job / "P0"
    p0.mkdir(exist_ok=False)
    assert subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() == HEAD
    sys.path.insert(0, str(repo))

    original_settings = source / "configs/diagnostics/motion_blur_v0.4.local.json"
    settings = read(original_settings)
    for key in ("baseline_root", "features_dir", "images_root", "gt_root", "runtime_root", "output_root"):
        path = Path(settings[key])
        settings[key] = str((source / path).resolve() if not path.is_absolute() else path.resolve())
    assert Path(settings["output_root"]) == full
    path = p0 / "motion_blur_v0.4.local.json"
    write(path, settings)
    run_tool = "tools/run_botsort_diagnostics.py"
    parity = load_command_json(execute(repo, p0 / "verify.log", run_tool, "verify", "--settings", path))
    historical_verification = read(full / "verification.json")
    assert parity["status"] == historical_verification["status"] == "PASS"
    historical_check = read(full / "round1_findings/historical_B_comparison.json")
    assert historical_check["status"] == "PASS"
    assert parity["sequences"] == historical_verification["sequences"]
    assert parity["reference_protocol_sha256"] == historical_verification["reference_protocol_sha256"]
    assert parity["observed_protocol_sha256"] == historical_verification["observed_protocol_sha256"]
    prediction_hashes = {seq: row["sha256"] for seq, row in parity["sequences"].items()}
    assert prediction_hashes == historical_check["prediction_sha256"]
    write(p0 / "frozen_parity.json", parity)

    frozen = read(full / "observed/protocols" / f"{TRACKER}.json")
    assert len(frozen["sequences"]) == 30 and sum(frozen["sequences"].values()) == 12590
    assert frozen["output_prediction_hashes"] == prediction_hashes
    source_analysis = read(full / "analysis/summary.json")
    assert source_analysis["status"] == "COMPLETE"
    assert source_analysis["run_protocol_sha256"] == parity["observed_protocol_sha256"]
    assert {seq: row["original_events_sha256"] for seq, row in parity["sequences"].items()} == source_analysis["original_events_sha256"]
    for name, expected in source_analysis["files_sha256"].items():
        assert sha(full / "analysis" / name) == expected, name

    preflight = load_command_json(execute(repo, p0 / "preflight.log", run_tool, "preflight", "--settings", path))
    probe = preflight["prediction_preflight"]
    assert preflight["status"] == probe["status"] == "READY"
    for key in PINNED:
        assert probe[key] == frozen[key], key
    code_changes = sorted({*probe["implementation_sha256"], *frozen["implementation_sha256"]} -
                          {name for name, digest in probe["implementation_sha256"].items()
                           if digest == frozen["implementation_sha256"].get(name)})
    write(p0 / "current_preflight.json", preflight)
    historical_eval = read(prior_eval / "evaluation_audit.json")
    truth = preflight["ground_truth_preflight"]["sha256"]
    assert truth == source_analysis["ground_truth"]["sha256"] == historical_eval["ground_truth"]["sha256"]
    from tools.evaluate_botsort_plus import verify_trackeval
    pinned = project / "external/TrackEval"
    evaluator = verify_trackeval(pinned, repo)
    assert evaluator["commit"] == historical_eval["evaluator"]["commit"] == TRACK_EVAL
    assert evaluator["source_files"] == historical_eval["evaluator"]["source_files"]
    write(p0 / "decoded_image_hashes.json", decoded_hashes(settings, frozen))

    establish = p0 / "trackeval"
    execute(repo, p0 / "trackeval.log", "tools/evaluate_botsort_plus.py", "--run-dir", full / "observed",
            "--trackers", TRACKER, "--gt-root", settings["gt_root"], "--trackeval-root", pinned,
            "--reference-tracker", TRACKER, "--output-dir", establish)
    audit = read(establish / "evaluation_audit.json")
    report = establish / "summary.json"
    historical_report = prior_eval / "summary.json"
    assert sha(report) == audit["summary_sha256"] and sha(historical_report) == historical_eval["summary_sha256"]
    previous_rows = {row["sequence"]: row for row in read(historical_report)["results"] if row["tracker"] == TRACKER}
    current_rows = {row["sequence"]: row for row in read(report)["results"]}
    assert current_rows.keys() == previous_rows.keys() and len(current_rows) == 31
    for sequence, row in current_rows.items():
        for metric in METRICS:
            assert math.isclose(row[metric], previous_rows[sequence][metric], rel_tol=0, abs_tol=1e-8), (sequence, metric)

    proof = dict(status="PASS", checkout_commit=HEAD, frozen_run=str(full), settings=str(path),
                 settings_sha256=sha(path), sequences=frozen["sequences"], frames=12590,
                 observed_protocol_sha256=parity["observed_protocol_sha256"], source_analysis_sha256=sha(full/"analysis/summary.json"),
                 detector_sha256=probe["detector_sha256"], rgb_encoder_sha256=probe["rgb_encoder_sha256"],
                 frozen_gt_sha256=truth, trackeval_commit=evaluator["commit"], trackeval_verified_files=evaluator["verified_source_files"],
                 current_implementation_changes_against_frozen=code_changes,
                 metrics=current_rows["COMBINED_SEQ"], frozen_parity_sha256=sha(p0/"frozen_parity.json"),
                 decoded_pixels_sha256=sha(p0/"decoded_image_hashes.json"))
    write(p0 / "verification.json", proof)
    print("P0_PASS", json.dumps(proof["metrics"]), flush=True)
    cohort = job / "E1/cohort"
    execute(repo, job / "E1_A.log", "tools/prepare_v04_lost_cohort.py", "--run-root", full, "--output", cohort)
    print("E1_A_COMPLETE", json.dumps(read(cohort/"cohort_summary.json")["missing_with_usable_detection"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    try:
        run(args.suite, args.job)
    except Exception:
        write(args.job / "first_failure.json", dict(status="FAIL", traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
