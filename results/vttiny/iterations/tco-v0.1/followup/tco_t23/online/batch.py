"""Run the TCO training capture, no-intervention gates and paired T3a metrics."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
import csv
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time

from replay import METHODS, sha256

ROLES = dict(zip(("B", "BG"), METHODS, strict=True))
HERE = Path(__file__).resolve().parent
METRICS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "FP", "FN", "IDSW")
PARITY_MODES = ("off", "record", "fallback")


def read(path):
    return json.loads(Path(path).read_bytes())


def t0_settings(root):
    return read(root / "prediction_settings.json")


def runtime_sources(args):
    return {name: sha256(path) for name, path in (
        ("replay.py", HERE / "replay.py"), ("collector.py", args.collector),
        ("model.py", HERE.parent / "model.py"))}


def selected_lengths(split, part):
    if split["schema"] != "tco_v0.1_training_split" or split["status"] != "LOCKED":
        raise ValueError("Use the locked TCO training partition")
    groups = split["training_sequence_to_source_video"]
    development = split["development_sequence_to_source_video"]
    if set(groups.values()) & set(development.values()):
        raise ValueError("The training plan shares source groups with the official scope")
    partitions = [split["split"][name]["sequences"] for name in ("fit", "calibration", "holdout")]
    if set(groups) != set().union(*partitions) or sum(map(len, partitions)) != len(groups):
        raise ValueError("Invalid train/calibration/holdout membership")
    owners = [set(groups[s] for s in part) for part in partitions]
    if sum(map(len, owners)) != len(set().union(*owners)):
        raise ValueError("A source group occurs in several learning subsets")
    if part == "all":
        return {seq: count for partition in partitions for seq, count in partition.items()}
    return split["split"][part]["sequences"]


def verify_cache(root, train, features, lengths, *, require_ready=True):
    identity = read(root / "T0/input_manifest.json")
    protocol = read(train / "protocol.json")
    expected = (identity["detector_sha256"], identity["rgb_encoder_sha256"])
    if (protocol["weight_sha256"], protocol["reid_sha256"]) != expected:
        raise ValueError("The training cache was not extracted with the frozen detector and RGB ReID")
    if any(len(protocol["sequences"].get(s, [])) != lengths[s] for s in lengths):
        raise ValueError("The frozen training detector frame map differs from the locked source split")
    unavailable = [s for s in lengths if not sequence_ready(train, features, s)]
    if require_ready and unavailable:
        raise ValueError(f"Frozen caches incomplete for {len(unavailable)} selected sequences: {unavailable[:8]}")
    return identity


def sequence_ready(baseline, features, sequence):
    return ((baseline / "detections" / (sequence + ".json.gz")).is_file()
            and (features / (sequence + ".npz")).is_file()
            and (features / (sequence + ".json")).is_file())


def train_prediction(args, lengths, *, require_ready=True):
    verify_cache(args.t0, args.train_baseline, args.train_features, lengths, require_ready=require_ready)
    settings = t0_settings(args.t0)
    settings.update(baseline_root=str(args.train_baseline.resolve()),
                    features_dir=str(args.train_features.resolve()),
                    images_root=str(args.images_root.resolve()),
                    gmc_mode="live")
    settings.pop("warp_sources", None)
    return settings


def new_output(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    return output


def start_job(args, root, settings, method, destination, mode, names, groups, selection=None):
    from experiments.vttiny_gmc.round3_joint.common import child_environment

    log = root / "logs" / (destination.relative_to(root).as_posix().replace("/", "_") + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    child = child_environment(settings)
    child["YOLO_CONFIG_DIR"] = str(root / "runtime_config")
    child["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    command = [sys.executable, "-Bu", str(HERE / "replay.py"), "--repo", str(args.repo),
               "--collector", str(args.collector), "--settings", str(root / "prediction_settings.json"),
               "--method", method, "--output-dir", str(destination), "--mode", mode,
               "--device", args.device, "--groups", str(groups), "--sequences", *names]
    if selection is not None:
        command.extend(("--selection", str(selection)))
    with log.open("xb") as stream:
        status = subprocess.run(command, env=child, cwd=args.repo, stdout=stream, stderr=subprocess.STDOUT)
    if status.returncode:
        raise RuntimeError(f"{method}/{mode} failed; see {log}")
    print(f"Complete {method}/{mode}", flush=True)


def settings_files(output, settings, groups):
    from experiments.vttiny_gmc.round3_joint.common import write_json

    write_json(output / "prediction_settings.json", settings)
    path = output / "causal_grouping.json"
    write_json(path, groups)
    return path


def launch_matrix(args, output, settings, matrix, names, groups, calibrator=None):
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(start_job, args, output, settings, method, output / destination, mode,
                               list(names), groups, calibrator if mode not in ("off", "capture") else None)
                   for method, destination, mode in matrix]
        for future in futures:
            future.result()


def collect_shards(args, output, settings, lengths, groups):
    """Dispatch disjoint native CPU replays as immutable feature sidecars arrive."""
    pool = ThreadPoolExecutor(max_workers=args.jobs)
    running, finished, unstarted = {}, {}, set(lengths)
    launch_time, number = time.monotonic(), 0
    try:
        while unstarted or running:
            available = [seq for seq in sorted(unstarted, key=lambda seq: (-lengths[seq], seq))
                         if sequence_ready(args.train_baseline, args.train_features, seq)]
            while available and len(running) < args.jobs and (len(available) >= 3 or len(available) == len(unstarted)):
                names = available[:8]
                del available[:8]
                number += 1
                folder = output / "shards" / f"B_{number:02}"
                job = pool.submit(start_job, args, output, settings, METHODS[0], folder, "capture", names, groups)
                running[job] = (folder, names)
                unstarted.difference_update(names)
                print(f"Capture shard {number}: {len(names)} clips, {sum(lengths[s] for s in names)} frames", flush=True)
            if running:
                done, _ = wait(running, timeout=4., return_when=FIRST_COMPLETED)
                for result in done:
                    result.result()
                    folder, names = running.pop(result)
                    finished[folder] = names
            elif unstarted:
                time.sleep(4.)
            if unstarted and time.monotonic() - launch_time > args.cache_wait_seconds:
                raise TimeoutError(f"Input wait elapsed with {len(unstarted)} training sequences unstarted")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return finished


def merge_capture(output, lengths, shards):
    """Combine disjoint completed shards without changing any native trace row."""
    from experiments.vttiny_gmc.round3_joint.common import write_json
    from tools.evaluate_botsort_plus import validate_runs

    target = output / "B"
    name = METHODS[0]
    stable = ("native_config", "detector_sha256", "rgb_encoder_sha256", "source_protocol_sha256",
              "encoder_contracts", "implementation_sha256", "boxmot_source_hashes", "config", "input_roots")
    mappings = ("sequences", "detection_fingerprints", "feature_hashes",
                "decoded_image_hashes", "output_prediction_hashes")
    nested_mappings = ("camera_manifests", "decision_manifests")
    feature_coverage = Counter()
    merged, per_shard = None, {}
    for folder, seqs in sorted(shards.items()):
        source = validate_runs(folder, [name])["protocols"][name]
        if source["sequences"] != {seq: lengths[seq] for seq in seqs}:
            raise ValueError("Capture shard cannot be joined to its frozen sequence assignment")
        if merged is None:
            merged = copy.deepcopy(source)
            merged.pop("validated_protocol_sha256", None)
            merged["subset"] = True
            for key in mappings:
                merged[key] = {}
            for key in nested_mappings:
                merged["joint_batch"][key] = {}
        elif any(source[key] != merged[key] for key in stable):
            raise ValueError("Capture shards do not share the pinned cache and tracker implementation")
        if set(source["sequences"]) & merged["sequences"].keys():
            raise ValueError("A training sequence occurs in multiple capture shards")
        for key in mappings:
            merged[key].update(source[key])
        for key in nested_mappings:
            merged["joint_batch"][key].update(source["joint_batch"][key])
        feature_coverage.update(source["feature_coverage"])
        per_shard[folder.relative_to(output).as_posix()] = {
            "sequences": source["sequences"], "source_protocol_sha256": source["validated_protocol_sha256"],
        }
        for view in ("trackers", "events", "decisions", "tco_capture", "tco_states", "warps", "actions"):
            shutil.copytree(folder / view, target / view, dirs_exist_ok=True)
    if merged is None or merged["sequences"] != lengths:
        raise ValueError("Capture shard union does not cover the locked training clips")
    merged["feature_coverage"] = feature_coverage
    merged["capture_merge"] = {"method": "Disjoint physical sequence files; no box nor fingerprint edits",
                               "shards": per_shard}
    write_json(target / "protocols" / f"{name}.json", merged)
    validate_runs(target, [name])


def capture(args):
    from experiments.vttiny_gmc.round3_joint.common import write_json

    split = read(args.split)
    lengths = selected_lengths(split, "all")
    settings = train_prediction(args, lengths, require_ready=not args.streaming_capture)
    output = new_output(args.output)
    groups = settings_files(output, settings, split["training_sequence_to_source_video"])
    if args.streaming_capture:
        shards = collect_shards(args, output, settings, lengths, groups)
        merge_capture(output, lengths, shards)
    else:
        launch_matrix(args, output, settings, [(METHODS[0], "B", "capture")], sorted(lengths), groups)
    protocol = output / "B" / "protocols" / f"{METHODS[0]}.json"
    if read(protocol)["sequences"] != lengths:
        raise ValueError("Train trace sequence contract differs from the locked split")
    write_json(output / "capture_manifest.json", {
        "status": "PASS", "split_sha256": sha256(args.split), "protocol_sha256": sha256(protocol),
        "frames": sum(lengths.values()), "sequences": lengths, "labels_in_prediction": False,
    })


def parity(args):
    from experiments.vttiny_gmc.round3_joint.common import assert_same_inputs, write_json
    from tools.evaluate_botsort_plus import validate_runs

    original = read(args.t0 / "T0" / "smoke_parity.json")
    if original["status"] != "PASS" or len(original["sequences"]) != 6:
        raise ValueError("Require the six original verified smoke sequences")
    original_inputs = read(args.t0 / "offline_settings.json")["development_source_groups"]
    settings = t0_settings(args.t0)
    output = new_output(args.output)
    groups = settings_files(output, settings, original_inputs)
    matrix = [(method, Path(variant) / role, variant)
              for role, method in ROLES.items() for variant in PARITY_MODES]
    launch_matrix(args, output, settings, matrix, original["sequences"], groups, args.selection)
    results = {}
    for role, method in ROLES.items():
        source = args.t0 / "smoke" / "off" / role
        reference = validate_runs(source, [method])["protocols"][method]
        results[role] = {}
        for variant in PARITY_MODES:
            here = output / variant / role
            got = validate_runs(here, [method])["protocols"][method]
            assert_same_inputs(reference, got, original["sequences"])
            stats = {}
            for sequence in original["sequences"]:
                expected = original["references"][role]["files"][sequence]
                checks = {"MOT": got["output_prediction_hashes"][sequence],
                          "events": sha256(here / "events" / method / f"{sequence}.jsonl"),
                          "state_chain": sha256(here / "tco_states" / method / f"{sequence}.sha256")}
                if any(expected[k] != digest for k, digest in checks.items()):
                    raise ValueError(f"{method}/{variant}/{sequence}: changed a frozen output without intervention")
                stats[sequence] = checks
            decision = got.get("tco_fixed_pair", {}).get("sequences", {})
            if variant != "off":
                if set(decision) != set(original["sequences"]):
                    raise ValueError("Missing shadow trace metadata")
                if not sum(x["counts"].get("evaluated", 0) for x in decision.values()):
                    raise ValueError("Shadow replay bypassed all model inference")
                if any(x["counts"].get("applied") for x in decision.values()):
                    raise ValueError("A shadow run updated a measurement")
            results[role][variant] = {"sequences": stats, "audit": decision}
    write_json(output / "TCO_runtime_parity.json", {
        "status": "PASS", "selection_sha256": sha256(args.selection), "roles": results,
        "t0_smoke_sha256": sha256(args.t0 / "T0" / "smoke_parity.json"),
        "prediction_settings_sha256": sha256(output / "prediction_settings.json"),
        "runtime_sources_sha256": runtime_sources(args),
    })


def project_predictions(entry, destination, alias, lengths):
    from experiments.vttiny_gmc.round3_joint.common import write_json
    from tools.evaluate_botsort_plus import validate_runs

    root, method = entry
    source = root / "protocols" / (method + ".json")
    protocol = validate_runs(root, [method])["protocols"][method]
    if protocol["sequences"] != lengths:
        raise ValueError("The prediction files do not cover the exact evaluation scope")
    clone = copy.deepcopy(protocol)
    clone.pop("validated_protocol_sha256", None)
    clone["evaluation_view"] = {"source": str(source), "source_sha256": sha256(source),
                                "real_tracker_name": method, "box_edits": False}
    clone["config"]["name"] = clone["name"] = alias
    # The generic production validator requires exactly 30 clips when subset is false.
    clone["subset"] = len(lengths) != 30
    root_dest = destination / "trackers" / alias / "data"
    root_dest.mkdir(parents=True)
    for sequence in lengths:
        shutil.copyfile(root / "trackers" / method / "data" / f"{sequence}.txt", root_dest / f"{sequence}.txt")
    write_json(destination / "protocols" / (alias + ".json"), clone)
    return protocol


def evaluate(args, output, names, runs, lengths, scope):
    from experiments.vttiny_gmc.round3_joint.common import assert_same_inputs, child_environment, write_json
    from tools.evaluate_botsort_plus import validate_gt, validate_runs

    evaluation_input = output / "evaluation_input"
    table = {alias: project_predictions(entry, evaluation_input, alias, lengths)
             for alias, entry in runs.items()}
    for method in names:
        off, on = table[method], table[f"{method}_tco"]
        assert_same_inputs(off, on, lengths)
        if off["native_config"] != on["native_config"] or off["config"] != on["config"]:
            raise ValueError("The paired references differ in their tracker configuration")
        if on["tco_fixed_pair"]["selection_sha256"] != sha256(args.selection):
            raise ValueError("Paired replay differs from the frozen selection")
    validate_runs(evaluation_input, list(runs))
    gt_hashes = validate_gt(args.gt, lengths)["sha256"]
    if scope == "official_30_development" and gt_hashes != read(
            args.t0 / "T0" / "input_manifest.json")["ground_truth"]["sha256"]:
        raise ValueError("Official RGB ground truth bytes differ from T0")
    argv = [sys.executable, str(args.repo / "tools/evaluate_botsort_plus.py"), "--run-dir", str(evaluation_input),
            "--trackers", *runs, "--reference-tracker", names[0], "--gt-root", str(args.gt),
            "--trackeval-root", str(args.trackeval), "--output-dir", str(output / "metrics")]
    with (output / "evaluation.log").open("xb") as log:
        subprocess.run(argv, cwd=args.repo, env=child_environment(t0_settings(args.t0)),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = read(output / "metrics" / "summary.json")
    combined = {row["tracker"]: {k: row[k] for k in METRICS} for row in summary["results"]
                if row["sequence"] == "COMBINED_SEQ"}
    changes = {method: {k: combined[f"{method}_tco"][k] - combined[method][k] for k in METRICS}
               for method in names}
    from offline_effects import write_changes
    write_changes(output, args.gt, args.trackeval, lengths, runs, names, summary)
    manifest = {
        "status": "PASS", "scope": scope, "selection_sha256": sha256(args.selection), "sequences": lengths,
        "training_split_sha256": sha256(args.split) if args.split else None,
        "runtime_sources_sha256": runtime_sources(args),
        "pooled_metrics": combined, "TCO_minus_reference": changes,
        "measurements": {m: table[f"{m}_tco"]["tco_fixed_pair"]["sequences"] for m in names},
        "ground_truth_sha256": gt_hashes, "evaluator_summary_sha256": sha256(output / "metrics" / "summary.json"),
    }
    write_json(output / "comparison.json", manifest)
    with (output / "comparison.csv").open("x", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=("reference", "mode", *METRICS))
        writer.writeheader()
        for method in names:
            for mode, values in (("off", combined[method]), ("on", combined[f"{method}_tco"]),
                                 ("on_minus_off", changes[method])):
                writer.writerow({"reference": method, "mode": mode, **values})


def verify_parity(args):
    checks = read(args.parity)
    if (checks["status"] != "PASS" or checks["selection_sha256"] != sha256(args.selection)
            or set(checks["roles"]) != set(ROLES)
            or checks["runtime_sources_sha256"] != runtime_sources(args)
            or checks["t0_smoke_sha256"] != sha256(args.t0 / "T0" / "smoke_parity.json")
            or sha256(args.parity.parent / "prediction_settings.json") != checks["prediction_settings_sha256"]
            or read(args.parity.parent / "prediction_settings.json") != t0_settings(args.t0)):
        raise ValueError("A frozen-mode parity proof is required before live measurements")


def holdout(args):
    verify_parity(args)
    split = read(args.split)
    lengths = selected_lengths(split, "holdout")
    settings = train_prediction(args, lengths)
    output = new_output(args.output)
    groups = settings_files(output, settings, split["training_sequence_to_source_video"])
    names = [METHODS[0]] if not args.include_bg else list(METHODS)
    matrix = [(m, Path("replay") / mode / role, mode)
              for role, m in ROLES.items() if m in names for mode in ("off", "active")]
    launch_matrix(args, output, settings, matrix, sorted(lengths), groups, args.selection)
    runs = {name + suffix: (output / destination, name) for name, destination, mode in matrix
            for suffix in ["" if mode == "off" else "_tco"]}
    evaluate(args, output, names, runs, lengths, "official_training_holdout")


def official(args):
    verify_parity(args)
    holdout_check = read(args.holdout)
    if (holdout_check["status"] != "PASS" or holdout_check["scope"] != "official_training_holdout"
            or holdout_check["selection_sha256"] != sha256(args.selection)
            or holdout_check["training_split_sha256"] != sha256(args.split)
            or holdout_check["sequences"] != selected_lengths(read(args.split), "holdout")
            or holdout_check["runtime_sources_sha256"] != runtime_sources(args)):
        raise ValueError("Require the training holdout replay report for this exact checkpoint")
    gate = read(args.t2_gate)
    if gate.get("allow_t3a") is not True or gate.get("selection_sha256") != sha256(args.selection):
        raise ValueError("T2 must approve the selected work point using training data alone")
    complete = read(args.t0 / "T0" / "T0_parity.json")
    lengths = read(args.t0 / "T0" / "input_manifest.json")["sequences"]
    if complete["status"] != "PASS" or len(lengths) != 30 or sum(lengths.values()) != 12590:
        raise ValueError("Require the fixed official 30-sequence T0 replay")
    original_inputs = read(args.t0 / "offline_settings.json")["development_source_groups"]
    settings = t0_settings(args.t0)
    if settings.get("gmc_mode") != "cached":
        raise ValueError("Official run must reuse the pinned GMC caches as well")
    output = new_output(args.output)
    groups = settings_files(output, settings, original_inputs)
    matrix = [(method, Path("active") / role, "active") for role, method in ROLES.items()]
    launch_matrix(args, output, settings, matrix, sorted(lengths), groups, args.selection)
    runs = {m: (args.t0 / "full" / "off" / role, m) for role, m in ROLES.items()}
    runs.update({m + "_tco": (output / destination, m) for m, destination, _ in matrix})
    for role, m in ROLES.items():
        p = read(runs[m][0] / "protocols" / (m + ".json"))
        expected = complete["full"]["references"][role]["files"]
        if p["output_prediction_hashes"] != {s: expected[s]["MOT"] for s in lengths}:
            raise ValueError(f"{m} archived off reference changed after T0")
    evaluate(args, output, list(METHODS), runs, lengths, "official_30_development")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("capture", "parity", "holdout", "official"))
    for name in ("repo", "collector", "t0", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("train-baseline", "train-features", "images-root", "split", "selection", "gt",
                 "trackeval", "parity", "holdout", "t2-gate"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cuda-devices", default="")
    parser.add_argument("--include-bg", action="store_true")
    parser.add_argument("--streaming-capture", action="store_true",
                        help="Capture ready training clips in small CPU shards while feature extraction finishes")
    parser.add_argument("--cache-wait-seconds", type=int, default=7200)
    args = parser.parse_args()
    if args.jobs < 1 or args.cache_wait_seconds < 1:
        parser.error("--jobs and --cache-wait-seconds must be positive")
    prerequisites = {
        "capture": ("train_baseline", "train_features", "images_root", "split"),
        "parity": ("selection",),
        "holdout": ("train_baseline", "train_features", "images_root", "split", "selection", "gt", "trackeval", "parity"),
        "official": ("selection", "split", "gt", "trackeval", "parity", "holdout", "t2_gate"),
    }
    for key in prerequisites[args.stage]:
        if getattr(args, key) is None:
            parser.error(f"--{key.replace('_', '-')} is required for {args.stage}")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    sys.path.insert(0, str(args.repo))
    globals()[args.stage](args)


if __name__ == "__main__":
    main()
