"""Run three new full W0 ablations, reuse two verified R4.1 references."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from tools.research.wem_r41 import batch as previous
from tools.research.wem_r41.assets import PREDICTION_KEYS
from .assets import (ROOT, METHODS, MODES, FULL_ARMS, SMOKE, digest, init, load, prepare,
                     read, sha, write)

legacy_imports = previous.legacy_imports
environment = previous.environment
completed_arm = previous.completed_arm
file_manifest = previous._file_manifest
immutable = previous._immutable
run_lock = previous.run_lock


def worker_source(source, mode):
    from capture import replace_once
    if mode == "lower_low":
        source = replace_once(source, "                self.cmc.apply = self.warp\n",
                              "                self.cmc.apply = self.warp\n"
                              "                self.track_low_thresh = .05\n")
    source = previous._worker_source(source)
    return replace_once(source, "                self.wem.finish(self, result)\n",
                        "                result = self.wem.publish(result)\n"
                        "                self.wem.finish(self, result)\n")


def predictor(mode):
    legacy_imports()
    from capture import ADAPTER_BLOB, WORKER_BLOB, pinned_source
    from .online import ClosedLoop, RGB_MODES
    adapter = importlib.import_module("botsort_plus.adapter_v2")
    worker = importlib.import_module("experiments.vttiny_gmc.round3_joint.worker")
    adapter_code = previous._adapter_source(pinned_source(adapter, ADAPTER_BLOB).decode("utf-8"))
    worker_code = worker_source(pinned_source(worker, WORKER_BLOB).decode("utf-8"), mode)
    exec(compile(adapter_code, adapter.__file__, "exec"), adapter.__dict__)

    def no_encoder(_settings):
        raise ValueError("R4.2 never extracts additional RGB appearance features")

    worker.__dict__.update(_r41_closed=ClosedLoop, _r41_encoder=no_encoder,
                           _r41_mode="off" if mode == "lower_low" else mode,
                           _r41_rgb_modes=RGB_MODES)
    exec(compile(worker_code, worker.__file__, "exec"), worker.__dict__)
    return worker


def child(bundle_path, mode, output, sequences):
    bundle = read(bundle_path)
    if bundle.get("schema") != "wem_r42_prediction_bundle_v1" or mode not in MODES:
        raise ValueError("Expected a fixed R4.2 prediction-only bundle and mode")
    if not sequences or len(sequences) != len(set(sequences)) or set(sequences) - set(bundle["sequences"]):
        raise ValueError("Unknown or duplicated sequence selection")
    if set(bundle["settings"]) - set(PREDICTION_KEYS):
        raise ValueError("Prediction bundle contains non-inference settings")
    for relative, expected in bundle["code_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Code changed after preflight: {relative}")
    settings = bundle["settings"]
    if settings.get("runtime_root"):
        sys.path.insert(0, str(Path(settings["runtime_root"]) / "tracking_runtime"))
    result = predictor(mode).run(settings, METHODS["BG"], Path(output), sequences, probe_mode="off")
    if result:
        return result
    path = Path(output) / "protocols" / (METHODS["BG"] + ".json")
    protocol = read(path)
    if protocol["native_config"].get("track_low_thresh") != .1:
        raise ValueError("All variants must originate from the frozen native 0.10 configuration")
    if mode == "lower_low":
        protocol["native_config"]["track_low_thresh"] = .05
        protocol["wem_low_control"] = dict(before=.1, effective=.05,
                                            camera_fingerprints_use_original_settings=True)
    protocol["wem_r42"] = dict(
        schema="wem_r42_online_run_v1", role="BG", mode=mode,
        code_sha256=bundle["code_sha256"], experiment=bundle["protocol"],
        prediction_bundle_sha256=sha(bundle_path), online_ground_truth=False,
        weak_state_feedback=mode in ("W0", "W0_relaxed_anchor"),
        state_chain_output_basis="canonical_native_output" if mode == "W0_emit_only" else "published_output",
        output_only_publication_boundary="after_complete_native_update" if mode == "W0_emit_only" else None,
    )
    write(path, protocol)
    return 0


def preserve_incomplete(batch, folder, log):
    if folder.exists() or log.exists():
        backup = batch / "superseded" / str(time.time_ns())
        backup.mkdir(parents=True)
        for path in (folder, log):
            if path.exists():
                path.rename(backup / path.name)
        print(f"PRESERVED incomplete {folder.name} at {backup}", flush=True)


def mark_arm(context, folder, key, names, provenance):
    write(folder / "arm_complete.json", dict(status="PREDICTION_COMPLETE", arm=key,
          fingerprint=context["fingerprint"], sequences=list(names), provenance=provenance,
          files=file_manifest(folder)))


def import_reference(context, batch, mode, names):
    key = f"BG_{mode}"
    reference = context["reuse"][key]
    folder, log = batch / key, batch / (key + ".log")
    if completed_arm(folder, key, context["fingerprint"], names):
        print(f"REUSE imported {key}", flush=True)
        return
    source = Path(reference["source_arm"])
    if (sha(source / "arm_complete.json") != reference["arm_manifest_sha256"]
            or not completed_arm(source, key, reference["fingerprint"], names)):
        raise ValueError(f"Reference changed after preflight: {key}")
    preserve_incomplete(batch, folder, log)
    # Copy, never hard-link or write through to the user's frozen earlier run.
    shutil.copytree(source, folder)
    if file_manifest(folder) != file_manifest(source):
        raise ValueError(f"Copied reference differs: {key}")
    original_marker = read(folder / "arm_complete.json")
    (folder / "arm_complete.json").unlink()
    write(folder / "r41_import.json", dict(provenance=reference, original_arm_receipt=original_marker))
    mark_arm(context, folder, key, names, reference)
    print(f"IMPORTED {key}: {len(names)} sequences; no tracking rerun", flush=True)


def track(context, batch, mode, names):
    key, method = f"BG_{mode}", METHODS["BG"]
    folder, log = batch / key, batch / (key + ".log")
    if completed_arm(folder, key, context["fingerprint"], names):
        print(f"REUSE {batch.name}/{key}", flush=True)
        return
    preserve_incomplete(batch, folder, log)
    command = [context["settings"]["runtime"]["python"], "-B", "-X", "utf8", str(ROOT / "tools/run_wem_r42.py"),
               "child", "--bundle", str(batch / "prediction_bundle.json"), "--output", str(folder),
               "--mode", mode, "--sequences", *names]
    env = environment(context)
    env["YOLO_CONFIG_DIR"] = str(batch / "runtime" / key)
    print(f"RUN {batch.name}/{key}: {len(names)} sequences", flush=True)
    with log.open("xb") as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    actual = read(folder / "protocols" / (method + ".json"))
    lengths = {name: context["bundle"]["sequences"][name] for name in names}
    if actual.get("status") != "COMPLETE" or actual.get("sequences") != lengths:
        raise ValueError(f"Incomplete run: {key}")
    mark_arm(context, folder, key, names, dict(kind="executed_r42", mode=mode,
                                               fingerprint=context["fingerprint"]))
    print(f"DONE {batch.name}/{key}", flush=True)


def inspect(context, batch, names):
    legacy_imports()
    from experiments.vttiny_gmc.round3_joint.common import assert_same_inputs
    from tools.evaluate_botsort_plus import _check_mot
    checks = {}
    for mode in MODES:
        key, method = f"BG_{mode}", METHODS["BG"]
        folder = batch / key
        if not completed_arm(folder, key, context["fingerprint"], names):
            raise ValueError(f"Incomplete arm: {key}")
        marker = read(folder / "arm_complete.json")
        actual = read(folder / "protocols" / (method + ".json"))
        checked = deepcopy(actual)
        if mode == "lower_low":
            if checked["native_config"].get("track_low_thresh") != .05:
                raise ValueError("The low-threshold control was not applied")
            checked["native_config"]["track_low_thresh"] = .1
        assert_same_inputs(context["references"]["BG"]["protocol"], checked, names)
        expected_scope = {name: context["bundle"]["sequences"][name] for name in names}
        if actual.get("status") != "COMPLETE" or actual.get("sequences") != expected_scope:
            raise ValueError(f"Incomplete protocol: {key}")
        provenance = marker["provenance"]
        if provenance.get("kind") == "executed_r42":
            recorded = actual.get("wem_r42", {})
            if (recorded.get("mode") != mode or recorded.get("role") != "BG"
                    or recorded.get("experiment") != context["bundle"]["protocol"]
                    or recorded.get("prediction_bundle_sha256") != sha(batch / "prediction_bundle.json")
                    or recorded.get("code_sha256") != context["bundle"]["code_sha256"]):
                raise ValueError(f"Run does not match the locked mode/bundle: {key}")
        elif provenance.get("kind") != "reused_r41" or key not in context["reuse"]:
            raise ValueError(f"Unknown reference provenance: {key}")
        details, totals = {}, Counter()
        for seq in names:
            length = context["bundle"]["sequences"][seq]
            receipt = read(folder / "capture_receipts" / method / (seq + ".json"))
            path = folder / "trackers" / method / "data" / (seq + ".txt")
            _check_mot(path, length)
            mot = sha(path)
            events_path = folder / "events" / method / (seq + ".jsonl")
            events, states = sha(events_path), sha(folder / "states" / method / (seq + ".sha256"))
            if receipt["counts"].get("frames") != length or mot != actual["output_prediction_hashes"][seq]:
                raise ValueError(f"Frame/output receipt differs: {key}/{seq}")
            old = context["parity"]["references"]["BG"]["files"][seq]
            if mode == "off" and (mot, events, states) != (old["mot_sha256"], old["events_sha256"], old["state_chain_sha256"]):
                raise ValueError(f"Frozen off parity differs: {seq}")
            if mode == "W0_emit_only":
                if (events, states) != (old["events_sha256"], old["state_chain_sha256"]):
                    raise ValueError(f"Output-only changed future canonical tracker state/events: {seq}")
                if (receipt["counts"].get("weak_state_updates") != 0
                        or receipt["counts"].get("weak_outputs_emitted") != receipt["counts"].get("committed", 0)):
                    raise ValueError(f"Output-only observation/state counts differ: {seq}")
            if mode in ("off", "lower_low") and receipt["counts"].get("committed", 0):
                raise ValueError(f"Disabled W emitted weak observations: {key}/{seq}")
            if mode == "W0" and key in context["reuse"]:
                source = Path(context["reuse"][key]["source_arm"])
                expected = (sha(source / "trackers" / method / "data" / (seq + ".txt")),
                            sha(source / "events" / method / (seq + ".jsonl")),
                            sha(source / "states" / method / (seq + ".sha256")))
                if (mot, events, states) != expected:
                    raise ValueError(f"W0 no longer reproduces R4.1: {seq}")
            camera = actual["joint_batch"]["camera_manifests"][seq]
            frozen = context["bundle"]["settings"]["warp_sources"]["adaptive_gmc"]
            if (camera.get("mode") != "cached" or camera["frames"] != length
                    or camera["source"]["sha256"] != frozen["sha256"][seq]
                    or camera["source"]["decoded_image_sha256"] != actual["decoded_image_hashes"][seq]):
                raise ValueError(f"Frozen camera input differs: {key}/{seq}")
            with events_path.open(encoding="utf-8") as stream:
                recovered = sum(json.loads(line).get("event") == "recovered" for line in stream)
            timing = read(folder / "timing" / method / (seq + ".json"))
            row = dict(mot=mot, native_events=events, states=states,
                       weak_events=sha(folder / "wem_events" / method / (seq + ".jsonl")),
                       counts=dict(receipt["counts"], B_recovered=recovered),
                       association_and_gmc_seconds=timing["association_and_gmc_seconds"],
                       replay_wall_seconds=timing["replay_wall_seconds"])
            for audit in ("wem_scores", "wem_decisions", "wem_frame_audit"):
                row[audit] = sha(folder / audit / method / (seq + ".jsonl.gz"))
            if mode in ("W0_relaxed_anchor", "W0_emit_only"):
                row["wem_r42_audit"] = sha(folder / "wem_r42_audit" / method / (seq + ".jsonl.gz"))
            details[seq] = row
            totals.update(row["counts"])
        elapsed = sum(row["association_and_gmc_seconds"] for row in details.values())
        wall = sum(row["replay_wall_seconds"] for row in details.values())
        checks[key] = dict(variant=mode, method=method, totals=dict(totals), per_sequence=details,
                           per_frame_ms_cached=1000 * elapsed / totals["frames"],
                           replay_frames_per_second=totals["frames"] / wall if wall else None,
                           provenance=provenance)
    result = dict(schema="wem_r42_verified_v1", status="PASS", scope=list(names),
                  scope_frames=sum(context["bundle"]["sequences"][seq] for seq in names),
                  fingerprint=context["fingerprint"], arms=checks,
                  state_chain_note="emit_only uses canonical native output; its emitted output is separately audited")
    write(batch / "verified.json", result)
    return result


def stage(context, which):
    batch = Path(context["r42_settings"]["output_root"]) / which
    batch.mkdir(parents=True, exist_ok=True)
    names = list(SMOKE) if which == "smoke" else list(context["bundle"]["sequences"])
    immutable(batch / "plan.json", dict(schema="wem_r42_batch_v1", stage=which,
              fingerprint=context["fingerprint"], sequences=names,
              frames=sum(context["bundle"]["sequences"][seq] for seq in names),
              arms=[f"BG_{mode}" for mode in MODES], experiment=context["bundle"]["protocol"],
              reused_references=list(context["reuse"]) if which == "full" else [],
              scientific_acceptance_gate=False))
    immutable(batch / "prediction_bundle.json", context["bundle"])
    if which == "full":
        for mode in ("off", "W0"):
            if f"BG_{mode}" in context["reuse"]:
                import_reference(context, batch, mode, names)
    pending = [mode for mode in MODES if not (which == "full" and f"BG_{mode}" in context["reuse"])]
    with ThreadPoolExecutor(max_workers=context["settings"]["runtime"]["workers"]) as pool:
        jobs = [pool.submit(track, context, batch, mode, names) for mode in pending]
        for job in jobs:
            job.result()
    inspect(context, batch, names)
    if which == "smoke":
        write(batch / "complete.json", dict(status="TECHNICAL_PARITY_PASS", fingerprint=context["fingerprint"],
                                             checks=sha(batch / "verified.json"), performance_gate=False))
    return batch


def evaluate(context, batch):
    names = list(context["bundle"]["sequences"])
    verified = inspect(context, batch, names)
    fingerprint = digest(dict(prediction=verified, inputs=context["receipt"]["input_sha256"]))
    marker = batch / "evaluation_complete.json"
    if marker.exists():
        saved = read(marker)
        if saved.get("fingerprint") != fingerprint:
            raise ValueError("Evaluation belongs to different predictions/inputs")
        for relative, expected in saved["files"].items():
            if sha(batch / relative) != expected:
                raise ValueError(f"Completed evaluation output changed: {relative}")
        return
    stale = [batch / name for name in ("eval_input", "metrics", "attribution", "report", "evaluate.log", "attribution.log")]
    if any(path.exists() for path in stale):
        backup = batch / "superseded" / ("evaluation_" + str(time.time_ns()))
        backup.mkdir(parents=True)
        for path in stale:
            if path.exists():
                path.rename(backup / path.name)
    view = batch / "eval_input"
    trackers = []
    for mode in MODES:
        key, method = f"BG_{mode}", METHODS["BG"]
        folder = batch / key
        protocol = read(folder / "protocols" / (method + ".json"))
        protocol["config"]["name"] = protocol["name"] = key
        directory = view / "trackers" / key / "data"
        directory.mkdir(parents=True)
        for seq in names:
            shutil.copyfile(folder / "trackers" / method / "data" / (seq + ".txt"), directory / (seq + ".txt"))
        write(view / "protocols" / (key + ".json"), protocol)
        trackers.append(key)
    inputs = context["evaluation"]
    command = [context["settings"]["runtime"]["python"], "-B", "-X", "utf8", str(ROOT / "tools/evaluate_botsort_plus.py"),
               "--run-dir", str(view), "--trackers", *trackers, "--reference-tracker", "BG_off",
               "--gt-root", inputs["ground_truth"]["root"], "--trackeval-root", inputs["trackeval"]["root"],
               "--output-dir", str(batch / "metrics")]
    with (batch / "evaluate.log").open("xb") as stream:
        subprocess.run(command, cwd=ROOT, env=environment(context), stdout=stream, stderr=subprocess.STDOUT, check=True)
    write(batch / "evaluation_inputs.json", inputs)
    command = [context["settings"]["runtime"]["python"], "-B", "-X", "utf8", str(ROOT / "tools/run_wem_r42.py"),
               "attribute", "--batch", str(batch), "--evaluation-inputs", str(batch / "evaluation_inputs.json")]
    with (batch / "attribution.log").open("xb") as stream:
        subprocess.run(command, cwd=ROOT, env=environment(context), stdout=stream, stderr=subprocess.STDOUT, check=True)
    files = {path.relative_to(batch).as_posix(): sha(path) for name in ("metrics", "attribution", "eval_input")
             for path in sorted((batch / name).rglob("*")) if path.is_file()}
    write(marker, dict(status="EVALUATION_COMPLETE", fingerprint=fingerprint, files=files))


def attribute(batch, evaluation_inputs):
    legacy_imports()
    from .offline import diagnose
    inputs = read(evaluation_inputs)
    diagnose(Path(batch), FULL_ARMS, list(inputs["sequences"]), inputs, METHODS)


def summarize(context, batch):
    from .reporting import report
    summary = report(batch, expected_sequences=list(context["bundle"]["sequences"]),
                     expected_arms=[f"BG_{mode}" for mode in MODES], full=True)
    write(batch / "complete.json", dict(status="FULL_BENCHMARK_COMPLETE", sequence_count=30,
          frames_per_arm=12590, arm_count=5, fingerprint=context["fingerprint"],
          checks=sha(batch / "verified.json"), report=sha(batch / "report/summary.json"),
          scientific_outcome="READ_REPORT_NOT_SOFTWARE_PASS", reused_references=list(context["reuse"])))
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("init", "check", "smoke", "full", "evaluate", "summarize", "child", "attribute"))
    result.add_argument("--settings", type=Path)
    result.add_argument("--from-r41-settings", type=Path)
    result.add_argument("--r41-batch", type=Path)
    result.add_argument("--reference-mode", choices=("reuse", "rerun"))
    result.add_argument("--bundle", type=Path)
    result.add_argument("--mode", choices=MODES)
    result.add_argument("--output", type=Path)
    result.add_argument("--sequences", nargs="+")
    result.add_argument("--batch", type=Path)
    result.add_argument("--evaluation-inputs", type=Path)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "child":
        if not all((args.bundle, args.mode, args.output, args.sequences)):
            raise ValueError("Child needs bundle, mode, output and sequences")
        return child(args.bundle, args.mode, args.output, args.sequences)
    if args.command == "attribute":
        if not args.batch or not args.evaluation_inputs:
            raise ValueError("Attribution needs batch and evaluation-inputs")
        attribute(args.batch, args.evaluation_inputs)
        return 0
    if not args.settings:
        raise ValueError("--settings is required")
    if args.command == "init":
        if not args.from_r41_settings:
            raise ValueError("init requires --from-r41-settings pointing to the existing local config")
        init(args.settings, args.from_r41_settings, args.r41_batch, args.reference_mode or "reuse")
        print(f"CREATED {args.settings}; inherited R4.1 inputs; configure only paths that moved")
        return 0
    if args.from_r41_settings or args.r41_batch or args.reference_mode:
        raise ValueError("--from-r41-settings, --r41-batch and --reference-mode apply to init; edit the local settings for later commands")
    context = prepare(load(args.settings))
    root = Path(context["r42_settings"]["output_root"])
    with run_lock(root):
        write(root / "inputs_verified.json", context["receipt"])
        if args.command == "check":
            print(f"INPUTS_VERIFIED_NOT_RUN: 30 sequences / 12,590 frames; 5 arms; {len(context['reuse'])} references reusable")
        elif args.command == "smoke":
            stage(context, "smoke")
        elif args.command == "full":
            stage(context, "smoke")
            batch = stage(context, "full")
            evaluate(context, batch)
            summarize(context, batch)
            print(f"FULL_BENCHMARK_COMPLETE: {batch / 'report/REPORT.md'}")
        else:
            batch = root / "full"
            evaluate(context, batch)
            if args.command == "summarize":
                summarize(context, batch)
    return 0
