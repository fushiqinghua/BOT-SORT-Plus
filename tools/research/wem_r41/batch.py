"""Run all ten causal W R4.1 arms, then evaluate the complete benchmark."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from .assets import (ROOT, METHODS, MODES, FULL_ARMS, SMOKE_ARMS, SMOKE, PREDICTION_KEYS,
                     digest, init, load, prepare, read, sha, write)


def legacy_imports():
    # The archived scripts have unqualified imports. Give those exact, pinned
    # directories precedence instead of depending on cwd or the user's PYTHONPATH.
    for relative in ("tools/research/vttiny_wem_r4", "tools/research/vttiny_wem_r3",
                     "tools/research/vttiny_wem_r3/observer", "tools", "."):
        path = str((ROOT / relative).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


def _adapter_source(source):
    from capture import replace_once
    source = replace_once(source, "    STrack.multi_predict(pool)\n",
                          "    STrack.multi_predict(pool)\n    tracker.wem.pre_gmc(pool, unconfirmed, high, low)\n")
    source = replace_once(source, "    STrack.multi_gmc(unconfirmed, warp)\n",
                          "    STrack.multi_gmc(unconfirmed, warp)\n"
                          "    tracker.wem.compensated(tracker, pool, unconfirmed, warp)\n")
    return replace_once(source, "    output = tracker._prepare_output(activated, refound, lost, removed).reshape(-1, 8)\n",
                        "    output = tracker.wem.finish_before_output(tracker, activated, refound, lost, removed,\n"
                        "                                             reserved, consumed, all_dets)\n")


def _worker_source(source):
    from capture import replace_once
    source = replace_once(source, "        sequence_iter = iter(paths)\n",
                          "        sequence_iter = iter(paths)\n"
                          "        wem_encoder = _r41_encoder(settings) if _r41_mode in _r41_rgb_modes else None\n")
    source = replace_once(source, "                self.audit_closed = False\n",
                          "                self.audit_closed = False\n"
                          "                self.wem = _r41_closed(paths[self.seq], self.seq, method, output, protocol,\n"
                          "                                       _r41_mode, settings, wem_encoder)\n")
    source = replace_once(source, "                result = super().update(detections, image, *a, **kw)\n",
                          "                self.wem.start(self, detections, image, kw['features'])\n"
                          "                result = super().update(detections, image, *a, **kw)\n"
                          "                self.wem.finish(self, result)\n")
    return replace_once(source, "            def finish_sequence_events(self):\n",
                        "            def finish_sequence_events(self):\n                self.wem.close()\n")


def predictor(mode, bundle):
    legacy_imports()
    from capture import ADAPTER_BLOB, WORKER_BLOB, pinned_source
    from .online import ClosedLoop, RGB_MODES
    adapter = importlib.import_module("botsort_plus.adapter_v2")
    worker = importlib.import_module("experiments.vttiny_gmc.round3_joint.worker")
    adapter_source = _adapter_source(pinned_source(adapter, ADAPTER_BLOB).decode("utf-8"))
    worker_source = _worker_source(pinned_source(worker, WORKER_BLOB).decode("utf-8"))

    def encoder(settings):
        from botsort_plus.feature_export import load_rgb_encoder
        weights = Path(bundle["rgb_weights"])
        if sha(weights) != bundle["protocol"]["input_identity"]["rgb_encoder_sha256"]:
            raise ValueError("Frozen ReID weights changed after preflight")
        result, meta = load_rgb_encoder(weights, bundle["rgb_device"], Path(settings["runtime_root"]) if settings.get("runtime_root") else None)
        if meta["sha256"] != bundle["protocol"]["input_identity"]["rgb_encoder_sha256"]:
            raise ValueError("Loaded a different RGB encoder")
        return result

    exec(compile(adapter_source, adapter.__file__, "exec"), adapter.__dict__)
    worker.__dict__.update(_r41_closed=ClosedLoop, _r41_encoder=encoder,
                           _r41_mode=mode, _r41_rgb_modes=RGB_MODES)
    exec(compile(worker_source, worker.__file__, "exec"), worker.__dict__)
    return worker


def child(bundle_path, role, mode, output, sequences):
    bundle = read(bundle_path)
    if bundle.get("schema") != "wem_r41_prediction_bundle_v1":
        raise ValueError("Expected prediction-only bundle")
    if role not in METHODS or mode not in (*MODES, "shadow", "force_reject"):
        raise ValueError("Unknown closed-loop arm")
    if not sequences or len(sequences) != len(set(sequences)) or set(sequences) - set(bundle["sequences"]):
        raise ValueError("Unknown or duplicate sequence selection")
    for path, expected in bundle["code_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Code changed after batch lock: {path}")
    settings = bundle["settings"]
    if set(settings) - set(PREDICTION_KEYS):
        raise ValueError("Prediction settings contain non-inference fields")
    if settings.get("runtime_root"):
        sys.path.insert(0, str(Path(settings["runtime_root"]) / "tracking_runtime"))
    worker = predictor(mode, bundle)
    rc = worker.run(settings, METHODS[role], Path(output), sequences, probe_mode="off")
    if rc:
        return rc
    path = Path(output) / "protocols" / (METHODS[role] + ".json")
    protocol = read(path)
    protocol["wem_r41"] = {
        "schema": "wem_r41_online_run_v1", "role": role, "mode": mode,
        "code_sha256": bundle["code_sha256"], "experiment": bundle["protocol"],
        "prediction_bundle_sha256": sha(bundle_path), "online_ground_truth": False,
        "causal_state_update": mode in ("W0", "W1", "W2", "W3"),
    }
    write(path, protocol)
    return 0


def environment(context, *, rgb=False):
    legacy_imports()
    from experiments.vttiny_gmc.round3_joint.common import child_environment
    env = child_environment(context["bundle"]["settings"])
    env["CUDA_VISIBLE_DEVICES"] = context["settings"]["runtime"]["cuda_visible_devices"] if rgb else ""
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _immutable(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(f"Existing batch has different inputs/code/settings; choose a new output_root: {path}")
    else:
        write(path, value)


@contextmanager
def run_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".wem_r41_running.json"
    value = {"pid": os.getpid(), "host": socket.gethostname()}
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        old = read(path)
        if old.get("host") != value["host"]:
            raise RuntimeError(f"Another host owns {path}; confirm its process has stopped before removing the stale lock")
        if os.name == "nt":
            # os.kill(pid, 0) is not a harmless liveness probe on Windows.
            # Preserve the lock and the possibly running experiment.
            raise RuntimeError(f"A Windows batch lock already exists at {path}; confirm the previous process has stopped before renaming the stale lock")
        if type(old.get("pid")) is not int or old["pid"] <= 0:
            raise RuntimeError(f"Invalid existing batch lock: {path}")
        try:
            os.kill(int(old["pid"]), 0)
        except ProcessLookupError:
            path.rename(path.with_name(path.name + f".stale.{time.time_ns()}"))
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        else:
            raise RuntimeError(f"A batch process is already using this output root: PID {old['pid']}")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream)
        yield
    finally:
        if path.exists() and read(path) == value:
            path.unlink()


def _file_manifest(folder):
    return {path.relative_to(folder).as_posix(): sha(path) for path in sorted(Path(folder).rglob("*"))
            if path.is_file() and path.name != "arm_complete.json"}


def completed_arm(folder, key, fingerprint, names):
    marker = Path(folder) / "arm_complete.json"
    if not marker.exists():
        return False
    saved = read(marker)
    if (saved.get("status") != "PREDICTION_COMPLETE" or saved.get("arm") != key or
            saved.get("fingerprint") != fingerprint or saved.get("sequences") != list(names)):
        raise ValueError(f"Completed arm belongs to a different experiment: {folder}")
    if saved.get("files") != _file_manifest(Path(folder)):
        raise ValueError(f"Completed arm output changed; preserve it and use a new output root: {folder}")
    return True


def track(context, batch, role, mode, names):
    key, method = f"{role}_{mode}", METHODS[role]
    folder, log = batch / key, batch / (key + ".log")
    if completed_arm(folder, key, context["fingerprint"], names):
        print(f"REUSE {batch.name}/{key}", flush=True)
        return
    if folder.exists() or log.exists():
        backup = batch / "superseded" / str(time.time_ns())
        backup.mkdir(parents=True)
        for path in (folder, log):
            if path.exists():
                path.rename(backup / path.name)
        print(f"PRESERVED incomplete {key} at {backup}", flush=True)
    command = [context["settings"]["runtime"]["python"], "-B", "-X", "utf8", str(ROOT / "tools/run_wem_r41.py"),
               "child", "--bundle", str(batch / "prediction_bundle.json"), "--output", str(folder),
               "--role", role, "--mode", mode, "--sequences", *names]
    env = environment(context, rgb=mode in ("W2", "W3", "shadow", "force_reject"))
    env["YOLO_CONFIG_DIR"] = str(batch / "runtime" / key)
    print(f"RUN {batch.name}/{key}: {len(names)} sequences", flush=True)
    with log.open("xb") as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    protocol = read(folder / "protocols" / (method + ".json"))
    expected = {name: context["bundle"]["sequences"][name] for name in names}
    if protocol.get("status") != "COMPLETE" or protocol["sequences"] != expected:
        raise ValueError(f"Arm did not finish its complete declared scope: {key}")
    write(folder / "arm_complete.json", {"status": "PREDICTION_COMPLETE", "arm": key,
          "fingerprint": context["fingerprint"], "sequences": list(names), "files": _file_manifest(folder)})
    print(f"DONE {batch.name}/{key}", flush=True)


def inspect(context, batch, arms, names):
    legacy_imports()
    from experiments.vttiny_gmc.round3_joint.common import assert_same_inputs
    from tools.evaluate_botsort_plus import _check_mot
    checks = {}
    for role, mode in arms:
        key, method = f"{role}_{mode}", METHODS[role]
        folder = batch / key
        if not completed_arm(folder, key, context["fingerprint"], names):
            raise ValueError(f"Incomplete arm: {key}")
        actual = read(folder / "protocols" / (method + ".json"))
        expected = context["references"][role]["protocol"]
        assert_same_inputs(expected, actual, names)
        if actual.get("status") != "COMPLETE" or set(actual["sequences"]) != set(names):
            raise ValueError(f"Incomplete arm protocol: {key}")
        details, totals = {}, Counter()
        for seq in names:
            length = context["bundle"]["sequences"][seq]
            receipt = read(folder / "capture_receipts" / method / (seq + ".json"))
            old = context["parity"]["references"][role]["files"][seq]
            mot_path = folder / "trackers" / method / "data" / (seq + ".txt")
            mot = sha(mot_path)
            _check_mot(mot_path, length)
            events_path = folder / "events" / method / (seq + ".jsonl")
            events, states = sha(events_path), sha(folder / "states" / method / (seq + ".sha256"))
            if receipt["counts"]["frames"] != length or mot != actual["output_prediction_hashes"][seq]:
                raise ValueError(f"Frame/output mismatch: {key}/{seq}")
            if mode in ("off", "shadow", "force_reject"):
                if (mot, events, states) != (old["mot_sha256"], old["events_sha256"], old["state_chain_sha256"]):
                    raise ValueError(f"Reference parity differs: {key}/{seq}")
                if receipt["counts"].get("committed", 0):
                    raise ValueError(f"Read-only control committed a weak observation: {key}/{seq}")
            camera = actual["joint_batch"]["camera_manifests"][seq]
            camera_name = "native" if role == "B" else "adaptive_gmc"
            frozen = context["bundle"]["settings"]["warp_sources"][camera_name]
            if (camera["frames"] != length or camera["mode"] != "cached" or
                    camera["source"]["sha256"] != frozen["sha256"][seq] or
                    camera["source"]["decoded_image_sha256"] != actual["decoded_image_hashes"][seq]):
                raise ValueError(f"GMC source drift: {key}/{seq}")
            with events_path.open(encoding="utf-8") as stream:
                b_recovered = sum(json.loads(line).get("event") == "recovered" for line in stream)
            weak = folder / "wem_events" / method / (seq + ".jsonl")
            timing = read(folder / "timing" / method / (seq + ".json"))
            row = {"mot": mot, "native_events": events, "states": states, "weak_events": sha(weak),
                   "counts": dict(receipt["counts"], B_recovered=b_recovered),
                   "association_and_gmc_seconds": timing["association_and_gmc_seconds"],
                   "replay_wall_seconds": timing["replay_wall_seconds"]}
            for audit in ("wem_scores", "wem_decisions", "wem_frame_audit"):
                row[audit] = sha(folder / audit / method / (seq + ".jsonl.gz"))
            details[seq] = row
            totals.update(row["counts"])
        elapsed = sum(row["association_and_gmc_seconds"] for row in details.values())
        wall = sum(row["replay_wall_seconds"] for row in details.values())
        checks[key] = {"variant": mode, "method": method, "totals": dict(totals), "per_sequence": details,
                       "per_frame_ms_cached": 1000 * elapsed / totals["frames"],
                       "replay_frames_per_second": totals["frames"] / wall if wall else None}
    for role in METHODS:
        if (role, "shadow") in arms and (role, "force_reject") in arms:
            for seq in names:
                left = checks[role + "_shadow"]["per_sequence"][seq]
                right = checks[role + "_force_reject"]["per_sequence"][seq]
                for stream in ("weak_events", "wem_scores", "wem_decisions", "wem_frame_audit"):
                    if left[stream] != right[stream]:
                        raise ValueError(f"Shadow and force-reject streams differ: {role}/{seq}/{stream}")
    result = {"schema": "wem_r41_verified_v1", "status": "PASS", "scope": list(names),
              "scope_frames": sum(context["bundle"]["sequences"][name] for name in names),
              "fingerprint": context["fingerprint"], "arms": checks}
    write(batch / "verified.json", result)
    return result


def stage(context, which):
    batch = Path(context["settings"]["paths"]["output_root"]) / which
    names = list(SMOKE) if which == "smoke" else list(context["bundle"]["sequences"])
    arms = SMOKE_ARMS if which == "smoke" else FULL_ARMS
    batch.mkdir(parents=True, exist_ok=True)
    plan = {"schema": "wem_r41_batch_v1", "stage": which, "fingerprint": context["fingerprint"],
            "sequences": names, "frames": sum(context["bundle"]["sequences"][name] for name in names),
            "arms": [f"{role}_{mode}" for role, mode in arms], "experiment": context["bundle"]["protocol"],
            "scientific_acceptance_gate": False}
    _immutable(batch / "plan.json", plan)
    _immutable(batch / "prediction_bundle.json", context["bundle"])
    workers = context["settings"]["runtime"]["workers"]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(track, context, batch, role, mode, names) for role, mode in arms]
        for job in jobs:
            job.result()
    inspect(context, batch, arms, names)
    if which == "smoke":
        write(batch / "complete.json", {"status": "TECHNICAL_PARITY_PASS", "fingerprint": context["fingerprint"],
                                       "checks": sha(batch / "verified.json"), "performance_gate": False})
    return batch


def evaluate(context, batch):
    """Only this post-prediction phase receives GT and evaluator paths."""
    names = list(context["bundle"]["sequences"])
    verified = inspect(context, batch, FULL_ARMS, names)
    evaluation_fingerprint = digest({"prediction": verified, "inputs": context["receipt"]["input_sha256"]})
    marker = batch / "evaluation_complete.json"
    if marker.exists():
        saved = read(marker)
        if saved.get("fingerprint") != evaluation_fingerprint:
            raise ValueError("Evaluation belongs to a different verified prediction set")
        for relative, expected in saved["files"].items():
            if sha(batch / relative) != expected:
                raise ValueError(f"Completed evaluation output changed: {relative}")
        return
    stale = [batch / part for part in ("eval_input", "metrics", "attribution", "report", "evaluate.log")]
    if any(path.exists() for path in stale):
        backup = batch / "superseded" / ("evaluation_" + str(time.time_ns()))
        backup.mkdir(parents=True)
        for path in stale:
            if path.exists():
                path.rename(backup / path.name)
    view = batch / "eval_input"
    trackers = []
    for role, mode in FULL_ARMS:
        key, method = f"{role}_{mode}", METHODS[role]
        folder = batch / key
        protocol = read(folder / "protocols" / (method + ".json"))
        protocol["config"]["name"] = key
        protocol["name"] = key
        directory = view / "trackers" / key / "data"
        directory.mkdir(parents=True)
        for seq in names:
            shutil.copyfile(folder / "trackers" / method / "data" / (seq + ".txt"), directory / (seq + ".txt"))
        write(view / "protocols" / (key + ".json"), protocol)
        trackers.append(key)
    inputs = context["evaluation"]
    command = [context["settings"]["runtime"]["python"], "-B", "-X", "utf8", str(ROOT / "tools/evaluate_botsort_plus.py"),
               "--run-dir", str(view), "--trackers", *trackers, "--reference-tracker", "B_off",
               "--gt-root", inputs["ground_truth"]["root"], "--trackeval-root", inputs["trackeval"]["root"],
               "--output-dir", str(batch / "metrics")]
    with (batch / "evaluate.log").open("xb") as stream:
        subprocess.run(command, cwd=ROOT, env=environment(context), stdout=stream, stderr=subprocess.STDOUT, check=True)
    # Attribution runs in its own Python process too (the configured environment,
    # never the launcher's possibly lightweight interpreter).
    evaluation_bundle = batch / "evaluation_inputs.json"
    write(evaluation_bundle, inputs)
    command = [context["settings"]["runtime"]["python"], "-B", "-X", "utf8", str(ROOT / "tools/run_wem_r41.py"),
               "attribute", "--batch", str(batch), "--evaluation-inputs", str(evaluation_bundle)]
    with (batch / "attribution.log").open("ab") as stream:
        subprocess.run(command, cwd=ROOT, env=environment(context), stdout=stream, stderr=subprocess.STDOUT, check=True)
    files = {path.relative_to(batch).as_posix(): sha(path) for name in ("metrics", "attribution", "eval_input")
             for path in sorted((batch / name).rglob("*")) if path.is_file()}
    write(marker, {"status": "EVALUATION_COMPLETE", "fingerprint": evaluation_fingerprint, "files": files})


def attribute(batch, evaluation_inputs):
    legacy_imports()
    from tools.research.vttiny_wem_r4.offline import diagnose
    inputs = read(evaluation_inputs)
    diagnose(Path(batch), FULL_ARMS, list(inputs["sequences"]), inputs, METHODS)


def summarize(context, batch):
    from .reporting import report
    if not (batch / "evaluation_complete.json").is_file():
        raise ValueError("Complete prediction and evaluation first")
    summary = report(batch, expected_sequences=list(context["bundle"]["sequences"]),
                     expected_arms=[f"{role}_{mode}" for role, mode in FULL_ARMS], full=True)
    write(batch / "complete.json", {"status": "FULL_BENCHMARK_COMPLETE", "sequence_count": 30,
          "frames_per_arm": 12590, "arm_count": 10, "fingerprint": context["fingerprint"],
          "checks": sha(batch / "verified.json"), "report": sha(batch / "report/summary.json"),
          "scientific_outcome": "READ_REPORT_NOT_SOFTWARE_PASS"})
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("init", "check", "smoke", "full", "evaluate", "summarize", "child", "attribute"))
    p.add_argument("--settings", type=Path)
    p.add_argument("--legacy-project-root", type=Path)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--role", choices=METHODS)
    p.add_argument("--mode", choices=(*MODES, "shadow", "force_reject"))
    p.add_argument("--output", type=Path)
    p.add_argument("--sequences", nargs="+")
    p.add_argument("--batch", type=Path)
    p.add_argument("--evaluation-inputs", type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "child":
        if not all((args.bundle, args.role, args.mode, args.output, args.sequences)):
            raise ValueError("Child needs bundle, role, mode, output and sequences")
        return child(args.bundle, args.role, args.mode, args.output, args.sequences)
    if args.command == "attribute":
        if not args.batch or not args.evaluation_inputs:
            raise ValueError("Attribution needs batch and evaluation-inputs")
        attribute(args.batch, args.evaluation_inputs)
        return 0
    if args.settings is None:
        raise ValueError("--settings is required")
    if args.command == "init":
        init(args.settings, args.legacy_project_root)
        print(f"Created {args.settings}; configure local paths and runtime before check")
        return 0
    context = prepare(load(args.settings))
    root = Path(context["settings"]["paths"]["output_root"])
    with run_lock(root):
        write(root / "inputs_verified.json", context["receipt"])
        if args.command == "check":
            print("INPUTS_VERIFIED_NOT_RUN: exact 30 sequences / 12,590 frames; 10 full arms")
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
            if args.command == "evaluate":
                evaluate(context, batch)
            else:
                # Recheck provenance and hashes rather than trusting a stale report.
                evaluate(context, batch)
                summarize(context, batch)
    return 0
