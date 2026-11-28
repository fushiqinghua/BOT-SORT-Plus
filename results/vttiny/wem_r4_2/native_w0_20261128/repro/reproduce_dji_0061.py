"""Rebuild the DJI_0061_2 CLEAR ledgers locally from frozen experiment outputs."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


SEQUENCE = "DJI_0061_2"
NATIVE = ("native_off", "native_W0")
HISTORICAL = ("B_off", "B_W0", "BG_off", "BG_W0")
METHOD = "native_botsort"
R41 = "3132e69fcca86ac5ce6d59271b2dff38454fc628"
R42_OUTPUT_ONLY = "54fc84d0126d1aab74cd26a7a98206a1fcf3f3a8d4db776ac6dbf4689f14e61f"
EVALUATOR = "3ea2cdcc710d38a2cdc9313518cf070ed5d38d4bc8af73cefc8e16a45d6dda2a"
NATIVE_EVENT_FIELDS = ("event", "frame", "track_id", "det_index", "stage")
WEAK_EVENT_FIELDS = ("event", "frame", "track_id", "det_index", "raw_score", "state_committed")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked(path, expected=None):
    raw = path.read_bytes()
    if expected is not None and digest(raw) != expected:
        raise ValueError(f"Frozen artifact differs: {path}")
    return raw


def read(path, expected=None):
    return json.loads(checked(path, expected))


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def project_events(path, raw, fields):
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for line in raw.splitlines():
            event = json.loads(line)
            row = {key: event[key] for key in fields if key in event}
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def ledger(result):
    frames = [sorted(frame.pop("gt_ids")) for frame in result["frames"]]
    for frame in result["frames"]:
        del frame["gt_boxes"]
        for row in frame["outputs"].values():
            del row["box_xywh"]
            row["gt_ious"] = {gid: iou for gid, iou in row["gt_ious"].items() if iou > 0}
    return frames


def native_output_only(source, runner, plan, bundle_file, settings, destination):
    """Keep the archived native hooks and insert only R4.2's delayed publication."""
    from experiments.vttiny_gmc.round3_joint.common import child_environment

    checked(source / "tools/research/wem_r42/online.py", R42_OUTPUT_ONLY)
    raw = checked(runner, plan["source_sha256"])
    source_text = raw.decode("utf-8")
    patches = (
        ("from tools.research.wem_r41.online import ClosedLoop", "from tools.research.wem_r42.online import ClosedLoop"),
        ("                self.wem.finish(self, result)\n",
         "                result = self.wem.publish(result)\n                self.wem.finish(self, result)\n"),
    )
    for before, after in patches:
        if source_text.count(before) != 1:
            raise ValueError("Unexpected archived native hook")
        source_text = source_text.replace(before, after)

    destination.mkdir(parents=True, exist_ok=False)
    child = destination / "replay_emit.py"
    child.write_text(source_text, encoding="utf-8", newline="\n")
    run = destination / "out"
    program = ('import runpy, sys; from pathlib import Path; '
               'ns = runpy.run_path(sys.argv[1]); '
               'raise SystemExit(ns["predict"](Path(sys.argv[2]), Path(sys.argv[3]), '
               '"W0_emit_only", [sys.argv[4]], Path(sys.argv[5])))')
    env = child_environment(settings)
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONWARNINGS="ignore", YOLO_CONFIG_DIR=str(destination / "runtime"))
    env["PYTHONPATH"] = os.pathsep.join([str(source), str(source / "tools"), env["PYTHONPATH"]])
    command = [sys.executable, "-B", "-c", program, str(child), str(source), str(bundle_file), SEQUENCE, str(run)]
    log = destination / "replay.log"
    with log.open("xb") as stream:
        process = subprocess.run(command, cwd=source, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"Output-only replay failed. See {log}\n{log.read_bytes()[-4000:].decode(errors='replace')}")
    return run


def export_arm(folders, reference, evaluate, official, output):
    mot, events, weak, state = folders
    source = {
        "mot": checked(mot, reference["mot"]),
        "native_events": checked(events, reference.get("native_events")),
        "weak_events": checked(weak, reference.get("weak_events")),
        "states": checked(state, reference["states"]),
    }
    result = evaluate()
    if official is not None and any(result["counts"][key] != official[key] for key in ("FP", "FN", "IDSW")):
        raise ValueError(f"Recovered CLEAR ledger differs from the retained evaluation: {mot}")
    frames = ledger(result)
    put(output / "clear.json", result)
    project_events(output / "native_events.jsonl", source["native_events"], NATIVE_EVENT_FIELDS)
    project_events(output / "weak_events.jsonl", source["weak_events"], WEAK_EVENT_FIELDS)
    return frames, {key: digest(value) for key, value in source.items()}


def arm_files(view, arm, source, method):
    return (view / arm / "data" / (SEQUENCE + ".txt"), source / "events" / method / (SEQUENCE + ".jsonl"),
            source / "wem_events" / method / (SEQUENCE + ".jsonl"), source / "states" / method / (SEQUENCE + ".sha256"))


def collect_camera(archive, reference_arms):
    arms = {}
    matrices = []
    for name in HISTORICAL:
        method = reference_arms[name]["method"]
        folder = archive / name
        receipt = read(folder / "arm_complete.json")
        protocol = read(folder / "protocols" / (method + ".json"), receipt["files"][f"protocols/{method}.json"])
        joint = protocol["joint_batch"]
        warp = folder / "warps" / method / (SEQUENCE + ".jsonl.gz")
        decisions = folder / "decisions" / method / (SEQUENCE + ".jsonl")
        raw = checked(warp, joint["camera_manifests"][SEQUENCE]["warp_sha256"])
        accepted = checked(decisions, joint["decision_manifests"][SEQUENCE]["data_sha256"])
        frames = [json.loads(line) for line in gzip.decompress(raw).splitlines()]
        matrices.append([frame["used"] for frame in frames])
        arms[name] = dict(frames=len(frames), warp_choices=dict(Counter(frame["choice"] for frame in frames)),
                          adaptive_triggers=sum(frame.get("trigger", False) for frame in frames),
                          lightglue_calls=sum(frame.get("model_called", False) for frame in frames),
                          dual_low_rejected_edges=sum(json.loads(line)["event"] == "dual_low_rejected_edge"
                                                     for line in accepted.splitlines()))
    return dict(sequence=SEQUENCE, ref=R41, arms=arms,
                identical_selected_matrices=all(m == matrices[0] for m in matrices[1:]))


def reproduce(args):
    sys.dont_write_bytecode = True
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    sys.path.insert(0, str(args.repo_root))
    checked(args.repo_root / "experiments/vttiny_gmc/round3_joint/analysis.py", EVALUATOR)
    from experiments.vttiny_gmc.round3_joint.analysis import evaluate_sequence, load_trackeval
    from experiments.vttiny_gmc.round3_joint.common import assert_same_inputs
    from botsort_plus.io import detection_files, detection_fingerprint, read_detection_cache, sequence_name

    native, archive = args.native_run, args.r41_run
    completion, old_done = read(native / "complete.json"), read(archive / "complete.json")
    if completion["status"] != "COMPLETE" or old_done["status"] != "FULL_BENCHMARK_COMPLETE":
        raise ValueError("Provide the completed native run and the R4.1 full directory")
    plan = read(native / "plan.json")
    native_receipts = read(native / "full/verified.json", completion["full_verification_sha256"])["results"][SEQUENCE]
    reference_arms = read(archive / "verified.json", old_done["checks"])["arms"]
    evaluation = read(archive / "evaluation_inputs.json", completion["frozen_evaluation_input_sha256"])
    recorded_metrics = read(native / "metrics/summary.json", completion["metric_sha256"])["results"]
    metrics = {entry["tracker"]: entry for entry in recorded_metrics if entry["sequence"] == SEQUENCE}
    if set(metrics) != set(NATIVE + HISTORICAL):
        raise ValueError("Retained evaluation is missing one or more original arms")
    bundle_file = archive / "prediction_bundle.json"
    settings = read(bundle_file, plan["prediction_bundle_sha256"])["settings"]
    gt_root = args.gt_root or Path(evaluation["ground_truth"]["root"])
    checked(gt_root / SEQUENCE / "gt/gt.txt", evaluation["ground_truth"]["sha256"][SEQUENCE])
    trackeval, audit = load_trackeval(args.trackeval_root or Path(evaluation["trackeval"]["root"]))
    detection_root = args.detections_root or Path(settings["baseline_root"]) / "detections"
    detection_path = next((p for p in detection_files(detection_root) if sequence_name(p) == SEQUENCE), None)
    if detection_path is None:
        raise FileNotFoundError(f"Missing frozen detection stream for {SEQUENCE}")
    cache = read_detection_cache(detection_path)
    view = native / "eval_input/trackers"
    protocol = read(native / "eval_input/protocols/native_W0.json")
    fingerprint = detection_fingerprint(cache)
    if fingerprint != protocol["detection_fingerprints"][SEQUENCE]:
        raise ValueError("Frozen detector cache differs")
    length = evaluation["sequences"][SEQUENCE]

    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "inputs"
    observed, hashes = None, {}
    for arm in NATIVE + HISTORICAL:
        if arm in NATIVE:
            source, method, receipt = native / "full", METHOD, native_receipts[arm]
        else:
            record = reference_arms[arm]
            source, method, receipt = archive, record["method"], record["per_sequence"][SEQUENCE]
        files = arm_files(view, arm, source / arm, method)
        evaluate = lambda arm=arm: evaluate_sequence(trackeval, gt_root, view, arm, SEQUENCE, length, args.output)
        frames, hashes[arm] = export_arm(files, receipt, evaluate, metrics[arm], output / arm)
        if observed is None:
            observed = frames
        elif observed != frames:
            raise ValueError("Official ground-truth frame scope differs among arms")

    emit = args.emit_run or native_output_only(args.r42_source or args.repo_root, args.native_runner, plan,
                                               bundle_file, settings, args.output / "emit_replay")
    emit_view = emit / "trackers"
    emit_protocol = read(emit / "protocols" / (METHOD + ".json"))
    assert_same_inputs(read(native / "full/native_off/protocols" / (METHOD + ".json")), emit_protocol, [SEQUENCE])
    counts = read(emit / "capture_receipts" / METHOD / (SEQUENCE + ".json"))["counts"]
    if counts["weak_state_updates"] != 0 or emit_protocol["native_w0"]["mode"] != "W0_emit_only":
        raise ValueError("The output-only run lacks the expected zero-feedback mode and receipt")
    off = hashes["native_off"]
    emit_reference = {"mot": emit_protocol["output_prediction_hashes"][SEQUENCE],
                      "states": off["states"], "native_events": off["native_events"]}
    emit_files = arm_files(emit_view, METHOD, emit, METHOD)
    evaluate_emit = lambda: evaluate_sequence(trackeval, gt_root, emit_view, METHOD, SEQUENCE, length, args.output)
    frames, hashes["native_emit_only"] = export_arm(emit_files, emit_reference, evaluate_emit, None,
                                                  output / "native_emit_only")
    if observed != frames:
        raise ValueError("Official ground-truth frame scope differs in the output-only run")
    put(output / "native_emit_only/receipt.json", dict(sequence=SEQUENCE, scorer=audit["commit"], counts=counts,
        canonical_equal_to_native_off=True, state_chain_sha256=off["states"], published_mot_sha256=emit_reference["mot"],
        source_sha256=emit_protocol["native_w0"]["runner_sha256"]))
    put(output / "gt_frames.json", [{str(gid): True for gid in ids} for ids in observed])
    # The analyzer indexes the fifth field by frozen detection row. No detector coordinates are exported.
    put(output / "detections.json", {frame["frame"]: [[None, None, None, None, row[4]] for row in frame["detections"]]
                                     for frame in cache["frames"]})
    put(output / "receipt.json", dict(sequence=SEQUENCE, native_completed=completion,
        rank_evaluator=audit["commit"], detection_fingerprint=fingerprint, source_sha256=hashes))
    put(args.output / "camera_evidence.json", collect_camera(archive, reference_arms))
    print(f"Reproduced seven CLEAR ledgers in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-run", type=Path, required=True, help="Completed original BoT-SORT W0 run directory")
    parser.add_argument("--r41-run", type=Path, required=True, help="Archived R4.1 run/full directory")
    parser.add_argument("--repo-root", type=Path, required=True, help="Source tree containing the pinned CLEAR evaluator")
    parser.add_argument("--output", type=Path, required=True, help="New local directory for private intermediate data")
    parser.add_argument("--emit-run", type=Path, help="Existing output-only out/ directory to verify and reuse")
    parser.add_argument("--r42-source", type=Path, help="Source tree for output-only replay, defaults to --repo-root")
    parser.add_argument("--native-runner", type=Path, default=Path(__file__).with_name("run_native.py"),
                        help="Exact archived native runner, defaults to adjacent run_native.py")
    parser.add_argument("--gt-root", type=Path, help="Relocated directory with the frozen official annotations")
    parser.add_argument("--trackeval-root", type=Path, help="Relocated directory with the frozen TrackEval source")
    parser.add_argument("--detections-root", type=Path, help="Relocated directory with the frozen detection stream")
    args = parser.parse_args()
    for key, path in vars(args).items():
        if path is not None:
            setattr(args, key, path.resolve())
    roots = (args.native_run, args.r41_run, args.repo_root, args.emit_run, args.r42_source)
    if any(root is not None and args.output.is_relative_to(root) for root in roots):
        parser.error("Choose an output directory outside the frozen runs and source trees")
    reproduce(args)


if __name__ == "__main__":
    main()
