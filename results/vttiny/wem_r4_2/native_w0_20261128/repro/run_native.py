"""Replay the pinned, original BoxMOT BoT-SORT with the archived strict W0 toggle."""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap


METHOD = "native_botsort"
MODES = ("off", "W0")
NAMES = tuple(f"native_{mode}" for mode in MODES)
HISTORICAL = ("B_off", "B_W0", "BG_off", "BG_W0")
REFERENCE_PROTOCOL_SHA = "7c18c28f2272c6fcf497bc6234c867a3472e0acbab23b73804d9744f315d2819"
SMOKE = ("DJI_0030_2", "DJI_0061_2", "DJI_0327_3")


def native_hooks(bot_sort):
    """Bind the original decorated update with only observation and W0 calls inserted."""
    from capture import replace_once

    original_class = inspect.getsource(bot_sort)
    nodes = {n.name: n for n in ast.parse(original_class).body[0].body if isinstance(n, ast.FunctionDef)}
    lines = original_class.splitlines(keepends=True)

    def source(name):
        node = nodes[name]
        start = min([node.lineno, *(dec.lineno for dec in node.decorator_list)])
        return textwrap.dedent("".join(lines[start - 1:node.end_lineno]))

    first = replace_once(source("_first_association"), "    STrack.multi_predict(strack_pool)\n",
        "    STrack.multi_predict(strack_pool)\n"
        "    low = dets[(dets[:, 4] > self.track_low_thresh) & (dets[:, 4] < self.track_high_thresh)]\n"
        "    self.wem.pre_gmc(strack_pool, unconfirmed, dets_first, low)\n")
    first = replace_once(first, "    STrack.multi_gmc(unconfirmed, warp)\n",
        "    STrack.multi_gmc(unconfirmed, warp)\n"
        "    self.wem.compensated(self, strack_pool, unconfirmed, warp)\n"
        "    self.native_unconfirmed = {int(track.id) for track in unconfirmed}\n")
    update = replace_once(source("update"),
        "    return self._prepare_output(\n        activated_stracks, refind_stracks, lost_stracks, removed_stracks\n    )",
        "    return self.finish_wem(\n        activated_stracks, refind_stracks, lost_stracks, removed_stracks, dets\n    )")
    module = sys.modules[bot_sort.__module__]
    namespace = dict(vars(module))
    exec(compile(first + "\n" + update, module.__file__, "exec"), namespace)
    return namespace["update"], namespace["_first_association"]


def predict(source, bundle_path, mode, sequences, output):
    from tools.research.wem_r41 import batch
    from tools.research.wem_r41.assets import PREDICTION_KEYS, read, sha, write

    bundle = read(bundle_path)
    if bundle["schema"] != "wem_r41_prediction_bundle_v1" or set(bundle["settings"]) - set(PREDICTION_KEYS):
        raise ValueError("Expected frozen inference-only R4.1 bundle")
    for relative, expected in bundle["code_sha256"].items():
        if sha(source / relative) != expected:
            raise ValueError(f"Frozen W0 integration source differs: {relative}")
    settings = bundle["settings"]
    if settings["gmc_mode"] != "cached" or set(sequences) - set(bundle["sequences"]):
        raise ValueError("Native W0 requires the frozen SOF stream and sequence scope")
    if settings.get("runtime_root"):
        sys.path.insert(0, str(Path(settings["runtime_root"]) / "tracking_runtime"))
    batch.legacy_imports()
    from tools.research.wem_r41.online import ClosedLoop
    from experiments.vttiny_gmc.round3_joint.camera import WarpCamera
    import botsort_plus.native_adapter as adapter
    import run_botsort_plus as original

    adapter.verify_upstream()
    config_path = output / "configs" / (METHOD + ".json")
    cfg = read(source / "configs/botsort_plus/12_baseline_ref.json")
    cfg["name"] = METHOD
    write(config_path, cfg)
    cli = ["--config", str(config_path), "--baseline-root", settings["baseline_root"],
           "--features-dir", settings["features_dir"], "--images-root", settings["images_root"],
           "--output-dir", str(output), "--trace", "--sequences", *sequences]
    if settings.get("runtime_root"):
        cli += ["--runtime-root", settings["runtime_root"]]
    arguments = original.parser().parse_args(cli)
    prior_prepare, prior_tracker = original.prepare, adapter.BoTSORTPlus
    prior_update = adapter.BotSort.update
    update, first = native_hooks(adapter.BotSort)

    def prepare(args):
        config, paths, protocol = prior_prepare(args)
        if config.enabled or config.raw_ir.enabled or config.native_overrides or not protocol["native_config"]["with_reid"]:
            raise ValueError("Original BoT-SORT must disable all Plus modules and use native ReID")
        if protocol["native_config"]["cmc_method"] != "sof" or protocol["native_config"]["track_low_thresh"] != .1:
            raise ValueError("Expected original SOF and native 0.10 threshold")
        order = iter(paths)
        identity = protocol["native_w0"] = dict(mode=mode, boxmot="16.0.11", original_native_association=True,
                   module_B=False, adaptive_camera=False, geometry_protection=False, formal_weak_update=mode == "W0",
                   anchor_rules="R4.1 strict W0", ground_truth_in_prediction=False,
                   prediction_bundle_sha256=sha(bundle_path), runner_sha256=sha(__file__), camera_manifests={})

        class Tracker(prior_tracker):
            _first_association = first

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.seq = next(order)
                self.joint_dual_low = self.matched_camera = False
                self.camera = WarpCamera(settings, "native", self.seq, protocol, self.cmc.apply, None,
                    output / "warps" / METHOD / (self.seq + ".jsonl.gz"))
                self.cmc.apply = self.warp
                self.wem = ClosedLoop(paths[self.seq], self.seq, METHOD, output, protocol, mode, settings, None)

            def warp(self, image, detections):
                self.joint_native_warp, used, self.joint_backup_warp, _ = self.camera.apply(
                    image, detections, self.frame_count)
                return used

            def update(self, dets, img, *args, **kwargs):
                self.wem.start(self, dets, img, kwargs["features"])
                result = super().update(dets, img, *args, **kwargs)
                self.wem.finish(self, result)
                return result

            def finish_wem(self, activated, refound, lost, removed, dets):
                for track in [*activated, *refound]:
                    tid, idx = int(track.id), int(track.det_ind)
                    if track.start_frame == self.frame_count:
                        event, stage = "native_birth", "native_birth"
                    elif tid in self.native_unconfirmed:
                        event, stage = "match", "unconfirmed"
                    else:
                        event, stage = "match", "high" if dets[idx, 4] > self.track_high_thresh else "low"
                    self.last_events.append(dict(event=event, stage=stage, track_id=tid, det_index=idx))
                # No residual B/A ownership exists on the native foundation.
                return self.wem.finish_before_output(self, activated, refound, lost, removed, set(), set(), dets)

            def finish_sequence_events(self):
                self.wem.close()
                identity["camera_manifests"][self.seq] = self.camera.close()
                return super().finish_sequence_events()

        adapter.BoTSORTPlus = Tracker
        return config, paths, protocol

    original.prepare, adapter.BotSort.update = prepare, update
    try:
        return original.run(arguments)
    finally:
        original.prepare, adapter.BoTSORTPlus, adapter.BotSort.update = prior_prepare, prior_tracker, prior_update


def check_predictions(folder, sequences, settings, reference):
    from tools.research.wem_r41.assets import read, sha, write
    from experiments.vttiny_gmc.round3_joint.common import assert_same_inputs
    from tools.evaluate_botsort_plus import _check_mot

    protocols = {arm: read(folder / arm / "protocols" / (METHOD + ".json")) for arm in NAMES}
    lengths = {seq: reference["sequences"][seq] for seq in sequences}
    result = {}
    for arm, protocol in protocols.items():
        if protocol["status"] != "COMPLETE" or protocol["sequences"] != lengths:
            raise ValueError(f"Missing complete prediction scope: {arm}")
        assert_same_inputs(reference, protocol, sequences)
    for seq, length in lengths.items():
        row = result[seq] = {}
        for arm, protocol in protocols.items():
            path = folder / arm / "trackers" / METHOD / "data" / (seq + ".txt")
            _check_mot(path, length)
            mot = sha(path)
            receipt = read(folder / arm / "capture_receipts" / METHOD / (seq + ".json"))
            camera = protocol["native_w0"]["camera_manifests"][seq]
            if (mot != protocol["output_prediction_hashes"][seq] or receipt["counts"]["frames"] != length
                    or camera["frames"] != length or camera["choices"] != {"sof": length}
                    or camera["source"]["sha256"] != settings["warp_sources"]["native"]["sha256"][seq]
                    or camera["source"]["decoded_image_sha256"] != protocol["decoded_image_hashes"][seq]):
                raise ValueError(f"Invalid output or frozen native GMC receipt: {arm}/{seq}")
            if arm == NAMES[0] and (mot != reference["output_prediction_hashes"][seq]
                                    or receipt["counts"].get("committed", 0)):
                raise ValueError(f"Disabled original BoT-SORT did not reproduce the archived native trajectory: {seq}")
            states = folder / arm / "states" / METHOD / (seq + ".sha256")
            events = folder / arm / "wem_events" / METHOD / (seq + ".jsonl")
            committed = [r["frame"] for r in map(json.loads, events.read_bytes().splitlines()) if r["event"] == "committed"]
            if len(committed) != receipt["counts"].get("committed", 0):
                raise ValueError(f"Weak commit receipt mismatch: {arm}/{seq}")
            row[arm] = dict(mot=mot, states=sha(states), weak_events=sha(events), first_commit=min(committed, default=None),
                            commits=len(committed), frames=length, warp_sha256=camera["source"]["sha256"])
        start = row[NAMES[1]]["first_commit"]
        prefix = length if start is None else start - 1
        files = [folder / arm / "states" / METHOD / (seq + ".sha256") for arm in NAMES]
        states = [p.read_bytes().splitlines() for p in files]
        if any(len(lines) != length for lines in states) or states[0][:prefix] != states[1][:prefix]:
            raise ValueError(f"W0 altered original state before its first weak update: {seq}")
        if start is None and row[NAMES[0]]["mot"] != row[NAMES[1]]["mot"]:
            raise ValueError(f"Zero-commit trajectory differs: {seq}")
        row["prefix_verified"] = prefix
    write(folder / "verified.json", dict(status="PASS", sequences=lengths, native_MOT_parity=True, results=result))
    return protocols


def evaluate(source, archived, output, full, protocols, reference, env, decisions):
    from tools.research.wem_r41.assets import read, sha, write
    from experiments.vttiny_gmc.round3_joint.common import assert_same_inputs

    receipts = read(archived / "verified.json")
    old_done = read(archived / "complete.json")
    old_eval = read(archived / "evaluation_complete.json")
    if old_done["checks"] != sha(archived / "verified.json") or old_eval["files"]["metrics/summary.json"] != sha(archived / "metrics/summary.json"):
        raise ValueError("Archived R4.1 metric or result receipt differs")
    inputs = read(archived / "evaluation_inputs.json")
    view = output / "eval_input"
    sequences = list(inputs["sequences"])
    if inputs["sequences"] != reference["sequences"]:
        raise ValueError("Evaluation scope differs from the 30-sequence native reference")
    archived_protocol_sha = {}
    for arm in (*NAMES, *HISTORICAL):
        if arm in protocols:
            folder, method, protocol = full / arm, METHOD, protocols[arm]
        else:
            folder = archived / arm
            method = receipts["arms"][arm]["method"]
            path = folder / "protocols" / (method + ".json")
            valid = read(folder / "arm_complete.json")
            if valid["files"][f"protocols/{method}.json"] != sha(path):
                raise ValueError(f"Imported protocol changed: {arm}")
            archived_protocol_sha[arm] = sha(path)
            protocol = read(path)
        assert_same_inputs(reference, protocol, sequences)
        protocol["name"] = protocol["config"]["name"] = arm
        dest = view / "trackers" / arm / "data"
        dest.mkdir(parents=True)
        for seq in sequences:
            path = folder / "trackers" / method / "data" / (seq + ".txt")
            expected = protocol["output_prediction_hashes"][seq]
            if sha(path) != expected or (arm in HISTORICAL and receipts["arms"][arm]["per_sequence"][seq]["mot"] != expected):
                raise ValueError(f"MOT fingerprint differs: {arm}/{seq}")
            shutil.copyfile(path, dest / (seq + ".txt"))
        write(view / "protocols" / (arm + ".json"), protocol)
    for seq in sequences:
        path = Path(inputs["ground_truth"]["root"]) / seq / "gt/gt.txt"
        if sha(path) != inputs["ground_truth"]["sha256"][seq]:
            raise ValueError(f"Official GT changed: {seq}")
    args = [sys.executable, "-B", str(source / "tools/evaluate_botsort_plus.py"), "--run-dir", str(view),
            "--trackers", *NAMES, *HISTORICAL, "--reference-tracker", NAMES[0], "--gt-root", inputs["ground_truth"]["root"],
            "--trackeval-root", inputs["trackeval"]["root"], "--output-dir", str(output / "metrics")]
    with (output / "evaluate.log").open("xb") as log:
        subprocess.run(args, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    def indexed(path):
        return {(item["tracker"], item["sequence"]): item for item in read(path)["results"]}

    current = indexed(output / "metrics/summary.json")
    previous = indexed(archived / "metrics/summary.json")
    native_previous = indexed(Path(decisions["run_dir"]) / "metrics/summary.json")
    for seq in [*sequences, "COMBINED_SEQ"]:
        for arm in HISTORICAL:
            if current[arm, seq] != previous[arm, seq]:
                raise ValueError(f"Scheduled re-evaluation failed to reproduce {arm}/{seq}")
        if {k: v for k, v in current[NAMES[0], seq].items() if k != "tracker"} != {
                k: v for k, v in native_previous["native", seq].items() if k != "tracker"}:
            raise ValueError(f"Native official metric differs from archived original: {seq}")
    write(output / "complete.json", dict(status="COMPLETE", sequence_count=len(sequences),
        frames_per_arm=sum(inputs["sequences"].values()), new_replays=list(NAMES), retained_references=list(HISTORICAL),
        native_trajectory_parity="PASS", prior_metric_parity="PASS", archived_protocol_sha256=archived_protocol_sha,
        frozen_evaluation_input_sha256=sha(archived / "evaluation_inputs.json"),
        full_verification_sha256=sha(full / "verified.json"), metric_sha256=sha(output / "metrics/summary.json")))
    print("COMPLETE", len(sequences), "sequences; six paired official results", flush=True)


def run(source, archived, output):
    from tools.research.wem_r41.assets import read, sha, write
    from experiments.vttiny_gmc.round3_joint.common import child_environment

    bundle_path = archived / "prediction_bundle.json"
    bundle = read(bundle_path)
    settings = bundle["settings"]
    reference_source = settings["warp_sources"]["native"]
    reference_path = Path(reference_source["run_dir"]) / "protocols" / (reference_source["method"] + ".json")
    if sha(reference_path) != REFERENCE_PROTOCOL_SHA:
        raise ValueError("Original native reference protocol differs")
    reference = read(reference_path)
    names = list(bundle["sequences"])
    if len(names) != 30 or sum(bundle["sequences"].values()) != 12590 or bundle["sequences"] != reference["sequences"]:
        raise ValueError("Expected the identical official 30 sequences and 12,590 frames")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "plan.json", dict(sequences=bundle["sequences"], smoke=SMOKE, toggles=MODES,
        native_reference_protocol_sha256=REFERENCE_PROTOCOL_SHA, prediction_bundle_sha256=sha(bundle_path),
        source_sha256=sha(__file__), additional_stage="strict R4.1 W0 on original BoxMOT native SOF/ReID"))
    env = child_environment(settings)
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONUTF8="1", PYTHONWARNINGS="ignore")
    for stage, chosen in (("smoke", SMOKE), ("full", names)):
        folder = output / stage
        folder.mkdir()

        def replay(mode):
            arm = f"native_{mode}"
            args = [sys.executable, "-B", "-u", __file__, "predict", "--source", str(source),
                    "--bundle", str(bundle_path), "--mode", mode, "--sequences", *chosen, "--output", str(folder / arm)]
            print(f"RUN {stage}/{arm}: {len(chosen)} sequences", flush=True)
            with (folder / (arm + ".log")).open("xb") as log:
                subprocess.run(args, cwd=source, env=dict(env, YOLO_CONFIG_DIR=str(folder / arm / "ultralytics")),
                               stdout=log, stderr=subprocess.STDOUT, check=True)

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(replay, MODES))
        protocols = check_predictions(folder, chosen, settings, reference)
        print(f"PASSED {stage}: native OFF parity and W0 pre-commit state parity", flush=True)
    evaluate(source, archived, output, folder, protocols, reference, env, reference_source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "predict"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--archived", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--sequences", nargs="+")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    if args.action == "predict":
        return predict(args.source, args.bundle, args.mode, args.sequences, args.output)
    run(args.source, args.archived, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
