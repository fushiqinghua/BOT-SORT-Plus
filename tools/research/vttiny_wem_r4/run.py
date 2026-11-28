"""Isolated exploratory W R4 runner and frozen-cache evaluation."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from capture import ADAPTER_BLOB, WORKER_BLOB, pinned_source, replace_once, sha


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
INTEGRATIONS = ROOT.parent
REPO = INTEGRATIONS / "botsort_plus_v0_4_motion_blur/BOT-SORT-Plus-followup-53faeccb"
R0 = INTEGRATIONS / "vttiny_wem_r01_v1/observer"
TCO = INTEGRATIONS / "botsort_plus_v0_4_motion_blur/tco_53faeccb/t01_v1"
LOCK = INTEGRATIONS / "vttiny_wem_r3_v1/feature_analysis/scores/full/calibration_lock.json"
WEIGHTS = INTEGRATIONS.parent / "outputs/yolo11m_rgb_rgbt_mot_test_v1/rgbt/weights/clip_veri.pt"
LOCK_SHA = "2cc752bd7d7f450d70d828b630d81892351f36d691267962441ee8ae87556719"
SMOKE = ("DJI_0030_2", "DJI_0061_2", "DJI_0327_3")
METHODS = {"B": "B_early_moderate", "BG": "evidence_both"}
PARITY = tuple((role, mode) for role in METHODS for mode in ("off", "shadow", "force_reject"))
ACTIVE = (("B", "lower_low"), ("B", "plain"), ("B", "coast"), ("B", "w_coast"), ("B", "w_full"),
          ("BG", "w_full"))

sys.path.insert(0, str(REPO))


def predictor(mode):
    from online import ClosedLoop, RGB_MODES
    from score_features import RGB_SHA

    adapter = importlib.import_module("botsort_plus.adapter_v2")
    source = pinned_source(adapter, ADAPTER_BLOB).decode()
    source = replace_once(source, "    STrack.multi_predict(pool)\n",
                          "    STrack.multi_predict(pool)\n    tracker.wem.pre_gmc(pool, unconfirmed, high, low)\n")
    source = replace_once(source, "    STrack.multi_gmc(unconfirmed, warp)\n",
                          "    STrack.multi_gmc(unconfirmed, warp)\n"
                          "    tracker.wem.compensated(tracker, pool, unconfirmed, warp)\n")
    source = replace_once(source, "    output = tracker._prepare_output(activated, refound, lost, removed).reshape(-1, 8)\n",
                          "    output = tracker.wem.finish_before_output(tracker, activated, refound, lost, removed,\n"
                          "                                             reserved, consumed, all_dets)\n")
    exec(compile(source, adapter.__file__, "exec"), adapter.__dict__)
    worker = importlib.import_module("experiments.vttiny_gmc.round3_joint.worker")
    source = pinned_source(worker, WORKER_BLOB).decode()
    if mode == "lower_low":
        source = replace_once(source, "                self.cmc.apply = self.warp\n",
                              "                self.cmc.apply = self.warp\n"
                              "                self.track_low_thresh = .05\n")
    source = replace_once(source, "        sequence_iter = iter(paths)\n",
                          "        sequence_iter = iter(paths)\n"
                          "        wem_encoder = _encoder(settings) if _observe_mode in _rgb_modes else None\n")
    source = replace_once(source, "                self.audit_closed = False\n",
                          "                self.audit_closed = False\n"
                          "                self.wem = _closed_loop(paths[self.seq], self.seq, method, output, protocol,\n"
                          "                                       _observe_mode, settings, wem_encoder)\n")
    source = replace_once(source, "                result = super().update(detections, image, *a, **kw)\n",
                          "                self.wem.start(self, detections, image, kw['features'])\n"
                          "                result = super().update(detections, image, *a, **kw)\n"
                          "                self.wem.finish(self, result)\n")
    source = replace_once(source, "            def finish_sequence_events(self):\n",
                          "            def finish_sequence_events(self):\n                self.wem.close()\n")

    def load_encoder(settings):
        from botsort_plus.feature_export import load_rgb_encoder
        if sha(WEIGHTS) != RGB_SHA:
            raise ValueError("Frozen RGB ReID weight fingerprint differs")
        encoder, meta = load_rgb_encoder(WEIGHTS, "cuda:0", Path(settings["runtime_root"]))
        if meta["sha256"] != RGB_SHA:
            raise ValueError("Loaded a different RGB encoder")
        return encoder

    worker.__dict__.update(_closed_loop=ClosedLoop, _encoder=load_encoder, _rgb_modes=RGB_MODES,
                           _observe_mode="off" if mode == "lower_low" else mode)
    exec(compile(source, worker.__file__, "exec"), worker.__dict__)
    return worker


def source_settings():
    from experiments.vttiny_gmc.round3_joint.common import read_json
    settings = read_json(R0 / "prediction_settings.json")
    if sha(LOCK) != LOCK_SHA or "gt_root" in settings or settings["gmc_mode"] != "cached":
        raise ValueError("Expected pinned reject-all conclusion and frozen prediction-only settings")
    return settings


def execute_child(args):
    from experiments.vttiny_gmc.round3_joint.common import read_json, write_json
    worker = predictor(args.mode)
    method = METHODS[args.role]
    rc = worker.run(source_settings(), method, args.output, args.sequences, probe_mode="off")
    if rc == 0 and args.mode == "lower_low":
        path = args.output / "protocols" / (method + ".json")
        protocol = read_json(path)
        if protocol["native_config"]["track_low_thresh"] != .1:
            raise ValueError("Low-threshold control must start from the frozen 0.10 tracker")
        protocol["native_config"]["track_low_thresh"] = .05
        protocol["wem_low_control"] = {"before": .1, "effective": .05, "camera_fingerprints_use_original_settings": True}
        write_json(path, protocol)
    return rc


def runtime(settings, gpu, enabled):
    from experiments.vttiny_gmc.round3_joint.common import child_environment
    env = child_environment(settings)
    env.update(CUDA_VISIBLE_DEVICES=gpu if enabled else "", PYTHONDONTWRITEBYTECODE="1",
               PYTHONWARNINGS="ignore", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    return env


def track(batch, role, mode, names, settings, gpu):
    from online import RGB_MODES
    folder = batch / (role + "_" + mode)
    log = folder.with_suffix(".log")
    with log.open("xb") as f:
        env = runtime(settings, gpu, mode in RGB_MODES)
        env["YOLO_CONFIG_DIR"] = str(batch / "runtime" / folder.name)
        subprocess.run([sys.executable, "-Bu", str(Path(__file__)), "child", "--output", str(folder),
                        "--role", role, "--mode", mode, "--sequences", *names],
                       env=env, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, check=True)
    print("completed", batch.name, role, mode, len(names), flush=True)


def inspect(batch, arms, names):
    from experiments.vttiny_gmc.round3_joint.common import assert_same_inputs, read_json, write_json
    baseline = read_json(TCO / "T0/reference_manifest.json")["references"]
    archived = read_json(R0 / "parity/full.json")
    if archived["status"] != "PASS" or not set(names) <= set(archived["sequences"]):
        raise ValueError("Pinned R0 proof does not cover experiment")
    checks = {}
    for role, mode in arms:
        key, method = role + "_" + mode, METHODS[role]
        folder = batch / key
        expected = read_json(Path(baseline[role]["run_dir"]) / "protocols" / (method + ".json"))
        actual = read_json(folder / "protocols" / (method + ".json"))
        checked = deepcopy(actual)
        if mode == "lower_low":
            assert actual["native_config"]["track_low_thresh"] == .05
            checked["native_config"]["track_low_thresh"] = expected["native_config"]["track_low_thresh"]
        assert_same_inputs(expected, checked, names)
        if actual["status"] != "COMPLETE" or set(actual["sequences"]) != set(names):
            raise ValueError(f"Incomplete scope for {key}")
        details, totals = {}, Counter()
        for seq in names:
            receipts = folder / "capture_receipts" / method / (seq + ".json")
            row = read_json(receipts)
            prior = archived["references"][role]["files"][seq]
            mot = sha(folder / "trackers" / method / "data" / (seq + ".txt"))
            events = sha(folder / "events" / method / (seq + ".jsonl"))
            state = sha(folder / "states" / method / (seq + ".sha256"))
            if mode in ("off", "shadow", "force_reject"):
                if (mot, events, state) != (prior["mot_sha256"], prior["events_sha256"], prior["state_chain_sha256"]):
                    raise ValueError(f"R4 R0 parity differs: {key}/{seq}")
                if row["counts"].get("committed", 0):
                    raise ValueError(f"Read-only arm committed an observation: {key}/{seq}")
            if row["counts"]["frames"] != actual["sequences"][seq] or mot != actual["output_prediction_hashes"][seq]:
                raise ValueError(f"Frame or prediction mismatch {key}/{seq}")
            with (folder / "events" / method / (seq + ".jsonl")).open(encoding="utf-8") as f:
                import json
                b_count = sum(json.loads(line).get("event") == "recovered" for line in f)
            log = folder / "wem_events" / method / (seq + ".jsonl")
            timing = read_json(folder / "timing" / method / (seq + ".json"))
            details[seq] = dict(mot=mot, native_events=events, states=state, weak_events=sha(log),
                                counts=dict(row["counts"], B_recovered=b_count),
                                association_and_gmc_seconds=timing["association_and_gmc_seconds"],
                                replay_wall_seconds=timing["replay_wall_seconds"])
            totals.update(details[seq]["counts"])
        wall = sum(item["replay_wall_seconds"] for item in details.values())
        tracker_seconds = sum(item["association_and_gmc_seconds"] for item in details.values())
        checks[key] = dict(variant=mode, method=method, totals=totals, per_sequence=details,
                           per_frame_ms_cached=1000*tracker_seconds/totals["frames"],
                           replay_frames_per_second=totals["frames"]/wall)
    for role in METHODS:
        if (role, "shadow") in arms and (role, "force_reject") in arms:
            for seq in names:
                if checks[role+"_shadow"]["per_sequence"][seq]["weak_events"] != checks[role+"_force_reject"]["per_sequence"][seq]["weak_events"]:
                    raise ValueError(f"Shadow and force-reject proposal stream differs: {role}/{seq}")
    result = dict(status="PASS", scope=list(names), r3_lock=LOCK_SHA, arms=checks)
    write_json(batch / "verified.json", result)
    return result


def evaluate(batch, arms, names, settings):
    from experiments.vttiny_gmc.round3_joint.common import read_json, write_json
    inputs = read_json(TCO / "T0/input_manifest.json")
    evaluation = batch / "eval_input"
    trackers = []
    for role, mode in arms:
        tracker, method = role + "_" + mode, METHODS[role]
        folder = batch / tracker
        protocol = read_json(folder / "protocols" / (method + ".json"))
        protocol["config"]["name"] = tracker
        name = evaluation / "trackers" / tracker / "data"
        name.mkdir(parents=True)
        for seq in names:
            src = folder / "trackers" / method / "data" / (seq + ".txt")
            shutil.copyfile(src, name / src.name)
        write_json(evaluation / "protocols" / (tracker + ".json"), protocol)
        trackers.append(tracker)
    args = [sys.executable, str(REPO / "tools/evaluate_botsort_plus.py"), "--run-dir", str(evaluation),
            "--trackers", *trackers, "--reference-tracker", "B_off", "--gt-root", inputs["ground_truth"]["root"],
            "--trackeval-root", inputs["trackeval"]["root"], "--output-dir", str(batch / "metrics")]
    with (batch / "evaluate.log").open("xb") as f:
        subprocess.run(args, cwd=REPO, env=runtime(settings, "", False), stdout=f, stderr=subprocess.STDOUT, check=True)
    from offline import diagnose
    diagnose(batch, arms, names, inputs, METHODS)


def main(args):
    from experiments.vttiny_gmc.round3_joint.common import read_json, write_json
    from online import SETTING
    target = ROOT / args.stage
    settings = source_settings()
    full = read_json(R0 / "parity/full.json")["sequences"]
    if len(full) != 30:
        raise ValueError("Expected 30 frozen official sequences")
    names = full if args.stage == "full" else list(SMOKE)
    parity_arms = PARITY if args.stage == "smoke" else (("B", "off"), ("BG", "off"))
    plan = dict(stage=args.stage, role="exploratory_r4_user_override_r3_stop",
                r3_decision="reject_all", prior_gate_failed=True, steps=names, frozen_diagnostic_point=SETTING,
                inspected_ground_truth_for_selection=False, scope="official 30-sequence development benchmark")
    if args.stage == "full":
        if any(read_json(ROOT / phase / "complete.json")["status"] != "PASS" for phase in ("smoke", "pilot")):
            raise ValueError("Full replay requires smoke and pilot completion")
    runs = parity_arms if args.stage == "smoke" else parity_arms + ACTIVE
    if args.repair_lower:
        if args.stage != "pilot" or read_json(target/"plan.json") != plan or (target/"complete.json").exists():
            raise ValueError("Expected the unfinished, unchanged exploratory pilot")
        others = tuple(arm for arm in runs if arm != ("B", "lower_low"))
        inspect(target, others, names)
        backup = ROOT / "superseded" / f"pilot_lower_{time.time_ns()}"
        backup.mkdir(parents=True)
        for path in (target / "B_lower_low", target / "B_lower_low.log"):
            if path.exists():
                path.rename(backup/path.name)
        track(target, "B", "lower_low", names, settings, args.gpu)
    else:
        target.mkdir(parents=False)
        write_json(target / "plan.json", plan)
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(track, target, role, mode, names, settings, args.gpu) for role, mode in runs]
            for f in futures:
                f.result()
    results = inspect(target, runs, names)
    if args.stage != "smoke":
        evaluate(target, runs, names, settings)
    write_json(target / "complete.json", dict(status="PASS", sequence_count=len(names),
               checks=sha(target/"verified.json")))
    print("PASS", args.stage, {a: row["totals"].get("committed",0) for a,row in results["arms"].items()}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("smoke", "pilot", "full", "child"))
    p.add_argument("--gpu", default="4")
    p.add_argument("--role", choices=METHODS)
    p.add_argument("--mode", choices=sorted({v for _, v in PARITY + ACTIVE}))
    p.add_argument("--output", type=Path)
    p.add_argument("--sequences", nargs="+")
    p.add_argument("--repair-lower", action="store_true")
    a = p.parse_args()
    if a.stage == "child":
        sys.exit(execute_child(a))
    main(a)
