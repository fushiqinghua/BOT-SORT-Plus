"""Run the frozen two-reference TCO observation audit with an explicit parity gate."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import subprocess
import sys

SRC = Path(__file__).resolve().parent
ROLES = {"B": "B_early_moderate", "BG": "evidence_both"}
PIN = "53faeccba267d8988174c18e1a538f759fa6dd07"


def reference_inputs(settings, paths):
    expected = common.read_json(BASELINE_JOB / "outputs/j0/manifest.json")
    assert expected["status"] == "PASS"
    lengths = common.sequence_lengths(settings)
    assert lengths == expected["sequences"] and len(lengths) == 30 and sum(lengths.values()) == 12590
    protocols, records = {}, {}
    for role, name in ROLES.items():
        prior = "B" if role == "B" else "evidence"
        ref = Path(paths["references"][role]["run_dir"])
        metadata = expected["references"][prior]
        protocol_file = ref / "protocols" / (name + ".json")
        protocol = validate_runs(ref, [name])["protocols"][name]
        assert metadata["protocol_sha256"] == common.sha256(protocol_file), role
        assert metadata["prediction_sha256"] == protocol["output_prediction_hashes"], role
        protocols[role] = protocol
        records[role] = {"tracker": name, "run_dir": str(ref), "protocol_sha256": common.sha256(protocol_file),
                         "trajectory_hashes": protocol["output_prediction_hashes"],
                         "config": protocol["config"], "implementation_sha256": protocol.get("implementation_sha256"),
                         "boxmot_source_hashes": protocol.get("boxmot_source_hashes"),
                         "camera_source": protocol.get("camera_manifest", protocol.get("case_fix"))}
    common.assert_same_inputs(protocols["B"], protocols["BG"], lengths)
    gt = validate_gt(paths["paths"]["gt_root"], lengths)
    evaluator = verify_trackeval(paths["paths"]["trackeval_root"], REPO)
    assert gt["sha256"] == expected["ground_truth"]["sha256"]
    assert evaluator["commit"] == expected["trackeval"]["commit"]
    baseline = protocols["B"]
    common.write_json(T0 / "input_manifest.json", {
        "schema": "tco_v0.1_input_manifest", "sequences": lengths, "frames": sum(lengths.values()),
        "ground_truth": gt, "trackeval": evaluator, "detector_sha256": baseline["detector_sha256"],
        "detection_fingerprints": baseline["detection_fingerprints"], "rgb_encoder_sha256": baseline["rgb_encoder_sha256"],
        "features": baseline["feature_hashes"], "decoded_rgb": baseline["decoded_image_hashes"],
        "groups_source": settings["grouping_provenance"], "cache_mode": settings["gmc_mode"],
        "warp_sources": settings["warp_sources"], "source_protocol_sha256": baseline["source_protocol_sha256"],
        "score_bands": {"low_exclusive": baseline["native_config"]["track_low_thresh"],
                        "high_exclusive": baseline["native_config"]["track_high_thresh"]},
    })
    common.write_json(T0 / "reference_manifest.json", {"commit": PIN, "references": records})
    return lengths, protocols


def launch(settings, stage, mode, role, names):
    method = ROLES[role]
    dest = ROOT / stage / mode / role
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        protocol_file = dest / "protocols" / (method + ".json")
        if protocol_file.is_file() and common.read_json(protocol_file)["status"] == "COMPLETE":
            return
        raise RuntimeError(f"An interrupted replay occupies {dest}; use a fresh batch directory")
    env = common.child_environment(settings)
    env.update(PYTHONPATH=os.pathsep.join([str(SRC), env["PYTHONPATH"]]), CUDA_VISIBLE_DEVICES="",
               YOLO_CONFIG_DIR=str(ROOT / "runtime_config"))
    command = [sys.executable, "-Bu", str(SRC / "collector.py"), "--settings", str(ROOT / "prediction_settings.json"),
               "--method", method, "--output-dir", str(dest), "--observer", mode,
               "--groups", settings["grouping_provenance"]["path"], "--sequences", *names]
    with (dest.parent / (role + ".log")).open("xb") as log:
        call = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    if call.returncode:
        raise RuntimeError(f"{stage}/{mode}/{role} failed; see {dest.parent / (role + '.log')}")


def check_parity(stage, names, protocols):
    output = {}
    for role, method in ROLES.items():
        root = ROOT / stage
        a, b = root / "off" / role, root / "on" / role
        reports = [validate_runs(p, [method])["protocols"][method] for p in (a, b)]
        ref = protocols[role]
        for report in reports:
            common.assert_same_inputs(ref, report, names)
        checks = {}
        for seq in names:
            expected = ref["output_prediction_hashes"][seq]
            ae, be = (p / "events" / method / (seq + ".jsonl") for p in (a, b))
            ad, bd = (p / "tco_states" / method / (seq + ".sha256") for p in (a, b))
            assert all(x["output_prediction_hashes"][seq] == expected for x in reports), (stage, role, seq, "MOT")
            assert common.sha256(ae) == common.sha256(be), (stage, role, seq, "events")
            assert common.sha256(ad) == common.sha256(bd), (stage, role, seq, "state-chain")
            trace = b / "tco_capture" / method / (seq + ".jsonl.gz")
            assert trace.is_file() and not (a / trace.relative_to(b)).exists(), (stage, role, seq, "capture")
            checks[seq] = {"MOT": expected, "events": common.sha256(ae), "state_chain": common.sha256(ad),
                           "runtime_capture": common.sha256(trace)}
        output[role] = {"method": method, "files": checks,
                        "runtime": reports[1].get("runtime", reports[1].get("boxmot_source_hashes"))}
    result = {"status": "PASS", "sequences": list(names), "references": output,
              "collector_sha256": common.sha256(SRC / "collector.py"),
              "settings_sha256": common.sha256(ROOT / "prediction_settings.json")}
    common.write_json(T0 / (stage + "_parity.json"), result)
    return result


def execute_stage(settings, stage, names, protocols):
    jobs = [(mode, role) for mode in ("off", "on") for role in ROLES]
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        scheduled = [pool.submit(launch, settings, stage, mode, role, names) for mode, role in jobs]
        for job in scheduled:
            job.result()
    return check_parity(stage, names, protocols)


def split_inputs(settings, stream):
    split = T1 / "split_manifest.json"
    if not split.is_file():
        T1.mkdir(exist_ok=True)
        subprocess.run([sys.executable, "-Bu", str(SRC / "split_t1.py"), "manifest", "--settings", str(LOCAL_CONFIG),
                        "--grouping", settings["grouping_provenance"]["path"], "--output", str(split)],
                       cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return split


def run_labeling(config, settings):
    parity = common.read_json(T0 / "T0_parity.json")
    if parity["status"] != "PASS" or len(parity["full"]["sequences"]) != 30:
        raise RuntimeError("Offline labeling requires the verified full run")
    env = common.child_environment(settings)
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), env["PYTHONPATH"]])
    log_path = ROOT / "label_t1.log"
    with log_path.open("ab") as stream:
        split = split_inputs(settings, stream)
        subprocess.run([sys.executable, "-Bu", str(SRC / "label_t1.py"),
                        "--run-root", str(ROOT / "full/on"), "--output", str(T1), "--config", str(config),
                        "--split-manifest", str(split)],
                       cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        review = T1 / "review_sample.csv"
        if not review.is_file():
            subprocess.run([sys.executable, "-Bu", str(SRC / "split_t1.py"), "review",
                            "--episodes", str(T1 / "episodes.jsonl.gz"), "--manifest", str(split),
                            "--output", str(review)], cwd=REPO, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        subprocess.run([sys.executable, "-Bu", str(SRC / "verify_labels.py"), str(T1),
                        str(BASELINE_JOB/"outputs/j2_evaluation/metrics/summary.json"),
                        str(T0/"input_manifest.json")], cwd=REPO, env=env,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    global ROOT, REPO, BASELINE_JOB, LOCAL_CONFIG, T0, T1
    global common, validate_gt, validate_runs, verify_trackeval

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["inventory", "smoke", "full", "labels"])
    parser.add_argument("--repo", type=Path, required=True, help=f"Checkout at {PIN}")
    parser.add_argument("--round3-job", type=Path, required=True, help="Archived Round 3 settings and J0/J2 verification outputs")
    parser.add_argument("--tco-config", type=Path, required=True, help="Local TCO config with absolute asset paths")
    parser.add_argument("--output-root", type=Path, required=True, help="Run directory shared by inventory, smoke, full and labels")
    args = parser.parse_args()
    ROOT, REPO = args.output_root.resolve(), args.repo.resolve()
    BASELINE_JOB, LOCAL_CONFIG = args.round3_job.resolve(), args.tco_config.resolve()
    T0, T1 = ROOT / "T0", ROOT / "T1"
    actual = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if actual != PIN:
        raise ValueError(f"Expected upstream commit {PIN}; found {actual}")
    sys.path.insert(0, str(REPO))
    from experiments.vttiny_gmc.round3_joint import common
    from tools.evaluate_botsort_plus import validate_gt, validate_runs, verify_trackeval

    settings = common.load_settings(BASELINE_JOB / "settings.json")
    paths = common.read_json(LOCAL_CONFIG)
    for value in [*paths["paths"].values(), *(entry["run_dir"] for entry in paths["references"].values())]:
        if not value or not Path(value).is_absolute():
            parser.error("The TCO data and reference paths in --tco-config must be absolute")
    paths["development_source_groups"] = settings["original_video_groups"]
    active = ROOT / "state.json"
    if args.stage == "labels":
        run_labeling(ROOT / "offline_settings.json", settings)
        common.write_json(active, {"status": "T1_OFFLINE_LABELS_COMPLETE"})
        return
    config = ROOT / "prediction_settings.json"
    predictions = common.prediction_settings(settings)
    if predictions["gmc_mode"] != "cached":
        raise ValueError("The archived T0/T1 run requires pinned cached GMC warps")
    if not settings.get("grouping_provenance", {}).get("path"):
        raise ValueError("The archived Round 3 settings must retain the explicit grouping file provenance")
    common.write_json(config, predictions)
    common.write_json(ROOT / "offline_settings.json", paths)
    lens, protocols = reference_inputs(settings, paths)
    if args.stage == "inventory":
        split = split_inputs(settings, None)
        print("Archived input contracts verified", len(lens), sum(lens.values()), split, flush=True)
        return
    common.write_json(T0 / "runtime_manifest.json", {
        "commit": PIN, "python": sys.version, "python_path": sys.executable,
        "source_sha256": {file.name: common.sha256(file) for file in sorted(SRC.glob("*.py"))},
        "pinned_dependencies": common.source_hashes(), "prediction_keys": sorted(predictions),
        "auxiliary_sampler_logs": True, "tracking_metrics_changed": False,
    })
    if args.stage == "smoke":
        common.write_json(active, {"status": "SMOKE_RUNNING"})
        execute_stage(settings, "smoke", list(common.SMOKE), protocols)
        common.write_json(active, {"status": "SMOKE_PASS"})
    else:
        smoke = common.read_json(T0 / "smoke_parity.json")
        assert smoke["status"] == "PASS" and smoke["sequences"] == list(common.SMOKE)
        assert smoke["collector_sha256"] == common.sha256(SRC / "collector.py")
        assert smoke["settings_sha256"] == common.sha256(config)
        common.write_json(active, {"status": "FULL_RUNNING"})
        full = execute_stage(settings, "full", list(lens), protocols)
        common.write_json(T0 / "T0_parity.json", {"status": "PASS", "smoke": smoke, "full": full})
        common.write_json(active, {"status": "FULL_PARITY_PASS"})
    print(common.read_json(active), flush=True)


if __name__ == "__main__":
    main()
