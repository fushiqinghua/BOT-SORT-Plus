"""Replay the exact W R0 observer on the archived conservative TCO training split."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
INTEGRATIONS = ROOT.parent.parent
SUITE = INTEGRATIONS / "botsort_plus_v0_4_motion_blur"
REPO = SUITE / "BOT-SORT-Plus-followup-53faeccb"
R0 = INTEGRATIONS / "vttiny_wem_r01_v1/observer"
TRAIN = SUITE / "tco_53faeccb/t23_v1"
MINI_ROLES = ("off", "on")
METHOD = "B_early_moderate"
PINS = {
    "split/training_split.json": "43329496f50105b87a9175d312f70b65b1dd3e050307697b20a0656709398c04",
    "training_inputs/verified.json": "b69d05eea9c39e09e941d5a890addf0e87c78a2028eacab0accbb9ecb6ba81b1",
    "training_inputs/protocol.json": "0e0b621de526e278b5a8a9f251b26270ef993b5e4f5cdc921510cc2918260e1e",
    "training_runs/capture_manifest.json": "956e1d6e032fd17b7b155f5063574ac0ff5a5f41989a2a71e110b56fbc5e7fb1",
    "training_runs/B/protocols/B_early_moderate.json":
        "ddb7f476ad97d870aae3acf8aa1c071ef190846cd62fcb267f8105014b0aa69a",
}
OBS_SHA = "d6d17d93138181329be3283c12ffa5336ecd83ccf1e65a4488213670a380d33a"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def decoded_sha(path):
    with gzip.open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(".writing")
    scratch.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                       encoding="utf-8")
    scratch.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_hashes():
    return {"capture.py": sha(R0/"capture.py"), "training_capture.py": sha(Path(__file__))}


def check_input(p, expected):
    require(sha(p) == expected, f"Frozen source differs: {p}")


def verify_settings(settings, assets):
    allowed = {"schema", "baseline_root", "features_dir", "images_root", "runtime_root",
               "vendor_dir", "gmc_mode", "python_user_base", "cuda_visible_devices"}
    require(not (set(settings) - allowed), "Online settings may not include GT or evaluation inputs")
    require(settings["gmc_mode"] == "live" and Path(settings["baseline_root"]).resolve() == assets.resolve()
            and Path(settings["features_dir"]).resolve() == (assets/"features").resolve(),
            "Prediction settings differ from the historical training replay")


def reference():
    for name, expected in PINS.items():
        check_input(TRAIN/name, expected)
    check_input(R0/"capture.py", OBS_SHA)
    split = read(TRAIN/"split/training_split.json")
    receipt = read(TRAIN/"training_runs/capture_manifest.json")
    frozen = read(TRAIN/"training_inputs/verified.json")
    ref_path = TRAIN / "training_runs/B/protocols" / (METHOD + ".json")
    ref = read(ref_path)
    partitions = {name: split["split"][name]["sequences"] for name in ("fit", "calibration", "holdout")}
    lengths = {seq: length for partition in partitions.values() for seq, length in partition.items()}
    groups = split["training_sequence_to_source_video"]
    validation = split["development_sequence_to_source_video"]
    names = set(lengths)
    training_groups = [set(groups[seq] for seq in data) for data in partitions.values()]
    require(split["status"] == "LOCKED" and receipt["status"] == "PASS" and frozen["status"] == "PASS"
            and ref["status"] == "COMPLETE" and not receipt["labels_in_prediction"], "Invalid archival status")
    require((len(names), sum(lengths.values())) == (74, 28931)
            and sum(map(len, partitions.values())) == 74 and names == groups.keys()
            and len(set.union(*training_groups)) == sum(map(len, training_groups))
            and not (set(groups.values()) & set(validation.values())) and not (names & validation.keys()),
            "Training source-proxy assignments are not disjoint from the locked 30 sequences")
    require(lengths == ref["sequences"] == receipt["sequences"] and receipt["split_sha256"] == PINS["split/training_split.json"]
            and receipt["protocol_sha256"] == sha(ref_path)
            and ref["source_protocol_sha256"] == PINS["training_inputs/protocol.json"]
            and ref["detector_sha256"] == frozen["detector_sha256"]
            and ref["rgb_encoder_sha256"] == frozen["rgb_encoder_sha256"], "Reference asset identity differs")
    settings_file = TRAIN / "training_runs/prediction_settings.json"
    settings = read(settings_file)
    verify_settings(settings, TRAIN/"training_inputs")
    dest = ROOT / "prediction_settings.json"
    if dest.exists():
        check_input(dest, sha(settings_file))
    else:
        dest.write_bytes(settings_file.read_bytes())
    return lengths, partitions, groups, split, ref, frozen, settings


def verify_source_files(names, lengths, ref, frozen):
    inputs = TRAIN/"training_inputs"
    out = {}
    for seq in names:
        receipt = frozen["per_sequence"][seq]
        det = inputs/"detections"/(seq+".json.gz")
        feat = inputs/"features"/(seq+".npz")
        meta = inputs/"features"/(seq+".json")
        for path, key in ((det, "detection_sha256"), (feat, "rgb_features_sha256"), (meta, "rgb_metadata_sha256")):
            check_input(path, receipt[key])
        require(receipt["frames"] == lengths[seq] and
                receipt["ordered_detection_fingerprint"] == ref["detection_fingerprints"][seq] and
                receipt["rgb_features_sha256"] == ref["feature_hashes"][seq]["features"] and
                receipt["rgb_metadata_sha256"] == ref["feature_hashes"][seq]["metadata"],
                f"Reference detector/ReID fingerprint differs for {seq}")
        out[seq] = {"frames": lengths[seq], **{key: receipt[key] for key in (
            "detection_sha256", "ordered_detection_fingerprint", "rgb_features_sha256", "rgb_metadata_sha256"
        )}}
    return out


def child(opts):
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(R0))
    from capture import overlays

    check_input(R0/"capture.py", OBS_SHA)
    params = read(ROOT/"prediction_settings.json")
    verify_settings(params, TRAIN/"training_inputs")
    worker = overlays()
    worker.__dict__["_wem_mode"] = opts.mode
    return worker.run(params, METHOD, opts.output, opts.sequences, probe_mode="off")


def launch(stage, mode, identifier, names, settings):
    sys.path.insert(0, str(REPO))
    from experiments.vttiny_gmc.round3_joint.common import child_environment

    parent = ROOT / stage / "shards" / mode
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / identifier
    log = parent / (identifier + ".log")
    success = parent / (identifier + ".json")
    expected = dict(status="PASS", mode=mode, sequences=names, code=source_hashes())
    if success.is_file() and read(success) == expected:
        return output
    if log.exists() or output.exists():
        archived = ROOT / "superseded" / f"{stage}_{mode}_{identifier}_{time.time_ns()}"
        archived.mkdir(parents=True)
        for item in (output, log, success):
            if item.exists():
                item.rename(archived / item.name)
    env = child_environment(settings)
    env.update(CUDA_VISIBLE_DEVICES="", YOLO_CONFIG_DIR=str(parent/(identifier+"_runtime_config")))
    with log.open("xb") as stream:
        subprocess.run([sys.executable,"-Bu",str(Path(__file__)), "child", "--mode", mode,
                        "--output", str(output), "--sequences", *names],
                       cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    write(success, expected)
    return output


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        os.link(source, target)
    else:
        require(sha(source) == sha(target), f"Canonical file already differs: {target}")


def parity(stage, shards, lengths, ref, inputs):
    from experiments.vttiny_gmc.round3_joint.common import assert_same_inputs

    historical = TRAIN/"training_runs/B"
    result = {}
    for names, folders in shards:
        rows = {mode: read(folder/"protocols"/(METHOD+".json")) for mode, folder in folders.items()}
        for mode in MINI_ROLES:
            require(rows[mode]["status"] == "COMPLETE" and set(rows[mode]["sequences"]) == set(names),
                    "Finished prediction has wrong sequence scope")
            assert_same_inputs(ref, rows[mode], names)
            require(rows[mode]["implementation_sha256"] == ref["implementation_sha256"]
                    and rows[mode]["boxmot_source_hashes"] == ref["boxmot_source_hashes"],
                    "Tracker implementation differs from historical B")
            require(rows[mode]["joint_batch"]["camera"] == "native"
                    and rows[mode]["joint_batch"]["gmc_mode"] == "live",
                    "Unexpected B GMC configuration")
        for seq in names:
            mot_rel = Path("trackers")/METHOD/"data"/(seq+".txt")
            evt_rel = Path("events")/METHOD/(seq+".jsonl")
            cam_rel = Path("warps")/METHOD/(seq+".jsonl.gz")
            state_rel = Path("states")/METHOD/(seq+".sha256")
            snap_rel = Path("snapshots")/METHOD/(seq+".jsonl.gz")
            owner_rel = Path("candidate_ownership")/METHOD/(seq+".jsonl.gz")
            receipt_rel = Path("capture_receipts")/METHOD/(seq+".json")
            expected_mot = ref["output_prediction_hashes"][seq]
            check_input(historical/mot_rel, expected_mot)
            expected_event = sha(historical/evt_rel)
            expected_camera = decoded_sha(historical/cam_rel)
            off, on = (folders[mode] for mode in MINI_ROLES)
            off_states, on_states = sha(off/state_rel), sha(on/state_rel)
            require(off_states == on_states, f"R0 formal state chain differs for {seq}")
            receipts = {}
            for mode, folder in folders.items():
                check_input(folder/mot_rel, expected_mot)
                check_input(folder/evt_rel, expected_event)
                require(decoded_sha(folder/cam_rel) == expected_camera,
                        f"Reference selected/raw/backup GMC frame records differ for {seq}")
                receipt = read(folder/receipt_rel)
                require(receipt["counts"]["frames"] == lengths[seq] and not receipt["gt_open_allowed"]
                        and not receipt["snapshot_applied"] and receipt["source"]["detection_sha256"] == inputs[seq]["detection_sha256"],
                        f"Invalid observer record for {seq}")
                receipts[mode] = receipt
            observed = receipts["on"]
            check_input(on/snap_rel, observed["snapshots_sha256"])
            check_input(on/owner_rel, observed["ownership_sha256"])
            canonical = ROOT/stage/"on/B"
            for rel in (mot_rel, evt_rel, cam_rel, state_rel, snap_rel, owner_rel, receipt_rel):
                link(on/rel, canonical/rel)
            result[seq] = dict(frames=lengths[seq], mot_sha256=expected_mot, events_sha256=expected_event,
                               state_chain_sha256=on_states, canonical_warps_decoded_sha256=expected_camera,
                               snapshot_file=str(canonical/snap_rel), ownership_file=str(canonical/owner_rel),
                               observation_proof=observed, frozen_inputs=inputs[seq])
    require(len(result) == len(inputs), "Shards failed to cover their requested inputs")
    return result


def balanced(names, lengths, n):
    shards = [[] for _ in range(n)]
    totals = [0] * n
    for seq in sorted(names, key=lambda s: (-lengths[s], s)):
        index = min(range(n), key=lambda i: totals[i])
        shards[index].append(seq)
        totals[index] += lengths[seq]
    return [sorted(s) for s in shards if s]


def run(stage):
    sys.path.insert(0, str(REPO))
    lengths, partitions, groups, split, ref, frozen, settings = reference()
    smoke = [min(part) for part in partitions.values()]
    if stage == "full":
        previous = read(ROOT/"parity/smoke.json")
        require(previous["status"] == "PASS" and previous["sequences"] == smoke and
                previous["code"] == source_hashes() and previous["settings_sha256"] == sha(ROOT/"prediction_settings.json"),
                "Full corpus requires smoke parity on the same code and settings")
    names = smoke if stage == "smoke" else sorted(lengths)
    source_files = verify_source_files(names, lengths, ref, frozen)
    assignments = balanced(names, lengths, 1 if stage == "smoke" else 6)
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [(chosen, {mode: pool.submit(launch, stage, mode, f"{i+1:02}", chosen, settings)
                          for mode in MINI_ROLES}) for i, chosen in enumerate(assignments)]
        shards = [(chosen, {mode: job.result() for mode, job in modes.items()}) for chosen, modes in jobs]
    results = parity(stage, shards, lengths, ref, source_files)
    frozen_result = dict(status="PASS", stage=stage, sequences=names,
                         frames_per_role=sum(lengths[s] for s in names), modes=list(MINI_ROLES),
                         references={"B": {"method": METHOD, "files": results}},
                         split_provenance=dict(proxy=True,
                                               source_video_provenance_verified=split["source_video_provenance_verified"],
                                               group_map={s:groups[s] for s in names},
                                               partitions={part: {s:data[s] for s in names if s in data}
                                                           for part,data in partitions.items()}),
                         source_pins=PINS, code=source_hashes(), online_w_updates=0, gt_open_allowed=False,
                         settings_sha256=sha(ROOT/"prediction_settings.json"))
    write(ROOT/"parity"/(stage+".json"), frozen_result)
    print(f'{stage} PASS {len(names)} clips / {frozen_result["frames_per_role"]} frames', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke","full","child"))
    parser.add_argument("--mode", choices=MINI_ROLES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sequences", nargs="+")
    opts = parser.parse_args()
    if opts.stage == "child":
        raise SystemExit(child(opts))
    run(opts.stage)
