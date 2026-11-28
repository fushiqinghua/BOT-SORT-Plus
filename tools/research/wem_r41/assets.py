"""Local path resolution and frozen input contracts; no tracking imports."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys


ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "wem_r41_settings"
METHODS = {"B": "B_early_moderate", "BG": "evidence_both"}
MODES = ("off", "W0", "W1", "W2", "W3")
SMOKE = ("DJI_0030_2", "DJI_0061_2", "DJI_0327_3")
FULL_ARMS = tuple((role, mode) for role in METHODS for mode in MODES)
SMOKE_ARMS = tuple((role, mode) for role in METHODS for mode in ("off", "shadow", "force_reject"))
PREDICTION_KEYS = ("schema", "baseline_root", "features_dir", "images_root", "runtime_root",
                   "vendor_dir", "gmc_mode", "warp_sources", "python_user_base", "cuda_visible_devices")
SETTINGS = {
    "schema": SCHEMA,
    "paths": {
        "prediction_settings": None, "r0_parity": None,
        "reference_manifest": "results/vttiny/iterations/tco-v0.1/T0/reference_manifest.json",
        "evaluation_manifest": "results/vttiny/iterations/tco-v0.1/T0/input_manifest.json",
        "rgb_weights": None, "output_root": "runs/wem_r4_1",
    },
    "prediction_overrides": {}, "evaluation_overrides": {}, "reference_overrides": {},
    "runtime": {"python": None, "cuda_visible_devices": "0", "rgb_device": "cuda:0", "workers": 1},
}


def read(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".writing")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def absolute(value, *, required=True):
    if value is None or value == "":
        if required:
            raise ValueError("A required path is unset; edit the generated local settings")
        return None
    text = os.path.expandvars(os.path.expanduser(str(value)))
    if re.search(r"\$\{|\$[A-Za-z_]|%[A-Za-z_][A-Za-z0-9_]*%", text):
        raise ValueError(f"Unresolved environment variable in path: {value}")
    path = Path(text)
    return str((path if path.is_absolute() else ROOT / path).resolve())


def init(path, legacy_project_root=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Settings already exist: {path}")
    data = deepcopy(SETTINGS)
    if legacy_project_root:
        project = Path(absolute(legacy_project_root))
        integrations = project / "integrations"
        data["paths"].update(
            prediction_settings=str(integrations / "vttiny_wem_r01_v1/observer/prediction_settings.json"),
            r0_parity=str(integrations / "vttiny_wem_r01_v1/observer/parity/full.json"),
            reference_manifest=str(integrations / "botsort_plus_v0_4_motion_blur/tco_53faeccb/t01_v1/T0/reference_manifest.json"),
            evaluation_manifest=str(integrations / "botsort_plus_v0_4_motion_blur/tco_53faeccb/t01_v1/T0/input_manifest.json"),
            rgb_weights=str(project / "outputs/yolo11m_rgb_rgbt_mot_test_v1/rgbt/weights/clip_veri.pt"),
        )
    write(path, data)
    return data


def load(path):
    supplied = read(path)
    if supplied.get("schema") != SCHEMA or set(supplied) - set(SETTINGS):
        raise ValueError(f"Expected {SCHEMA} and documented top-level fields")
    data = deepcopy(SETTINGS)
    for key in ("paths", "runtime", "prediction_overrides", "evaluation_overrides", "reference_overrides"):
        if not isinstance(supplied.get(key, {}), dict):
            raise ValueError(f"{key} must be an object")
        data[key].update(supplied.get(key, {}))
    if set(data["paths"]) != set(SETTINGS["paths"]) or set(data["runtime"]) != set(SETTINGS["runtime"]):
        raise ValueError("Unknown paths/runtime field")
    data["paths"] = {key: absolute(value) for key, value in data["paths"].items()}
    runtime = data["runtime"]
    workers = runtime["workers"]
    if type(workers) is not int or workers < 1:
        raise ValueError("runtime.workers must be a positive integer")
    python = runtime["python"] or sys.executable
    runtime["python"] = shutil.which(python) or absolute(python)
    if not Path(runtime["python"]).is_file():
        raise FileNotFoundError(f"Python executable not found: {python}")
    if not isinstance(runtime["cuda_visible_devices"], str):
        raise ValueError("cuda_visible_devices must be a string")
    if not isinstance(runtime["rgb_device"], str) or not re.fullmatch(r"cpu|cuda:\d+", runtime["rgb_device"]):
        raise ValueError("rgb_device must be cpu or cuda:<visible index>")
    if set(data["prediction_overrides"]) - set(PREDICTION_KEYS):
        raise ValueError("Prediction overrides cannot contain evaluation/unknown keys")
    if set(data["evaluation_overrides"]) - {"gt_root", "trackeval_root"}:
        raise ValueError("Unknown evaluation override")
    if set(data["reference_overrides"]) - set(METHODS):
        raise ValueError("Unknown reference role")
    data["settings_source"] = {"path": str(Path(path).resolve()), "sha256": sha(path)}
    return data


def source_manifest():
    files = set()
    for folder in ("botsort_plus", "experiments/vttiny_gmc", "tools/research/wem_r41",
                   "tools/research/wem_r31", "tools/research/vttiny_wem_r3", "tools/research/vttiny_wem_r4"):
        files.update((ROOT / folder).rglob("*.py"))
    files.update(ROOT / path for path in (
        "tools/run_wem_r41.py", "tools/run_botsort_plus.py", "tools/evaluate_botsort_plus.py",
        "configs/research/wem_r4_1_protocol.json", "configs/botsort_plus/B_early_moderate.json",
        "provenance/trackeval.json"))
    return {str(path.relative_to(ROOT)).replace(os.sep, "/"): sha(path) for path in sorted(files)}


def _pin_source(path, expected):
    raw = (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
    actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if actual != expected:
        raise ValueError(f"Frozen integration source differs: {path}")


def prepare(settings):
    """Check full coverage and produce separate prediction/evaluation payloads."""
    paths = settings["paths"]
    protocol = read(ROOT / "configs/research/wem_r4_1_protocol.json")
    from tools.research.wem_r31.protocol import ENGINE
    if (protocol.get("schema") != "wem_r41_protocol_v1" or protocol.get("engine") != ENGINE
            or protocol.get("full_modes") != list(MODES)):
        raise ValueError("Fixed R4.1 policy protocol differs from the implemented W0--W3 engine")
    for path, expected in protocol["upstream_git_blobs"].items():
        _pin_source(path, expected)
    for key in ("prediction_settings", "r0_parity", "reference_manifest", "evaluation_manifest", "rgb_weights"):
        if not Path(paths[key]).is_file():
            raise FileNotFoundError(f"paths.{key}: {paths[key]}")
    prediction = read(paths["prediction_settings"])
    if "gt_root" in prediction or "references" in prediction or "evaluation" in prediction:
        raise ValueError("Use the R0 prediction-only settings, not combined GT/evaluation settings")
    prediction = {key: value for key, value in prediction.items() if key in PREDICTION_KEYS}
    prediction.update(settings["prediction_overrides"])
    if prediction.get("gmc_mode") != "cached":
        raise ValueError("This batch requires the frozen cached GMC streams")
    for key in ("baseline_root", "features_dir", "images_root", "runtime_root", "vendor_dir", "python_user_base"):
        if key in ("baseline_root", "features_dir", "images_root") or prediction.get(key):
            prediction[key] = absolute(prediction.get(key))
    for key in ("baseline_root", "features_dir", "images_root"):
        if not Path(prediction[key]).is_dir():
            raise FileNotFoundError(f"prediction.{key}: {prediction[key]}")
    if not {"native", "adaptive_gmc"} <= set(prediction.get("warp_sources", {})):
        raise ValueError("Both native and adaptive_gmc frozen warp sources are required")
    for camera, source in prediction["warp_sources"].items():
        source["run_dir"] = absolute(source["run_dir"])
        source.setdefault("method", camera)
    source_protocol = Path(prediction["baseline_root"]) / "protocol.json"
    baseline = read(source_protocol)
    lengths = {name: len(frames) for name, frames in baseline["sequences"].items()}
    if lengths != protocol["sequences"] or len(lengths) != 30 or sum(lengths.values()) != 12590:
        raise ValueError("Expected the exact frozen 30-sequence, 12,590-frame benchmark")
    inputs, refs, parity = (read(paths[key]) for key in ("evaluation_manifest", "reference_manifest", "r0_parity"))
    if inputs["sequences"] != lengths or inputs["frames"] != sum(lengths.values()):
        raise ValueError("Evaluation input manifest has different sequence coverage")
    for key in ("detector_sha256", "rgb_encoder_sha256", "source_protocol_sha256"):
        if inputs[key] != protocol["input_identity"][key]:
            raise ValueError(f"Frozen input manifest differs: {key}")
    if sha(source_protocol) != inputs["source_protocol_sha256"]:
        raise ValueError("Baseline protocol bytes differ from the frozen experiment")
    if sha(paths["rgb_weights"]) != inputs["rgb_encoder_sha256"]:
        raise ValueError("RGB ReID weights differ from the frozen encoder")
    if parity.get("status") != "PASS" or set(parity.get("sequences", [])) != set(lengths):
        raise ValueError("R0 parity must cover the complete frozen 30 sequences")
    references = {}
    for role, method in METHODS.items():
        archived = refs["references"][role]
        override = settings["reference_overrides"].get(role, {})
        if set(override) - {"run_dir"}:
            raise ValueError("Only reference run_dir may be relocated")
        directory = Path(absolute(override.get("run_dir") or archived["run_dir"]))
        reference_path = directory / "protocols" / (method + ".json")
        reference = read(reference_path)
        if sha(reference_path) != archived["protocol_sha256"]:
            raise ValueError(f"Frozen {role} reference protocol differs")
        if reference.get("status") != "COMPLETE" or reference["sequences"] != lengths:
            raise ValueError(f"Incomplete {role} reference")
        if reference["native_config"].get("track_low_thresh") != .1:
            raise ValueError("The native low threshold must stay at 0.10")
        for key in protocol["input_identity"]:
            if reference[key] != inputs[key]:
                raise ValueError(f"Reference input differs: {role}/{key}")
        for name in lengths:
            old = parity["references"][role]["files"][name]
            if old["mot_sha256"] != archived["trajectory_hashes"][name]:
                raise ValueError(f"R0 and frozen trajectory receipts disagree: {role}/{name}")
            if any(not re.fullmatch(r"[0-9a-f]{64}", old.get(key, "")) for key in
                   ("mot_sha256", "events_sha256", "state_chain_sha256")):
                raise ValueError(f"Missing R0 parity fingerprints: {role}/{name}")
        references[role] = {"run_dir": str(directory), "protocol": reference}
    # Hash the exact cached embeddings and warp inputs. Decode/image and detector
    # row fingerprints are checked by the original worker in every arm.
    assets = {key: sha(paths[key]) for key in ("prediction_settings", "r0_parity", "reference_manifest",
                                             "evaluation_manifest", "rgb_weights")}
    for name in lengths:
        for suffix, label in ((".npz", "features"), (".json", "metadata")):
            file = Path(prediction["features_dir"]) / (name + suffix)
            if sha(file) != inputs["features"][name][label]:
                raise ValueError(f"Frozen feature cache differs: {name}/{label}")
    for camera in ("native", "adaptive_gmc"):
        source = prediction["warp_sources"][camera]
        root, method = Path(source["run_dir"]), source["method"]
        camera_protocol = read(root / "protocols" / (method + ".json"))
        if camera_protocol.get("status") != "COMPLETE" or camera_protocol["sequences"] != lengths:
            raise ValueError(f"Incomplete frozen camera stream: {camera}")
        hashes = {}
        for name in lengths:
            value = sha(root / "warps" / method / (name + ".jsonl.gz"))
            known = source.get("sha256", {}).get(name)
            if known is not None and known != value:
                raise ValueError(f"Changed cached warp: {camera}/{name}")
            hashes[name] = value
        source["sha256"] = hashes
        assets["warps_" + camera] = {"protocol": sha(root / "protocols" / (method + ".json")), "files": hashes}
    evaluation = deepcopy(inputs)
    evaluation["ground_truth"]["root"] = absolute(settings["evaluation_overrides"].get("gt_root") or inputs["ground_truth"]["root"])
    evaluation["trackeval"]["root"] = absolute(settings["evaluation_overrides"].get("trackeval_root") or inputs["trackeval"]["root"])
    if not (Path(evaluation["trackeval"]["root"]) / "trackeval/__init__.py").is_file():
        raise FileNotFoundError("The original TrackEval checkout is required")
    from tools.evaluate_botsort_plus import verify_trackeval
    assets["trackeval_verified"] = verify_trackeval(evaluation["trackeval"]["root"], ROOT)
    for name in lengths:
        file = Path(evaluation["ground_truth"]["root"]) / name / "gt/gt.txt"
        if sha(file) != inputs["ground_truth"]["sha256"][name]:
            raise ValueError(f"Frozen GT changed: {name}")
    output = Path(paths["output_root"])
    for key in ("baseline_root", "features_dir"):
        if output == Path(prediction[key]) or output.is_relative_to(Path(prediction[key])):
            raise ValueError("Use an output directory outside frozen input folders")
    code = source_manifest()
    prediction["cuda_visible_devices"] = settings["runtime"]["cuda_visible_devices"]
    bundle = {"schema": "wem_r41_prediction_bundle_v1", "settings": prediction,
              "rgb_weights": paths["rgb_weights"], "rgb_device": settings["runtime"]["rgb_device"],
              "code_sha256": code, "sequences": lengths, "protocol": protocol}
    receipt = {"schema": "wem_r41_inputs_v1", "status": "INPUTS_VERIFIED_NOT_RUN",
               "sequences": lengths, "frames": sum(lengths.values()), "input_sha256": assets,
               "code_sha256": code, "settings_sha256": settings["settings_source"]["sha256"],
               "prediction_bundle_sha256": digest(bundle), "protocol": protocol}
    return {"settings": settings, "bundle": bundle, "evaluation": evaluation, "references": references,
            "parity": parity, "receipt": receipt, "fingerprint": digest(receipt)}
