"""Configuration and provenance shared by the round-3 command line tools."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
BASE_COMMIT = "7d549bc8bbddfefe99fbb8119ea13d40a4b5bd6d"
SCHEMA = "v04_joint_batch_v1"
B_CONFIG_DIGEST = "71628cbb98971354fd9efc6dacc564ae1bc9e121614a1146b46a9e2f0ec698a4"
SMOKE = ["DJI_0030_2", "DJI_0169_3", "DJI_0109_1", "DJI_0173_2", "DJI_0221_2", "DJI_0028_3"]
METHODS = {
    "B_early_moderate": {"camera": "native", "dual_low": False, "evidence_both": False},
    "B_skip_only": {"camera": "skip_only", "dual_low": False, "evidence_both": False},
    "B_adaptive_no_dual_low": {"camera": "adaptive_gmc", "dual_low": False, "evidence_both": False},
    "B_early_moderate_dual_low": {"camera": "adaptive_gmc", "dual_low": True, "evidence_both": False},
    "evidence_both": {"camera": "adaptive_gmc", "dual_low": True, "evidence_both": True},
}
CONTROLS = ["B_skip_only", "B_adaptive_no_dual_low"]
ROLES = {"native": "native", "B": "B_early_moderate", "G": "dual_low", "BG": "B_early_moderate_dual_low",
         "evidence": "evidence_both"}
INPUT_FIELDS = ("native_config", "source_protocol_sha256", "detector_sha256", "rgb_encoder_sha256",
                "encoder_contracts")
SEQUENCE_FIELDS = ("detection_fingerprints", "feature_hashes", "decoded_image_hashes")


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".writing")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def new_directory(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Output already exists; use a new directory: {path}")
    path.mkdir(parents=True)
    return path


def absolute(value):
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return str(path.resolve() if path.is_absolute() else (REPO / path).resolve())


def load_settings(path):
    data = read_json(path)
    if data.get("schema") != SCHEMA:
        raise ValueError(f"Expected schema={SCHEMA}; use the round-3 example or init command")
    data = copy.deepcopy(data)
    for key in ("baseline_root", "features_dir", "images_root", "gt_root", "trackeval_root", "output_root",
                "runtime_root", "vendor_dir", "python_user_base", "grouping_file", "smoke_manifest"):
        if data.get(key) is not None:
            data[key] = absolute(data[key])
    if data.get("gmc_mode", "live") not in ("live", "cached"):
        raise ValueError("gmc_mode must be live or cached")
    data.setdefault("gmc_mode", "live")
    for key, entry in data.get("references", {}).items():
        if not isinstance(entry, dict) or not entry.get("run_dir"):
            raise ValueError(f"references.{key} needs run_dir")
        entry.setdefault("tracker", ROLES.get(key, key))
        for field in ("run_dir", "events_dir", "metrics_file"):
            if entry.get(field):
                entry[field] = absolute(entry[field])
    for key, entry in data.get("warp_sources", {}).items():
        if key not in ("native", "skip_only", "adaptive_gmc"):
            raise ValueError(f"Unknown warp source {key}")
        entry["run_dir"] = absolute(entry["run_dir"])
        entry.setdefault("method", key)
    if not data.get("original_video_groups"):
        grouping = Path(data.get("grouping_file") or REPO / "results/vttiny/iterations/v0.4-motion-blur/round2-e1-e2-e3/E3/A/E3_sequence_summary.csv")
        if grouping.is_file():
            with grouping.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            data["original_video_groups"] = {r["sequence"]: r["original_video"] for r in rows}
            data["grouping_provenance"] = {"path": str(grouping), "sha256": sha256(grouping),
                                          "semantics": "Reuse archived explicit source-video mapping; no new filename inference"}
    data["settings_source"] = {"path": str(Path(path).resolve()), "sha256": sha256(path)}
    return data


def frozen_config(method):
    if method not in METHODS:
        raise ValueError(f"Unknown experiment {method}")
    cfg = read_json(REPO / "configs/botsort_plus/B_early_moderate.json")
    if json_digest(cfg) != B_CONFIG_DIGEST:
        raise ValueError("B_early_moderate configuration differs from the frozen experimental base")
    cfg["name"] = method
    return cfg


def sequence_lengths(settings):
    protocol = read_json(Path(settings["baseline_root"]) / "protocol.json")
    lengths = {name: len(frames) for name, frames in protocol["sequences"].items()}
    explicit = settings.get("sequence_lengths")
    if explicit is not None and explicit != lengths:
        raise ValueError("Explicit sequence_lengths disagrees with frozen detection protocol")
    return lengths


def select_sequences(settings, sequences=None):
    lengths = sequence_lengths(settings)
    chosen = list(sequences or settings.get("sequences") or lengths)
    if not chosen or len(chosen) != len(set(chosen)) or set(chosen) - set(lengths):
        raise ValueError("Empty, duplicated or unknown sequence selection")
    return {s: lengths[s] for s in chosen}


def source_hashes():
    paths = set((REPO / "botsort_plus").rglob("*.py"))
    paths.update((REPO / "experiments/vttiny_gmc/round3_joint").glob("*.py"))
    paths.update(REPO / p for p in (
        "tools/run_v04_joint_batch.py", "tools/run_botsort_plus.py", "tools/evaluate_botsort_plus.py",
        "experiments/vttiny_gmc/native_botsort_gmc_factorial_v1/frozen_gmc.py",
        "experiments/vttiny_gmc/native_botsort_gmc_factorial_v1/gmc_variants.py",
        "experiments/vttiny_gmc/native_botsort_gmc_0030_fp_v1/guard.py",
        "experiments/vttiny_gmc/b_gmc_recovery/forensics/lost_camera_margin.py"))
    return {p.relative_to(REPO).as_posix(): sha256(p) for p in sorted(paths) if p.is_file()}


def assert_same_inputs(left, right, sequences, *, decoded=True):
    for field in INPUT_FIELDS:
        if left.get(field) is None or left.get(field) != right.get(field):
            raise ValueError(f"Mixed input contract: {field}")
    for field in SEQUENCE_FIELDS:
        if field == "decoded_image_hashes" and not decoded:
            continue
        for seq in sequences:
            a, b = left.get(field, {}).get(seq), right.get(field, {}).get(seq)
            if a is None or a != b:
                raise ValueError(f"Mixed input contract: {field}/{seq}")
    for seq in sequences:
        if left["sequences"].get(seq) != right["sequences"].get(seq):
            raise ValueError(f"Frame count mismatch: {seq}")


def child_environment(settings):
    env = dict(os.environ)
    paths = [str(REPO), str(REPO / "tools")]
    if settings.get("runtime_root"):
        paths.append(str(Path(settings["runtime_root"]) / "tracking_runtime"))
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    if settings.get("python_user_base"):
        env["PYTHONUSERBASE"] = settings["python_user_base"]
    if settings.get("cuda_visible_devices") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(settings["cuda_visible_devices"])
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[variable] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def prediction_settings(settings):
    # Prediction receives no GT path, manual labels, sequence-specific policy,
    # offline outcome labels or historical future trajectory statistics.
    allowed = ("schema", "baseline_root", "features_dir", "images_root", "runtime_root", "vendor_dir",
               "gmc_mode", "warp_sources", "python_user_base", "cuda_visible_devices")
    return {key: settings[key] for key in allowed if key in settings}
