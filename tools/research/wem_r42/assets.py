"""Frozen inputs and safe reuse of completed R4.1 reference predictions."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import sys

from tools.research.wem_r41 import assets as previous
from tools.research.wem_r41.batch import completed_arm

ROOT = previous.ROOT
read, write, sha, digest = previous.read, previous.write, previous.sha, previous.digest
METHODS = {"BG": "evidence_both"}
MODES = ("off", "W0", "lower_low", "W0_relaxed_anchor", "W0_emit_only")
FULL_ARMS = tuple(("BG", mode) for mode in MODES)
SMOKE = previous.SMOKE
SCHEMA = "wem_r42_settings"
SETTINGS = dict(schema=SCHEMA, r41_settings=None, r41_batch=None, reference_mode="reuse",
                output_root="runs/wem_r4_2", runtime={})
PROTOCOL_PATH = "configs/research/wem_r4_2_protocol.json"
PARAMETERS = dict(native_low_default=.1, native_low_control=.05, weak_score_range=[.05, .1],
                  native_iou_floor=.5, anchor_iou_floor=.5, near_owner_iou=.9, top_geometry_groups=5)
STATE_BASIS = dict(W0_emit_only="canonical native output before weak publication; emitted output separately audited",
                   others="actual published output")


def validate_protocol(protocol):
    if (protocol.get("schema") != "wem_r42_protocol_v1" or protocol.get("full_modes") != list(MODES)
            or protocol.get("reference_method") != METHODS["BG"] or protocol.get("frames_per_arm") != 12590
            or protocol.get("source_commit") != "3a6e6c4d01d3d3ee04f1a9c952e25a4790f04e77"
            or protocol.get("new_full_arms") != [f"BG_{mode}" for mode in MODES[2:]]
            or protocol.get("parameters") != PARAMETERS or protocol.get("state_chain_output_basis") != STATE_BASIS
            or protocol.get("controls") != dict(ground_truth_in_prediction=False, parameter_search=False,
                                                scientific_acceptance_gate=False)):
        raise ValueError("Fixed R4.2 protocol differs from the implemented matrix, parameters or controls")


def init(path, r41_settings, r41_batch=None, reference_mode="reuse"):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Settings already exist: {path}")
    if reference_mode not in ("reuse", "rerun"):
        raise ValueError("reference_mode must be reuse or rerun")
    source = Path(previous.absolute(r41_settings))
    if not source.is_file():
        raise FileNotFoundError(f"Existing R4.1 local settings required: {source}")
    old = previous.load(source)
    batch = previous.absolute(r41_batch) if r41_batch else str(Path(old["paths"]["output_root"]) / "full")
    value = dict(SETTINGS, r41_settings=str(source), r41_batch=batch, reference_mode=reference_mode)
    write(path, value)
    return value


def load(path):
    supplied = read(path)
    if supplied.get("schema") != SCHEMA or set(supplied) - set(SETTINGS):
        raise ValueError(f"Expected {SCHEMA} and documented top-level fields")
    value = deepcopy(SETTINGS)
    value.update(supplied)
    if value["reference_mode"] not in ("reuse", "rerun"):
        raise ValueError("reference_mode must be reuse or rerun")
    if not isinstance(value["runtime"], dict) or set(value["runtime"]) - set(previous.SETTINGS["runtime"]):
        raise ValueError("Unknown runtime override")
    value["r41_settings"] = previous.absolute(value["r41_settings"])
    value["r41_batch"] = previous.absolute(value["r41_batch"], required=False)
    value["output_root"] = previous.absolute(value["output_root"])
    value["settings_source"] = {"path": str(Path(path).resolve()), "sha256": sha(path)}
    return value


def runtime_overrides(base, overrides):
    runtime = deepcopy(base)
    runtime.update(overrides)
    if type(runtime["workers"]) is not int or runtime["workers"] < 1:
        raise ValueError("runtime.workers must be a positive integer")
    executable = runtime["python"] or sys.executable
    runtime["python"] = shutil.which(executable) or previous.absolute(executable)
    if not Path(runtime["python"]).is_file():
        raise FileNotFoundError(f"Python executable not found: {executable}")
    if not isinstance(runtime["cuda_visible_devices"], str):
        raise ValueError("cuda_visible_devices must be a string")
    if not isinstance(runtime["rgb_device"], str) or not previous.re.fullmatch(r"cpu|cuda:\d+", runtime["rgb_device"]):
        raise ValueError("rgb_device must be cpu or cuda:<visible index>")
    return runtime


def normalized_source_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def reference_sources(settings, context):
    """Read and verify references; never edit the original R4.1 directory."""
    if settings["reference_mode"] == "rerun":
        return {}
    if not settings["r41_batch"]:
        raise ValueError("reuse requires r41_batch pointing at the old full/ directory; use rerun if unavailable")
    source = Path(settings["r41_batch"])
    output = Path(settings["output_root"])
    if source == output or source.is_relative_to(output) or output.is_relative_to(source.parent):
        raise ValueError("Use a separate R4.2 output_root outside the original R4.1 run")
    completion = read(source / "complete.json")
    verified = read(source / "verified.json")
    bundle = read(source / "prediction_bundle.json")
    plan = read(source / "plan.json")
    fingerprint = verified.get("fingerprint")
    if (completion.get("status") != "FULL_BENCHMARK_COMPLETE" or verified.get("status") != "PASS"
            or completion.get("sequence_count") != 30 or completion.get("frames_per_arm") != 12590
            or completion.get("arm_count") != 10 or not fingerprint
            or completion.get("fingerprint") != fingerprint or plan.get("fingerprint") != fingerprint
            or completion.get("checks") != sha(source / "verified.json")
            or completion.get("report") != sha(source / "report/summary.json")):
        raise ValueError("R4.1 full completion and verification receipts disagree")
    names = list(context["bundle"]["sequences"])
    if (bundle.get("schema") != "wem_r41_prediction_bundle_v1"
            or bundle.get("protocol") != context["r41_protocol"]
            or bundle.get("sequences") != context["bundle"]["sequences"]
            or set(verified.get("scope", [])) != set(names)
            or len(verified.get("scope", [])) != len(names)
            or verified.get("scope_frames") != 12590
            or set(plan.get("sequences", [])) != set(names)
            or set(verified.get("arms", {})) != {f"{r}_{m}" for r, m in previous.FULL_ARMS}
            or set(plan.get("arms", [])) != {f"{r}_{m}" for r, m in previous.FULL_ARMS}):
        raise ValueError("R4.1 reference scope or fixed policy differs")
    old_codes = bundle.get("code_sha256", {})
    if set(old_codes) != set(context["r41_code_sha256"]):
        raise ValueError("R4.1 reference does not identify the complete frozen integration source set")
    eol_equivalent = []
    for relative, expected in old_codes.items():
        current = ROOT / relative
        if current.is_absolute() and not current.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Reference source path escapes repository")
        if sha(current) != expected:
            if normalized_source_hash(current) != expected:
                raise ValueError(f"R4.1 source differs from the reference: {relative}")
            eol_equivalent.append(relative)
    from experiments.vttiny_gmc.round3_joint.common import assert_same_inputs
    references = {}
    for mode in ("off", "W0"):
        key, method = f"BG_{mode}", METHODS["BG"]
        folder = source / key
        if not completed_arm(folder, key, fingerprint, names):
            raise ValueError(f"Incomplete R4.1 reference: {key}")
        actual = read(folder / "protocols" / (method + ".json"))
        recorded = actual.get("wem_r41", {})
        if (actual.get("status") != "COMPLETE" or actual.get("sequences") != context["bundle"]["sequences"]
                or recorded.get("role") != "BG" or recorded.get("mode") != mode
                or recorded.get("experiment") != context["r41_protocol"]
                or recorded.get("code_sha256") != old_codes
                or recorded.get("prediction_bundle_sha256") != sha(source / "prediction_bundle.json")):
            raise ValueError(f"R4.1 arm protocol differs: {key}")
        assert_same_inputs(context["references"]["BG"]["protocol"], actual, names)
        expected_camera = context["bundle"]["settings"]["warp_sources"]["adaptive_gmc"]["sha256"]
        for seq in names:
            camera = actual["joint_batch"]["camera_manifests"][seq]
            row = verified["arms"][key]["per_sequence"][seq]
            files = {
                "mot": folder / "trackers" / method / "data" / (seq + ".txt"),
                "native_events": folder / "events" / method / (seq + ".jsonl"),
                "states": folder / "states" / method / (seq + ".sha256"),
            }
            if (camera.get("mode") != "cached" or camera["source"]["sha256"] != expected_camera[seq]
                    or camera["source"]["decoded_image_sha256"] != actual["decoded_image_hashes"][seq]
                    or any(sha(path) != row[label] for label, path in files.items())
                    or actual["output_prediction_hashes"][seq] != row["mot"]):
                raise ValueError(f"R4.1 reference files/warp differ: {key}/{seq}")
            if mode == "off":
                old = context["parity"]["references"]["BG"]["files"][seq]
                if any(row[label] != old[receipt] for label, receipt in (
                        ("mot", "mot_sha256"), ("native_events", "events_sha256"), ("states", "state_chain_sha256"))):
                    raise ValueError(f"R4.1 off is not the frozen reference: {seq}")
        references[key] = dict(kind="reused_r41", source_batch=str(source), source_arm=str(folder),
                               fingerprint=fingerprint, completion_sha256=sha(source / "complete.json"),
                               arm_manifest_sha256=sha(folder / "arm_complete.json"),
                               protocol_sha256=sha(folder / "protocols" / (method + ".json")),
                               eol_equivalent_source_files=eol_equivalent)
    return references


def prepare(settings):
    inherited = previous.load(settings["r41_settings"])
    original_output = Path(inherited["paths"]["output_root"])
    destination = Path(settings["output_root"])
    if destination == original_output or destination.is_relative_to(original_output) or original_output.is_relative_to(destination):
        raise ValueError("R4.2 output_root must be separate from the original R4.1 output_root")
    inherited["paths"]["output_root"] = str(destination)
    inherited["runtime"] = runtime_overrides(inherited["runtime"], settings["runtime"])
    context = previous.prepare(inherited)
    protocol = read(ROOT / PROTOCOL_PATH)
    validate_protocol(protocol)
    context["r41_protocol"] = deepcopy(context["bundle"]["protocol"])
    context["r41_code_sha256"] = deepcopy(context["bundle"]["code_sha256"])
    code = context["bundle"]["code_sha256"]
    additions = list((ROOT / "tools/research/wem_r42").glob("*.py"))
    additions += [ROOT / "tools/run_wem_r42.py", ROOT / PROTOCOL_PATH]
    for file in additions:
        code[file.relative_to(ROOT).as_posix()] = sha(file)
    protocol["sequences"] = context["bundle"]["sequences"]
    protocol["input_identity"] = context["r41_protocol"]["input_identity"]
    context["bundle"]["schema"] = "wem_r42_prediction_bundle_v1"
    context["bundle"]["protocol"] = protocol
    context["bundle"]["r41_policy_protocol"] = context["r41_protocol"]
    context["r42_settings"] = settings
    context["reuse"] = reference_sources(settings, context)
    receipt = context["receipt"]
    receipt.update(schema="wem_r42_inputs_v1", protocol=protocol, code_sha256=code,
                   settings_sha256=settings["settings_source"]["sha256"],
                   r41_settings_sha256=sha(settings["r41_settings"]),
                   prediction_bundle_sha256=digest(context["bundle"]),
                   reference_mode=settings["reference_mode"], reference_reuse=context["reuse"])
    context["fingerprint"] = digest(receipt)
    return context
