"""Live or provenance-checked frozen camera transforms, with exact frame order."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .common import REPO, assert_same_inputs, read_json, sha256

FACTOR = REPO / "experiments/vttiny_gmc/native_botsort_gmc_factorial_v1"
IDENTITY = np.eye(2, 3, dtype=np.float64)


def _reference():
    return read_json(REPO / "experiments/vttiny_gmc/reference.json")["gmc_factorial"]


def _canonical_sha(path):
    # Git checkouts on Windows may use CRLF; only newline representation may differ.
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _verify_sources():
    reference = _reference()
    accepted = {}
    for name in ("frozen_gmc.py", "gmc_variants.py"):
        path = FACTOR / name
        raw, canonical = sha256(path), _canonical_sha(path)
        expected = reference["source_sha256"][name]
        if expected not in (raw, canonical):
            raise ValueError(f"Archived GMC selector source differs from frozen base: {path}")
        accepted[name] = {expected, raw, canonical}
    return reference, accepted


def _module_origin(name, expected, *, package=False):
    """Path insertion must not silently reuse a previously imported other module."""
    names = [key for key in sys.modules if key == name or (package and key.startswith(name + "."))]
    expected = Path(expected).resolve()
    for key in names:
        module = sys.modules[key]
        filename = getattr(module, "__file__", None)
        if module is None or not filename:
            raise RuntimeError(f"Cannot verify preloaded {key} origin")
        actual = Path(filename).resolve()
        correct = expected in actual.parents if package else actual == expected
        if not correct:
            raise RuntimeError(f"Preloaded {key} is outside the pinned source: {actual}")


def _validate_cache_contract(flags, camera):
    reference, source_hashes = _verify_sources()
    expected = {"native": (False, False), "skip_only": (True, False), "adaptive_gmc": (True, True)}[camera]
    if (type(flags.get("skip")) is not bool or type(flags.get("lightglue")) is not bool
            or (flags["skip"], flags["lightglue"]) != expected):
        raise ValueError(f"Cached candidate-set flags disagree with {camera}")
    if flags.get("policy") != reference["policy"]:
        raise ValueError("Cached GMC thresholds differ from the frozen selection policy")
    if flags.get("kalman") != reference["kalman"] or flags.get("past_outputs_modified") is not False:
        raise ValueError("Cached GMC run does not declare the original KF and unchanged past outputs")
    for name, accepted in source_hashes.items():
        if flags.get("source_sha256", {}).get(name) not in accepted:
            raise ValueError(f"Cached GMC source provenance differs: {name}")
    model = flags.get("model")
    if expected[1]:
        if not isinstance(model, dict):
            raise ValueError("Cached LightGlue camera has no model provenance")
        for key in ("commit", "weights", "source_sha256"):
            if model.get(key) != reference["model"][key]:
                raise ValueError(f"Cached LightGlue model provenance differs: {key}")
    elif model is not None:
        raise ValueError("Non-LightGlue candidate set must not declare a LightGlue model")


def _validate_record(camera, native, used, record, recorded_backup=None):
    allowed = {"sof"} if camera == "native" else ({"sof", "identity"} if camera == "skip_only"
                                                  else {"sof", "identity", "lightglue"})
    choice = record.get("choice")
    eligible = record.get("eligible", [])
    if choice not in allowed:
        raise ValueError("Selected warp is not a candidate of the declared camera policy")
    if (not isinstance(eligible, list) or not all(isinstance(item, str) for item in eligible)
            or len(set(eligible)) != len(eligible)
            or any(item not in allowed - {"sof"} for item in eligible)):
        raise ValueError("Eligible alternatives contradict the declared candidate set")
    if type(record.get("trigger", False)) is not bool or type(record.get("model_called", False)) is not bool:
        raise ValueError("Camera trigger/model_called flags must be booleans")
    if choice == "sof":
        if eligible:
            raise ValueError("SOF selected although an alternative is marked eligible")
        if not np.array_equal(native, used):
            raise ValueError("SOF choice differs from its native matrix")
    elif choice not in eligible:
        raise ValueError("Selected alternative is absent from eligible candidates")
    if choice != "sof" and not record.get("trigger", False):
        raise ValueError("Alternative selected without the declared trigger")
    if choice == "identity" and not np.array_equal(IDENTITY, used):
        raise ValueError("Identity choice contains a non-identity warp")
    if camera != "adaptive_gmc" and record.get("model_called", False):
        raise ValueError("Non-LightGlue candidate set claims a model call")
    if camera == "native" and record.get("trigger", False):
        raise ValueError("Native-only camera must not claim an adaptive trigger")
    if camera == "adaptive_gmc" and record.get("trigger", False) and not record.get("model_called", False):
        raise ValueError("Adaptive trigger did not call the declared LightGlue candidate")
    if choice == "lightglue" and not record.get("model_called", False):
        raise ValueError("LightGlue selected without a model call")
    backup = IDENTITY.copy() if "identity" in eligible else native.copy()
    if recorded_backup is not None and not np.array_equal(matrix(recorded_backup, "backup"), backup):
        raise ValueError("Recorded backup disagrees with the candidate eligibility rule")
    return backup


def matrix(value, label):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2, 3) or not np.isfinite(result).all():
        raise ValueError(f"Invalid {label} affine matrix")
    return result.copy()


def setup_matcher(settings, camera):
    # Import the exact archived selector rather than reimplement its thresholds.
    reference, _ = _verify_sources()
    _module_origin("frozen_gmc", FACTOR / "frozen_gmc.py")
    _module_origin("gmc_variants", FACTOR / "gmc_variants.py")
    sys.path.insert(0, str(FACTOR))
    if settings.get("gmc_mode", "live") == "cached" or camera != "adaptive_gmc":
        return None, None
    vendor = Path(settings.get("vendor_dir") or "")
    if not vendor.is_dir() or not settings.get("vendor_dir"):
        raise FileNotFoundError("Live adaptive GMC requires vendor_dir from the previous LightGlue experiment")
    model = reference["model"]
    source = vendor / ("LightGlue-" + model["commit"])
    for folder, hashes in ((vendor / "torch/hub/checkpoints", model["weights"]),
                           (source / "lightglue", model["source_sha256"])):
        for name, expected in hashes.items():
            path = folder / name
            matches = sha256(path) == expected
            if path.suffix == ".py":
                matches = matches or _canonical_sha(path) == expected
            if not matches:
                raise ValueError(f"Pinned LightGlue provenance mismatch: {folder / name}")
    _module_origin("lightglue", source / "lightglue", package=True)
    import os
    os.environ["TORCH_HOME"] = str(vendor / "torch")
    sys.path[:0] = [str(vendor / "dependencies"), str(source)]
    from frozen_gmc import Matcher
    import torch
    # TORCH_HOME alone loses to an earlier torch.hub.set_dir() in this process.
    # Bind the actual constructor cache to the exact files just verified above.
    torch.hub.set_dir(str(vendor / "torch/hub"))
    original_download = torch.hub.download_url_to_file
    def reject_download(*args, **kwargs):
        raise RuntimeError("LightGlue setup attempted a network download; restore the pinned local weights")
    # Cache lookup is still the archived constructor's operation, but never downloads.
    torch.hub.download_url_to_file = reject_download
    try:
        matcher = Matcher()
    finally:
        torch.hub.download_url_to_file = original_download
    _module_origin("lightglue", source / "lightglue", package=True)
    return matcher, model


class WarpCamera:
    def __init__(self, settings, camera, sequence, protocol, native_apply, matcher, output):
        self.camera, self.sequence = camera, sequence
        self.expected_frames = protocol["sequences"][sequence]
        self.native_apply = native_apply
        self.index = 0
        self.cached = settings.get("gmc_mode", "live") == "cached"
        self.reader = None
        self.cache_protocol = None
        self.source = None
        self.selector = None
        self.closed = False
        self.choices = {}
        if self.cached:
            source = settings.get("warp_sources", {}).get(camera)
            if not source:
                raise ValueError(f"Missing warp_sources.{camera}; never reuse a different candidate-set cache")
            root, method = Path(source["run_dir"]), source["method"]
            pp = root / "protocols" / (method + ".json")
            prior = read_json(pp)
            if prior.get("status") != "COMPLETE":
                raise ValueError(f"Incomplete source camera run: {pp}")
            flags = prior.get("gmc_factorial", {})
            _validate_cache_contract(flags, camera)
            assert_same_inputs(prior, protocol, [sequence], decoded=False)
            path = root / "warps" / method / (sequence + ".jsonl.gz")
            digest = sha256(path)
            known = source.get("sha256", {}).get(sequence)
            if known is not None and known != digest:
                raise ValueError(f"Frozen warp SHA-256 differs: {sequence}")
            self.source = {"path": str(path), "sha256": digest, "protocol_sha256": sha256(pp),
                           "method": method, "decoded_image_sha256": prior["decoded_image_hashes"][sequence]}
            self.cache_protocol = prior
            self.reader = gzip.open(path, "rt", encoding="utf-8")
        elif camera != "native":
            if camera == "adaptive_gmc" and matcher is None:
                raise ValueError("Live adaptive GMC requires the verified LightGlue matcher; no skip-only fallback")
            _verify_sources()
            _module_origin("gmc_variants", FACTOR / "gmc_variants.py")
            from gmc_variants import Selector
            self.selector = Selector(True, matcher if camera == "adaptive_gmc" else None)
        self.path = Path(output)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = gzip.open(self.path, "xt", encoding="utf-8", compresslevel=3)

    def apply(self, image, detections, frame):
        if self.closed or frame != self.index + 1 or frame > self.expected_frames:
            raise ValueError(f"Camera stream frame order violation: {self.sequence}/{frame}")
        if self.cached:
            line = self.reader.readline()
            if not line:
                raise ValueError(f"Truncated camera cache at {self.sequence}/{frame}")
            row = json.loads(line)
            if row.get("frame") != frame:
                raise ValueError(f"Camera cache frame mismatch at {self.sequence}/{frame}")
            native, used = matrix(row["native"], "native"), matrix(row["used"], "used")
            record = {k: v for k, v in row.items() if k not in ("native", "used", "backup", "frame")}
            backup = _validate_record(self.camera, native, used, record, row.get("backup"))
        else:
            native = matrix(self.native_apply(image, detections), "native")
            if self.selector is None:
                used, record = native.copy(), {"choice": "sof", "trigger": False, "model_called": False}
            else:
                used, record = self.selector.apply(image, detections, native)
                used = matrix(used, "selected")
            backup = _validate_record(self.camera, native, used, record)
        self.index = frame
        choice = record["choice"]
        self.choices[choice] = self.choices.get(choice, 0) + 1
        self.writer.write(json.dumps({"frame": frame, "native": native.tolist(), "used": used.tolist(),
                                      "backup": backup.tolist(), **record}, allow_nan=False,
                                     separators=(",", ":")) + "\n")
        return native, used, backup, record

    def close(self):
        if self.closed:
            return self.manifest
        try:
            if self.index != self.expected_frames:
                raise ValueError(f"Camera frame count mismatch: {self.index}/{self.expected_frames}")
            if self.reader is not None and self.reader.readline():
                raise ValueError("Extra frame in frozen warp stream")
        finally:
            if self.reader is not None:
                self.reader.close()
            self.writer.close()
            self.closed = True
        self.manifest = {"frames": self.index, "candidate_set": self.camera, "choices": self.choices,
                         "mode": "cached" if self.cached else "live", "source": self.source,
                         "warp_sha256": sha256(self.path)}
        return self.manifest

    def abort(self):
        if not self.closed:
            if self.reader is not None:
                self.reader.close()
            self.writer.close()
            self.closed = True
