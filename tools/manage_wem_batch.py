#!/usr/bin/env python3
"""Prepare a WEM research batch; never execute a tracker or accept a research gate.

Uses only the Python standard library. Paths in settings are repository-relative
unless absolute. Exit codes: 0 requested operation completed; 2 invalid input or
unsafe destination; 3 unavailable local inputs, unprepared batch, or stale state.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "176f2a81613c34d16a860765ea7be0845116b534"
SETTINGS_SCHEMA = "wem_v0.1_settings"
PROTOCOL_SCHEMA = "wem_v0.1_protocol"
TEMPLATE = "configs/research/wem_v0.1.example.json"
PROTECTED_DIRS = (
    ".git", "botsort_plus", "configs", "docs", "experiments", "provenance",
    "results", "tests", "tools", "external",
)
PATH_KINDS = {
    "detections_dir": "directory",
    "baseline_root": "directory",
    "features_dir": "directory",
    "images_root": "directory",
    "gt_root": "directory",
    "trackeval_root": "directory",
    "source_group_manifest": "file",
}
REFERENCE_NAMES = {"B": "B_early_moderate", "BG": "evidence_both"}
NOT_CHECKED = [
    "Detection contents, actual runtime masks, feature validity and data fingerprints.",
    "Frozen-reference reproduction, instrumentation parity, or any R0 scientific gate.",
    "Ground-truth contents, identity labels, mother-video separation or data leakage.",
    "R1b state-aware opportunity, R2/R3 evidence, W implementation or R4 metrics.",
]


class BatchError(ValueError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_constant(value: str) -> None:
    raise BatchError(f"Non-finite JSON constant is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BatchError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_finite(value: Any) -> None:
    # parse_constant does not see overflowed numeric literals such as 1e999.
    if isinstance(value, float) and not math.isfinite(value):
        raise BatchError("Non-finite JSON numeric value is not allowed")
    if isinstance(value, dict):
        for item in value.values():
            _require_finite(item)
    elif isinstance(value, list):
        for item in value:
            _require_finite(item)


def read_object(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        obj = json.loads(raw.decode("utf-8-sig"), parse_constant=_reject_constant,
                         object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BatchError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise BatchError(f"JSON root must be an object: {path}")
    _require_finite(obj)
    return obj, raw


def resolve_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise BatchError(f"{label} must be a non-empty path string")
    if "\x00" in value or ("$" + "{") in value:
        raise BatchError(f"{label} contains a NUL or unresolved placeholder")
    value = value.strip()
    windows = PureWindowsPath(value)
    if os.name != "nt" and (windows.drive or windows.root) and not value.startswith("/"):
        raise BatchError(f"{label} is a Windows path on a non-Windows host")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def validate_settings(settings: dict[str, Any]) -> None:
    expected = {"schema", "base_commit", "output_root", "protocol_file",
                "thresholds", "paths", "references"}
    if set(settings) != expected:
        raise BatchError(f"Settings keys must be exactly {sorted(expected)}")
    if settings["schema"] != SETTINGS_SCHEMA or settings["base_commit"] != BASE_COMMIT:
        raise BatchError("Settings schema or base_commit does not match WEM v0.1")
    resolve_path(settings["output_root"], "output_root")
    resolve_path(settings["protocol_file"], "protocol_file")
    thresholds = settings["thresholds"]
    if not isinstance(thresholds, dict) or set(thresholds) != {"low", "high"}:
        raise BatchError("thresholds must contain exactly low and high")
    for name, expected_value in (("low", 0.1), ("high", 0.5)):
        value = thresholds[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or value != expected_value):
            raise BatchError(f"thresholds.{name} must be {expected_value} for the frozen reference")
    paths = settings["paths"]
    if not isinstance(paths, dict) or set(paths) != set(PATH_KINDS):
        raise BatchError(f"paths must contain exactly {sorted(PATH_KINDS)}")
    for key, value in paths.items():
        if value is not None:
            resolve_path(value, f"paths.{key}")
    references = settings["references"]
    if not isinstance(references, dict) or set(references) != set(REFERENCE_NAMES):
        raise BatchError("references must contain exactly B and BG")
    for key, tracker in REFERENCE_NAMES.items():
        item = references[key]
        if not isinstance(item, dict) or set(item) != {"tracker", "run_dir"}:
            raise BatchError(f"references.{key} must contain tracker and run_dir")
        if item["tracker"] != tracker:
            raise BatchError(f"references.{key}.tracker must be {tracker}")
        if item["run_dir"] is not None:
            resolve_path(item["run_dir"], f"references.{key}.run_dir")


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("base_commit") != BASE_COMMIT:
        raise BatchError("Protocol schema or base_commit does not match WEM v0.1")


def safe_output(settings: dict[str, Any], settings_path: Path) -> Path:
    output = resolve_path(settings["output_root"], "output_root")
    root = REPO_ROOT.resolve()
    if output == root or output in root.parents:
        raise BatchError("output_root must not be the repository or its ancestor")
    for name in PROTECTED_DIRS:
        if _under(output, (root / name).resolve()):
            raise BatchError(f"output_root overlaps repository source tree: {name}")
    assets = [(settings_path.resolve(), False),
              (resolve_path(settings["protocol_file"], "protocol_file"), False)]
    for key, kind in PATH_KINDS.items():
        if settings["paths"][key] is not None:
            assets.append((resolve_path(settings["paths"][key], key), kind == "directory"))
    for key in REFERENCE_NAMES:
        value = settings["references"][key]["run_dir"]
        if value is not None:
            assets.append((resolve_path(value, key), True))
    for asset, is_directory in assets:
        if _under(asset, output) or (is_directory and _under(output, asset)):
            raise BatchError(f"output_root overlaps declared input: {asset}")
    return output


def _path_state(value: Any, label: str, kind: str) -> dict[str, Any]:
    if value is None:
        return {"path": None, "status": "UNSET"}
    path = resolve_path(value, label)
    good = path.is_dir() if kind == "directory" else path.is_file()
    return {"path": str(path), "status": "PRESENT" if good else
            ("WRONG_KIND" if path.exists() else "MISSING")}


def format_command(args: list[str], *, windows: bool | None = None) -> str:
    """Quote copyable commands for PowerShell on Windows and POSIX sh otherwise."""
    use_windows = os.name == "nt" if windows is None else windows
    if use_windows:
        # PowerShell single-quoted strings do not expand dollars or backticks.
        # An embedded apostrophe is represented by two apostrophes. The call
        # operator is required because the executable itself is a quoted string.
        return "& " + " ".join("'" + value.replace("'", "''") + "'" for value in args)
    return shlex.join(args)


def inspect(settings_path: Path) -> dict[str, Any]:
    settings, raw_settings = read_object(settings_path)
    validate_settings(settings)
    protocol_path = resolve_path(settings["protocol_file"], "protocol_file")
    protocol, raw_protocol = read_object(protocol_path)
    validate_protocol(protocol)
    output = safe_output(settings, settings_path)
    paths = {key: _path_state(value, key, PATH_KINDS[key])
             for key, value in settings["paths"].items()}
    references = {key: {"tracker": item["tracker"],
                        **_path_state(item["run_dir"], key, "directory")}
                  for key, item in settings["references"].items()}
    args = None
    ready = paths["detections_dir"]["status"] == "PRESENT"
    if ready:
        args = ["python", "-X", "utf8", str(REPO_ROOT / "tools/audit_wem_s0.py"),
                "--detections-dir", paths["detections_dir"]["path"],
                "--output-dir", str(output / "r1a_s0"),
                "--low-thresh", str(settings["thresholds"]["low"]),
                "--high-thresh", str(settings["thresholds"]["high"])]
    command = None if args is None else format_command(args)
    public = {
        "status": "INPUT_PATHS_PRESENT" if ready else "NEEDS_LOCAL_DETECTIONS_PATH",
        "meaning": "Path checks only; no data, parity or scientific gate has passed.",
        "settings": str(settings_path), "output_root": str(output),
        "ready_for_r1a_inventory": ready,
        "protocol_sha256": sha256_bytes(raw_protocol),
        "settings_sha256": sha256_bytes(raw_settings),
        "paths": paths, "references": references,
        "later_assets_missing": [
            name for name, state in paths.items()
            if name != "detections_dir" and state["status"] != "PRESENT"
        ] + [f"reference_{key}" for key, value in references.items()
             if value["status"] != "PRESENT"],
        "next_command": command, "next_command_args": args,
        "next_command_shell": "powershell" if os.name == "nt" else "posix",
        "not_checked": NOT_CHECKED,
    }
    return {"settings": settings, "settings_raw": raw_settings,
            "protocol_raw": raw_protocol, "protocol_path": protocol_path,
            "output": output, "report": public}


def source_head() -> str | None:
    if not (REPO_ROOT / ".git").exists():
        return None
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=True, timeout=10)
        value = result.stdout.strip()
        return value if len(value) == 40 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _snapshot_errors(info: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    output = info["output"]
    manifest_path = output / "batch_manifest.json"
    if not manifest_path.is_file():
        return None, ["No batch_manifest.json; this destination is not a prepared WEM batch."]
    try:
        manifest, _ = read_object(manifest_path)
    except BatchError as exc:
        return None, [str(exc)]
    errors = []
    if manifest.get("schema") != "wem_v0.1_prepared" or manifest.get("status") != "PREPARED_NOT_RUN":
        errors.append("Unexpected prepared manifest schema/status.")
    if manifest.get("output_root") != str(output):
        errors.append("Prepared output_root differs from current destination.")
    snapshots = manifest.get("snapshots")
    if not isinstance(snapshots, dict):
        return manifest, errors + ["Prepared snapshots must be an object."]
    for name, content in (("prepared_settings.json", info["settings_raw"]),
                          ("prepared_protocol.json", info["protocol_raw"])):
        expected = sha256_bytes(content)
        snapshot = snapshots.get(name)
        recorded = snapshot.get("sha256") if isinstance(snapshot, dict) else None
        try:
            actual = sha256_bytes((output / name).read_bytes())
        except OSError:
            actual = None
        if recorded != expected or actual != expected:
            errors.append(f"Changed or missing snapshot: {name}")
    return manifest, errors


def prepare(info: dict[str, Any]) -> tuple[dict[str, Any], int]:
    if not info["report"]["ready_for_r1a_inventory"]:
        return {**info["report"], "status": "NOT_PREPARED_MISSING_DETECTIONS"}, 3
    output = info["output"]
    if output.exists():
        manifest, errors = _snapshot_errors(info)
        if errors:
            raise BatchError("Refusing to overwrite existing destination: " + "; ".join(errors))
        return {"status": "PREPARED_NOT_RUN", "idempotent": True,
                "output_root": str(output), "next_command": info["report"]["next_command"],
                "next_command_shell": info["report"]["next_command_shell"],
                "experiment_acceptance": "NOT_EVALUATED"}, 0
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.prepare-", dir=output.parent))
    try:
        snapshots = {}
        for name, raw in (("prepared_settings.json", info["settings_raw"]),
                          ("prepared_protocol.json", info["protocol_raw"])):
            (staging / name).write_bytes(raw)
            snapshots[name] = {"sha256": sha256_bytes(raw), "bytes": len(raw)}
        manifest = {
            "schema": "wem_v0.1_prepared", "status": "PREPARED_NOT_RUN",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "base_commit": BASE_COMMIT, "observed_source_head": source_head(),
            "output_root": str(output), "settings_source": info["report"]["settings"],
            "protocol_source": str(info["protocol_path"]), "snapshots": snapshots,
            "thresholds": info["settings"]["thresholds"],
            "references": info["settings"]["references"],
            "experiment_acceptance": "NOT_EVALUATED", "not_checked": NOT_CHECKED,
        }
        (staging / "batch_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        # Rename into a previously unused destination; never replace an existing batch.
        if output.exists():
            raise BatchError("Destination appeared during preparation; choose another output_root")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"status": "PREPARED_NOT_RUN", "idempotent": False,
            "output_root": str(output), "next_command": info["report"]["next_command"],
            "next_command_shell": info["report"]["next_command_shell"],
            "experiment_acceptance": "NOT_EVALUATED"}, 0


def status(info: dict[str, Any]) -> tuple[dict[str, Any], int]:
    if not info["output"].exists():
        return {"status": "NOT_PREPARED", "output_root": str(info["output"]),
                "experiment_acceptance": "NOT_EVALUATED"}, 3
    manifest, errors = _snapshot_errors(info)
    if errors:
        return {"status": "STALE_OR_INVALID_PREPARATION", "errors": errors,
                "output_root": str(info["output"]),
                "experiment_acceptance": "NOT_EVALUATED"}, 3
    return {
        "status": "PREPARED_NOT_RUN", "output_root": str(info["output"]),
        "snapshot_integrity": "MATCHES_CURRENT_SETTINGS_AND_PROTOCOL",
        "observed_source_head_at_preparation": manifest.get("observed_source_head"),
        "r1a_summary_present": (info["output"] / "r1a_s0/summary.json").is_file(),
        "r1a_summary_meaning": "Presence only; this command does not inspect or accept experiment results.",
        "current_input_paths": info["report"]["paths"],
        "experiment_acceptance": "NOT_EVALUATED", "not_checked": NOT_CHECKED,
    }, 0


def init(output_value: str) -> tuple[dict[str, Any], int]:
    output = resolve_path(output_value, "init output")
    local_settings = (
        output.parent == (REPO_ROOT / "configs/research").resolve()
        and output.name.endswith(".local.json")
    )
    for name in PROTECTED_DIRS:
        if _under(output, (REPO_ROOT / name).resolve()) and not (
                name == "configs" and local_settings):
            raise BatchError(
                "Local settings must be outside source trees or a new "
                "configs/research/*.local.json file")
    if output == REPO_ROOT or output in REPO_ROOT.parents:
        raise BatchError("init output must be a new settings file")
    template, raw = read_object(REPO_ROOT / TEMPLATE)
    validate_settings(template)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as stream:
            stream.write(raw)
    except FileExistsError as exc:
        raise BatchError(f"Refusing to overwrite existing local settings: {output}") from exc
    return {"status": "LOCAL_TEMPLATE_CREATED", "settings": str(output),
            "next_step": "Set paths.detections_dir. Optional later-stage paths may remain null.",
            "experiment_acceptance": "NOT_EVALUATED"}, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init_parser = commands.add_parser(
        "init", help="Create local settings without overwriting; use configs/research/wem_v0.1.local.json")
    init_parser.add_argument("--output", required=True)
    for name in ("check", "prepare", "status"):
        child = commands.add_parser(name)
        child.add_argument("--settings", required=True)
        if name == "check":
            child.add_argument("--strict", action="store_true",
                               help="Exit 3 if the R1a detections directory is unavailable")
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            report, exit_code = init(args.output)
        else:
            info = inspect(resolve_path(args.settings, "settings"))
            if args.command == "check":
                report = info["report"]
                exit_code = 3 if args.strict and not report["ready_for_r1a_inventory"] else 0
            elif args.command == "prepare":
                report, exit_code = prepare(info)
            else:
                report, exit_code = status(info)
    except (BatchError, OSError) as exc:
        report, exit_code = {"status": "ERROR", "error": str(exc)}, 2
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
