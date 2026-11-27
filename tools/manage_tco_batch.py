#!/usr/bin/env python3
"""Prepare and inspect a TCO research batch; never train or run a tracker.

All relative paths are resolved from the repository containing this script.
Exit codes: 0 = requested preparation/check completed; 2 = invalid input or
unsafe destination; 3 = missing assets/unprepared batch (2 with check --strict).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "tco_v0.1_settings"
BASE_COMMIT = "235fe4097066b5023cc84ca8cd6edd5757fd658c"
DEFAULT_TEMPLATE = "configs/research/tco_v0.1.example.json"
CONSTRAINTS = {
    "rgb_only": True,
    "existing_detections_only": True,
    "create_new_ids": False,
    "modify_high_score": False,
    "allow_below_threshold": False,
    "allow_future_frames": False,
    "gt_in_prediction": False,
    "allow_ir": False,
    "preserve_original_low_score_band": True,
}
PATH_KINDS = {
    "baseline_root": "directory",
    "features_dir": "directory",
    "images_root": "directory",
    "gt_root": "directory",
    "trackeval_root": "directory",
    "train_images_root": "directory",
    "train_gt_root": "directory",
    "split_manifest": "json_file",
}
REFERENCE_TRACKERS = {"B": "B_early_moderate", "BG": "evidence_both"}
NOT_CHECKED = [
    "Dataset contents, decoded pixels, annotations and detection/feature compatibility.",
    "Mother-video separation, detector-training overlap or data leakage.",
    "Historical prediction byte parity and full input/source fingerprint equality.",
    "Spatial encoder adapter, cohort exporter, TCO model, training and online integration.",
    "Any real-data metric, T0 reproduction or later experimental acceptance gate.",
]


class ProtocolError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bad_constant(value: str) -> None:
    raise ProtocolError(f"Non-finite JSON constant is not allowed: {value}")


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=_bad_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"JSON root must be an object: {path}")
    return value


def resolve_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{label} must be a non-empty path string")
    if "\x00" in value or "${" in value:
        raise ProtocolError(f"{label} contains an unresolved placeholder or NUL")
    value = value.strip()
    # Accept native absolute paths, but do not reinterpret a Windows drive path
    # as a repository-relative path when this script is running on POSIX.
    win_path = PureWindowsPath(value)
    if os.name != "nt" and (win_path.drive or win_path.root):
        if not value.startswith("/"):
            raise ProtocolError(f"{label} is a Windows path on a non-Windows host")
    if ".." in value.replace("\\", "/").split("/"):
        raise ProtocolError(f"{label} must not contain '..'; use an absolute asset path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _same_or_under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _check_constraints(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be an object")
    if set(value) != set(CONSTRAINTS):
        raise ProtocolError(f"{label} keys must be exactly {sorted(CONSTRAINTS)}")
    for key, expected in CONSTRAINTS.items():
        if value[key] is not expected:
            raise ProtocolError(f"{label}.{key} must be the JSON boolean {str(expected).lower()}")


def validate_settings(settings: dict[str, Any]) -> None:
    if settings.get("schema") != SCHEMA:
        raise ProtocolError(f"settings.schema must equal {SCHEMA}")
    _check_constraints(settings.get("constraints"), "settings.constraints")
    for key in ("protocol_file", "tasks_file", "output_root"):
        resolve_path(settings.get(key), key)
    paths = settings.get("paths")
    if not isinstance(paths, dict) or set(paths) != set(PATH_KINDS):
        raise ProtocolError(f"settings.paths keys must be exactly {sorted(PATH_KINDS)}")
    for key, value in paths.items():
        if value is not None:
            resolve_path(value, f"paths.{key}")
    references = settings.get("references")
    if not isinstance(references, dict) or set(references) != set(REFERENCE_TRACKERS):
        raise ProtocolError("settings.references must contain exactly B and BG")
    for role, tracker in REFERENCE_TRACKERS.items():
        ref = references[role]
        if not isinstance(ref, dict) or set(ref) != {"run_dir", "tracker"}:
            raise ProtocolError(f"references.{role} needs exactly run_dir and tracker")
        if ref["tracker"] != tracker:
            raise ProtocolError(f"references.{role}.tracker must equal {tracker}")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", ref["tracker"]):
            raise ProtocolError(f"Unsafe tracker name for references.{role}")
        if ref["run_dir"] is not None:
            resolve_path(ref["run_dir"], f"references.{role}.run_dir")


def _check_contract_identity(value: dict[str, Any], label: str, schema: str) -> None:
    if value.get("schema") != schema:
        raise ProtocolError(f"{label}.schema must equal {schema}")
    if value.get("base_commit") != BASE_COMMIT:
        raise ProtocolError(f"{label}.base_commit must equal {BASE_COMMIT}")


def _validate_tasks(tasks: dict[str, Any], *, require_contract: bool = False) -> None:
    if require_contract:
        _check_contract_identity(tasks, "tasks", "tco_v0.1_tasks")
    rows = tasks.get("tasks")
    if not isinstance(rows, list) or not rows:
        raise ProtocolError("tasks_file must contain a non-empty tasks list")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
            raise ProtocolError("Every task must be an object with a non-empty string id")
        if row["id"] in seen:
            raise ProtocolError(f"Duplicate task id: {row['id']}")
        seen.add(row["id"])
    if not require_contract:
        return
    dependencies: dict[str, list[str]] = {}
    for row in rows:
        items = row.get("depends_on")
        if not isinstance(items, list) or any(not isinstance(item, str) or item not in seen for item in items):
            raise ProtocolError(f"Task {row['id']} depends_on must be a list of existing task IDs")
        if len(items) != len(set(items)):
            raise ProtocolError(f"Task {row['id']} has duplicate dependencies")
        dependencies[row["id"]] = items
    remaining = {key: set(value) for key, value in dependencies.items()}
    while remaining:
        ready = {key for key, value in remaining.items() if not value}
        if not ready:
            raise ProtocolError(f"Task dependency cycle detected among: {sorted(remaining)}")
        remaining = {key: value - ready for key, value in remaining.items() if key not in ready}


def _safe_output(settings: dict[str, Any], settings_path: Path) -> Path:
    output = resolve_path(settings["output_root"], "output_root")
    if output == REPO_ROOT or output in REPO_ROOT.parents:
        raise ProtocolError("output_root must not be the repository or its ancestor")
    # Protect version-controlled source trees and all declared input assets.
    for name in (".git", "botsort_plus", "configs", "docs", "experiments", "provenance", "results", "tests", "tools"):
        protected = (REPO_ROOT / name).resolve()
        if _same_or_under(output, protected):
            raise ProtocolError(f"output_root overlaps protected repository path: {protected}")
    asset_paths: list[tuple[Path, bool]] = [(settings_path, False)]
    for key, kind in PATH_KINDS.items():
        if settings["paths"][key] is not None:
            asset_paths.append((resolve_path(settings["paths"][key], key), kind == "directory"))
    for role in REFERENCE_TRACKERS:
        value = settings["references"][role]["run_dir"]
        if value is not None:
            asset_paths.append((resolve_path(value, role), True))
    for key in ("protocol_file", "tasks_file"):
        asset_paths.append((resolve_path(settings[key], key), False))
    for asset, is_directory in asset_paths:
        if _same_or_under(asset, output) or (is_directory and _same_or_under(output, asset)):
            raise ProtocolError(f"output_root overlaps declared input: {asset}")
    return output


def inspect(settings_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    settings = read_object(settings_path)
    validate_settings(settings)
    output = _safe_output(settings, settings_path)
    checks: list[dict[str, Any]] = []
    missing: list[str] = []
    errors: list[str] = []
    sources: dict[str, dict[str, str]] = {
        "settings": {"path": str(settings_path), "sha256": sha256(settings_path)}
    }
    parsed: dict[str, Any] = {}

    def json_source(name: str, path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            missing.append(f"{name}: missing file {path}")
            checks.append({"item": name, "path": str(path), "status": "MISSING"})
            return None
        try:
            data = read_object(path)
        except ProtocolError as exc:
            errors.append(str(exc))
            checks.append({"item": name, "path": str(path), "status": "INVALID_JSON"})
            return None
        sources[name] = {"path": str(path), "sha256": sha256(path)}
        checks.append({"item": name, "path": str(path), "status": "JSON_OBJECT_READABLE", "sha256": sources[name]["sha256"]})
        return data

    for key in ("protocol_file", "tasks_file"):
        path = resolve_path(settings[key], key)
        parsed[key] = json_source(key, path)
    if parsed["protocol_file"] is not None:
        try:
            _check_contract_identity(parsed["protocol_file"], "protocol", "tco_v0.1_protocol")
            _check_constraints(parsed["protocol_file"].get("constraints"), "protocol.constraints")
        except ProtocolError as exc:
            errors.append(str(exc))
    if parsed["tasks_file"] is not None:
        try:
            _validate_tasks(parsed["tasks_file"], require_contract=True)
        except ProtocolError as exc:
            errors.append(str(exc))

    for key, kind in PATH_KINDS.items():
        value = settings["paths"][key]
        if value is None:
            missing.append(f"paths.{key}: not configured")
            checks.append({"item": key, "status": "NOT_CONFIGURED"})
            continue
        path = resolve_path(value, key)
        if kind == "json_file":
            json_source(key, path)
        else:
            exists = path.is_dir()
            checks.append({"item": key, "path": str(path), "status": "DIRECTORY_EXISTS_ONLY" if exists else "MISSING_DIRECTORY"})
            if not exists:
                missing.append(f"paths.{key}: missing directory {path}")

    for role, tracker in REFERENCE_TRACKERS.items():
        value = settings["references"][role]["run_dir"]
        if value is None:
            missing.append(f"references.{role}.run_dir: not configured")
            checks.append({"item": f"reference_{role}", "status": "NOT_CONFIGURED"})
            continue
        run_dir = resolve_path(value, f"references.{role}.run_dir")
        json_source(f"reference_{role}", run_dir / "protocols" / f"{tracker}.json")

    report = {
        "schema": "tco_v0.1_preparation_check",
        "status": "INVALID" if errors else "INPUTS_MISSING" if missing else "READY_TO_PREPARE_ONLY",
        "experimental_execution": "NOT_RUN",
        "output_root": str(output),
        "errors": errors,
        "missing": missing,
        "checks": checks,
        "fingerprints": sources,
        "not_checked": NOT_CHECKED,
        "scope": "Paths and small metadata only. No detector, tracker, model or dataset evaluation ran.",
    }
    return settings, report, parsed


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def init_config(template: Path, destination: Path) -> dict[str, Any]:
    settings = read_object(template)
    validate_settings(settings)
    if destination.exists():
        raise ProtocolError(f"Refusing to overwrite configuration: {destination}")
    if not destination.parent.is_dir():
        raise ProtocolError(f"Configuration parent must already exist: {destination.parent}")
    write_json(destination, settings, exclusive=True)
    return {"status": "CONFIG_CREATED", "path": str(destination), "experimental_execution": "NOT_RUN"}


def prepare(settings_path: Path) -> tuple[dict[str, Any], int]:
    settings, report, parsed = inspect(settings_path)
    if report["errors"]:
        return report, 2
    if report["missing"]:
        return report, 3
    output = resolve_path(settings["output_root"], "output_root")
    if output.exists():
        raise ProtocolError(f"Refusing to overwrite existing output_root: {output}")
    # All input inspection completed before making any output directories.
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.prepare.lock"
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ProtocolError(f"Another preparation or stale lock exists: {lock_path}") from exc
    temporary: Path | None = None
    try:
        with os.fdopen(lock_fd, "w", encoding="utf-8") as lock:
            lock.write(f"pid={os.getpid()}\n")
        if output.exists():
            raise ProtocolError(f"Refusing to overwrite existing output_root: {output}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.preparing-", dir=output.parent))
        frozen = temporary / "frozen"
        frozen.mkdir()
        snapshots: dict[str, dict[str, str]] = {}
        for name, record in report["fingerprints"].items():
            source = Path(record["path"])
            if sha256(source) != record["sha256"]:
                raise ProtocolError(f"Input metadata changed during preparation: {source}")
            destination = frozen / f"{name}.json"
            shutil.copyfile(source, destination)
            if sha256(destination) != record["sha256"]:
                raise ProtocolError(f"Snapshot copy mismatch: {source}")
            snapshots[name] = {"file": f"frozen/{name}.json", "sha256": record["sha256"]}
        ledger = {
            "schema": "tco_v0.1_task_ledger",
            "status": "PREPARED_NOT_RUN",
            "tasks": [
                {"id": row["id"], "title": row.get("title", row["id"]), "state": "PLANNED", "evidence": []}
                for row in parsed["tasks_file"]["tasks"]
            ],
            "note": "Preparation does not pass T0 or implement the cohort/model/training/online experiments.",
        }
        write_json(temporary / "task_ledger.json", ledger)
        write_json(temporary / "preparation_check.json", report)
        manifest = {
            "schema": "tco_v0.1_prepared_batch",
            "status": "PREPARED_NOT_RUN",
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "design_base_commit": BASE_COMMIT,
            "preparation_tool": {"path": "tools/manage_tco_batch.py", "sha256": sha256(Path(__file__).resolve())},
            "experimental_execution": "NOT_RUN",
            "sources": report["fingerprints"],
            "snapshots": snapshots,
            "constraints": CONSTRAINTS,
            "not_checked": NOT_CHECKED,
            "ledger_initial_sha256": sha256(temporary / "task_ledger.json"),
            "preparation_check_sha256": sha256(temporary / "preparation_check.json"),
        }
        write_json(temporary / "batch_manifest.json", manifest)
        if output.exists():
            raise ProtocolError(f"Output was created by another process: {output}")
        # The cooperative exclusive lock protects concurrent runs of this tool;
        # rename publishes the completed directory as one operation.
        os.rename(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)
        lock_path.unlink(missing_ok=True)
    return {"status": "PREPARED_NOT_RUN", "output_root": str(output), "task_count": len(ledger["tasks"]), "experimental_execution": "NOT_RUN", "next_action": "Follow T0 in the experiment plan; preparation is not a reproduction result."}, 0


def status(settings_path: Path) -> tuple[dict[str, Any], int]:
    settings = read_object(settings_path)
    validate_settings(settings)
    output = _safe_output(settings, settings_path)
    manifest_path = output / "batch_manifest.json"
    if not manifest_path.is_file():
        return {"status": "NOT_PREPARED", "output_root": str(output), "experimental_execution": "NOT_RUN"}, 3
    manifest = read_object(manifest_path)
    if manifest.get("schema") != "tco_v0.1_prepared_batch":
        raise ProtocolError("Unrecognized batch manifest schema")
    if manifest.get("design_base_commit") != BASE_COMMIT:
        raise ProtocolError("Batch manifest design_base_commit does not match this design")
    _check_constraints(manifest.get("constraints"), "manifest.constraints")
    mismatches: list[str] = []
    tool_source_changed = manifest.get("preparation_tool", {}).get("sha256") != sha256(Path(__file__).resolve())
    if tool_source_changed:
        mismatches.append("preparation tool source changed since this batch was prepared")
    for name, record in manifest.get("snapshots", {}).items():
        relative = record.get("file")
        if not isinstance(relative, str) or ".." in relative.replace("\\", "/").split("/"):
            raise ProtocolError(f"Invalid snapshot path: {name}")
        snapshot = (output / relative).resolve()
        if not _same_or_under(snapshot, output):
            raise ProtocolError(f"Snapshot escapes output_root: {name}")
        if not snapshot.is_file() or sha256(snapshot) != record.get("sha256"):
            mismatches.append(f"frozen snapshot: {name}")
    original_settings_sha = manifest.get("sources", {}).get("settings", {}).get("sha256")
    if sha256(settings_path) != original_settings_sha:
        mismatches.append("current settings differ from the prepared settings")
    check_path = output / "preparation_check.json"
    if not check_path.is_file() or sha256(check_path) != manifest.get("preparation_check_sha256"):
        mismatches.append("preparation_check.json")
    ledger = read_object(output / "task_ledger.json")
    _validate_tasks(ledger)
    ledger_changed = sha256(output / "task_ledger.json") != manifest.get("ledger_initial_sha256")
    return {
        "status": "SNAPSHOT_MISMATCH" if mismatches else "LEDGER_CHANGED_UNVERIFIED" if ledger_changed else "PREPARED_NOT_RUN",
        "output_root": str(output),
        "mismatches": mismatches,
        "ledger_changed": ledger_changed,
        "tool_source_changed": tool_source_changed,
        "ledger_reported_tasks": ledger["tasks"],
        "experimental_acceptance": "NOT_VALIDATED_BY_THIS_TOOL",
        "note": "Task labels are user-supplied records, never automatic proof of implementation or experiment success.",
    }, 2 if mismatches else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Copy a UTF-8 local configuration without overwriting")
    init.add_argument("--settings-template", default=DEFAULT_TEMPLATE)
    init.add_argument("--output-config", required=True)
    check = sub.add_parser("check", help="Inspect configuration, paths and small metadata; run no experiments")
    check.add_argument("--settings", required=True)
    check.add_argument("--strict", action="store_true", help="Return 2 rather than 3 for missing assets")
    prep = sub.add_parser("prepare", help="Freeze metadata and create a fresh PLANNED task ledger")
    prep.add_argument("--settings", required=True)
    inspect_status = sub.add_parser("status", help="Read prepared state; validate snapshots, never certify results")
    inspect_status.add_argument("--settings", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init":
            result = init_config(resolve_path(args.settings_template, "settings-template"), resolve_path(args.output_config, "output-config"))
            code = 0
        else:
            settings_path = resolve_path(args.settings, "settings")
            if args.command == "check":
                _, result, _ = inspect(settings_path)
                code = 2 if result["errors"] else (2 if args.strict else 3) if result["missing"] else 0
            elif args.command == "prepare":
                result, code = prepare(settings_path)
            else:
                result, code = status(settings_path)
    except (ProtocolError, OSError) as exc:
        result, code = {"status": "INVALID", "error": str(exc), "experimental_execution": "NOT_RUN"}, 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
