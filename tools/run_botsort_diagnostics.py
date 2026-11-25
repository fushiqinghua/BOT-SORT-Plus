#!/usr/bin/env python3
"""Run four observational diagnostics on one frozen B reference.

Prediction runs in separate GT-free child processes. The offline analysis only
starts after input identities and MOT files pass an on/off parity check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from botsort_plus.config import PlusConfig
from botsort_plus.io import sha256, write_json
from tools.evaluate_botsort_plus import validate_gt, validate_runs

BASE_COMMIT = "667eb40a6aa624c35fb86eb73656bb334b107a11"
CONFIG = REPO / "configs/botsort_plus/B_early_moderate.json"
# Hash of the complete EFFECTIVE configuration at BASE_COMMIT, including all
# dataclass defaults. JSON formatting and Windows line endings do not affect it.
FROZEN_CONFIG_SHA256 = "ac9816cad232ed814f4d810b377e9f146daefc65338f75a46f282e698e6a333d"
SETTINGS_SCHEMA = "botsort_diagnostics_settings_v1"
SUITE_SCHEMA = "botsort_diagnostics_suite_v1"
KEYS = {"schema", "baseline_root", "features_dir", "images_root", "gt_root",
        "output_root", "runtime_root", "sequences", "max_review"}
IDENTITY_KEYS = ("sequences", "subset", "native_config", "detector_sha256",
                 "rgb_encoder_sha256", "source_protocol_sha256",
                 "detection_fingerprints", "encoder_contracts", "feature_hashes",
                 "implementation_sha256", "boxmot_source_hashes", "decoded_image_hashes")


def read_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def resolve_path(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Paths must be nonempty strings")
    path = Path(os.path.expandvars(value)).expanduser()
    return (path if path.is_absolute() else REPO / path).resolve()


def settings(args):
    data = read_json(args.settings)
    if data.get("schema") != SETTINGS_SCHEMA or set(data) - KEYS:
        raise ValueError(f"Expected {SETTINGS_SCHEMA}; unknown keys: {sorted(set(data)-KEYS)}")
    for key in ("baseline_root", "features_dir", "images_root", "gt_root", "output_root"):
        data[key] = resolve_path(data.get(key))
    data["runtime_root"] = resolve_path(data["runtime_root"]) if data.get("runtime_root") else None
    if getattr(args, "output_root", None):
        data["output_root"] = args.output_root.resolve()
    if getattr(args, "sequences", None):
        data["sequences"] = list(args.sequences)
    sequences = data.get("sequences")
    if sequences is not None and (not isinstance(sequences, list) or not sequences
            or any(not isinstance(v, str) or not v for v in sequences)
            or len(sequences) != len(set(sequences))):
        raise ValueError("sequences must be null or a nonempty list of unique names")
    maximum = data.setdefault("max_review", 120)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
        raise ValueError("max_review must be a nonnegative integer")
    for key in ("baseline_root", "features_dir", "images_root", "gt_root"):
        # Prevent diagnostics from writing inside any source dataset/cache.
        if data["output_root"] == data[key] or data[key] in data["output_root"].parents:
            raise ValueError(f"output_root must be outside {key}")
    cfg = frozen_config()
    if (cfg.name != "B_early_moderate" or cfg.adapter_mode != "v2"
            or not cfg.recovery.enabled or cfg.recovery.early_evidence != "moderate"
            or cfg.birth.enabled or cfg.identity.enabled or cfg.appearance.enabled
            or cfg.raw_ir.enabled or cfg.recovery.output_prediction_frames
            or cfg.native_overrides):
        raise ValueError("The diagnostic reference configuration was changed; use the frozen B_early_moderate")
    data["tracker_name"] = cfg.name
    return data


def frozen_config(path=None):
    cfg = PlusConfig.load(path or CONFIG)
    payload = json.dumps(cfg.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    if hashlib.sha256(payload.encode("utf-8")).hexdigest() != FROZEN_CONFIG_SHA256:
        raise ValueError("The complete B_early_moderate configuration differs from the frozen reference")
    return cfg


def serialize_settings(data):
    return {k: str(v) if isinstance(v, Path) else v for k, v in data.items()}


def command(data, output, *, diagnostics=False, preflight=False):
    result = [sys.executable, "-X", "utf8", str(REPO / "tools/run_botsort_plus.py"),
              "--baseline-root", str(data["baseline_root"]), "--features-dir", str(data["features_dir"]),
              "--images-root", str(data["images_root"]), "--config", str(CONFIG),
              "--output-dir", str(output), "--trace"]
    if data.get("runtime_root"):
        result += ["--runtime-root", str(data["runtime_root"])]
    if data.get("sequences"):
        result += ["--sequences", *data["sequences"]]
    if diagnostics:
        result.append("--diagnostics")
    if preflight:
        result.append("--preflight")
    return result


def child_env():
    result = os.environ.copy()
    result["PYTHONUTF8"] = "1"
    result["PYTHONIOENCODING"] = "utf-8"
    return result


def run_child(argv, log=None):
    print("Running:", subprocess.list2cmdline(argv), flush=True)
    if log is None:
        completed = subprocess.run(argv, cwd=REPO, env=child_env(), text=True,
                                   encoding="utf-8", capture_output=True)
        if completed.returncode:
            raise RuntimeError(completed.stdout + "\n" + completed.stderr)
        return completed.stdout
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(argv, cwd=REPO, env=child_env(), text=True,
                                   encoding="utf-8", errors="replace", stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        try:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            status = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if status:
        raise RuntimeError(f"Replay failed with exit code {status}; see {log}")
    return None


def preflight(data):
    # This child checks only detector/features/images. GT is checked separately
    # below; it never enters the prediction command or its process.
    output = run_child(command(data, data["output_root"] / "observed", diagnostics=True, preflight=True))
    report = json.loads(output)
    if report.get("status") != "READY":
        raise ValueError("Prediction inputs are not READY")
    gt = validate_gt(data["gt_root"], report["sequences"])
    from botsort_plus.diagnostics.common import load_gt
    gt["structure"] = {}
    for sequence, frames in report["sequences"].items():
        rows, ignored = load_gt(data["gt_root"] / sequence / "gt" / "gt.txt", frames)
        gt["structure"][sequence] = {"frames": frames,
                                      "marked_annotations": sum(map(len, rows.values())),
                                      "ignored_zero_marked_rows": ignored}
    runtime = data.get("runtime_root") or os.environ.get("TRACKING_RUNTIME_ROOT")
    verify_code = (
        "import sys,json; from pathlib import Path; "
        "runtime=sys.argv[1]; "
        "sys.path.insert(0,str(Path(runtime)/'tracking_runtime')) if runtime else None; "
        "from botsort_plus.native_adapter import verify_upstream; "
        "from botsort_plus.adapter_v2 import verify_v2_upstream; "
        "from botsort_plus.diagnostics.recorder import verify_cmc_sources; "
        "print(json.dumps(dict(verify_upstream(),**verify_v2_upstream(),**verify_cmc_sources())))"
    )
    verified = run_child([sys.executable, "-X", "utf8", "-c", verify_code, str(runtime or "")])
    return {"schema": SUITE_SCHEMA, "status": "READY", "prediction_preflight": report,
            "ground_truth_preflight": gt, "runtime_check": verified.strip(),
            "note": "GT was inspected only by offline preflight, not by the prediction child."}


def verify_pair(reference, observed, name, output=None):
    ref = validate_runs(reference, [name])
    obs = validate_runs(observed, [name])
    a, b = ref["protocols"][name], obs["protocols"][name]
    for key in IDENTITY_KEYS:
        if not a.get(key) or not b.get(key) or a[key] != b[key]:
            # False subset is a valid complete-data declaration.
            if key == "subset" and a.get(key) is False and b.get(key) is False:
                continue
            raise ValueError(f"On/off input or implementation mismatch: {key}")
    if a["config"] != b["config"]:
        raise ValueError("On/off tracker configuration mismatch")
    if a.get("diagnostics", {}).get("enabled", False):
        raise ValueError("Reference run must have diagnostics disabled")
    diag = b.get("diagnostics", {})
    if (not diag.get("enabled") or diag.get("status") != "COMPLETE"
            or diag.get("schema") != "botsort_motion_diagnostics_v1"):
        raise ValueError("Observed diagnostic trace is incomplete")
    if set(diag.get("traces", {})) != set(b["sequences"]):
        raise ValueError("Observed trace set differs from replay sequences")
    comparisons = {}
    for seq in a["sequences"]:
        left = Path(reference) / "trackers" / name / "data" / f"{seq}.txt"
        right = Path(observed) / "trackers" / name / "data" / f"{seq}.txt"
        equal = left.read_bytes() == right.read_bytes()
        if not equal:
            raise ValueError(f"Diagnostics changed MOT output for {seq}; no causal diagnosis will run")
        # Both runs are made with --trace. Recovery and accepted-match events
        # must also remain byte-identical, even when final boxes happen to agree.
        ref_events = Path(reference) / "events" / name / f"{seq}.jsonl"
        obs_events = Path(observed) / "events" / name / f"{seq}.jsonl"
        if ref_events.read_bytes() != obs_events.read_bytes():
            raise ValueError(f"Diagnostics changed original tracking events for {seq}")
        relative = Path("diagnostics") / name / f"{seq}.jsonl.gz"
        trace = Path(observed) / relative
        manifest_relative = relative.with_suffix("").with_suffix(".json")
        manifest_path = Path(observed) / manifest_relative
        declaration = diag["traces"][seq]
        if (Path(declaration.get("path", "")) != relative
                or Path(declaration.get("manifest_path", "")) != manifest_relative
                or declaration.get("status") != "COMPLETE"
                or declaration.get("records") != b["sequences"][seq]
                or declaration.get("manifest_sha256") != sha256(manifest_path)
                or declaration.get("data_sha256") != sha256(trace)):
            raise ValueError(f"Incomplete or changed diagnostic trace for {seq}")
        manifest = read_json(manifest_path)
        if (manifest.get("schema") != diag["schema"] or manifest.get("status") != "COMPLETE"
                or manifest.get("sequence") != seq or manifest.get("trace_filename") != trace.name
                or manifest.get("records") != b["sequences"][seq]
                or manifest.get("expected_frames") != b["sequences"][seq]
                or manifest.get("data_sha256") != declaration["data_sha256"]):
            raise ValueError(f"Invalid diagnostic manifest for {seq}")
        comparisons[seq] = {"byte_identical": True, "sha256": sha256(right),
                            "original_events_byte_identical": True,
                            "original_events_sha256": sha256(obs_events),
                            "trace_sha256": declaration["data_sha256"],
                            "frames": a["sequences"][seq]}
    result = {"schema": "botsort_diagnostics_parity_v1", "status": "PASS", "tracker": name,
              "reference_protocol_sha256": a["validated_protocol_sha256"],
              "observed_protocol_sha256": b["validated_protocol_sha256"],
              "compared_identity_fields": list(IDENTITY_KEYS), "sequences": comparisons,
              "meaning": "Identical inputs, MOT bytes and original events with diagnostics off/on; "
                         "trace files verified against COMPLETE manifests. Not an accuracy result."}
    if output:
        if Path(output).exists():
            raise FileExistsError(output)
        write_json(output, result)
    return result


def analyze(data, output=None):
    from botsort_plus.diagnostics.analysis import analyze_run
    root, name = data["output_root"], data["tracker_name"]
    verify_pair(root / "reference", root / "observed", name)
    return analyze_run(root / "observed", name, data["gt_root"], output or root / "analysis",
                       reference_run_root=root / "reference", reference_tracker_name=name)


def blur(data, output=None, manual_labels=None, audit_dir=None):
    from botsort_plus.diagnostics.blur import analyze_blur
    root, name = data["output_root"], data["tracker_name"]
    verify_pair(root / "reference", root / "observed", name)
    return analyze_blur(root / "observed", name, data["baseline_root"], data["features_dir"],
                        data["images_root"], audit_dir or root / "analysis", output or root / "blur",
                        manual_labels=manual_labels, max_review=data["max_review"])


def package_evidence(root, *, include_traces=False, output=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    dest = Path(output).resolve() if output else root.parent / f"{root.name}_evidence.zip"
    if dest.exists():
        raise FileExistsError(f"Evidence archive already exists: {dest}")
    if dest.suffix.lower() != ".zip":
        raise ValueError("Evidence package output must end with .zip")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Do not follow symlinks while packaging evidence: {path}")
        if not path.is_file() or path.suffix in (".tmp", ".zip"):
            continue
        rel = path.relative_to(root)
        if "diagnostics" in rel.parts and path.suffix == ".gz":
            if not include_traces or not path.name.endswith(".jsonl.gz"):
                continue
        elif path.suffix.lower() not in {".json", ".jsonl", ".csv", ".md", ".txt", ".png", ".jpg"}:
            continue
        if path.name == "gt.txt" or "detections" in rel.parts or "features" in rel.parts:
            continue
        files.append(path)
    if not files:
        raise ValueError("No diagnostic evidence to package")
    manifest = {"schema": "botsort_diagnostics_evidence_v1", "include_traces": include_traces,
                "files": {str(p.relative_to(root)).replace(os.sep, "/"): sha256(p) for p in files},
                "note": "Packages diagnostic reports, protocols, logs, MOT results and review sheets from this "
                        "experiment root. Feature arrays, weights, gt.txt and partial compressed files are excluded. "
                        "Review sheets contain source imagery. Original external inputs are not traversed."}
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(".zip.tmp")
    if temp.exists():
        raise FileExistsError(temp)
    try:
        with zipfile.ZipFile(temp, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files:
                archive.write(path, str(path.relative_to(root)))
            archive.writestr("EVIDENCE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
        temp.replace(dest)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return {"path": str(dest), "sha256": sha256(dest), "files": len(files)}


def run_suite(data):
    if data["output_root"].exists():
        raise FileExistsError("Choose a new output_root. Existing experiments are never overwritten or silently resumed.")
    default_archive = data["output_root"].parent / f"{data['output_root'].name}_evidence.zip"
    if default_archive.exists() or default_archive.with_suffix(".zip.tmp").exists():
        raise FileExistsError(f"Choose another output_root; its evidence archive already exists: {default_archive}")
    readiness = preflight(data)
    root = data["output_root"]
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "preflight.json", readiness)
    source_files = [CONFIG, Path(__file__), REPO / "tools/run_botsort_plus.py"]
    source_files += sorted((REPO / "botsort_plus").rglob("*.py"))
    experiment = {"schema": SUITE_SCHEMA, "status": "RUNNING", "base_commit": BASE_COMMIT,
                  "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "settings": serialize_settings(data), "tracker_config": PlusConfig.load(CONFIG).to_dict(),
                  "source_sha256": {str(p.relative_to(REPO)).replace(os.sep, "/"): sha256(p) for p in source_files},
                  "prediction_reads_gt": False, "stages": {},
                  "research_scope": "Observational D1-D4. No new filter, association, ReID training or accuracy claim."}
    write_json(root / "experiment.json", experiment)
    try:
        for stage, flag in (("reference", False), ("observed", True)):
            run_child(command(data, root / stage, diagnostics=flag), root / "logs" / f"{stage}.txt")
            experiment["stages"][stage] = "COMPLETE"
            write_json(root / "experiment.json", experiment)
        verify_pair(root / "reference", root / "observed", data["tracker_name"], root / "verification.json")
        experiment["stages"]["parity"] = "PASS"
        analyze(data)
        experiment["stages"]["D1_D2_D3"] = "COMPLETE"
        write_json(root / "experiment.json", experiment)
        blur(data)
        experiment["stages"]["D4_quality_proxies"] = "COMPLETE"
        experiment["stages"]["D4_natural_blur_labels"] = "PENDING_MANUAL_REVIEW"
        experiment["status"] = "COMPLETE"
    except BaseException:
        experiment["status"] = "FAILED"
        experiment["error"] = traceback.format_exc()
        raise
    finally:
        write_json(root / "experiment.json", experiment)
    try:
        packed = package_evidence(root)
    except Exception:
        experiment["status"] = "ANALYSIS_COMPLETE_PACKAGING_FAILED"
        experiment["packaging_error"] = traceback.format_exc()
        experiment["recovery"] = "Analysis is complete. Retry package --output with a fresh .zip destination."
        write_json(root / "experiment.json", experiment)
        raise
    print(json.dumps({"status": "COMPLETE", "output_root": str(root), "evidence": packed,
                      "next": "Review blur/manual_review.csv; natural blur attribution remains pending."},
                     ensure_ascii=False, indent=2))
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run", "verify", "analyze", "blur", "package"):
        sub = commands.add_parser(name)
        sub.add_argument("--settings", type=Path, required=True)
        sub.add_argument("--output-root", type=Path, help="Override experiment root (use a fresh one for run)")
        if name in ("preflight", "run"):
            sub.add_argument("--sequences", nargs="+", help="Optional smoke-test sequence subset")
        if name in ("analyze", "blur", "package"):
            sub.add_argument("--output", type=Path, help="Fresh analysis directory or .zip destination")
        if name == "blur":
            sub.add_argument("--manual-labels", type=Path)
            sub.add_argument("--audit-dir", type=Path)
        if name == "package":
            sub.add_argument("--include-traces", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    data = settings(args)
    root, name = data["output_root"], data["tracker_name"]
    if args.command == "run":
        return run_suite(data)
    if args.command == "preflight":
        result = preflight(data)
    elif args.command == "verify":
        result = verify_pair(root / "reference", root / "observed", name)
    elif args.command == "analyze":
        result = analyze(data, args.output)
    elif args.command == "blur":
        result = blur(data, args.output, args.manual_labels, args.audit_dir)
    else:
        result = package_evidence(root, include_traces=args.include_traces, output=args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
