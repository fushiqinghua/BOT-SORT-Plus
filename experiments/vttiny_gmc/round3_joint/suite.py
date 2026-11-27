"""Orchestrate J0-J3 and E3 closure; J4 waits for actual diagnostic evidence."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import zipfile

from .common import (REPO, SCHEMA, BASE_COMMIT, METHODS, CONTROLS, ROLES, SMOKE,
                     read_json, write_json, sha256, json_digest, load_settings,
                     new_directory, frozen_config, select_sequences, source_hashes,
                     assert_same_inputs, child_environment, prediction_settings)


def contract(settings):
    return {"schema": SCHEMA, "prediction_settings": prediction_settings(settings),
            "source_sha256": source_hashes(), "B_config": frozen_config("B_early_moderate"),
            "methods": METHODS}


def run_child(settings, arguments, log):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(map(str, arguments)), flush=True)
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen([sys.executable, "-X", "utf8", "-u", *map(str, arguments)],
                                   cwd=REPO, env=child_environment(settings), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f"Child exited {code}; see {log}")


def resolved_file(settings, directory):
    path = Path(directory) / "resolved_settings.json"
    write_json(path, settings)
    return path


def reference_entries(settings):
    result = {}
    source = settings.get("references", {})
    for role, tracker in ROLES.items():
        entry = source.get(role, source.get(tracker))
        if entry is None:
            if role == "evidence":
                continue
            raise ValueError(f"references.{role} is required; point it at the existing joint full run")
        result[role] = dict(entry, tracker=entry.get("tracker", tracker))
        if result[role]["tracker"] != tracker:
            raise ValueError(f"Reference role {role} requires historical tracker {tracker}")
    return result


def validate_references(settings, sequences):
    from tools.evaluate_botsort_plus import validate_runs
    protocols, metadata = {}, {}
    for role, entry in reference_entries(settings).items():
        root, name = Path(entry["run_dir"]), entry["tracker"]
        checked = validate_runs(root, [name])
        p = checked["protocols"][name]
        if not set(sequences) <= set(p["sequences"]):
            raise ValueError(f"Reference {role} lacks requested sequences")
        protocols[role] = p
        metadata[role] = {"run_dir": str(root), "tracker": name,
                          "protocol_sha256": sha256(root / "protocols" / (name + ".json")),
                          "prediction_sha256": {s: p["output_prediction_hashes"][s] for s in sequences}}
    for role, p in protocols.items():
        assert_same_inputs(protocols["B"], p, sequences)
    expected_b = dict(protocols["B"]["config"])
    from botsort_plus.config import PlusConfig
    current_b = PlusConfig.load(REPO / "configs/botsort_plus/B_early_moderate.json").to_dict()
    expected_b.pop("name", None)
    current_b.pop("name", None)
    if expected_b != current_b:
        raise ValueError("Historical B configuration does not match the frozen current B")
    return protocols, metadata


def track(settings, method, output, sequences, *, probe="off", preflight=False):
    output = Path(output)
    clean = output / "prediction_settings.json"
    if not clean.exists():
        write_json(clean, prediction_settings(settings))
    elif read_json(clean) != prediction_settings(settings):
        raise ValueError("Output contains a different prediction contract")
    args = ["-m", "experiments.vttiny_gmc.round3_joint.worker", "--settings", clean,
            "--method", method, "--output-dir", output, "--probe-mode", probe,
            "--sequences", *sequences]
    if preflight:
        args.append("--preflight")
    run_child(settings, args, output / "logs" / (method + ".log"))
    if not preflight:
        # The parent compares provenance; prediction itself never reads GT or
        # historical trajectory outcomes. Reject changed same-path inputs.
        entry = reference_entries(settings)["B"]
        reference = read_json(Path(entry["run_dir"]) / "protocols" / (entry["tracker"] + ".json"))
        pp = output / "protocols" / (method + ".json")
        current = read_json(pp)
        try:
            assert_same_inputs(reference, current, sequences)
        except ValueError as error:
            current.update(status="FAILED", error=str(error))
            write_json(pp, current)
            raise


def preflight(settings, output, sequences=None):
    output = new_directory(output)
    report = {"schema": SCHEMA, "status": "RUNNING", "base_commit": BASE_COMMIT}
    write_json(output / "manifest.json", report)
    try:
        lengths = select_sequences(settings, sequences)
        for key in ("baseline_root", "features_dir", "images_root", "gt_root", "trackeval_root"):
            if not settings.get(key) or not Path(settings[key]).is_dir():
                raise FileNotFoundError(f"Configure an existing {key}: {settings.get(key)}")
        for seq in lengths:
            if not settings.get("original_video_groups", {}).get(seq):
                raise ValueError(f"Missing explicit source-video group: {seq}")
        protocols, references = validate_references(settings, lengths)
        from tools.evaluate_botsort_plus import verify_trackeval, validate_gt
        evaluation = verify_trackeval(settings["trackeval_root"], REPO)
        gt = validate_gt(settings["gt_root"], lengths)
        # Source and input verification runs in a GT-isolated child.
        track(settings, "B_adaptive_no_dual_low", output / "tracking_preflight", lengths, preflight=True)
        prepared = read_json(output / "tracking_preflight/preflight.json")
        assert_same_inputs(protocols["B"], prepared, lengths, decoded=False)
        if settings["gmc_mode"] == "cached":
            for camera in ("skip_only", "adaptive_gmc", "native"):
                src = settings.get("warp_sources", {}).get(camera)
                if not src:
                    raise ValueError(f"Cached mode needs warp_sources.{camera}")
                for seq in lengths:
                    path = Path(src["run_dir"]) / "warps" / src["method"] / (seq + ".jsonl.gz")
                    if not path.is_file():
                        raise FileNotFoundError(path)
        report.update(status="PASS", sequences=lengths, frames=sum(lengths.values()), references=references,
                      trackeval=evaluation, ground_truth=gt, contract=contract(settings),
                      contract_sha256=json_digest(contract(settings)),
                      cache_note="Cached per-frame shape/order/policy/pixel checks also run during tracking",
                      dataset_execution="Preflight only; no new tracking accuracy measured")
    except Exception:
        report.update(status="FAILED", error=traceback.format_exc())
        write_json(output / "manifest.json", report)
        raise
    write_json(output / "manifest.json", report)
    return report


def compare_runs(left_root, right_root, method, sequences, *, expected_reference=None):
    from tools.evaluate_botsort_plus import validate_runs
    left = validate_runs(left_root, [method])["protocols"][method]
    right = validate_runs(right_root, [method])["protocols"][method]
    assert_same_inputs(left, right, sequences)
    results = {}
    for seq in sequences:
        if left["output_prediction_hashes"][seq] != right["output_prediction_hashes"][seq]:
            raise ValueError(f"Observer changed MOT: {method}/{seq}")
        a = Path(left_root) / "events" / method / (seq + ".jsonl")
        b = Path(right_root) / "events" / method / (seq + ".jsonl")
        if not a.is_file() or not b.is_file() or sha256(a) != sha256(b):
            raise ValueError(f"Observer changed or omitted events: {method}/{seq}")
        if expected_reference and left["output_prediction_hashes"][seq] != expected_reference["output_prediction_hashes"][seq]:
            raise ValueError(f"Replay differs from historical {method}/{seq}; stop and inspect input/runtime/warps")
        results[seq] = {"MOT": left["output_prediction_hashes"][seq], "events": sha256(a)}
    return results


def smoke(settings, output, sequences=None):
    output = new_directory(output)
    target = select_sequences(settings)
    default_smoke = [s for s in SMOKE if s in target] or list(target)
    chosen = select_sequences(settings, sequences or default_smoke)
    protocols, metadata = validate_references(settings, chosen)
    report = {"status": "RUNNING", "sequences": chosen, "reference_manifest": metadata,
              "contract_sha256": json_digest(contract(settings)), "pairs": {}}
    write_json(output / "verification.json", report)
    try:
        # Native B tests core hooks; BG tests the archived guard; new arms test
        # both candidate sets. Each actual observer has an off/on comparison.
        track(settings, "B_early_moderate", output / "base", chosen)
        b = read_json(output / "base/protocols/B_early_moderate.json")
        assert_same_inputs(protocols["B"], b, chosen)
        if any(b["output_prediction_hashes"][s] != protocols["B"]["output_prediction_hashes"][s] for s in chosen):
            raise ValueError("B replay does not reproduce the frozen historical MOT")
        for method in ["B_early_moderate_dual_low", *CONTROLS]:
            track(settings, method, output / "off", chosen)
            track(settings, method, output / "on", chosen, probe="full")
            report["pairs"][method] = compare_runs(output / "off", output / "on", method, chosen,
                expected_reference=protocols["BG"] if method == "B_early_moderate_dual_low" else None)
        report.update(status="PASS", note="Historical B/BG MOT equality plus all observer off/on MOT and event equality")
    except Exception:
        report.update(status="FAILED", error=traceback.format_exc())
        write_json(output / "verification.json", report)
        raise
    write_json(output / "verification.json", report)
    return report


def require_smoke(settings, sequences=None):
    path = Path(settings.get("smoke_manifest") or Path(settings["output_root"]) / "smoke/verification.json")
    check = read_json(path)
    if check.get("status") != "PASS" or check.get("contract_sha256") != json_digest(contract(settings)):
        raise ValueError("Run smoke successfully with these inputs, settings and source before the full experiment")
    target = select_sequences(settings, sequences)
    required = set(target) & set(SMOKE) or set(target)
    if not required <= set(check.get("sequences", {})):
        raise ValueError(f"Smoke scope is insufficient; verify these sequences first: {sorted(required)}")
    for role, entry in reference_entries(settings).items():
        pp = Path(entry["run_dir"]) / "protocols" / (entry["tracker"] + ".json")
        saved = check.get("reference_manifest", {}).get(role, {})
        if saved.get("protocol_sha256") != sha256(pp):
            raise ValueError(f"Reference changed after smoke: {role}")
    return sha256(path)


def controls(settings, output, sequences=None):
    gate = require_smoke(settings, sequences)
    output = new_directory(output)
    chosen = select_sequences(settings, sequences)
    report = {"status": "RUNNING", "smoke_sha256": gate, "sequences": chosen, "methods": CONTROLS}
    write_json(output / "manifest.json", report)
    try:
        for method in CONTROLS:
            track(settings, method, output, chosen, probe="full" if method == "B_adaptive_no_dual_low" else "off")
        report["status"] = "COMPLETE_TRACKING_PENDING_EVALUATION"
    except Exception:
        report.update(status="FAILED", error=traceback.format_exc())
        write_json(output / "manifest.json", report)
        raise
    write_json(output / "manifest.json", report)
    return report


def probe_run(settings, method, output, sequences=None):
    gate = require_smoke(settings, sequences)
    output = new_directory(output)
    chosen = select_sequences(settings, sequences)
    track(settings, method, output, chosen, probe="full")
    if method in ("B_early_moderate", "B_early_moderate_dual_low", "evidence_both"):
        roles = {v: k for k, v in ROLES.items()}
        entries = reference_entries(settings)
        role = roles[method]
        if role not in entries:
            raise ValueError("Provide the historical evidence reference before auditing it")
        expected = read_json(Path(entries[role]["run_dir"]) / "protocols" / (entries[role]["tracker"] + ".json"))
        actual = read_json(output / "protocols" / (method + ".json"))
        assert_same_inputs(expected, actual, chosen)
        if any(expected["output_prediction_hashes"][s] != actual["output_prediction_hashes"][s] for s in chosen):
            write_json(output / "verification.json", {"status": "FAILED", "reason": "historical MOT differs"})
            raise ValueError("Read-only replay differs from historical MOT; do not interpret its traces")
    result = {"status": "PASS", "smoke_sha256": gate, "method": method, "sequences": chosen}
    write_json(output / "verification.json", result)
    return result


def analyze(settings, output, sequences=None):
    # Child handles exclusive output creation; keep its settings/log outside it.
    output = Path(output)
    holder = new_directory(output.with_name(output.name + "_job"))
    config = resolved_file(settings, holder)
    args = ["-m", "experiments.vttiny_gmc.round3_joint.analysis", "--settings", config, "--output", output]
    args += ["--sequences", *select_sequences(settings, sequences)]
    run_child(settings, args, holder / "analysis.log")
    return read_json(output / "manifest.json")


def project_reference(entry, target, sequences, *, expected=None):
    """Copy immutable MOT into an explicit evaluation-only subset view."""
    from tools.evaluate_botsort_plus import validate_runs
    root, method = Path(entry["run_dir"]), entry["tracker"]
    p = validate_runs(root, [method])["protocols"][method]
    if expected is not None:
        assert_same_inputs(expected, p, sequences)
    src_path = root / "protocols" / (method + ".json")
    view = copy.deepcopy(p)
    view["evaluation_view"] = {"source_protocol": str(src_path), "source_sha256": sha256(src_path),
                               "purpose": "exact MOT subset for official re-evaluation; no trajectory editing"}
    full_names = set(p["sequences"])
    for field in ("sequences", "output_prediction_hashes", "detection_fingerprints", "feature_hashes", "decoded_image_hashes"):
        view[field] = {s: p[field][s] for s in sequences}
    view["subset"] = p["subset"] or set(sequences) != full_names
    dest = Path(target) / "trackers" / method / "data"
    dest.mkdir(parents=True, exist_ok=False)
    for seq in sequences:
        shutil.copyfile(root / "trackers" / method / "data" / (seq + ".txt"), dest / (seq + ".txt"))
    write_json(Path(target) / "protocols" / (method + ".json"), view)
    return view


def evaluate(settings, run_dir, output, sequences=None):
    output = new_directory(output)
    chosen = select_sequences(settings, sequences)
    refs = reference_entries(settings)
    expected = read_json(Path(refs["B"]["run_dir"]) / "protocols" / (refs["B"]["tracker"] + ".json"))
    entries = list(refs.values()) + [{"run_dir": str(run_dir), "tracker": m} for m in CONTROLS]
    view = output / "evaluation_input"
    names = []
    for entry in entries:
        project_reference(entry, view, chosen, expected=expected)
        names.append(entry["tracker"])
    run_child(settings, [REPO / "tools/evaluate_botsort_plus.py", "--run-dir", view, "--trackers", *names,
                         "--reference-tracker", refs["B"]["tracker"], "--gt-root", settings["gt_root"],
                         "--trackeval-root", settings["trackeval_root"], "--output-dir", output / "metrics"],
              output / "evaluate.log")
    return read_json(output / "metrics/summary.json")


def probe_summary(settings, run_dir, method, output, sequences=None):
    output = Path(output)
    holder = new_directory(output.with_name(output.name + "_job"))
    config = resolved_file(settings, holder)
    args = ["-m", "experiments.vttiny_gmc.round3_joint.probe_analysis", "--settings", config,
            "--run-dir", run_dir, "--tracker", method, "--output-dir", output]
    args += ["--sequences", *select_sequences(settings, sequences)]
    run_child(settings, args, holder / "analysis.log")
    return {"status": "COMPLETE", "output": str(output)}


def package(root, *, include_traces=False, archive_path=None):
    root = Path(root).resolve()
    dest = Path(archive_path).resolve() if archive_path else root.parent / (root.name + "_evidence.zip")
    temporary = dest.with_suffix(dest.suffix + ".tmp")
    if dest.exists() or temporary.exists():
        raise FileExistsError(dest)
    members = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(root)
        if p.suffix in (".tmp", ".writing"):
            continue
        if "evaluation_input" in rel.parts or "tracking_preflight" in rel.parts:
            continue
        if p.name == "gt.txt" or p.suffix.lower() in (".pt", ".pth", ".npz", ".jpg", ".jpeg", ".mp4", ".zip"):
            continue
        if not include_traces and any(k in rel.parts for k in ("probes", "warps")):
            continue
        members.append((p, rel))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(temporary, "x", zipfile.ZIP_DEFLATED) as archive:
        for path, relative in members:
            archive.write(path, relative.as_posix())
    temporary.replace(dest)
    return {"path": str(dest), "sha256": sha256(dest), "files": len(members), "include_traces": include_traces}


def init_settings(from_settings, destination, *, joint_run=None, factorial_run=None, evidence_run=None, vendor=None):
    example = read_json(REPO / "configs/diagnostics/joint_v0.4_round3.example.json")
    if from_settings:
        old = read_json(from_settings)
        for key in ("baseline_root", "features_dir", "images_root", "gt_root", "runtime_root", "trackeval_root", "python_user_base"):
            if key in old:
                example[key] = old[key]
    if joint_run:
        for role in ("native", "B", "G", "BG"):
            example["references"][role]["run_dir"] = str(joint_run)
    if evidence_run:
        example["references"]["evidence"] = {"run_dir": str(evidence_run), "tracker": "evidence_both"}
    if factorial_run:
        example["gmc_mode"] = "cached"
        example["warp_sources"] = {m: {"run_dir": str(factorial_run), "method": m}
                                   for m in ("native", "skip_only", "adaptive_gmc")}
    if vendor:
        example["vendor_dir"] = str(vendor)
    path = Path(destination)
    if path.exists():
        raise FileExistsError(path)
    write_json(path, example)
    return {"status": "CONFIG_CREATED", "path": str(path), "next": "Fill local references/vendor or warp paths, then preflight"}


def run_batch(settings):
    root = new_directory(settings["output_root"])
    state = {"schema": SCHEMA, "status": "RUNNING", "base_commit": BASE_COMMIT,
             "stages": {}, "E1": "STOP_P1_P2_REUSE_DIAGNOSTICS", "E2": "STOP_FIXED_191",
             "E3": "PENDING_HUMAN_REVIEW", "J4": "AWAITING_DIAGNOSTIC_DECISION"}
    write_json(root / "batch_status.json", state)
    resolved_file(settings, root)
    steps = [
        ("J0", lambda: preflight(settings, root / "j0")),
        ("J1", lambda: analyze(settings, root / "j1")),
        ("smoke", lambda: smoke(settings, root / "smoke")),
        ("J2", lambda: controls(settings, root / "j2")),
        ("J2_evaluation", lambda: evaluate(settings, root / "j2", root / "j2_evaluation")),
        ("J3_BG", lambda: probe_run(settings, "B_early_moderate_dual_low", root / "j3_bg")),
        ("J3_F_summary", lambda: probe_summary(settings, root / "j2", "B_adaptive_no_dual_low", root / "j3_f_analysis")),
        ("J3_BG_summary", lambda: probe_summary(settings, root / "j3_bg", "B_early_moderate_dual_low", root / "j3_bg_analysis")),
    ]
    try:
        for key, function in steps:
            result = function()
            state["stages"][key] = {"status": result.get("status", "COMPLETE")}
            write_json(root / "batch_status.json", state)
        from .e3_review import run as review
        result = review("prepare", output_dir=root / "e3_review")
        state["stages"]["E3_prepare"] = {"status": result.get("status", "PREPARED")}
        state["status"] = "COMPLETE_EXECUTABLE_STAGES"
    except Exception:
        state.update(status="FAILED", error=traceback.format_exc())
        write_json(root / "batch_status.json", state)
        raise
    write_json(root / "batch_status.json", state)
    state["evidence"] = package(root)
    write_json(root / "batch_status.json", state)
    return state


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create a local JSON from the existing diagnostics settings")
    init.add_argument("--from-settings", type=Path)
    init.add_argument("--output-config", type=Path, required=True)
    init.add_argument("--joint-run", type=Path)
    init.add_argument("--factorial-run", type=Path)
    init.add_argument("--evidence-run", type=Path)
    init.add_argument("--vendor", type=Path)
    for name in ("preflight", "analyze", "smoke", "run-controls", "probe", "evaluate", "probe-summary", "run", "status", "package"):
        sub = commands.add_parser(name)
        sub.add_argument("--settings", type=Path, required=True)
        sub.add_argument("--output-dir", type=Path)
        sub.add_argument("--sequences", nargs="+")
        if name in ("evaluate", "probe-summary"):
            sub.add_argument("--run-dir", type=Path)
        if name in ("probe", "probe-summary"):
            sub.add_argument("--method", choices=METHODS, default="B_early_moderate_dual_low")
        if name == "package":
            sub.add_argument("--include-traces", action="store_true")
            sub.add_argument("--archive", type=Path, help="A new ZIP path, e.g. after completing human review")
    for name in ("e3-prepare", "e3-check", "e3-report"):
        sub = commands.add_parser(name)
        sub.add_argument("--input", type=Path)
        sub.add_argument("--output-dir", type=Path)
        sub.add_argument("--source-root", type=Path)
        sub.add_argument("--causal-signals", type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "init":
            result = init_settings(args.from_settings, args.output_config, joint_run=args.joint_run,
                                   factorial_run=args.factorial_run, evidence_run=args.evidence_run, vendor=args.vendor)
        elif args.command.startswith("e3-"):
            from .e3_review import main as review_main
            command = [args.command[3:]]
            for key in ("input", "output_dir", "source_root", "causal_signals"):
                value = getattr(args, key)
                if value is not None:
                    command += ["--" + key.replace("_", "-"), str(value)]
            return review_main(command)
        else:
            settings = load_settings(args.settings)
            root = Path(settings["output_root"])
            seq = args.sequences
            if args.command == "run":
                if args.output_dir is not None:
                    settings["output_root"] = str(args.output_dir.resolve())
                if seq:
                    settings["sequences"] = seq
                result = run_batch(settings)
            elif args.command == "preflight":
                result = preflight(settings, args.output_dir or root / "j0", seq)
            elif args.command == "analyze":
                result = analyze(settings, args.output_dir or root / "j1", seq)
            elif args.command == "smoke":
                result = smoke(settings, args.output_dir or root / "smoke", seq)
            elif args.command == "run-controls":
                result = controls(settings, args.output_dir or root / "j2", seq)
            elif args.command == "probe":
                result = probe_run(settings, args.method, args.output_dir or root / "j3_bg", seq)
            elif args.command == "evaluate":
                result = evaluate(settings, args.run_dir or root / "j2", args.output_dir or root / "j2_evaluation", seq)
            elif args.command == "probe-summary":
                result = probe_summary(settings, args.run_dir or root / "j3_bg", args.method,
                                       args.output_dir or root / "j3_bg_analysis", seq)
            elif args.command == "package":
                result = package(root, include_traces=args.include_traces, archive_path=args.archive)
            else:
                path = root / "batch_status.json"
                result = read_json(path) if path.is_file() else {"status": "INDIVIDUAL_STAGES", "root": str(root),
                         "manifests": [str(p.relative_to(root)) for p in root.rglob("manifest.json")]}
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
