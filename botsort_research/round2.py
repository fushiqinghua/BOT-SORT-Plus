"""Plan, execute and collect the second experiment batch using separate processes.

No shell command strings are executed. Each job owns a new output directory;
completed jobs can be resumed only with unchanged source, inputs and outputs.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import zipfile

from .io import detection_files, dump_json, load_json, sha256
from .runner import DEFAULT_CACHE, ROOT, source_manifest


CONFIG_ROOT = ROOT / "configs/research/round2"
DEFAULT_SPLIT = ROOT / "configs/research/split_retrospective.json"
PHASES = ("preflight", "m2", "m1", "m3", "coverage")


def _absolute(path):
    return str(Path(path).expanduser().resolve()) if path is not None else None


def _tree_hashes(paths):
    result = {}
    for path in paths:
        path = Path(path)
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for file in files:
            if not file.is_file():
                raise FileNotFoundError(file)
            result[str(file.resolve())] = sha256(file)
    return result


def make_plan(output, *, cache_dir, image_root, gt_root, split_path=DEFAULT_SPLIT,
              embeddings_dir=None, runtime_root=None, trackeval_root=None, config_root=CONFIG_ROOT):
    """Freeze explicit commands, config snapshots and non-image input hashes."""
    from .calibration import validate_split
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new batch directory: {output}")
    files = detection_files(cache_dir)
    split = load_json(split_path)
    validate_split(split, available=files)
    names = split["calibration"] + split["evaluation"]
    if not Path(image_root).is_dir():
        raise FileNotFoundError(image_root)
    inputs = [files[name] for name in names] + [Path(gt_root) / name / "gt/gt.txt" for name in names]
    if embeddings_dir:
        inputs += [Path(embeddings_dir) / (name + suffix) for name in names for suffix in (".npz", ".json")]
    hashes = _tree_hashes(inputs)
    required_configs = ("default", "iou_080", "iou_085", "iou_090", "nwd_8", "nwd_12_8", "nwd_20")
    config_paths = [Path(config_root) / (name + ".json") for name in required_configs]
    if any(not path.is_file() for path in config_paths):
        raise FileNotFoundError(f"All seven Round2 configs are required in {config_root}")
    output.mkdir(parents=True)
    snapshots = output / "config"
    snapshots.mkdir()
    for path in config_paths:
        shutil.copyfile(path, snapshots / path.name)
    dump_json(snapshots / "split.json", split)
    plan_path = output / "plan.json"
    split_arg = str(snapshots / "split.json")
    default = str(snapshots / "default.json")
    common = ["--cache-dir", _absolute(cache_dir), "--image-root", _absolute(image_root), "--split", split_arg]
    if runtime_root:
        common += ["--runtime-root", _absolute(runtime_root)]
    eval_options = ["--gt-root", _absolute(gt_root)]
    if trackeval_root:
        eval_options += ["--trackeval-root", _absolute(trackeval_root)]
    jobs, evaluation_runs = [], []

    def job(identity, phase, args, outputs, dependencies=(), module="botsort_research.cli"):
        jobs.append({"id": identity, "phase": phase, "module": module,
                     "args": list(map(str, args)), "outputs": list(map(str, outputs)),
                     "dependencies": list(dependencies)})
        return identity

    doctor = ["doctor", "--cache-dir", _absolute(cache_dir), "--image-root", _absolute(image_root), "--check-runtime"]
    if runtime_root:
        doctor += ["--runtime-root", _absolute(runtime_root)]
    if embeddings_dir:
        doctor += ["--embeddings-dir", _absolute(embeddings_dir)]
    job("doctor", "preflight", doctor + ["--output", output / "doctor.json"], [output / "doctor.json"])

    def replay(identity, phase, variants, *, config=default, cmc="sof", subset="evaluation", dependencies=("baseline_analyze",), appearance=False, model=False, compare=None):
        run = output / "runs" / identity
        args = ["run", *common, "--config", config, "--subset", subset, "--variants", *variants,
                "--cmc", cmc, "--trace", "--output-trace", "--output", run]
        if appearance:
            args += ["--embeddings-dir", _absolute(embeddings_dir)]
        if model:
            args += ["--appearance-model", output / "appearance_calibration/model.json"]
        job(identity, phase, args, [run], dependencies)
        previous = identity
        if compare:
            reference = ROOT / "results/vttiny/yolo11m_rgb_rgbt/rgbt/trackers" / compare[1] / "data"
            previous = job(identity + "_compare", phase,
                ["compare-baseline", "--run-dir", run, "--variant", compare[0], "--reference-dir", reference],
                [run / "baseline_comparison.json"], (previous,))
        previous = job(identity + "_evaluate", phase, ["evaluate", "--run-dir", run, *eval_options],
                       [run / "metrics"], (previous,))
        last = job(identity + "_analyze", phase,
                   ["analyze-run", "--run-dir", run, "--gt-root", _absolute(gt_root), "--output", run / "analysis"],
                   [run / "analysis"], (previous,))
        if subset == "evaluation":
            evaluation_runs.append(str(run))
        return run, last

    baseline, _ = replay("baseline", "preflight", ["native_iou", "native_iou_det"],
                          dependencies=("doctor",), compare=("native_iou", "botsort_no_reid"))
    replay("m2_core", "m2", ["m2_nll_baseline_r", "m2_nll_floor", "m2_nll_floor_det",
        "m2_nll_floor_native_update", "m2_nll_floor_native_update_det", "iou_floor_update",
        "m2_nll_first_native_update", "m2_nll_second_native_update"])

    # Strong simple controls are selected on nine calibration sequences, never
    # by examining all their evaluation results and choosing the best after.
    control_runs, control_done = {"iou": [], "nwd": []}, {"iou": [], "nwd": []}
    for family, configurations, variant in (
        ("iou", ("iou_080", "iou_085", "iou_090"), "iou_relaxed"),
        ("nwd", ("nwd_8", "nwd_12_8", "nwd_20"), "nwd_native_update"),
    ):
        for filename in configurations:
            run, done = replay("cal_" + filename, "m2", [variant],
                config=str(snapshots / (filename + ".json")), subset="calibration")
            control_runs[family].append(str(run))
            control_done[family].append(done)
        selected = output / ("selected_" + family)
        select_id = job("select_" + family, "m2",
            ["select-control", "--family", family, "--runs", *control_runs[family], "--output", selected],
            [selected], control_done[family], module="botsort_research.round2")
        replay("control_" + family, "m2", [variant], config=str(selected / "config.json"), dependencies=(select_id,))

    # Calibration runs inspect committed hypotheses; the script does not
    # optimize M1 defaults against the development-evaluation sequences.
    m1_calibration_runs, m1_calibration_done = [], []
    for mode in ("none", "m1_native_scale", "m1_mask_grid", "m1_fb", "m1_reliable"):
        run, done = replay("cal_" + mode, "m1", ["native_iou"], cmc=mode, subset="calibration")
        m1_calibration_runs.append(run)
        m1_calibration_done.append(done)
        replay("m1_" + mode, "m1", ["native_iou"], cmc=mode, dependencies=(done,))
    # SOF on the same calibration subset completes the M1 control comparison.
    run, done = replay("cal_sof", "m1", ["native_iou"], subset="calibration")
    m1_calibration_runs.append(run)
    m1_calibration_done.append(done)
    job("m1_calibration_summary", "m1",
        ["summarize", "--runs", *m1_calibration_runs, "--output", output / "m1_calibration_comparison"],
        [output / "m1_calibration_comparison"], m1_calibration_done)

    if embeddings_dir:
        _, native_done = replay("appearance_native", "m3", ["native_reid"], appearance=True,
                                compare=("native_reid", "botsort_reid"))
        calibration = output / "appearance_calibration"
        calibrated = job("appearance_calibrate", "m3",
            ["calibrate-appearance", "--cache-dir", _absolute(cache_dir), "--gt-root", _absolute(gt_root),
             "--embeddings-dir", _absolute(embeddings_dir), "--split", split_arg, "--output", calibration],
            [calibration], (native_done,))
        replay("appearance_experiments", "m3", ["reid_fixed", "reid_gap", "reid_recovery_fixed", "reid_recovery"],
               appearance=True, model=True, dependencies=(calibrated,))

    for subset in ("calibration", "evaluation"):
        destination = output / ("coverage_" + subset)
        job("coverage_" + subset, "coverage", ["detection-coverage", "--cache-dir", _absolute(cache_dir),
            "--gt-root", _absolute(gt_root), "--split", split_arg, "--subset", subset, "--output", destination],
            [destination], ("baseline_analyze",))

    plan = {"schema_version": 1, "batch": "round2", "status": "PLANNED",
        "interpretation": "retrospective_development_only", "root": str(output), "repo_root": str(ROOT),
        "source": source_manifest(), "python": sys.executable,
        "paths": {"cache_dir": _absolute(cache_dir), "image_root": _absolute(image_root), "gt_root": _absolute(gt_root),
                  "embeddings_dir": _absolute(embeddings_dir), "runtime_root": _absolute(runtime_root),
                  "trackeval_root": _absolute(trackeval_root)},
        "input_sha256": hashes, "config_sha256": _tree_hashes([snapshots]),
        "split": split, "jobs": jobs, "evaluation_runs": evaluation_runs,
        "disabled": [] if embeddings_dir else ["m3: --embeddings-dir was not supplied; all M3 jobs are explicitly omitted"],
        "notes": ["Execute with the frozen runtime and pinned TrackEval.",
                  "Each replay/evaluation/calibration job runs in its own process: online GT guard stays active.",
                  "No parameter is selected from the 21 evaluation sequences by this orchestrator.",
                  "Images are hashed by each replay; final comparisons reject differing image manifests.",
                  "M3 uses only existing high-score RGB features; low-score feature extraction is future work."]}
    dump_json(plan_path, plan)
    lines = ["# 第二批实验执行计划", "", "此文件由 plan 生成。命令中的路径已经固定，执行器逐任务记录状态和日志。", "",
             f"共 {len(jobs)} 个任务；M3 {'启用' if embeddings_dir else '未启用：缺少 --embeddings-dir'}。", "",
             "```bash", f"{shlex.quote(sys.executable)} -m botsort_research.round2 run --plan {shlex.quote(str(plan_path))} --phase all --resume", "```", ""]
    for value in jobs:
        command = [sys.executable, "-m", value["module"], *value["args"]]
        lines += [f"## {value['id']} ({value['phase']})", "", "```bash", shlex.join(command), "```", ""]
    (output / "commands.md").write_text("\n".join(lines), encoding="utf-8")
    return plan


def _verify_hashes(hashes, description):
    for filename, expected in hashes.items():
        path = Path(filename)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"{description} changed or missing: {path}")


def _dependency_order(jobs, phases):
    by_id = {job["id"]: job for job in jobs}
    if len(by_id) != len(jobs):
        raise ValueError("Job IDs must be unique")
    needed, visiting = set(), set()

    def include(identity):
        if identity in needed:
            return
        if identity in visiting or identity not in by_id:
            raise ValueError("Invalid or cyclic job dependencies")
        visiting.add(identity)
        for dependency in by_id[identity]["dependencies"]:
            include(dependency)
        visiting.remove(identity)
        needed.add(identity)

    for job in jobs:
        if job["phase"] in phases:
            include(job["id"])
    # Plan construction is topological; reject a manually edited bad order.
    result, done = [], set()
    for job in jobs:
        if job["id"] not in needed:
            continue
        if set(job["dependencies"]) - done:
            raise ValueError("Jobs must be in dependency order")
        result.append(job)
        done.add(job["id"])
    return result


def execute_plan(plan_path, *, phase="all", resume=False, retry_failed=False):
    plan_path = Path(plan_path).resolve()
    plan = load_json(plan_path)
    root = Path(plan["root"])
    if plan.get("batch") != "round2" or plan_path != root / "plan.json" or Path(plan["repo_root"]) != ROOT:
        raise ValueError("This plan must be executed at its original repository and batch paths")
    current_source = source_manifest()
    if current_source["python_sha256"] != plan["source"]["python_sha256"]:
        raise ValueError("Research source changed since planning; create a new batch plan")
    _verify_hashes(plan["input_sha256"], "Input")
    _verify_hashes(plan["config_sha256"], "Config snapshot")
    phases = set(PHASES if phase == "all" else [phase])
    if phase == "m3" and plan["disabled"]:
        raise ValueError("This plan omitted M3. Create another plan with --embeddings-dir; do not silently skip M3")
    selected = _dependency_order(plan["jobs"], phases)
    state_path = root / "state.json"
    state = load_json(state_path) if state_path.exists() else {"plan_sha256": sha256(plan_path), "jobs": {}}
    if state.get("plan_sha256") != sha256(plan_path):
        raise ValueError("Plan changed after execution began")
    log_root = root / "logs"
    log_root.mkdir(exist_ok=True)
    for job in selected:
        identity = job["id"]
        previous = state["jobs"].get(identity)
        if previous and previous.get("status") == "COMPLETE":
            if not resume:
                raise FileExistsError(f"{identity} is complete; pass --resume to verify and skip it")
            _verify_hashes(previous["output_sha256"], "Completed job output")
            print(f"SKIP verified {identity}", flush=True)
            continue
        existing = [Path(path) for path in job["outputs"] if Path(path).exists()]
        if existing:
            if not retry_failed:
                raise FileExistsError(f"Partial/unrecorded outputs for {identity}; inspect logs, then use --retry-failed to archive and retry")
            archived = root / "failed" / (identity + "_" + str(time.time_ns()))
            archived.mkdir(parents=True)
            for index, path in enumerate(existing):
                shutil.move(str(path), str(archived / f"{index}_{path.name}"))
        command = [sys.executable, "-m", job["module"], *job["args"]]
        record = {"status": "RUNNING", "command": command, "started_unix": time.time()}
        state["jobs"][identity] = record
        dump_json(state_path, state)
        log = log_root / (identity + ".log")
        print(f"RUN {identity}: log={log}", flush=True)
        try:
            with log.open("a", encoding="utf-8") as stream:
                stream.write("\nCOMMAND " + shlex.join(command) + "\n")
                stream.flush()
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
            if result.returncode:
                raise RuntimeError(f"{identity} failed (exit {result.returncode}); inspect {log}")
            record["output_sha256"] = _tree_hashes(job["outputs"])
            if not record["output_sha256"]:
                raise RuntimeError(f"{identity} produced no reviewable output")
            record["status"] = "COMPLETE"
            record["finished_unix"] = time.time()
            dump_json(state_path, state)
        except BaseException as exc:
            record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", finished_unix=time.time())
            dump_json(state_path, state)
            raise
    return state


def select_control(family, runs, output):
    """Predeclared ranking: HOTA, IDF1, fewer CLEAR errors, then listed order."""
    from .reporting import summarize_runs
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    variant = {"iou": "iou_relaxed", "nwd": "nwd_native_update"}[family]
    entries = []
    for index, path in enumerate(runs):
        run = Path(path)
        manifest = load_json(run / "experiment.json")
        if manifest.get("subset") != "calibration" or manifest.get("variants") != [variant]:
            raise ValueError("Simple-control selection accepts only matching single-variant calibration runs")
        if set(manifest["sequences"]) != set(manifest["split"]["calibration"]):
            raise ValueError("Select controls on the complete calibration subset")
        summary = load_json(run / "metrics/summary.json")
        values = [r for r in summary["results"] if r["sequence"] == "COMBINED_SEQ" and r["tracker"] == variant]
        if len(values) != 1:
            raise ValueError("Missing unique control metrics")
        row = values[0]
        entries.append((run, manifest, row, index))
    if not entries:
        raise ValueError("Need control candidates")
    # Reuse the comparison guard to reject changed GT/cache/image/source outputs.
    summarize_runs(runs, output / "calibration_comparison")
    chosen = max(entries, key=lambda x: (x[2]["HOTA"], x[2]["IDF1"],
                    -(x[2]["FP"] + x[2]["FN"] + x[2]["IDSW"]), -x[3]))
    config = copy.deepcopy(chosen[1]["config"])
    config["selection"] = {"family": family, "subset": "calibration", "source_run": str(chosen[0])}
    dump_json(output / "config.json", config)
    selection = {"status": "COMPLETE", "family": family, "selected_run": str(chosen[0]),
        "ranking": ["highest unrounded HOTA", "highest IDF1", "fewest FP+FN+IDSW", "first predeclared candidate"],
        "candidates": [{"run": str(run), "metrics": row, "experiment_sha256": sha256(run / "experiment.json")}
                       for run, _, row, _ in entries]}
    dump_json(output / "selection.json", selection)
    return selection


def collect(plan_path, output, *, allow_partial=False):
    """Create a small, commit-ready evidence folder; keep full traces in the batch."""
    from .reporting import summarize_runs
    plan = load_json(plan_path)
    root, output = Path(plan["root"]), Path(output)
    if output.exists():
        raise FileExistsError(output)
    state = load_json(root / "state.json")
    if state.get("plan_sha256") != sha256(plan_path):
        raise ValueError("Plan changed after execution began")
    for receipt in state["jobs"].values():
        if receipt.get("status") == "COMPLETE":
            _verify_hashes(receipt["output_sha256"], "Completed job output")
    expected = [job["id"] for job in plan["jobs"]]
    incomplete = [identity for identity in expected if state["jobs"].get(identity, {}).get("status") != "COMPLETE"]
    if incomplete and not allow_partial:
        raise ValueError(f"Batch incomplete: {incomplete[:6]}; --allow-partial explicitly labels a partial export")
    complete = []
    for value in plan["evaluation_runs"]:
        run = Path(value)
        identity = run.name + "_analyze"
        if state["jobs"].get(identity, {}).get("status") == "COMPLETE":
            # Analysis also verified predictions, source indices and official CLEAR counts.
            _verify_hashes(state["jobs"][identity]["output_sha256"], "Analysis output")
            complete.append(run)
    if not complete:
        raise ValueError("No complete, evaluated and analyzed evaluation runs")
    summarize_runs(complete, output / "comparison")
    sequence_rows = []
    baseline_rows = {r["sequence"]: r for r in load_json(root / "runs/baseline/metrics/summary.json")["results"]
                     if r["tracker"] == "native_iou"}
    for run in complete:
        manifest = load_json(run / "experiment.json")
        for row in load_json(run / "metrics/summary.json")["results"]:
            base = baseline_rows[row["sequence"]]
            sequence_rows.append({"run": run.name, "cmc": manifest["cmc"], **row,
                **{"delta_" + key: row[key] - base[key] for key in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "FP", "FN", "IDSW")}})
        destination = output / "runs" / run.name
        destination.mkdir(parents=True)
        for filename in ("experiment.json", "baseline_comparison.json", "metrics/summary.json"):
            source = run / filename
            if source.is_file():
                target = destination / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        # Agent-defined diagnostic summaries are small; detailed frame/case
        # tables remain available in the full archive for targeted follow-up.
        for source in (run / "analysis").glob("*"):
            if source.is_file() and ("summary" in source.name or source.suffix == ".md" or
                                    source.name in ("case_samples.csv", "clear_verification.csv", "recovery_cases.csv")):
                target = destination / "analysis" / source.name
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(source, target)
    with (output / "sequence_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sequence_rows[0]))
        writer.writeheader()
        writer.writerows(sequence_rows)
    extras = {"selected_iou": "select_iou", "selected_nwd": "select_nwd",
              "appearance_calibration": "appearance_calibrate", "coverage_calibration": "coverage_calibration",
              "coverage_evaluation": "coverage_evaluation", "m1_calibration_comparison": "m1_calibration_summary"}
    for name, identity in extras.items():
        source = root / name
        if state["jobs"].get(identity, {}).get("status") == "COMPLETE":
            destination = output / name
            destination.mkdir()
            for file in source.glob("*"):
                if file.is_file() and (file.suffix == ".json" or file.name in ("coverage.csv", "comparison.csv", "comparison.md")):
                    shutil.copyfile(file, destination / file.name)
    dump_json(output / "execution.json", {"status": "PARTIAL" if incomplete else "COMPLETE",
        "interpretation": "retrospective_development_only", "source": plan["source"],
        "plan_sha256": sha256(plan_path), "disabled": plan["disabled"], "incomplete_jobs": incomplete,
        "completed_jobs": [identity for identity in expected if identity not in incomplete],
        "missing_optional_exports": [name for name, identity in extras.items()
                                     if state["jobs"].get(identity, {}).get("status") != "COMPLETE"],
        "full_evidence_directory": str(root)})
    (output / "README.md").write_text(
        "# 第二批实验结果\n\n本目录由 `botsort_research.round2 collect` 生成。\n\n"
        "- 合并指标与差值见 `comparison/comparison.csv`。\n"
        "- 逐序列指标见 `sequence_metrics.csv`；不能把逐序列百分比均值当作 COMBINED_SEQ。\n"
        "- 输出完整性、未启用模块和未完成任务见 `execution.json`。\n"
        "- 每组 `runs/*/analysis` 为与 TrackEval CLEAR 核对后的诊断汇总。\n"
        "- 帧级错因、源检测索引、全部 MOT 轨迹和日志保留在完整 batch 目录；用 `pack` 打包。\n\n"
        "## 实验者补充\n\n请补充运行环境、是否修改参数、异常及代表性帧号。所有真实指标来自本地运行，此处无预填实验结论。\n",
        encoding="utf-8")
    return {"output": str(output), "runs": len(complete), "incomplete_jobs": incomplete}


def pack(plan_path, output):
    plan = load_json(plan_path)
    root, output = Path(plan["root"]), Path(output).resolve()
    if output.exists() or output == root or root in output.parents:
        raise ValueError("Use a new ZIP path outside the batch directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in root.rglob("*") if p.is_file() and "failed" not in p.relative_to(root).parts)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file in files:
            archive.write(file, Path(root.name) / file.relative_to(root))
    result = {"filename": output.name, "sha256": sha256(output), "bytes": output.stat().st_size,
              "files": len(files), "excluded": ["archived failed attempts"]}
    dump_json(output.with_suffix(output.suffix + ".json"), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="Freeze paths, input hashes, configs and all batch commands")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--gt-root", type=Path, required=True)
    p.add_argument("--embeddings-dir", type=Path)
    p.add_argument("--runtime-root", type=Path)
    p.add_argument("--trackeval-root", type=Path)
    p.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("run", help="Execute dependency-ordered jobs; logs and receipts are stored in the batch")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--phase", choices=("all", *PHASES), default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--retry-failed", action="store_true", help="Archive incomplete output paths before retry; never overwrite/delete them")
    p = sub.add_parser("select-control", help="Internal calibration-only selection of a simple control")
    p.add_argument("--family", choices=("iou", "nwd"), required=True)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("collect", help="Export compact results suitable for committing to the current branch")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    p = sub.add_parser("pack", help="Zip complete traces/trajectories/logs for later error analysis")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = make_plan(args.output, cache_dir=args.cache_dir, image_root=args.image_root, gt_root=args.gt_root,
                split_path=args.split, embeddings_dir=args.embeddings_dir, runtime_root=args.runtime_root,
                trackeval_root=args.trackeval_root)
            print(f"PLANNED {len(result['jobs'])} jobs: {args.output / 'plan.json'}")
            for note in result["disabled"]:
                print("OMITTED " + note)
        elif args.command == "run":
            execute_plan(args.plan, phase=args.phase, resume=args.resume, retry_failed=args.retry_failed)
            print(f"COMPLETE phase={args.phase}; collect results after the requested phases finish")
        elif args.command == "select-control":
            result = select_control(args.family, args.runs, args.output)
            print(f"SELECTED {result['selected_run']}")
        elif args.command == "collect":
            print(json.dumps(collect(args.plan, args.output, allow_partial=args.allow_partial), ensure_ascii=False))
        else:
            print(json.dumps(pack(args.plan, args.output), ensure_ascii=False))
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, ImportError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
