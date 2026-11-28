"""Portable CLI for the W R3.1 fixed offline comparisons."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile

from . import inputs, protocol

ROOT = Path(__file__).resolve().parents[3]


def validate_protocol(root):
    path = root / "configs/research/wem_r3_1_protocol.json"
    data, fingerprint = inputs.read_json(path)
    if data != protocol.PROTOCOL:
        raise ValueError("Published fixed R3.1 protocol changed; use a separately versioned experiment")
    return path, fingerprint


def init_settings(path):
    path = Path(path).resolve()
    if not path.name.endswith(".local.json"):
        raise ValueError("Local settings filename must end with .local.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(copy.deepcopy(protocol.SETTINGS), ensure_ascii=False, indent=2) + "\n")
    return {"status": "SETTINGS_CREATED", "settings": str(path), "next": "Fill paths, then run check and run"}


def check_inputs(settings_path, root=ROOT):
    _, protocol_sha = validate_protocol(root)
    settings = inputs.load_settings(settings_path, root)
    bundle = inputs.load_bundle(settings)
    return settings, bundle, protocol_sha


def input_summary(bundle):
    return {"scope": bundle.scope, "reference": protocol.REFERENCE,
            "sequence_count": len(bundle.partitions), "raw_feature_edges": len(bundle.rows),
            "partitions": {part: {"sequences": sum(p == part for p in bundle.partitions.values()),
                                  "raw_feature_edges": sum(bundle.partitions[row["sequence"]] == part for row in bundle.rows)}
                           for part in protocol.PARTITIONS},
            "enabled_online": False}


def run_comparison(settings_path, root=ROOT):
    # Import numerical dependencies only for the actual comparison.
    from . import engine, reporting
    import numpy
    import scipy

    settings, bundle, protocol_sha = check_inputs(settings_path, root)
    decisions = reporting.evaluate(bundle)
    comparison, per_sequence, per_source = reporting.build_tables(bundle, decisions)
    deltas = reporting.paired_deltas(decisions)
    reviews = reporting.review_queue(bundle, decisions)
    archive = root / "results/vttiny/wem_v0.1/r3/calibration/fixed_diagnostics.csv"
    historical = reporting.check_historical_controls(comparison, archive) if bundle.scope == "full" else {
        "status": "NOT_APPLICABLE_SMOKE", "reason": "The published controls describe the full 74-sequence caches"}
    summary = dict(status="OFFLINE_COMPARISON_COMPLETE", input=input_summary(bundle),
                   operating_policy="unchanged_reject_all", enabled_online=False,
                   operating_point_selection="NOT_PERFORMED", scientific_acceptance="NOT_EVALUATED",
                   holdout_status=protocol.PROTOCOL["holdout_status"], historical_control_parity=historical,
                   units={"accepted": "selected track/candidate-group edges", "useful_target_frames": "deduplicated GT target frames",
                          "useful_fn_episodes": "deduplicated (sequence, GT identity, baseline FN episode start)",
                          "rejection_counts": "raw edges; reasons may overlap", "rates": "null when denominator is zero"},
                   invariants={"W1_W2_same_eligible_edges": True, "W3_eligible_subset_of_W2": True,
                               "scoring_reads_offline_labels": False},
                   methods=list(engine.METHODS), comparison=comparison,
                   review_queue_groups=len(reviews),
                   limitations=["This is a frozen-state comparison, not HOTA/IDF1/MOTA or an online update replay",
                                "Unknown and ignored labels are not counted as correct or incorrect",
                                "Natural negative evidence remains sparse; source groups are proxies, not verified mother videos",
                                "W_old changes multiple factors relative to W2 and is only a historical reference"])
    target = settings["paths"]["output_dir"]
    # All inputs, calculations, and full-control parity finish before publication.
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".wem_r31_", dir=target.parent))
    try:
        (stage / "summary.json").write_text(reporting.json_text(summary), encoding="utf-8")
        for name, rows in (("comparison.csv", comparison), ("per_sequence.csv", per_sequence),
                           ("per_source.csv", per_source), ("method_deltas.csv", deltas)):
            reporting.write_csv(stage / name, rows)
        reporting.write_csv(stage / "review_queue.csv", reviews,
                            ("case_id", "partition", "sequence", "frame", "track_id", "adopted_by", "human_review_status"))
        reporting.write_decisions(stage / "decisions.jsonl.gz", decisions)
        (stage / "protocol.json").write_text(reporting.json_text(protocol.PROTOCOL), encoding="utf-8")
        code_paths = sorted(Path(__file__).parent.glob("*.py")) + [root / "tools/run_wem_r31_ablation.py"]
        manifest = dict(schema="wem_r31_run_manifest_v1", status=summary["status"],
                        created_utc=datetime.now(timezone.utc).isoformat(), base_commit=protocol.BASE_COMMIT,
                        scope=bundle.scope, settings_sha256=settings["settings_sha256"], protocol_sha256=protocol_sha,
                        input_files=bundle.receipts,
                        code_sha256={str(path.relative_to(root)).replace(os.sep, "/"): inputs.digest(path.read_bytes())
                                     for path in code_paths},
                        historical_controls_sha256=inputs.digest(archive.read_bytes()) if bundle.scope == "full" else None,
                        runtime={"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__},
                        output_sha256={path.name: inputs.digest(path.read_bytes()) for path in sorted(stage.iterdir())},
                        enabled_online=False)
        (stage / "manifest.json").write_text(reporting.json_text(manifest), encoding="utf-8")
        if target.exists():
            raise ValueError("Output appeared during processing; refusing to replace it")
        stage.rename(target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {"status": summary["status"], "output_dir": str(target), "methods": summary["methods"],
            "review_queue_groups": len(reviews), "historical_control_parity": historical["status"],
            "enabled_online": False, "scientific_acceptance": "NOT_EVALUATED"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "check", "run"))
    parser.add_argument("--settings", type=Path, required=True,
                        help="Local paths JSON; relative data paths resolve from the repository root")
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            result = init_settings(args.settings)
        elif args.command == "check":
            _, bundle, protocol_sha = check_inputs(args.settings)
            result = {"status": "INPUTS_VERIFIED_NOT_RUN", **input_summary(bundle), "protocol_sha256": protocol_sha}
        else:
            result = run_comparison(args.settings)
    except (ValueError, KeyError, TypeError, OSError, EOFError, ImportError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
