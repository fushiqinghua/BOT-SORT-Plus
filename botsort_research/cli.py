"""Command-line entry point for staged M2 experiments and M1/M3 diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .io import detection_files, dump_json, image_path, load_detections, load_json, selected_sequences
from .runner import DEFAULT_CACHE, DEFAULT_CONFIG, ROOT, VARIANTS


def _selection(args):
    from .calibration import validate_split
    files = detection_files(args.cache_dir)
    split = load_json(args.split) if getattr(args, "split", None) else None
    if split:
        validate_split(split, available=files)
    names = selected_sequences(files, split, getattr(args, "subset", "evaluation"), getattr(args, "sequences", None))
    return files, split, names


def _new_path(path):
    if Path(path).exists():
        raise FileExistsError(f"Use a new output path; existing outputs are preserved: {path}")


def _doctor(args):
    files, split, names = _selection(args)
    totals = {"sequences": len(names), "frames": 0, "detections": 0, "empty_detection_frames": 0}
    missing, model_hashes = [], set()
    for name in names:
        cache = load_detections(files[name])
        model_hashes.add(cache.get("model_sha256"))
        for frame in cache["frames"]:
            totals["frames"] += 1
            totals["detections"] += len(frame["detections"])
            totals["empty_detection_frames"] += not bool(frame["detections"])
            if args.image_root:
                path = image_path(frame, args.image_root)
                if not path.is_file():
                    missing.append(str(path))
    if len(model_hashes) != 1 or None in model_hashes:
        raise ValueError("Detection caches must identify one fixed detector checkpoint")
    result = {"cache_validation": "PASS", **totals, "detector_sha256": next(iter(model_hashes)),
              "images_checked": bool(args.image_root), "missing_image_count": len(missing),
              "first_missing_images": missing[:5], "runtime_checked": bool(args.check_runtime),
              "interpretation": "retrospective_development_only"}
    if args.check_runtime:
        from .runner import configure_runtime
        configure_runtime(args.runtime_root)
        from .adapter import verify_runtime
        result["runtime"] = verify_runtime()
    if args.output:
        _new_path(args.output)
        dump_json(args.output, result)
    if missing:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise FileNotFoundError("Images are missing; check --image-root before replay")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def cache_options(p, selection=True):
        p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE, help="Frozen per-sequence .json or .json.gz detections")
        if selection:
            p.add_argument("--split", type=Path, help="Recording-disjoint split JSON")
            p.add_argument("--subset", choices=("calibration", "evaluation"), default="evaluation")
            p.add_argument("--sequences", nargs="+", help="Optional subset of sequence names, checked against the split")

    p = sub.add_parser("doctor", help="Validate cached detections, optional image paths and BoxMOT source compatibility")
    cache_options(p)
    p.add_argument("--image-root", type=Path)
    p.add_argument("--check-runtime", action="store_true")
    p.add_argument("--runtime-root", type=Path)
    p.add_argument("--output", type=Path)

    p = sub.add_parser("make-split", help="Create a grouped retrospective development split; never a new independent test")
    cache_options(p, selection=False)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--calibration-fraction", type=float, default=0.3)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("calibrate", help="Offline GT calibration of detection measurement noise")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--gt-root", type=Path, required=True, help="Directory containing <sequence>/gt/gt.txt")
    p.add_argument("--min-samples", type=int, default=100)
    p.add_argument("--max-pairs-per-sequence", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("run", help="Replay M2 variants with fixed detections; no GT allowed")
    cache_options(p)
    p.add_argument("--image-root", type=Path, required=True, help="Official image root containing test2017/")
    p.add_argument("--runtime-root", type=Path, help="Optional original runtime directory containing tracking_runtime/boxmot")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--noise-model", type=Path)
    p.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS[:3]))
    p.add_argument("--cmc", choices=("sof", "none"), default="sof")
    p.add_argument("--trace", action="store_true", help="Write per-stage M2 matches and gate rejection counts")
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("compare-baseline", help="Compare native_iou replay against archived no-ReID trajectories")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, default=ROOT / "results/vttiny/yolo11m_rgb_rgbt/rgbt/trackers/botsort_no_reid/data")
    p.add_argument("--atol", type=float, default=1e-6)

    p = sub.add_parser("evaluate", help="Evaluate completed replay with the pinned TrackEval implementation")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--gt-root", type=Path, required=True)
    p.add_argument("--trackeval-root", type=Path)

    p = sub.add_parser("summarize", help="Compare evaluated runs only when their caches, GT and sequence sets agree")
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("diagnose-m1", help="Inspect native sparse-flow geometry proxies; images required")
    cache_options(p)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--max-pairs", type=int, help="Debug-only per-sequence frame-pair cap; omit for complete diagnostics")
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("diagnose-m3", help="Offline coverage/gap/appearance diagnostics; embeddings optional")
    cache_options(p)
    p.add_argument("--gt-root", type=Path, required=True)
    p.add_argument("--embeddings-dir", type=Path, help="Original per-sequence .npz and .json feature caches")
    p.add_argument("--predictions-dir", type=Path, help="MOT .txt directory; use archived botsort_reid/data first")
    p.add_argument("--max-pairs-per-sequence", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = _doctor(args)
        elif args.command == "make-split":
            from .calibration import build_split
            _new_path(args.output)
            result = build_split(detection_files(args.cache_dir), seed=args.seed, fraction=args.calibration_fraction)
            dump_json(args.output, result)
        elif args.command == "calibrate":
            from .calibration import calibrate, validate_split
            split = load_json(args.split)
            validate_split(split, available=detection_files(args.cache_dir))
            result = calibrate(args.cache_dir, args.gt_root, split["calibration"], args.output,
                               split=split, min_samples=args.min_samples,
                               max_pairs_per_sequence=args.max_pairs_per_sequence, seed=args.seed)
        elif args.command == "run":
            from .runner import replay
            result = replay(args.cache_dir, args.image_root, args.output, variants=args.variants,
                            config_path=args.config, noise_path=args.noise_model, split_path=args.split,
                            subset=args.subset, sequences=args.sequences, runtime_root=args.runtime_root,
                            cmc=args.cmc, trace=args.trace)
        elif args.command == "compare-baseline":
            from .runner import compare_native
            result = compare_native(args.run_dir, args.reference_dir, atol=args.atol)
        elif args.command == "evaluate":
            from .runner import evaluate_run
            result = evaluate_run(args.run_dir, args.gt_root, args.trackeval_root)
        elif args.command == "summarize":
            from .reporting import summarize_runs
            result = summarize_runs(args.runs, args.output)
        elif args.command == "diagnose-m1":
            from .cmc import diagnose_motion
            files, split, names = _selection(args)
            _new_path(args.output)
            if args.max_pairs is not None and args.max_pairs < 1:
                raise ValueError("--max-pairs must be positive")
            summaries = []
            for name in names:
                frames = load_detections(files[name])["frames"]
                summaries.append(diagnose_motion(frames, lambda frame: image_path(frame, args.image_root),
                                                 args.output / name, max_pairs=args.max_pairs, sequence=name))
            result = {"interpretation": "GMC geometry proxies, not true camera-motion error", "split": split,
                      "subset": args.subset if split else "all_or_explicit", "sequences": summaries}
            dump_json(args.output / "manifest.json", result)
        elif args.command == "diagnose-m3":
            from .diagnostics import diagnose_appearance
            files, split, names = _selection(args)
            _new_path(args.output)
            result = diagnose_appearance(args.cache_dir, args.gt_root, args.output, sequences=names,
                                         embeddings_dir=args.embeddings_dir, predictions_dir=args.predictions_dir,
                                         max_pairs_per_sequence=args.max_pairs_per_sequence, seed=args.seed)
            dump_json(args.output / "manifest.json", {"split": split, "subset": args.subset if split else "all_or_explicit",
                                                       "sequences": names, "interpretation": "retrospective_development_only"})
        else:
            raise AssertionError("Unreachable command")
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, ImportError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    # Full manifests live in output files; print only a small completion summary.
    if args.command in ("doctor", "make-split", "compare-baseline"):
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"COMPLETE {args.command}: {getattr(args, 'output', None) or args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
