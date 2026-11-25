"""Causal cache replay and independent TrackEval evaluation."""
from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import BASELINE_COMMIT
from .io import (detection_files, dump_json, forbid_gt, image_path, load_detections,
                 load_json, selected_sequences, sha256)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "results/vttiny/yolo11m_rgb_rgbt/rgbt/detections"
DEFAULT_CONFIG = ROOT / "configs/research/m2_default.json"
VARIANTS = ("native_iou", "m2_nll_baseline_r", "m2_nll_floor", "m2_nll_calibrated",
            "native_iou_det", "m2_nll_floor_det", "m2_nll_floor_native_update",
            "m2_nll_floor_native_update_det", "iou_floor_update", "m2_nll_first_native_update",
            "m2_nll_second_native_update", "iou_relaxed", "nwd_native_update",
            "native_reid", "reid_fixed", "reid_gap", "reid_recovery_fixed", "reid_recovery")
APPEARANCE_VARIANTS = ("native_reid", "reid_fixed", "reid_gap", "reid_recovery_fixed", "reid_recovery")
CMC_MODES = ("sof", "none", "m1_native_scale", "m1_mask_grid", "m1_fb", "m1_reliable")


def configure_runtime(runtime_root=None):
    root = runtime_root or os.environ.get("TRACKING_RUNTIME_ROOT")
    if root:
        root = Path(root).expanduser().resolve()
        package_root = root / "tracking_runtime"
        if not (package_root / "boxmot").is_dir():
            raise FileNotFoundError(f"Expected {package_root / 'boxmot'}; --runtime-root contains tracking_runtime")
        sys.path.insert(0, str(package_root))
    return root


def source_manifest():
    files = sorted((ROOT / "botsort_research").glob("*.py"))
    # The pooled evaluation adapter is repository code too; pin it alongside
    # the research package rather than relying only on upstream TrackEval.
    files += [ROOT / "tools/evaluate_vttiny.py", ROOT / "tools/vttiny_tracking_common.py"]
    result = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"git_commit": commit, "working_tree_dirty": dirty, "python_sha256": result}


def _runtime_manifest(tracker):
    import cv2
    import torch
    import boxmot
    from boxmot.trackers.botsort import botsort, botsort_track
    from boxmot.motion.kalman_filters.aabb import xywh_kf
    from boxmot.motion.cmc import sof
    paths = [Path(inspect.getfile(m)) for m in (botsort, botsort_track, xywh_kf, sof)]
    try:
        version = importlib.metadata.version("boxmot")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {"python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "opencv": cv2.__version__,
            "boxmot_distribution": version, "boxmot_package": getattr(boxmot, "__version__", None),
            "boxmot_source": {str(p): sha256(p) for p in paths},
            "max_time_lost": int(tracker.max_time_lost), "threads": 1}


class IdentityCMC:
    """Explicit CMC-off ablation; never used as an image-read failure fallback."""
    def apply(self, img, dets=None):
        return np.eye(2, 3, dtype=float)


def _prepare_experiment(config_path, noise_path, split_path, files, subset, sequences, variants):
    from .calibration import validate_split
    config = load_json(config_path)
    if config.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError("Config must identify the frozen baseline commit")
    if config["tracker"].get("with_reid"):
        raise ValueError("First M2 experiments require with_reid=false; appearance fusion is not calibrated")
    if config["tracker"].get("cmc_method") != "sof":
        raise ValueError("Config must preserve cmc_method=sof; use --cmc none for the explicit ablation")
    image_shape = config.get("expected_image_shape")
    if not isinstance(image_shape, list) or len(image_shape) != 2 or any(not isinstance(v, int) or v <= 0 for v in image_shape):
        raise ValueError("Config requires positive expected_image_shape=[height,width] in native coordinates")
    split = load_json(split_path) if split_path else None
    if split:
        validate_split(split, available=files)
    names = selected_sequences(files, split, subset, sequences)
    fitted = load_json(noise_path) if noise_path else None
    if "m2_nll_calibrated" in variants:
        if not fitted or not split:
            raise ValueError("Calibrated replay needs both --noise-model and --split")
        if subset != "evaluation":
            raise ValueError("Calibrated replay is restricted to the disjoint evaluation partition")
        provenance = fitted.get("provenance", {})
        fitted_sequences = provenance.get("sequences", provenance.get("calibration_sequences", []))
        if not fitted_sequences or not set(fitted_sequences).issubset(split["calibration"]):
            raise ValueError("Noise model must identify calibration sequences within this split")
        if set(fitted_sequences) & set(names):
            raise ValueError("Calibration and replay sequences overlap")
        if fitted.get("noise", {}).get("mode") != "calibrated":
            raise ValueError("Expected a fitted calibrated noise model")
        inputs = provenance.get("inputs", {})
        for name in fitted_sequences:
            expected = inputs.get(name, {}).get("detections_sha256")
            if not expected or sha256(files[name]) != expected:
                raise ValueError(f"Calibration cache changed or has no provenance hash: {name}")
        detector_hashes = provenance.get("detector_model_sha256", [])
        if len(detector_hashes) != 1 or not detector_hashes[0]:
            raise ValueError("Calibrated noise must identify exactly one frozen detector checkpoint")
    return config, split, names, fitted


def replay(cache_dir, image_root, output, *, variants=("native_iou", "m2_nll_baseline_r", "m2_nll_floor"),
           config_path=DEFAULT_CONFIG, noise_path=None, split_path=None, subset="evaluation",
           sequences=None, runtime_root=None, cmc="sof", trace=False, output_trace=False,
           embeddings_dir=None, appearance_model_path=None):
    """Run current-frame prediction only. A run directory is immutable once started."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory to avoid mixing experiments: {output}")
    if not variants or len(set(variants)) != len(variants) or set(variants) - set(VARIANTS):
        raise ValueError(f"Choose unique variants from {VARIANTS}")
    if cmc not in CMC_MODES:
        raise ValueError(f"CMC must be one of {CMC_MODES}")
    files = detection_files(cache_dir)
    config, split, names, fitted = _prepare_experiment(config_path, noise_path, split_path, files, subset, sequences, variants)
    appearance_variants = set(variants) & set(APPEARANCE_VARIANTS)
    appearance_model = load_json(appearance_model_path) if appearance_model_path else None
    if appearance_variants:
        if embeddings_dir is None:
            raise ValueError("Appearance replay requires --embeddings-dir with matching cached high-score RGB features")
        from .appearance import load_replay_features, dense_frame_features, validate_appearance_model
        from .appearance_adapter import build_appearance_tracker
        if appearance_variants - {"native_reid"}:
            if appearance_model is None or split is None or subset != "evaluation":
                raise ValueError("Calibrated appearance variants need --appearance-model, --split and the disjoint evaluation subset")
            validate_appearance_model(appearance_model, split, cache_dir, embeddings_dir, names)
    overrides = config.get("variant_overrides", {})
    if set(overrides) - set(VARIANTS):
        raise ValueError("Unknown variant in variant_overrides")
    for value in overrides.values():
        if set(value) - {"tracker", "m2"}:
            raise ValueError("Variant overrides accept tracker and m2 sections only")
    configure_runtime(runtime_root)
    import cv2
    import torch
    from .adapter import build_tracker
    from .output_policy import apply_output_policy, output_policy_for_variant
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    # Configuration/provenance may contain a fitted noise model, never live GT.
    sys.addaudithook(forbid_gt)
    record = {"schema_version": 2, "status": "RUNNING", "baseline_commit": BASELINE_COMMIT,
              "interpretation": "retrospective_development_only", "config": config,
              "split": split, "subset": subset if split else "all_or_explicit_sequences",
              "variants": list(variants), "cmc": cmc, "sequences": names,
              "source": source_manifest(), "cache_dir": str(Path(cache_dir).resolve()),
              "image_root": str(Path(image_root).resolve()), "cache_sha256": {},
              "frames": {}, "image_manifest_sha256": {}, "timing": [],
              "noise_model": fitted, "noise_model_sha256": sha256(noise_path) if noise_path else None,
              "appearance_model": appearance_model,
              "appearance_model_sha256": sha256(appearance_model_path) if appearance_model_path else None,
              "embeddings_dir": str(Path(embeddings_dir).resolve()) if embeddings_dir else None,
              "feature_inputs": {}, "output_trace": bool(output_trace), "effective_variants": {},
              "notes": ["Same frozen detections, pooled object class 0; no extra NMS.",
                        "No GT is loaded during prediction; no gap interpolation or retrospective outputs.",
                        "Association time excludes image decoding, hashing, detector and ReID inference.",
                        "Old official test is a development benchmark, not a newly independent holdout."]}
    output.mkdir(parents=True)
    dump_json(output / "experiment.json", record)
    try:
        detector_hash = None
        for name in names:
            payload = load_detections(files[name])
            current_hash = payload.get("model_sha256")
            if not current_hash or (detector_hash is not None and current_hash != detector_hash):
                raise ValueError("Replay requires caches from exactly one identified detector checkpoint")
            detector_hash = current_hash
            if "m2_nll_calibrated" in variants and current_hash != fitted["provenance"]["detector_model_sha256"][0]:
                raise ValueError("Calibration and replay use different detector checkpoints")
            record["detector_model_sha256"] = detector_hash
            record["cache_sha256"][name] = sha256(files[name])
            record["frames"][name] = len(payload["frames"])
            features, feature_info = None, None
            if appearance_variants:
                features, feature_info = load_replay_features(files[name], payload["frames"], embeddings_dir, name)
                record["feature_inputs"][name] = feature_info
                if "encoder_sha256" in record and record["encoder_sha256"] != feature_info["encoder_sha256"]:
                    raise ValueError("Appearance replay requires one fixed encoder checkpoint")
                if "feature_dimension" in record and record["feature_dimension"] != feature_info["dimension"]:
                    raise ValueError("Appearance replay requires one fixed feature dimension")
                record["encoder_sha256"] = feature_info["encoder_sha256"]
                record["feature_dimension"] = feature_info["dimension"]
                if appearance_model is not None:
                    expected_enc = appearance_model.get("provenance", {}).get("encoder_sha256")
                    if expected_enc and expected_enc != feature_info["encoder_sha256"]:
                        raise ValueError("Calibration and replay use different appearance encoders")
            image_hashes = []
            for variant in variants:
                params = copy.deepcopy(config["m2"])
                tracker_config = copy.deepcopy(config["tracker"])
                override = overrides.get(variant, {})
                tracker_config.update(override.get("tracker", {}))
                for key, value in override.get("m2", {}).items():
                    if isinstance(value, dict) and isinstance(params.get(key), dict):
                        params[key].update(value)
                    else:
                        params[key] = value
                if variant == "m2_nll_calibrated":
                    params["noise"] = fitted["noise"]
                policy = "kalman" if variant in APPEARANCE_VARIANTS else output_policy_for_variant(variant)
                record["effective_variants"][variant] = {"tracker": tracker_config, "m2": params,
                    "output_policy": policy, "appearance": config.get("appearance") if variant in APPEARANCE_VARIANTS else None}
                dest = output / "trackers" / variant / "data" / f"{name}.txt"
                dest.parent.mkdir(parents=True, exist_ok=True)
                trace_path = output / "traces" / variant / f"{name}.jsonl"
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                context = trace_path.open("w", encoding="utf-8") if trace else contextlib.nullcontext(None)
                output_trace_path = output / "outputs" / variant / f"{name}.jsonl"
                if output_trace:
                    output_trace_path.parent.mkdir(parents=True, exist_ok=True)
                output_context = output_trace_path.open("w", encoding="utf-8") if output_trace else contextlib.nullcontext(None)
                with context as trace_stream, output_context as output_stream:
                    def emit(event):
                        if trace_stream is not None:
                            trace_stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                    if variant in APPEARANCE_VARIANTS:
                        tracker = build_appearance_tracker(variant, tracker_config, config.get("appearance"),
                                                           appearance_model, trace_callback=emit if trace else None)
                    else:
                        tracker = build_tracker(variant, tracker_config, params, trace_callback=emit if trace else None)
                    record["effective_variants"][variant]["actual_with_reid"] = bool(tracker.with_reid)
                    if hasattr(tracker, "research_appearance"):
                        record["effective_variants"][variant]["appearance_runtime"] = tracker.research_appearance
                    if hasattr(tracker, "research_update_mode"):
                        record["effective_variants"][variant]["update_noise_mode"] = tracker.research_update_mode
                    if cmc == "none":
                        tracker.cmc = IdentityCMC()
                    elif cmc != "sof":
                        from .robust_cmc import build_cmc
                        tracker.cmc = build_cmc(cmc, config.get("m1"), trace_callback=emit if trace else None)
                    record.setdefault("runtime", _runtime_manifest(tracker))
                    record.setdefault("reviewed_runtime", tracker.research_runtime_info)
                    rows, association_seconds = 0, 0.0
                    started = time.perf_counter()
                    with dest.open("w", encoding="utf-8", newline="") as stream:
                        writer = csv.writer(stream, lineterminator="\n")
                        for index, frame in enumerate(payload["frames"]):
                            path = image_path(frame, image_root)
                            img = cv2.imread(str(path))
                            if img is None:
                                raise FileNotFoundError(f"Cannot decode {path}; --image-root must contain test2017")
                            if list(img.shape[:2]) != config["expected_image_shape"]:
                                raise ValueError(f"Image shape {img.shape[:2]} disagrees with cached native coordinate shape {config['expected_image_shape']}: {path}")
                            if variant == variants[0]:
                                image_hashes.append({"frame": frame["frame"], "path": str(path), "sha256": sha256(path), "shape": list(img.shape)})
                            det = np.asarray(frame["detections"], dtype=np.float32).reshape(-1, 6).copy()
                            det[:, 5] = 0
                            start = time.perf_counter()
                            if variant in APPEARANCE_VARIANTS:
                                embs = dense_frame_features(frame, features, feature_info["dimension"])
                                predicted = tracker.update(det, img, embs)
                            else:
                                predicted = tracker.update(det, img)
                            association_seconds += time.perf_counter() - start
                            filtered = np.asarray(predicted, dtype=float).reshape(-1, 8).copy()
                            predicted = apply_output_policy(filtered, det, policy)
                            if output_stream is not None:
                                observations = []
                                for k, row in enumerate(predicted):
                                    index_det = row[7]
                                    if index_det != int(index_det) or not 0 <= int(index_det) < len(det):
                                        raise ValueError("Tracker output cannot be mapped to a current detection")
                                    observations.append({"track_id": int(row[4]), "det_index": int(index_det),
                                        "score": float(row[5]), "detection_box": det[int(index_det), :4].tolist(),
                                        "kalman_box": filtered[k, :4].tolist(), "output_box": row[:4].tolist()})
                                output_stream.write(json.dumps({"frame": frame["frame"], "policy": policy,
                                    "observations": observations}, allow_nan=False) + "\n")
                            seen = set()
                            for row in predicted:
                                if len(row) != 8 or not np.isfinite(row).all():
                                    raise ValueError(f"Invalid tracker output: {variant}/{name}/{frame['frame']}")
                                x, y, x2, y2, tid, score, cls, _ = row
                                if tid != int(tid) or tid <= 0 or tid in seen or cls != 0 or x2 <= x or y2 <= y:
                                    raise ValueError(f"Invalid ID/class/box: {variant}/{name}/{frame['frame']}")
                                seen.add(tid)
                                writer.writerow([frame["frame"], int(tid), x, y, x2-x, y2-y, score, -1, -1, -1])
                                rows += 1
                    entry = {"variant": variant, "sequence": name, "frames": len(payload["frames"]), "rows": rows,
                             "association_seconds": association_seconds, "replay_seconds": time.perf_counter()-started,
                             "predictions_sha256": sha256(dest)}
                    if hasattr(tracker.cmc, "summary"):
                        entry["cmc_summary"] = tracker.cmc.summary()
                    if output_trace:
                        output_stream.flush()
                        entry["output_trace_sha256"] = sha256(output_trace_path)
                    if trace:
                        trace_stream.flush()
                        entry["trace_sha256"] = sha256(trace_path)
                    record["timing"].append(entry)
                    print(f"REPLAY {variant} {name} frames={entry['frames']} rows={rows}", flush=True)
            image_manifest = output / "inputs" / f"{name}.images.json"
            dump_json(image_manifest, image_hashes)
            record["image_manifest_sha256"][name] = sha256(image_manifest)
            dump_json(output / "experiment.json", record)
        record["status"] = "COMPLETE"
        dump_json(output / "experiment.json", record)
        return record
    except Exception as exc:
        record["status"] = "FAILED"
        record["error"] = f"{type(exc).__name__}: {exc}"
        dump_json(output / "experiment.json", record)
        raise


def compare_native(run_dir, reference_dir, atol=1e-6, variant="native_iou"):
    run_dir, reference_dir = Path(run_dir), Path(reference_dir)
    record = load_json(run_dir / "experiment.json")
    if variant not in ("native_iou", "native_reid"):
        raise ValueError("Only native_iou/native_reid are frozen-baseline comparisons")
    if record.get("status") != "COMPLETE" or variant not in record["variants"]:
        raise ValueError(f"Need a complete run containing {variant}")
    if record["cmc"] != "sof":
        raise ValueError("Native baseline comparison needs the frozen SOF CMC")
    results = []
    for seq in record["sequences"]:
        a = run_dir / "trackers" / variant / "data" / f"{seq}.txt"
        b = reference_dir / f"{seq}.txt"
        def read_mot(p):
            if not p.read_text().strip():
                return np.empty((0, 10))
            return np.loadtxt(p, delimiter=",", ndmin=2)
        actual, expected = read_mot(a), read_mot(b)
        same = actual.shape == expected.shape and np.allclose(actual, expected, atol=atol, rtol=0)
        results.append({"sequence": seq, "match": bool(same), "actual_rows": len(actual), "reference_rows": len(expected)})
    result = {"status": "PASS" if all(r["match"] for r in results) else "FAIL", "variant": variant,
              "absolute_tolerance": atol, "results": results}
    dump_json(run_dir / "baseline_comparison.json", result)
    if result["status"] != "PASS":
        raise ValueError("Native replay differs from the archive; inspect baseline_comparison.json before interpreting M2")
    return result


def evaluate_run(run_dir, gt_root, trackeval_root=None):
    """Run pinned TrackEval in a separate process from prediction's GT guard."""
    run_dir = Path(run_dir).resolve()
    record = load_json(run_dir / "experiment.json")
    if record.get("status") != "COMPLETE":
        raise ValueError("Only complete runs may be evaluated")
    destination = run_dir / "metrics"
    if destination.exists():
        raise FileExistsError(f"Metrics already exist: {destination}")
    root = Path(trackeval_root or os.environ.get("TRACKEVAL_ROOT", ROOT / "external/TrackEval")).resolve()
    pinned = load_json(ROOT / "provenance/trackeval.json")
    fingerprints = pinned["package_sha256"]
    mismatches, actual_fingerprints, line_ending_only = [], {}, []
    for name, expected in fingerprints.items():
        path = root / name
        if not path.is_file():
            mismatches.append(name)
            continue
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        actual_fingerprints[name] = actual
        if actual == expected:
            continue
        # The archived Windows runtime fingerprints contain CRLF. Linux git
        # checkouts use LF; accept only that exact byte transformation.
        archived_newlines = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        if hashlib.sha256(archived_newlines).hexdigest() == expected:
            line_ending_only.append(name)
        else:
            mismatches.append(name)
    if mismatches:
        raise ValueError(f"TrackEval must match provenance/trackeval.json; missing/changed: {mismatches[:5]}")
    # Check predicted files are exactly those completed by replay.
    for entry in record["timing"]:
        path = run_dir / "trackers" / entry["variant"] / "data" / (entry["sequence"] + ".txt")
        if sha256(path) != entry["predictions_sha256"]:
            raise ValueError(f"Prediction changed after run: {path}")
    os.environ["TRACKEVAL_ROOT"] = str(root)
    sys.path.insert(0, str(ROOT / "tools"))
    from evaluate_vttiny import evaluate
    result = evaluate("rgb", "test", run_dir / "trackers", record["variants"], destination,
                      seqs=record["frames"], gt_folder=Path(gt_root).resolve())
    result["interpretation"] = record["interpretation"]
    result["trackeval_provenance"] = {"commit": pinned["commit"], "actual_source_sha256": actual_fingerprints,
                                      "accepted_line_ending_only_differences": line_ending_only}
    result["experiment_manifest_sha256"] = sha256(run_dir / "experiment.json")
    result["gt_sha256"] = {s: sha256(Path(gt_root) / s / "gt/gt.txt") for s in record["sequences"]}
    dump_json(destination / "summary.json", result)
    combined = [r for r in result["results"] if r["sequence"] == "COMBINED_SEQ"]
    fields = ("tracker", "HOTA", "DetA", "AssA", "IDF1", "MOTA", "FP", "FN", "IDSW")
    lines = ["# M2 experiment results", "", "Retrospective development comparison; COMBINED_SEQ, not a mean of sequence percentages.", "",
             "|"+"|".join(fields)+"|", "|"+"|".join(["---"]*len(fields))+"|"]
    for row in combined:
        lines.append("|"+"|".join(str(row[f]) if isinstance(row[f], (str, int)) else f"{row[f]:.4f}" for f in fields)+"|")
    (destination / "comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result
