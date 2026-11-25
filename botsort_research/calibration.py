"""Offline detector-noise calibration; never imported by the tracking update path.

Ground truth is allowed here only. The fitted parameters contain no per-frame GT.
An automatic split of the archived test set is retrospective development, not an
independent test set: that set has already informed checkpoint/model selection.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment, nnls


def flight_group(sequence: str) -> str:
    """Keep differently cut clips with the same DJI recording prefix together."""
    match = re.match(r"^(DJI_\d+)(?:_|$)", sequence, flags=re.IGNORECASE)
    return match.group(1).upper() if match else sequence


def build_split(sequence_names: Iterable[str], seed: int = 0, fraction: float = 0.3) -> dict:
    names = sorted(sequence_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("Sequence names must be nonempty and unique.")
    if not 0 < fraction < 1:
        raise ValueError("Calibration fraction must lie strictly between 0 and 1.")
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(flight_group(name), []).append(name)
    if len(groups) < 2:
        raise ValueError("At least two recording groups are needed for a disjoint split.")
    # A hash ranking is stable across Python/NumPy versions and input order.
    order = sorted(groups, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
    count = min(len(order) - 1, max(1, round(len(order) * fraction)))
    calibration_groups = set(order[:count])
    result = {
        "schema_version": 1,
        "interpretation": "retrospective_development_only",
        "grouping": "DJI flight prefix",
        "seed": seed,
        "requested_calibration_group_fraction": fraction,
        "calibration": [n for n in names if flight_group(n) in calibration_groups],
        "evaluation": [n for n in names if flight_group(n) not in calibration_groups],
        "limitations": [
            "The archived official test set already informed model/checkpoint selection; this split is not a fresh independent test.",
            "A DJI recording prefix is a grouping proxy, not proof of independent flights, sites, subjects or capture times.",
            "Replace this manifest with an explicitly audited recording-level split before making generalization claims.",
        ],
    }
    validate_split(result, available=names)
    return result


def validate_split(split: Mapping, available: Iterable[str] | None = None) -> None:
    """Reject both exact-sequence overlap and same-recording clip overlap."""
    if split.get("schema_version") != 1:
        raise ValueError("Unsupported split schema_version; expected 1.")
    partitions = {}
    for key in ("calibration", "evaluation"):
        values = split.get(key)
        if not isinstance(values, list) or not values or not all(isinstance(s, str) and s for s in values):
            raise ValueError(f"{key} must be a nonempty list of sequence names.")
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate sequences in {key}.")
        partitions[key] = set(values)
    overlap = partitions["calibration"] & partitions["evaluation"]
    if overlap:
        raise ValueError(f"Calibration/evaluation overlap: {sorted(overlap)}")
    group_overlap = {flight_group(s) for s in partitions["calibration"]} & {flight_group(s) for s in partitions["evaluation"]}
    if group_overlap:
        raise ValueError(f"Recording-group leakage across partitions: {sorted(group_overlap)}")
    if available is not None:
        unknown = set.union(*partitions.values()) - set(available)
        if unknown:
            raise ValueError(f"Split refers to unavailable detection caches: {sorted(unknown)}")
    if not isinstance(split.get("interpretation"), str) or not split["interpretation"]:
        raise ValueError("A split must explicitly state its interpretation.")


def _xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    size = boxes[:, 2:] - boxes[:, :2]
    return np.column_stack((boxes[:, :2] + size / 2, size))


def select_pairs(
    detections: np.ndarray,
    ground_truth: np.ndarray,
    *,
    center_gate: float = 1.5,
    size_ratio_gate: float = 4.0,
    ambiguity_margin: float = 0.2,
    min_score: float = 0.1,
) -> tuple[np.ndarray, dict]:
    """Select conservative one-to-one calibration pairs without an IoU cutoff.

    Returns original detection/GT indices. GT columns start with xyxy, followed
    optionally by ID. The geometric assignment and BOTH row/column ambiguity
    margins intentionally exclude crowded/duplicate detections. Selection bias
    remains; diagnostics report it and these pairs are not an oracle evaluation.
    """
    det = np.asarray(detections, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    if det.size == 0:
        det = np.empty((0, 6))
    if gt.size == 0:
        gt = np.empty((0, 5))
    if det.ndim != 2 or det.shape[1] < 5 or gt.ndim != 2 or gt.shape[1] < 4:
        raise ValueError("Detections require xyxy, score; GT requires xyxy.")
    if not np.isfinite(det).all() or not np.isfinite(gt).all():
        raise ValueError("Nonfinite boxes/scores cannot be calibrated.")
    if not (center_gate > 0 and size_ratio_gate > 1 and ambiguity_margin >= 0 and 0 <= min_score <= 1):
        raise ValueError("Invalid pair selection gates.")
    if np.any(det[:, 2:4] <= det[:, :2]) or np.any(gt[:, 2:4] <= gt[:, :2]):
        raise ValueError("All calibration boxes must have positive width and height.")
    if np.any((det[:, 4] < 0) | (det[:, 4] > 1)):
        raise ValueError("Detection scores must lie in [0, 1].")
    kept = np.flatnonzero(det[:, 4] > min_score)
    stats = {"detections": len(det), "ground_truth": len(gt), "below_tracking_score": len(det) - len(kept),
             "candidate_pairs": 0, "assigned_pairs": 0, "ambiguous_pairs": 0, "accepted_pairs": 0}
    if not len(kept) or not len(gt):
        return np.empty((0, 2), dtype=np.int64), stats
    dz = _xyxy_to_xywh(det[kept, :4])
    gz = _xyxy_to_xywh(gt[:, :4])
    scale = np.maximum(np.maximum(dz[:, None, 2:], gz[None, :, 2:]), 6.0)
    delta = (dz[:, None, :2] - gz[None, :, :2]) / scale
    center = np.linalg.norm(delta, axis=2)
    log_ratio = np.abs(np.log(dz[:, None, 2:] / gz[None, :, 2:]))
    valid = (center <= center_gate) & np.all(log_ratio <= np.log(size_ratio_gate), axis=2)
    cost = center + 0.25 * log_ratio.sum(axis=2)
    stats["candidate_pairs"] = int(valid.sum())
    # Explicit rejection columns avoid forcing a geometrically impossible pair.
    reject = center_gate + 0.5 * np.log(size_ratio_gate) + 1.0
    augmented = np.full((len(dz), len(gz) + len(dz)), reject)
    augmented[:, :len(gz)] = np.where(valid, cost, reject * 1000)
    rows, cols = linear_sum_assignment(augmented)
    accepted = []
    for row, col in zip(rows, cols):
        if col >= len(gz) or not valid[row, col]:
            continue
        stats["assigned_pairs"] += 1
        row_competitors = np.flatnonzero(valid[row])
        row_competitors = row_competitors[row_competitors != col]
        col_competitors = np.flatnonzero(valid[:, col])
        col_competitors = col_competitors[col_competitors != row]
        row_alt = np.min(cost[row, row_competitors]) if len(row_competitors) else np.inf
        col_alt = np.min(cost[col_competitors, col]) if len(col_competitors) else np.inf
        if row_alt - cost[row, col] < ambiguity_margin or col_alt - cost[row, col] < ambiguity_margin:
            stats["ambiguous_pairs"] += 1
            continue
        accepted.append((kept[row], col))
    stats["accepted_pairs"] = len(accepted)
    return np.asarray(accepted, dtype=np.int64).reshape(-1, 2), stats


def fit_noise_model(measurements: np.ndarray, residuals: np.ndarray, scores: np.ndarray, *, min_samples: int = 100) -> tuple[dict, dict]:
    """Nonnegative second-moment regression with explicit scale normalization."""
    z = np.asarray(measurements, dtype=np.float64)
    r = np.asarray(residuals, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != 4 or r.shape != z.shape or s.shape != (len(z),):
        raise ValueError("Expected measurements/residuals N x 4 and scores N.")
    if not isinstance(min_samples, int) or min_samples < 4:
        raise ValueError("min_samples must be an integer of at least four.")
    if len(z) < min_samples:
        raise ValueError(f"Insufficient unambiguous calibration pairs: {len(z)} < {min_samples}. Do not substitute evaluation data.")
    if not np.isfinite(z).all() or not np.isfinite(r).all() or not np.isfinite(s).all():
        raise ValueError("Calibration inputs must be finite.")
    if np.any(z[:, 2:] <= 0) or np.any((s < 0) | (s > 1)):
        raise ValueError("Positive box sizes and scores in [0,1] required.")
    dimensions = z[:, [2, 3, 2, 3]]
    coefficients, ranks, conditions = [], [], []
    predictions = np.empty_like(r)
    for axis in range(4):
        design = np.column_stack((np.ones(len(z)), dimensions[:, axis] ** 2, (1 - s) ** 2))
        scale = np.maximum(np.sqrt(np.mean(design ** 2, axis=0)), 1e-12)
        normalized = design / scale
        weights, _ = nnls(normalized, r[:, axis] ** 2)
        beta = weights / scale
        coefficients.append(beta)
        ranks.append(int(np.linalg.matrix_rank(normalized)))
        cond = float(np.linalg.cond(normalized))
        conditions.append(cond if np.isfinite(cond) else None)
        predictions[:, axis] = np.maximum(design @ beta, 1e-6)
    coefficients = np.asarray(coefficients)
    noise = {"mode": "calibrated", "floor_std": np.sqrt(coefficients[:, 0]).tolist(),
             "size_scale": np.sqrt(coefficients[:, 1]).tolist(),
             "score_scale": np.sqrt(coefficients[:, 2]).tolist(), "min_variance": 1e-6}
    standardized = r ** 2 / predictions
    diagnostics = {
        "samples": len(z), "axis_order": ["cx", "cy", "w", "h"],
        "design_rank": ranks, "design_condition_number": conditions,
        "residual_mean": np.mean(r, axis=0).tolist(), "residual_median": np.median(r, axis=0).tolist(),
        "residual_rmse": np.sqrt(np.mean(r ** 2, axis=0)).tolist(),
        "standardized_residual_rms": np.sqrt(np.mean(standardized, axis=0)).tolist(),
        "in_sample_detection_residual_4d_coverage_at_9_4877": float(np.mean(standardized.sum(axis=1) <= 9.4877)),
        "warnings": [
            "Fit uses squared residuals, including annotation error and systematic detector bias; it is not a causal decomposition of camera and detector noise.",
            "Geometric matching and ambiguity exclusion truncate the residual population; coverage is in-sample and conditional on selected pairs.",
            "Adjacent frames are correlated. Pair count is not an independent-sample count; use recording-level evaluation.",
            "Zero-mean diagonal Gaussian measurement error remains a modeling assumption, not a validated distribution.",
        ] + (["Some design matrices are rank deficient; coefficients are not separately identifiable."] if min(ranks) < 3 else []),
    }
    return noise, diagnostics


def _stratum(z: np.ndarray, score: float) -> str:
    side = float(np.sqrt(z[2] * z[3]))
    size = "lt8" if side < 8 else "8to16" if side < 16 else "16to32" if side < 32 else "ge32"
    confidence = "low_0.1to0.5" if score < 0.5 else "boundary_0.5" if score == 0.5 else "high_gt0.5"
    return f"{size}/{confidence}"


def calibrate(cache_dir, gt_root, sequences, output=None, *, split=None, min_samples=100, max_pairs_per_sequence=20000, seed=0) -> dict:
    """Fit parameters from calibration sequences, with input hashes and exclusions.

    ``output`` receives the complete JSON object; its ``noise`` field is passed to
    NoiseModel.from_dict. Callers must retain the provenance alongside that model.
    """
    from .io import detection_files, load_detections, load_gt, sha256

    output = Path(output) if output is not None else None
    if output is not None and output.exists():
        raise FileExistsError(f"Refusing to overwrite calibration result: {output}")
    names = list(sequences)
    if not names or len(names) != len(set(names)):
        raise ValueError("Calibration sequences must be nonempty and unique.")
    if not isinstance(max_pairs_per_sequence, int) or max_pairs_per_sequence < 1:
        raise ValueError("max_pairs_per_sequence must be a positive integer.")
    files = detection_files(Path(cache_dir))
    unknown = set(names) - set(files)
    if unknown:
        raise ValueError(f"Missing detection caches: {sorted(unknown)}")
    if split is not None:
        validate_split(split, available=files)
        if set(names) != set(split["calibration"]):
            raise ValueError("Calibration sequence list must exactly equal split['calibration'].")
    rng = np.random.default_rng(seed)
    samples, sequence_stats, hashes = [], {}, {}
    detector_hashes = set()
    all_strata, retained_strata = Counter(), Counter()
    for sequence in sorted(names):
        payload = load_detections(files[sequence])
        gt = load_gt(Path(gt_root), sequence)
        frame_ids = {int(f["frame"]) for f in payload["frames"]}
        if set(gt) - frame_ids:
            raise ValueError(f"GT frames absent from cache in {sequence}; check dataset/coordinate alignment.")
        if payload.get("model_sha256"):
            detector_hashes.add(payload["model_sha256"])
        stats = Counter()
        reservoir, seen = [], 0
        for frame in payload["frames"]:
            det = np.asarray(frame["detections"], dtype=np.float64).reshape(-1, 6)
            truth = np.asarray(gt.get(int(frame["frame"]), np.empty((0, 5))), dtype=np.float64).reshape(-1, 5)
            pairs, counts = select_pairs(det, truth)
            stats.update(counts)
            stats["frames"] += 1
            for d, g in pairs:
                z = _xyxy_to_xywh(det[d:d + 1, :4])[0]
                residual = z - _xyxy_to_xywh(truth[g:g + 1, :4])[0]
                score = float(det[d, 4])
                item = (z, residual, score)
                all_strata[_stratum(z, score)] += 1
                seen += 1
                if len(reservoir) < max_pairs_per_sequence:
                    reservoir.append(item)
                else:
                    index = int(rng.integers(seen))
                    if index < max_pairs_per_sequence:
                        reservoir[index] = item
        for z, _, score in reservoir:
            retained_strata[_stratum(z, score)] += 1
        samples.extend(reservoir)
        stats["retained_pairs"] = len(reservoir)
        sequence_stats[sequence] = dict(stats)
        gt_path = Path(gt_root) / sequence / "gt" / "gt.txt"
        hashes[sequence] = {"detections_sha256": sha256(files[sequence]), "gt_sha256": sha256(gt_path)}
    if len(detector_hashes) > 1:
        raise ValueError("Calibration mixes detector checkpoints; use one frozen detector cache.")
    z = np.asarray([x[0] for x in samples], dtype=np.float64).reshape(-1, 4)
    r = np.asarray([x[1] for x in samples], dtype=np.float64).reshape(-1, 4)
    scores = np.asarray([x[2] for x in samples], dtype=np.float64)
    noise, fit = fit_noise_model(z, r, scores, min_samples=min_samples)
    result = {
        "schema_version": 1, "noise": noise,
        "provenance": {
            "purpose": "offline_calibration_only_no_gt_during_tracking",
            "interpretation": split["interpretation"] if split is not None else "development_only_no_evaluation_partition_verified",
            "calibration_sequences": sorted(names),
            "evaluation_sequences": sorted(split["evaluation"]) if split is not None else [],
            "detector_model_sha256": sorted(detector_hashes), "inputs": hashes,
            "seed": seed, "max_pairs_per_sequence": max_pairs_per_sequence,
            "pair_selection": {"center_gate": 1.5, "size_ratio_gate": 4.0, "ambiguity_margin": 0.2, "min_score_strictly_above": 0.1, "iou_gate": None},
            "fit": "Nonnegative least squares of squared residuals on [1, detection_size^2, (1-score)^2], separately for cx,cy,w,h; uniform reservoir cap per sequence.",
        },
        "diagnostics": {**fit, "sequences": sequence_stats,
                        "accepted_pair_strata": dict(sorted(all_strata.items())),
                        "retained_pair_strata": dict(sorted(retained_strata.items()))},
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        # Exclusive creation also protects against a simultaneous experiment.
        with output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    return result
