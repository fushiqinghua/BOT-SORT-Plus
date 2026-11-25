"""Gaussian likelihood association and exact assignment with explicit rejection."""
import numpy as np
from scipy.optimize import linear_sum_assignment


DEFAULT_STAGE = {
    "gate_chi2": 18.4668269529,  # chi2(df=4), 0.999; a starting hypothesis.
    "max_center_distance_px": 64.0,
    "max_size_ratio": 4.0,
    "miss_cost": 8.0,
    "unassigned_cost": 8.0,
}


def stage_config(config=None):
    config = dict(config or {})
    extra = set(config) - set(DEFAULT_STAGE)
    if extra:
        raise ValueError(f"Unknown association parameters: {sorted(extra)}")
    merged = {**DEFAULT_STAGE, **config}
    for name, value in merged.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if name in {"miss_cost", "unassigned_cost"}:
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        elif value <= 0:
            raise ValueError(f"{name} must be positive")
    if merged["max_size_ratio"] < 1:
        raise ValueError("max_size_ratio must be at least one")
    return merged


def gaussian_cost(means, covariances, measurements, scores, noise, config=None):
    """Return NLL matrix and diagnostics; invalid/gated pairs have +inf cost.

    z=(cx,cy,w,h), S=H P H.T + R, C=.5*(r.T S^-1 r+logdet(S)+4log(2pi)).
    All coordinates are native-image pixels; the density reference is fixed at
    one pixel in each coordinate. The constant is retained when comparing to
    miss + unassigned costs. gate_reason bits: 1=Mahalanobis, 2=center,
    4=size ratio, 8=invalid predicted state or non-positive-definite S.
    """
    cfg = stage_config(config)
    means = np.asarray(means, dtype=np.float64).reshape(-1, 8)
    covariances = np.asarray(covariances, dtype=np.float64).reshape(-1, 8, 8)
    measurements = np.asarray(measurements, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n, m = len(means), len(measurements)
    if len(covariances) != n or len(scores) != m:
        raise ValueError("State, covariance, measurement and score counts disagree")
    if not np.isfinite(measurements).all() or np.any(measurements[:, 2:] <= 0):
        raise ValueError("Detections must be finite with positive sizes")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be finite and lie in [0,1]")
    costs = np.full((n, m), np.inf)
    maha = np.full((n, m), np.nan)
    logdet = np.full((n, m), np.nan)
    reasons = np.zeros((n, m), dtype=np.uint8)
    measurement_covariances = np.full((n, m, 4, 4), np.nan)
    constant = 4 * np.log(2 * np.pi)
    for i, (mean, p) in enumerate(zip(means, covariances)):
        if not np.isfinite(mean).all() or not np.isfinite(p).all() or np.any(mean[2:4] <= 0):
            reasons[i, :] = 8
            continue
        p4 = 0.5 * (p[:4, :4] + p[:4, :4].T)
        for j, (z, score) in enumerate(zip(measurements, scores)):
            r = z - mean[:4]
            if np.linalg.norm(r[:2]) > cfg["max_center_distance_px"]:
                reasons[i, j] |= 2
            ratios = z[2:] / mean[2:4]
            if np.max(np.maximum(ratios, 1.0 / ratios)) > cfg["max_size_ratio"]:
                reasons[i, j] |= 4
            noise_cov = noise.covariance(z, score, mean[:4])
            measurement_covariances[i, j] = noise_cov
            s = p4 + noise_cov
            try:
                chol = np.linalg.cholesky(s)
                whitened = np.linalg.solve(chol, r)
            except np.linalg.LinAlgError:
                reasons[i, j] |= 8
                continue
            maha[i, j] = float(whitened @ whitened)
            logdet[i, j] = float(2 * np.log(np.diag(chol)).sum())
            if maha[i, j] > cfg["gate_chi2"]:
                reasons[i, j] |= 1
            if reasons[i, j] == 0:
                costs[i, j] = 0.5 * (maha[i, j] + logdet[i, j] + constant)
    return costs, {"mahalanobis2": maha, "logdet": logdet, "gate_reason": reasons,
                   "measurement_covariance": measurement_covariances}


def _cost_vector(value, length, name):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim == 0:
        a = np.full(length, float(a))
    if a.shape != (length,) or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError(f"{name} must be nonnegative finite scalar or length {length}")
    return a


def assign_with_dummies(costs, miss_cost, unassigned_cost):
    """Minimize matched cost + explicit costs for every unmatched row/column.

    One private dummy per real row and column guarantees a feasible assignment
    even when every candidate is forbidden. No IoU threshold is applied to NLL.
    Equal-cost match-vs-reject ties choose rejection before solving.
    """
    costs = np.asarray(costs, dtype=np.float64)
    if costs.ndim != 2 or np.isnan(costs).any() or np.isneginf(costs).any():
        raise ValueError("Costs must be a 2-D matrix of finite values or +inf")
    n, m = costs.shape
    miss = _cost_vector(miss_cost, n, "miss_cost")
    unassigned = _cost_vector(unassigned_cost, m, "unassigned_cost")
    if not n or not m:
        return np.empty((0, 2), dtype=int), np.arange(n), np.arange(m)
    augmented = np.full((n + m, n + m), np.inf)
    allowed_costs = costs.copy()
    allowed_costs[allowed_costs >= miss[:, None] + unassigned[None, :]] = np.inf
    augmented[:n, :m] = allowed_costs
    augmented[np.arange(n), m + np.arange(n)] = miss
    augmented[n + np.arange(m), np.arange(m)] = unassigned
    augmented[n:, m:] = 0.0
    rows, cols = linear_sum_assignment(augmented)
    matches = np.array([(r, c) for r, c in zip(rows, cols) if r < n and c < m], dtype=int).reshape(-1, 2)
    unmatched_tracks = np.setdiff1d(np.arange(n), matches[:, 0], assume_unique=True)
    unmatched_detections = np.setdiff1d(np.arange(m), matches[:, 1], assume_unique=True)
    return matches, unmatched_tracks, unmatched_detections


DEFAULT_NWD = {"normalizer_px": 12.8, "first_threshold": 0.8, "second_threshold": 0.5}


def nwd_config(config=None):
    """NWD is a simple distance control, not a fitted probabilistic model."""
    config = dict(config or {})
    if set(config) - set(DEFAULT_NWD):
        raise ValueError(f"Unknown NWD parameters: {sorted(set(config) - set(DEFAULT_NWD))}")
    values = {**DEFAULT_NWD, **config}
    if not np.isfinite(values["normalizer_px"]) or values["normalizer_px"] <= 0:
        raise ValueError("NWD normalizer_px must be finite and positive")
    for key in ("first_threshold", "second_threshold"):
        if not np.isfinite(values[key]) or not 0 < values[key] < 1:
            raise ValueError(f"NWD {key} must lie strictly between zero and one")
    return values


def nwd_cost(predicted_xywh, measurements, *, normalizer_px=12.8, config=None):
    """Return 1 - exp(-W2/C) with boxes mapped to (cx,cy,w/2,h/2).

    W2 is Euclidean distance in this four-coordinate representation (the
    Gaussian box interpretation of Wang et al., arXiv:2110.13389). The physical
    center/size gates are the same as M2; there is no covariance or chi2 gate.
    The numeric C is a research starting value and must be selected on the
    calibration partition, not claimed as calibrated for this dataset.
    """
    nwd_config({"normalizer_px": normalizer_px})
    cfg = stage_config(config)
    predicted = np.asarray(predicted_xywh, dtype=np.float64).reshape(-1, 4)
    measured = np.asarray(measurements, dtype=np.float64).reshape(-1, 4)
    if not np.isfinite(measured).all() or np.any(measured[:, 2:] <= 0):
        raise ValueError("Detections must be finite with positive sizes")
    n, m = len(predicted), len(measured)
    cost = np.full((n, m), np.inf)
    reasons = np.zeros((n, m), dtype=np.uint8)
    for i, box in enumerate(predicted):
        if not np.isfinite(box).all() or np.any(box[2:] <= 0):
            reasons[i, :] = 8
            continue
        delta = measured - box
        reasons[i, np.linalg.norm(delta[:, :2], axis=1) > cfg["max_center_distance_px"]] |= 2
        ratios = measured[:, 2:] / box[2:]
        reasons[i, np.max(np.maximum(ratios, 1.0 / ratios), axis=1) > cfg["max_size_ratio"]] |= 4
        delta[:, 2:] *= 0.5
        distance = np.linalg.norm(delta, axis=1)
        values = -np.expm1(-distance / normalizer_px)
        valid = reasons[i] == 0
        cost[i, valid] = values[valid]
    return cost, reasons
