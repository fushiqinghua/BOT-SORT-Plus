"""Detection measurement covariance in native-image pixel coordinates.

No ground truth is read here. Calibration is a separate offline operation.
"""
from dataclasses import dataclass
from typing import Mapping

import numpy as np


def _four_nonnegative(value, name):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (4,) or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError(f"{name} must contain four finite nonnegative values")
    return tuple(float(x) for x in a)


@dataclass(frozen=True)
class NoiseModel:
    """Diagonal R; coefficient names denote standard deviations, not variances.

    In floor/calibrated modes, variance[k] = floor_std[k]**2 +
    (size_scale[k] * [w,h,w,h][k])**2 + score_scale[k]**2*(1-score)**2.
    Baseline mode reproduces the R used by v16.0.11 STrack.update: that method
    does not pass detection confidence to the Kalman filter.
    """
    mode: str = "floor"
    floor_std: tuple = (1.0, 1.0, 1.5, 1.5)
    size_scale: tuple = (0.05, 0.05, 0.05, 0.05)
    score_scale: tuple = (2.0, 2.0, 3.0, 3.0)
    min_variance: float = 1e-6

    def __post_init__(self):
        if self.mode not in {"baseline", "floor", "calibrated"}:
            raise ValueError(f"Unknown noise mode: {self.mode}")
        for name in ("floor_std", "size_scale", "score_scale"):
            object.__setattr__(self, name, _four_nonnegative(getattr(self, name), name))
        if not np.isfinite(self.min_variance) or self.min_variance <= 0:
            raise ValueError("min_variance must be finite and positive")

    @classmethod
    def from_dict(cls, config: Mapping):
        allowed = {"mode", "floor_std", "size_scale", "score_scale", "min_variance"}
        extra = set(config) - allowed
        if extra:
            raise ValueError(f"Unknown noise parameters: {sorted(extra)}")
        return cls(**dict(config))

    def to_dict(self):
        return {"mode": self.mode, "floor_std": list(self.floor_std),
                "size_scale": list(self.size_scale), "score_scale": list(self.score_scale),
                "min_variance": float(self.min_variance)}

    def covariance(self, measurement_xywh, score, predicted_xywh=None):
        z = np.asarray(measurement_xywh, dtype=np.float64)
        if z.shape != (4,) or not np.isfinite(z).all() or np.any(z[2:] <= 0):
            raise ValueError("measurement_xywh must be finite with positive width/height")
        if not np.isfinite(score) or not 0 <= float(score) <= 1:
            raise ValueError("detection score must be in [0, 1]")
        if self.mode == "baseline":
            if predicted_xywh is None:
                raise ValueError("baseline R requires predicted_xywh")
            p = np.asarray(predicted_xywh, dtype=np.float64)
            if p.shape != (4,) or not np.isfinite(p).all() or np.any(p[2:] <= 0):
                raise ValueError("predicted_xywh must be finite with positive width/height")
            # Do not add a floor to this ablation: it must be the original R.
            variance = (0.05 * p[[2, 3, 2, 3]]) ** 2
        else:
            dimensions = z[[2, 3, 2, 3]]
            variance = (np.square(self.floor_std)
                        + np.square(np.asarray(self.size_scale) * dimensions)
                        + np.square(self.score_scale) * (1.0 - float(score)) ** 2)
            variance = np.maximum(variance, self.min_variance)
        return np.diag(variance)


def correct_with_covariance(mean, covariance, measurement, measurement_covariance):
    """XYWH correction with explicit R and Joseph-form covariance update.

    This is deliberately independent of BoxMOT so its numerical properties can
    be tested without a GPU or model dependencies. Prediction/Q is unchanged.
    """
    mean = np.asarray(mean, dtype=np.float64)
    p = np.asarray(covariance, dtype=np.float64)
    z = np.asarray(measurement, dtype=np.float64)
    r = np.asarray(measurement_covariance, dtype=np.float64)
    if mean.shape != (8,) or p.shape != (8, 8) or z.shape != (4,) or r.shape != (4, 4):
        raise ValueError("Expected mean(8), covariance(8,8), measurement(4), R(4,4)")
    if not all(np.isfinite(a).all() for a in (mean, p, z, r)):
        raise ValueError("Kalman correction inputs must be finite")
    if not np.allclose(r, r.T, atol=1e-10):
        raise ValueError("R must be symmetric")
    np.linalg.cholesky(r)  # Positive definite R is part of the public contract.
    p = 0.5 * (p + p.T)
    s = p[:4, :4] + r
    chol = np.linalg.cholesky(s)
    gain = np.linalg.solve(chol.T, np.linalg.solve(chol, p[:, :4].T)).T
    updated_mean = mean + gain @ (z - mean[:4])
    ikh = np.eye(8)
    ikh[:, :4] -= gain
    updated_cov = ikh @ p @ ikh.T + gain @ r @ gain.T
    return updated_mean, 0.5 * (updated_cov + updated_cov.T)
