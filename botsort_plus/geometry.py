"""Scale-aware geometry. Covariance changes search eligibility, not similarity.

All distances are measured in the original detector/RGB coordinate system.
These defaults are initial hypotheses, not parameters optimized on VT-Tiny-MOT.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class GeometryConfig:
    pixel_floor: float = 4.0
    center_gate_scale: float = 2.0
    max_center_pixels: float = 48.0
    max_scale_ratio: float = 2.5
    uncertainty_sigma: float = 2.0
    tiny_size: float = 16.0
    size_weight: float = 0.10

    def __post_init__(self):
        values = vars(self)
        if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in values.values()):
            raise ValueError("Geometry parameters must be finite numbers")
        if min(self.pixel_floor, self.center_gate_scale, self.max_center_pixels, self.tiny_size) <= 0:
            raise ValueError("Geometry scales must be positive")
        if self.max_scale_ratio <= 1 or self.uncertainty_sigma < 0 or not 0 <= self.size_weight < 1:
            raise ValueError("Invalid scale ratio, uncertainty or weight")


def box_iou(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    inter = np.maximum(0.0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    area_a = np.maximum(0, a[2:] - a[:2]).prod()
    area_b = np.maximum(0, b[2:] - b[:2]).prod()
    return float(inter / max(area_a + area_b - inter, 1e-12))


def warp_box(box, warp):
    box, warp = np.asarray(box, float), np.asarray(warp, float)
    if warp.shape != (2, 3) or not np.isfinite(warp).all():
        raise ValueError("GMC transform must be a finite 2x3 affine matrix")
    x1, y1, x2, y2 = box
    corners = np.array([[x1,y1,1], [x1,y2,1], [x2,y1,1], [x2,y2,1]]) @ warp.T
    return np.r_[corners.min(axis=0), corners.max(axis=0)]


def geometry_cost(pred_box, obs_box, config=None, center_cov=None):
    """Return a nonnegative cost, or inf when a bounded geometric gate fails.

    Tiny boxes smoothly emphasize center distance over IoU. Pixel floors avoid
    exploding costs for a 1-pixel displacement. An uncertain KF may search a
    wider region, but its center residual is STILL divided by the original,
    fixed scale. Thus increasing covariance never improves a finite pair cost.
    """
    c = config or GeometryConfig()
    a, b = np.asarray(pred_box, float), np.asarray(obs_box, float)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite(np.r_[a,b]).all():
        raise ValueError("Geometry requires finite xyxy boxes")
    wa, wb = a[2:] - a[:2], b[2:] - b[:2]
    if min(*wa, *wb) <= 0:
        return float("inf")
    ratios = np.maximum(wa / wb, wb / wa)
    if ratios.max() > c.max_scale_ratio:
        return float("inf")
    size = float(min(np.sqrt(wa.prod()), np.sqrt(wb.prod())))
    base_radius = max(c.pixel_floor, c.center_gate_scale * size)
    extension = 0.0
    if center_cov is not None:
        cov = np.asarray(center_cov, float)
        if cov.shape != (2, 2) or not np.isfinite(cov).all():
            raise ValueError("Center covariance must be finite 2x2")
        eig = np.linalg.eigvalsh((cov + cov.T) / 2)
        if eig.min() < -1e-6:
            raise ValueError("Center covariance must be positive semidefinite")
        extension = c.uncertainty_sigma * np.sqrt(max(0.0, eig.max()))
    delta = float(np.linalg.norm((a[:2] + a[2:] - b[:2] - b[2:]) / 2))
    if delta > min(c.max_center_pixels, base_radius + extension):
        return float("inf")
    large_fraction = min(1.0, size / c.tiny_size)
    iou_weight = (0.20 + 0.40 * large_fraction) * (1 - c.size_weight)
    center_weight = 1 - c.size_weight - iou_weight
    size_cost = float(np.log(ratios).mean() / np.log(c.max_scale_ratio))
    return (iou_weight * (1 - box_iou(a, b))
            + center_weight * delta / base_radius + c.size_weight * size_cost)
