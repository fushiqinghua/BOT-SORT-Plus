"""Causal, staged M1 camera-motion experiments in native image coordinates.

This module deliberately does not replace or monkey-patch BoxMOT's SOF.  The
runner installs an instance only for a named M1 ablation.  Image correspondence
quality is a proxy for reliability, never camera-motion ground truth.

The sparse-flow lifecycle follows BoxMOT v16.0.11 (AGPL-3.0); its license is
retained in provenance/BOXMOT_LICENSE.  Estimation and reliability checks below
are separate experimental code.
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np

from .cmc import affine_residuals, spatial_coverage


SUPPORTED_CMC_MODES = (
    "m1_native_scale", "m1_mask_grid", "m1_fb", "m1_reliable",
)

DEFAULT_CONFIG = {
    "scale": 0.15,
    "ransac_native_px": 3.0,
    "max_corners": 1000,
    "grid": 4,
    "mask_margin_native_px": 3.0,
    "mask_score_threshold": 0.1,
    "fb_native_px": 2.0,
    "min_reliable_points": 8,
    "min_holdout_points": 4,
    "min_inlier_fraction": 0.5,
    "min_grid_coverage": 0.25,
    "max_spatial_condition": 100.0,
    "min_affine_scale": 0.8,
    "max_affine_scale": 1.25,
    "max_holdout_median_native_px": 3.0,
    "max_translation_disagreement_native_px": 3.0,
}


def _config(config):
    config = dict(config or {})
    extra = set(config) - set(DEFAULT_CONFIG)
    if extra:
        raise ValueError(f"Unknown M1 parameters: {sorted(extra)}")
    value = {**DEFAULT_CONFIG, **config}
    for key, number in value.items():
        if isinstance(number, bool) or not np.isfinite(number):
            raise ValueError(f"M1 {key} must be a finite number")
        if key in {"max_corners", "grid", "min_reliable_points", "min_holdout_points"}:
            if number != int(number) or number < 1:
                raise ValueError(f"M1 {key} must be a positive integer")
            value[key] = int(number)
        elif number < 0:
            raise ValueError(f"M1 {key} must be nonnegative")
    if not 0 < value["scale"] <= 1:
        raise ValueError("M1 scale must be in (0,1]")
    for key in ("ransac_native_px", "fb_native_px", "max_spatial_condition",
                "min_affine_scale", "max_holdout_median_native_px",
                "max_translation_disagreement_native_px"):
        if value[key] <= 0:
            raise ValueError(f"M1 {key} must be positive")
    if value["max_affine_scale"] < value["min_affine_scale"]:
        raise ValueError("M1 affine scale limits are reversed")
    if value["min_reliable_points"] < 4 or value["min_holdout_points"] < 2:
        raise ValueError("M1 needs >=4 reliable fit points and >=2 holdout points")
    for key in ("mask_score_threshold", "min_inlier_fraction", "min_grid_coverage"):
        if not 0 <= value[key] <= 1:
            raise ValueError(f"M1 {key} must lie in [0,1]")
    if value["max_corners"] < value["grid"] ** 2:
        raise ValueError("M1 max_corners must cover at least one corner per grid cell")
    return value


def scaled_to_native(points, native_shape, scaled_shape):
    """Invert OpenCV resize's pixel-center convention, using actual dimensions.

    Rounded width and height generally have slightly different scale factors.
    Fitting the partial affine in these native coordinates avoids treating an
    anisotropically resized image as an exactly isotropic camera coordinate.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    height, width = native_shape[:2]
    small_height, small_width = scaled_shape[:2]
    if min(height, width, small_height, small_width) <= 0:
        raise ValueError("Image dimensions must be positive")
    scales = np.array([small_width / width, small_height / height])
    return (points + 0.5) / scales - 0.5


def _spatial_condition(points):
    """Condition of centered point geometry; infinity for collinear support."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) < 3:
        return math.inf
    covariance = np.cov(points, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] <= 1e-9:
        return math.inf
    return float(eigenvalues[-1] / eigenvalues[0])


def _finite_or_none(value):
    return float(value) if value is not None and np.isfinite(value) else None


class RobustCMC:
    """Previous-to-current sparse-flow CMC with explicitly staged additions.

    ``m1_native_scale`` changes the fit coordinate system and RANSAC threshold.
    ``m1_mask_grid`` additionally masks foreground and balances feature cells.
    ``m1_fb`` additionally rejects inconsistent forward/backward flow.
    ``m1_reliable`` additionally validates a held-out correspondence subset and
    chooses partial affine, supported translation, or flagged identity.

    The callback receives one JSON-safe event on every ``apply`` call, including
    initialization, insufficient texture, and fallbacks. No GT is accepted.
    """

    def __init__(self, mode="m1_reliable", config=None, trace_callback=None):
        if mode not in SUPPORTED_CMC_MODES:
            raise ValueError(f"Unknown M1 CMC mode {mode!r}")
        import cv2
        self.cv2 = cv2
        self.mode = mode
        self.config = _config(config)
        self.use_mask_grid = mode != "m1_native_scale"
        self.use_fb = mode in ("m1_fb", "m1_reliable")
        self.use_reliability = mode == "m1_reliable"
        self.research_config = {
            "mode": mode, **self.config,
            "mask_grid": self.use_mask_grid, "forward_backward_filter": self.use_fb,
            "heldout_reliability": self.use_reliability,
            "coordinate_system": "native_pixels_resize_center_corrected",
            "estimator": "partial_affine_ransac",
            "translation_gate": "none",
            "quality_is_camera_ground_truth": False,
        }
        self.feature_params = dict(maxCorners=self.config["max_corners"],
                                   qualityLevel=0.01, minDistance=1, blockSize=3,
                                   useHarrisDetector=False, k=0.04)
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        self.lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=self.criteria)
        self.trace_callback = trace_callback
        self.previous = None
        self.points = None
        self.native_shape = None
        self.frame = 0
        self.last_report = None
        self.status_counts = Counter()
        self.model_counts = Counter()

    def summary(self):
        return {"frames": self.frame, "status_counts": dict(self.status_counts),
                "model_counts": dict(self.model_counts), "parameters": self.research_config}

    def _mask(self, shape, native_shape, dets):
        mask = np.full(shape, 255, dtype=np.uint8)
        if not self.use_mask_grid or dets is None:
            return mask
        dets = np.asarray(dets)
        if not dets.size:
            return mask
        if dets.ndim != 2 or dets.shape[1] < 4 or not np.isfinite(dets).all():
            raise ValueError("M1 requires finite Nx4+ detections")
        if np.any(dets[:, 2:4] <= dets[:, :2]):
            raise ValueError("M1 foreground boxes must have positive width/height")
        height, width = shape
        native_height, native_width = native_shape
        scales = np.array([width / native_width, height / native_height])
        margin = self.config["mask_margin_native_px"]
        for det in dets:
            if len(det) >= 5 and det[4] <= self.config["mask_score_threshold"]:
                continue
            lo = np.floor((det[:2] - margin) * scales).astype(int)
            hi = np.ceil((det[2:4] + margin) * scales).astype(int)
            x1, y1 = np.maximum(lo, 0)
            x2, y2 = np.minimum(hi, [width, height])
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 0
        return mask

    def _features(self, gray, mask):
        if not self.use_mask_grid:
            return self.cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        grid = self.config["grid"]
        height, width = gray.shape
        xs, ys = np.linspace(0, width, grid + 1, dtype=int), np.linspace(0, height, grid + 1, dtype=int)
        # Equal quotas cap a richly textured moving region's influence.  No
        # redistribution from empty cells: the spatial-support gate sees that.
        quota = self.config["max_corners"] // (grid * grid)
        found = []
        params = {**self.feature_params, "maxCorners": quota}
        for gy in range(grid):
            for gx in range(grid):
                cell = np.zeros_like(mask)
                cell[ys[gy]:ys[gy+1], xs[gx]:xs[gx+1]] = mask[ys[gy]:ys[gy+1], xs[gx]:xs[gx+1]]
                points = self.cv2.goodFeaturesToTrack(gray, mask=cell, **params)
                if points is not None:
                    found.append(points)
        return np.concatenate(found) if found else None

    def _remember(self, gray, mask, native_shape, fallback_points=None, refine=False):
        points = self._features(gray, mask)
        if (points is None or len(points) < 4) and fallback_points is not None:
            points = fallback_points.copy()
        if refine and points is not None and len(points) >= 4 and min(gray.shape) > 12:
            self.cv2.cornerSubPix(gray, points, (5, 5), (-1, -1), self.criteria)
            # Refinement can move a corner into foreground; enforce the mask
            # again so even the first pair obeys its detection exclusions.
            if self.use_mask_grid:
                points = points[self._inside_mask(points, mask)]
        self.previous, self.points, self.native_shape = gray.copy(), points, native_shape

    @staticmethod
    def _inside_mask(points, mask):
        points = np.asarray(points).reshape(-1, 2)
        h, w = mask.shape
        valid = np.isfinite(points).all(axis=1)
        valid &= (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
        rows = np.flatnonzero(valid)
        xy = np.floor(points[rows]).astype(int)
        valid[rows] &= mask[xy[:, 1], xy[:, 0]] != 0
        return valid

    def _event(self):
        return {"stage": "cmc", "frame": self.frame, "mode": self.mode,
                "status": None, "estimated": False, "model": "identity", "reasons": [],
                "keypoints": 0, "tracked": 0, "after_mask": 0, "after_fb": 0,
                "fb_median_native_px": None, "fb_p90_native_px": None,
                "fit_points": 0, "holdout_points": 0, "inliers": 0,
                "inlier_fraction": None, "inlier_grid_coverage": 0.0,
                "spatial_condition": None, "residual_median_native_px": None,
                "residual_p90_native_px": None, "holdout_median_native_px": None,
                "holdout_inlier_fraction": None, "affine_scale": None,
                "translation_native_px": None, "rotation_degrees": None,
                "translation_disagreement_native_px": None,
                "affine_rejection_reasons": [], "translation_rejection_reasons": [],
                "warp": np.eye(2, 3).tolist()}

    def _finish(self, report, warp=None):
        if warp is not None:
            warp = np.asarray(warp, dtype=np.float64)
            if warp.shape != (2, 3) or not np.isfinite(warp).all():
                raise RuntimeError("M1 produced an invalid camera transform")
            report["warp"] = warp.tolist()
            report["translation_native_px"] = float(np.linalg.norm(warp[:, 2]))
            report["rotation_degrees"] = float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0])))
        self.last_report = report
        self.status_counts[report["status"]] += 1
        self.model_counts[report["model"]] += 1
        if self.trace_callback is not None:
            self.trace_callback(report)
        return np.asarray(report["warp"], dtype=np.float64)

    def _quality(self, before, after, warp, mask, hold_before, hold_after, native_shape):
        residuals = affine_residuals(before, after, warp)
        inliers = np.asarray(mask, dtype=bool).reshape(-1)
        accepted = residuals[inliers]
        held = affine_residuals(hold_before, hold_after, warp)
        condition = _spatial_condition(before[inliers])
        return {"fit_points": len(before), "holdout_points": len(held), "inliers": int(inliers.sum()),
                "inlier_fraction": float(inliers.mean()) if len(inliers) else 0.0,
                "inlier_grid_coverage": spatial_coverage(before[inliers], native_shape, self.config["grid"]),
                "spatial_condition": _finite_or_none(condition),
                "residual_median_native_px": float(np.median(accepted)) if len(accepted) else None,
                "residual_p90_native_px": float(np.quantile(accepted, 0.9)) if len(accepted) else None,
                "holdout_median_native_px": float(np.median(held)) if len(held) else None,
                "holdout_inlier_fraction": float(np.mean(held <= self.config["ransac_native_px"])) if len(held) else None,
                "affine_scale": float(np.hypot(warp[0, 0], warp[1, 0]))}

    def _reliability_reasons(self, quality):
        reasons = []
        cfg = self.config
        if quality["inliers"] < cfg["min_reliable_points"]:
            reasons.append("insufficient_inlier_support")
        if quality["inlier_fraction"] < cfg["min_inlier_fraction"]:
            reasons.append("low_inlier_fraction")
        if quality["inlier_grid_coverage"] < cfg["min_grid_coverage"]:
            reasons.append("poor_spatial_coverage")
        if quality["spatial_condition"] is None or quality["spatial_condition"] > cfg["max_spatial_condition"]:
            reasons.append("ill_conditioned_spatial_support")
        if not cfg["min_affine_scale"] <= quality["affine_scale"] <= cfg["max_affine_scale"]:
            reasons.append("implausible_scale")
        if quality["holdout_points"] < cfg["min_holdout_points"]:
            reasons.append("insufficient_holdout_support")
        elif (quality["holdout_inlier_fraction"] < cfg["min_inlier_fraction"] or
              quality["holdout_median_native_px"] > cfg["max_holdout_median_native_px"]):
            reasons.append("heldout_residual_failed")
        return reasons

    def estimate_points(self, previous_native, current_native, native_shape):
        """Fit a frame pair from validated correspondences; useful for geometry QA.

        Returns ``(warp, report_fields)`` and never mutates image/history state.
        Only ``m1_reliable`` reserves every fifth correspondence for validation.
        Held-out points are not used in RANSAC, translation fit, or refitting.
        """
        before = np.asarray(previous_native, dtype=float).reshape(-1, 2)
        after = np.asarray(current_native, dtype=float).reshape(-1, 2)
        if before.shape != after.shape or not np.isfinite(before).all() or not np.isfinite(after).all():
            raise ValueError("M1 point pairs must be matching finite Nx2 arrays")
        identity = np.eye(2, 3)
        report = self._event()
        if len(before) < 4:
            report.update(status="insufficient_correspondences", reasons=["fewer_than_four_pairs"])
            return identity, report
        held = np.zeros(len(before), dtype=bool)
        if self.use_reliability:
            # Deterministic membership independent of image residual or GT.
            held[::5] = True
        fit_before, fit_after = before[~held], after[~held]
        hold_before, hold_after = before[held], after[held]
        report.update(fit_points=len(fit_before), holdout_points=len(hold_before))
        if len(fit_before) < 4:
            report.update(status="insufficient_correspondences", reasons=["insufficient_fit_pairs"])
            return identity, report
        warp, mask = self.cv2.estimateAffinePartial2D(
            fit_before, fit_after, method=self.cv2.RANSAC,
            ransacReprojThreshold=self.config["ransac_native_px"],
            maxIters=2000, confidence=0.99, refineIters=10)
        valid_affine = (warp is not None and mask is not None and np.isfinite(warp).all()
                        and np.linalg.det(warp[:, :2]) > 1e-12)
        if valid_affine:
            quality = self._quality(fit_before, fit_after, warp, mask, hold_before, hold_after, native_shape)
            report.update(quality)
            reasons = self._reliability_reasons(quality) if self.use_reliability else []
        else:
            reasons = ["affine_estimation_failed"]
        if not reasons:
            report.update(status="estimated_partial_affine", model="partial_affine", estimated=True)
            return warp, report
        report["affine_rejection_reasons"] = reasons
        if not self.use_reliability:
            report.update(status="affine_estimation_failed", reasons=reasons)
            return identity, report

        # A weaker, two-parameter model may be supported when rotation/scale is
        # unstable. It must independently pass the same held-out/spatial tests.
        shifts = fit_after - fit_before
        translation = np.median(shifts, axis=0)
        mask = np.linalg.norm(shifts - translation, axis=1) <= self.config["ransac_native_px"]
        if mask.any():
            translation = np.median(shifts[mask], axis=0)
            mask = np.linalg.norm(shifts - translation, axis=1) <= self.config["ransac_native_px"]
        translated = np.eye(2, 3)
        translated[:, 2] = translation
        quality = self._quality(fit_before, fit_after, translated, mask, hold_before, hold_after, native_shape)
        translation_reasons = self._reliability_reasons(quality)
        # Median shifts of supported grid cells should agree with each other.
        # This is a spatial consistency check, not a maximum camera translation.
        h, w = native_shape[:2]
        cells = np.floor(fit_before / [w, h] * self.config["grid"]).astype(int)
        medians = []
        for cell in np.unique(cells[mask], axis=0):
            belonging = mask & np.all(cells == cell, axis=1)
            if belonging.sum() >= 2:
                medians.append(np.median(shifts[belonging], axis=0))
        disagreement = max((float(np.linalg.norm(value - translation)) for value in medians), default=math.inf)
        report["translation_disagreement_native_px"] = _finite_or_none(disagreement)
        if disagreement > self.config["max_translation_disagreement_native_px"]:
            translation_reasons.append("spatial_translation_disagreement")
        if translation_reasons:
            report.update(status="unreliable_identity_fallback", reasons=reasons + translation_reasons,
                          translation_rejection_reasons=translation_reasons)
            return identity, report
        report.update(quality)
        report.update(status="estimated_translation_fallback", model="translation", estimated=True,
                      reasons=reasons)
        return translated, report

    def apply(self, img, dets=None):
        if (img is None or not isinstance(img, np.ndarray) or img.ndim != 3 or
                img.shape[2] != 3 or img.dtype != np.uint8 or min(img.shape[:2]) < 2):
            raise ValueError("M1 expects a nonempty uint8 BGR image")
        self.frame += 1
        cv2 = self.cv2
        native_shape = tuple(img.shape[:2])
        h, w = native_shape
        small_shape = (max(2, round(w * self.config["scale"])), max(2, round(h * self.config["scale"])))
        gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), small_shape, interpolation=cv2.INTER_LINEAR)
        current_mask = self._mask(gray.shape, native_shape, dets)
        report = self._event()
        if min(gray.shape) < 12:
            self.previous, self.points, self.native_shape = gray.copy(), None, native_shape
            report.update(status="image_too_small_for_flow", reasons=["scaled_dimension_below_twelve"])
            return self._finish(report)
        if self.native_shape is not None and (self.native_shape != native_shape or self.previous.shape != gray.shape):
            self._remember(gray, current_mask, native_shape, refine=True)
            report.update(status="reset_image_shape_changed", reasons=["previous_and_current_shapes_differ"])
            return self._finish(report)
        if self.previous is None or self.points is None or len(self.points) < 4:
            self._remember(gray, current_mask, native_shape, refine=True)
            report["keypoints"] = 0 if self.points is None else len(self.points)
            report["status"] = "initialized_no_pair_estimate" if report["keypoints"] >= 4 else "insufficient_initial_features"
            return self._finish(report)
        report["keypoints"] = len(self.points)
        following, status, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, self.points, None, **self.lk_params)
        if following is None or status is None:
            self._remember(gray, current_mask, native_shape)
            report.update(status="forward_flow_failed", reasons=["lk_no_correspondences"])
            return self._finish(report)
        before = self.points.reshape(-1, 2)
        after = following.reshape(-1, 2)
        valid = status.reshape(-1).astype(bool) & np.isfinite(before).all(axis=1) & np.isfinite(after).all(axis=1)
        report["tracked"] = int(valid.sum())
        # Both image bounds are mandatory. Foreground masking is staged.
        valid &= self._inside_mask(before, np.full(self.previous.shape, 255, dtype=np.uint8))
        valid &= self._inside_mask(after, current_mask if self.use_mask_grid else np.full_like(current_mask, 255))
        before, after = before[valid], after[valid]
        report["after_mask"] = len(before)
        if self.use_fb and len(before):
            backwards, back_status, _ = cv2.calcOpticalFlowPyrLK(
                gray, self.previous, after.reshape(-1, 1, 2), None, **self.lk_params)
            if backwards is None or back_status is None:
                before, after = before[:0], after[:0]
                report["reasons"].append("backward_flow_failed")
            else:
                back = backwards.reshape(-1, 2)
                good = back_status.reshape(-1).astype(bool) & np.isfinite(back).all(axis=1)
                fb = np.linalg.norm(scaled_to_native(back, native_shape, gray.shape) -
                                    scaled_to_native(before, native_shape, gray.shape), axis=1)
                if good.any():
                    report["fb_median_native_px"] = float(np.median(fb[good]))
                    report["fb_p90_native_px"] = float(np.quantile(fb[good], 0.9))
                good &= fb <= self.config["fb_native_px"]
                before, after = before[good], after[good]
        report["after_fb"] = len(before)
        native_before = scaled_to_native(before, native_shape, gray.shape)
        native_after = scaled_to_native(after, native_shape, gray.shape)
        warp, estimate = self.estimate_points(native_before, native_after, native_shape)
        for key in ("frame", "stage", "mode", "keypoints", "tracked", "after_mask", "after_fb",
                    "fb_median_native_px", "fb_p90_native_px"):
            estimate.pop(key, None)
        estimate["reasons"] = report["reasons"] + estimate["reasons"]
        report.update(estimate)
        fallback = after.reshape(-1, 1, 2).astype(np.float32) if len(after) >= 4 else None
        self._remember(gray, current_mask, native_shape, fallback_points=fallback)
        return self._finish(report, warp if report["estimated"] else None)


def build_cmc(mode, config=None, trace_callback=None):
    return RobustCMC(mode=mode, config=config, trace_callback=trace_callback)
