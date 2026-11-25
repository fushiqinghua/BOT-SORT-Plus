"""Offline M1 sparse-flow diagnostics; no GT and no tracker state mutation.

The probe follows BoxMOT v16.0.11 SOF's feature/LK/RANSAC settings. It emits
quality proxies that native SOF does not expose. It is NOT camera-motion truth,
and should not replace the native tracker for the SOF/on vs SOF/off experiment.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def spatial_coverage(points, shape, grid=4):
    """Fraction of occupied grid cells; images without points have zero coverage."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    height, width = shape[:2]
    if grid < 1 or height <= 0 or width <= 0:
        raise ValueError("Invalid image shape/grid")
    if len(points) == 0:
        return 0.
    valid = np.isfinite(points).all(axis=1) & (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)
    cells = np.floor(points[valid] / [width, height] * grid).astype(int)
    return len(np.unique(cells, axis=0)) / float(grid * grid)


def affine_residuals(previous, current, warp):
    previous, current = (np.asarray(p, dtype=float).reshape(-1, 2) for p in (previous, current))
    warp = np.asarray(warp, dtype=float)
    if previous.shape != current.shape or warp.shape != (2, 3):
        raise ValueError("Points/warp shape mismatch")
    return np.linalg.norm(previous @ warp[:, :2].T + warp[:, 2] - current, axis=1)


class SOFProbe:
    """Sequential SOF-style replay with explicit estimate failure status.

    ``warp`` maps previous image coordinates to the current image. All reported
    residual/translation distances are native pixels; RANSAC's 3-pixel gate is
    applied in the downscaled image (20 native pixels at scale=.15).
    """

    def __init__(self, scale=.15):
        if not 0 < scale <= 1:
            raise ValueError("Diagnostic scale must be in (0,1]")
        try:
            import cv2
        except ImportError as error:
            raise ImportError("M1 motion diagnostics require OpenCV; install opencv-python-headless in the experiment environment.") from error
        self.cv2, self.scale = cv2, float(scale)
        self.feature_params = dict(maxCorners=1000, qualityLevel=.01, minDistance=1,
                                   blockSize=3, useHarrisDetector=False, k=.04)
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01)
        self.lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=self.criteria)
        self.previous, self.points, self.initialized = None, None, False

    def _features(self, gray):
        return self.cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)

    def _reset(self, gray):
        self.previous = gray.copy()
        self.points = self._features(gray)
        self.initialized = self.points is not None and len(self.points) >= 4

    def step(self, image):
        cv2 = self.cv2
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a nonempty BGR image")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (0, 0), fx=self.scale, fy=self.scale,
                          interpolation=cv2.INTER_LINEAR)
        identity = np.eye(2, 3, dtype=np.float64)
        report = dict(status=None, estimated=False, keypoints=0, tracked=0, inliers=0,
                      inlier_fraction=None, inlier_grid_coverage=0.,
                      residual_median_native_px=None, residual_p90_native_px=None,
                      fb_median_native_px=None, fb_p90_native_px=None,
                      translation_native_px=None, rotation_degrees=None,
                      affine_scale=None, warp=identity.tolist())
        if self.previous is not None and self.previous.shape != gray.shape:
            self._reset(gray)
            report["status"] = "reset_image_shape_changed"
            return report
        if not self.initialized or self.previous is None or self.points is None:
            points = self._features(gray)
            report["keypoints"] = 0 if points is None else len(points)
            if points is not None and len(points) >= 4:
                cv2.cornerSubPix(gray, points, (5, 5), (-1, -1), self.criteria)
                self.initialized = True
                report["status"] = "initialized_no_pair_estimate"
            else:
                self.initialized = False
                report["status"] = "insufficient_initial_features"
            self.previous, self.points = gray.copy(), points
            return report
        report["keypoints"] = len(self.points)
        following, status, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, self.points, None, **self.lk_params)
        if following is None or status is None:
            self._reset(gray)
            report["status"] = "forward_flow_failed"
            return report
        valid = status.reshape(-1).astype(bool)
        before, after = self.points[valid], following[valid]
        report["tracked"] = len(before)
        if len(before) < 4:
            self._reset(gray)
            report["status"] = "insufficient_tracked_features"
            return report
        # Explicit defaults are OpenCV's native estimateAffinePartial2D defaults.
        warp, inlier_mask = cv2.estimateAffinePartial2D(before, after, method=cv2.RANSAC,
            ransacReprojThreshold=3., maxIters=2000, confidence=.99, refineIters=10)
        if warp is None or not np.isfinite(warp).all() or inlier_mask is None:
            report["status"] = "affine_estimation_failed"
        else:
            accepted = inlier_mask.reshape(-1).astype(bool)
            residual = affine_residuals(before, after, warp) / self.scale
            report["inliers"] = int(accepted.sum())
            report["inlier_fraction"] = float(accepted.mean())
            report["inlier_grid_coverage"] = spatial_coverage(before[accepted], gray.shape)
            if accepted.any():
                report["residual_median_native_px"] = float(np.median(residual[accepted]))
                report["residual_p90_native_px"] = float(np.quantile(residual[accepted], .9))
            # This backward pass is observational: it never removes points or
            # changes the transform estimated above.
            backwards, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, after, None, **self.lk_params)
            if backwards is not None and back_status is not None:
                consistent = back_status.reshape(-1).astype(bool)
                fb = np.linalg.norm(backwards.reshape(-1, 2) - before.reshape(-1, 2), axis=1)[consistent] / self.scale
                fb = fb[np.isfinite(fb)]
                if len(fb):
                    report["fb_median_native_px"] = float(np.median(fb))
                    report["fb_p90_native_px"] = float(np.quantile(fb, .9))
            native = np.asarray(warp, dtype=float).copy()
            native[:, 2] /= self.scale
            report.update(status="estimated", estimated=True, warp=native.tolist(),
                          translation_native_px=float(np.linalg.norm(native[:, 2])),
                          rotation_degrees=float(np.degrees(np.arctan2(native[1, 0], native[0, 0]))),
                          affine_scale=float(np.hypot(native[0, 0], native[1, 0])))
        refreshed = self._features(gray)
        self.points = after.copy() if refreshed is None or len(refreshed) < 4 else refreshed.copy()
        self.previous, self.initialized = gray.copy(), True
        return report


def diagnose_motion(frames, image_resolver, output_dir, max_pairs=None, scale=.15,
                    sequence="sequence"):
    """Write per-frame-pair motion quality, loading only images and frame IDs.

    ``image_resolver(frame_dict)`` returns a local image path. Frame input must
    include every intermediate frame; ``max_pairs`` takes the first N pairs,
    not a stride that would change the optical-flow displacement distribution.
    """
    if max_pairs is not None and max_pairs < 1:
        raise ValueError("max_pairs must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    probe, statuses, estimates, rows = SOFProbe(scale), Counter(), [], 0
    previous_frame = None
    columns = ["sequence", "frame", "previous_frame", "status", "estimated", "keypoints",
        "tracked", "inliers", "inlier_fraction", "inlier_grid_coverage",
        "residual_median_native_px", "residual_p90_native_px", "fb_median_native_px",
        "fb_p90_native_px", "translation_native_px", "rotation_degrees", "affine_scale",
        "a00", "a01", "tx", "a10", "a11", "ty"]
    with (output / "pairs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in frames:
            if max_pairs is not None and rows >= max_pairs + 1:
                break
            frame = int(record["frame"])
            if previous_frame is not None and frame != previous_frame + 1:
                raise ValueError("Motion diagnostics require consecutive frames")
            path = Path(image_resolver(record))
            image = probe.cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(f"Cannot decode image for {sequence}, frame {frame}: {path}")
            result = probe.step(image)
            statuses[result["status"]] += 1
            row = {"sequence": sequence, "frame": frame, "previous_frame": previous_frame, **result}
            warp = row.pop("warp")
            row.update(dict(zip(("a00", "a01", "tx", "a10", "a11", "ty"), np.asarray(warp).reshape(-1))))
            writer.writerow(row)
            if result["estimated"]:
                estimates.append(result)
            previous_frame, rows = frame, rows + 1
    if rows == 0:
        raise ValueError("No frames supplied")
    distributions = {}
    for name in ("inlier_fraction", "inlier_grid_coverage", "residual_p90_native_px",
                 "fb_p90_native_px", "translation_native_px", "rotation_degrees", "affine_scale"):
        values = [record[name] for record in estimates if record[name] is not None]
        distributions[name] = {"samples": len(values), "median": float(np.median(values)) if values else None,
                               "p90": float(np.quantile(values, .9)) if values else None}
    summary = {
        "status": "COMPLETE", "sequence": sequence, "frames": rows,
        "purpose": "OFFLINE_SOF_QUALITY_PROXIES_NOT_CAMERA_GROUND_TRUTH",
        "opencv_version": probe.cv2.__version__,
        "parameters": {"scale": scale, "RANSAC_threshold_scaled_pixels": 3.,
                       "RANSAC_threshold_native_pixels": 3. / scale,
                       "feature": probe.feature_params, "lk": {"window": [21, 21], "max_level": 3},
                       "mask": "none", "forward_backward_filter": False},
        "status_counts": dict(statuses), "estimated_pairs": len(estimates),
        "quality_distributions": distributions,
        "interpretation": [
            "The first frame initializes state and has no pair estimate.",
            "Identity warps accompanying a failure are flagged estimated=false; they are not valid zero-motion measurements.",
            "A high inlier ratio or low residual can describe moving water; neither proves correct camera motion.",
            "SOF on/off tracking with identical detections and TrackEval is the causal performance experiment.",
            "This independent diagnostic replay follows native SOF settings; use the original BoxMOT SOF in tracking runs.",
            "No GT, IMU, camera intrinsics, segmentation, or cross-modal registration is assumed.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                   allow_nan=False) + "\n", encoding="utf-8")
    return summary
