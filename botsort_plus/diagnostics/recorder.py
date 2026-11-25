"""Read-only, GT-free telemetry for the pinned v2 BoT-SORT + B replay.

The observer copies native state at explicit boundaries. It never substitutes a
motion model or an association cost. GMC diagnostics rerun deterministic LK on
copies, but never fit another transform or consume OpenCV's RANSAC RNG. Native
RANSAC's mask is not exposed by SOF: every reported inlier statistic is a proxy.
"""
from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np


SCHEMA = "botsort_motion_diagnostics_v1"
SOF_SOURCE_HASHES = {
    "motion/cmc/sof.py": "98c146fd9bd7b3a35de2743b000bb87c8cbdbec3e57740c78c538b4a0554aa37",
    "motion/cmc/base_cmc.py": "a6e29fd8dd132940e2f806bf4488efce58b783ced8bf26cf77d25ae0beb6a4bb",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_cmc_sources():
    """Validate extra sources used by read-only SOF instrumentation at preflight."""
    import boxmot
    native_root = Path(boxmot.__file__).resolve().parent
    hashes = {relative: file_sha256(native_root / relative) for relative in SOF_SOURCE_HASHES}
    if hashes != SOF_SOURCE_HASHES:
        raise RuntimeError("Diagnostics require pinned BoxMOT 16.0.11 SOF and preprocessing sources")
    return hashes


def _detached(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _detached(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_detached(item) for item in value]
    return deepcopy(value)


def _snapshots(tracks, kalman_filter=None):
    """Copy every native state, including Lost/Removed entries in native pools."""
    result, seen = [], set()
    for track in tracks:
        tid = int(track.id)
        if tid in seen:
            continue
        seen.add(tid)
        mean = np.asarray(track.mean).copy()
        covariance = np.asarray(track.covariance).copy()
        if mean.shape != (8,) or covariance.shape != (8, 8):
            raise ValueError("Diagnostics require native XYWH 8D Kalman state")
        item = {
            "track_id": tid, "state": int(track.state),
            "is_activated": bool(track.is_activated),
            "last_seen": int(track.end_frame),
            "last_observation_det_index": int(track.det_ind),
            "det_index": int(track.det_ind),
            "mean": mean.tolist(), "covariance": covariance.tolist(),
            "xyxy": np.asarray(track.xyxy).copy().tolist(),
        }
        if kalman_filter is not None:
            # Passing copies protects the live state even if a future project()
            # implementation changes from its current non-mutating behavior.
            projected_mean, projected_covariance = kalman_filter.project(mean, covariance)
            item["projected_mean"] = np.asarray(projected_mean).copy().tolist()
            item["projected_covariance"] = np.asarray(projected_covariance).copy().tolist()
        result.append(item)
    return result


def validate_capture_config(config, native, *, position_policy=None, position_rgb_first=False):
    """Reject unsupported paths instead of silently collecting partial evidence."""
    if (config.adapter_mode != "v2" or not config.recovery.enabled or
            config.birth.enabled or config.appearance.enabled or config.identity.enabled or
            config.raw_ir.enabled or config.recovery.output_prediction_frames or
            position_policy is not None or position_rgb_first):
        raise ValueError("--diagnostics requires v2 + B, with A/C/C2/raw-IR/motion-output/position refinement disabled")
    if native.get("cmc_method", "sof") != "sof":
        raise ValueError("--diagnostics supports the pinned native SOF camera motion only")


def _cmc_before(cmc):
    from boxmot.motion.cmc.sof import SOF
    if type(cmc) is not SOF:
        raise ValueError("Diagnostics require the exact native SOF object")
    if not 0 < float(cmc.scale) <= 1 or not cmc.grayscale:
        raise ValueError("Diagnostics require grayscale SOF with resize scale in (0,1]")
    return {
        "initialized": bool(cmc.initialized), "scale": float(cmc.scale),
        "prev_frame": None if cmc.prev_frame is None else cmc.prev_frame.copy(),
        "prev_keypoints": None if cmc.prev_keypoints is None else cmc.prev_keypoints.copy(),
        "lk_params": deepcopy(cmc.lk_params),
    }


def _gmc_quality(before, cmc, warp, image_shape):
    """Describe the actual returned warp using independently recomputed LK.

    Residuals use original image pixels. The inlier proxy threshold is 3 pixels
    in SOF's resized image (the native OpenCV default); it is NOT its RANSAC
    membership mask. Coverage is the source-point bounding-box area divided by
    original image area. Identity warps are never equated with estimation failure.
    """
    import cv2
    warp = np.asarray(warp, dtype=float).copy()
    scale = before["scale"]
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid SOF resize scale")
    points = before["prev_keypoints"]
    result = {
        "warp": warp.tolist(), "scale": scale,
        "status": "unobserved", "fallback_reason": None,
        "feature_count": 0 if points is None else int(len(points)),
        "tracked_count": None, "inlier_count_proxy": None,
        "inlier_ratio_proxy": None, "inlier_threshold_px": 3. / scale,
        "reprojection_median_px": None, "reprojection_p90_px": None,
        "spatial_coverage": None,
        "translation_px": warp[:, 2].tolist(),
        "rotation_deg": math.degrees(math.atan2(warp[1, 0], warp[0, 0])),
        "scale_factor": float(np.linalg.norm(warp[:, 0])),
        "native_inlier_mask_observed": False,
        "quality_method": "deterministic_LK_replay_against_native_warp",
    }
    if not before["initialized"] or before["prev_frame"] is None or points is None:
        result["status"] = "initialized" if cmc.initialized else "initialization_insufficient_features"
        if not cmc.initialized:
            result["fallback_reason"] = "initialization_insufficient_features"
        return result
    # SOF stores the current preprocessed image after each native apply().
    # Copy every input; never touch native keypoints/frames or run RANSAC again.
    current = cmc.prev_frame.copy()
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        before["prev_frame"].copy(), current, points.copy(), None, **before["lk_params"])
    if next_points is None or status is None:
        result.update(status="flow_failed", fallback_reason="flow_failed", tracked_count=0)
        return result
    valid = status.reshape(-1) == 1
    source = points.reshape(-1, 2)[valid].astype(float) / scale
    target = next_points.reshape(-1, 2)[valid].astype(float) / scale
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite deterministic SOF diagnostic correspondence")
    result["tracked_count"] = int(len(source))
    if len(source):
        predicted = source @ warp[:, :2].T + warp[:, 2]
        residual = np.linalg.norm(predicted - target, axis=1)
        inlier = residual <= result["inlier_threshold_px"]
        extent = np.ptp(source, axis=0)
        result.update(
            inlier_count_proxy=int(inlier.sum()), inlier_ratio_proxy=float(inlier.mean()),
            reprojection_median_px=float(np.median(residual)),
            reprojection_p90_px=float(np.percentile(residual, 90)),
            spatial_coverage=float(np.prod(extent) / (image_shape[0] * image_shape[1])),
        )
    if len(source) < 4:
        result.update(status="insufficient_tracks", fallback_reason="insufficient_tracks")
    elif np.array_equal(warp, np.eye(2, 3)):
        # SOF returns identity both for a valid no-motion fit and for an
        # estimateAffinePartial2D failure. Its API does not distinguish these.
        result["status"] = "identity_ambiguous"
    else:
        result["status"] = "estimated"
    return result


class DiagnosticRecorder:
    """One compressed JSONL record per frame; COMPLETE only after validation."""

    def __init__(self, path, *, sequence, expected_frames, metadata=None):
        self.path = Path(path)
        if not self.path.name.endswith(".jsonl.gz"):
            raise ValueError("Diagnostic path must end with .jsonl.gz")
        self.sequence = str(sequence)
        self.expected_frames = int(expected_frames)
        if self.expected_frames < 1:
            raise ValueError("Expected frame count must be positive")
        self.manifest_path = self.path.with_name(self.path.name[:-9] + ".json")
        self.partial_path = self.path.with_name(self.path.name[:-3] + ".partial.gz")
        if any(path.exists() for path in (self.path, self.partial_path, self.manifest_path)):
            raise FileExistsError(f"Diagnostic sequence already exists: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._raw = self.partial_path.open("xb")
        self._stream = gzip.GzipFile(filename="", fileobj=self._raw, mode="wb", mtime=0)
        self.records = 0
        self._frame = None
        self._frame_tracks = []
        self._closed = False
        self.manifest = {
            "schema": SCHEMA, "status": "RUNNING", "sequence": self.sequence,
            "expected_frames": self.expected_frames, "records": 0,
            "data_sha256": None, "trace_filename": self.path.name,
            "metadata": _detached(metadata or {}),
            "field_semantics": {
                "image_sha256": "SHA256 of contiguous decoded BGR image pixel bytes; image_shape and image_dtype recorded",
                "projected_covariance": "native H P H^T + R, after GMC and before association/update",
                "last_observation_det_index": "native det_ind in last observed frame, not necessarily current frame",
                "after_update": "unique active/lost/current-frame input tracks plus current removals, not historical removed pool",
                "gmc_inliers": "proxy threshold on deterministic LK replay; native RANSAC mask is not observed",
                "gmc_residuals": "all LK-success correspondences against returned native warp, original pixels",
                "gmc_coverage": "bounding-box area of successful source LK points / original image area",
                "gmc_identity": "identity can be valid no motion or a hidden estimation failure; it is not counted as proven failure",
            },
        }
        self._write_manifest()

    def _write_manifest(self):
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.manifest, ensure_ascii=False, allow_nan=False,
                                        indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.manifest_path)

    def begin(self, tracker, frame, detections, pool, unconfirmed, image):
        if self._closed or self._frame is not None or int(frame) != self.records + 1:
            raise ValueError("Diagnostic frames must be complete, contiguous, and start at 1")
        if self.records == 0:
            self.manifest["diagnostic_native_sha256"] = verify_cmc_sources()
        all_dets = np.asarray(detections)
        self._frame_tracks = list(pool) + list(unconfirmed)
        self._cmc = _cmc_before(tracker.cmc)
        self._image_shape = tuple(image.shape)
        self._frame = {
            "schema": SCHEMA, "sequence": self.sequence, "frame": int(frame),
            "image_sha256": hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
            "image_shape": list(image.shape), "image_dtype": str(image.dtype),
            "detections": [], "before_prediction": _snapshots(pool),
            "unconfirmed_before_gmc": _snapshots(unconfirmed),
        }
        for index, row in enumerate(all_dets):
            feature = tracker._packets[index].get("rgb")
            self._frame["detections"].append({
                "det_index": index, "xyxy": row[:4].copy().tolist(),
                "score": float(row[4]), "rgb_valid": feature is not None,
                "rgb_quality": None if feature is None else float(feature.quality),
            })

    def predicted(self, pool):
        if self._frame is None or "after_prediction" in self._frame:
            raise ValueError("Diagnostic prediction boundary out of order")
        self._frame["after_prediction"] = _snapshots(pool)

    def compensated(self, tracker, pool, unconfirmed, warp):
        if self._frame is None or "after_prediction" not in self._frame or "after_gmc" in self._frame:
            raise ValueError("Diagnostic GMC boundary out of order")
        self._frame["after_gmc"] = _snapshots(pool, tracker.kalman_filter)
        self._frame["unconfirmed"] = _snapshots(unconfirmed, tracker.kalman_filter)
        self._frame["gmc"] = _gmc_quality(self._cmc, tracker.cmc, warp, self._image_shape)
        self._cmc = None

    def finish(self, tracker, output, removed):
        if self._frame is None or "after_gmc" not in self._frame:
            raise ValueError("Diagnostic frame is missing native prediction/GMC boundaries")
        self._frame["after_update"] = _snapshots(
            list(tracker.active_tracks) + list(tracker.lost_stracks) + list(removed) + self._frame_tracks)
        self._frame["output"] = np.asarray(output).copy().reshape(-1, 8).tolist()
        self._frame["events"] = _detached(tracker.last_events)
        encoded = json.dumps(self._frame, ensure_ascii=False, allow_nan=False,
                             separators=(",", ":")).encode("utf-8") + b"\n"
        self._stream.write(encoded)
        self.records += 1
        self._frame = None
        self._frame_tracks = []

    def _close_stream(self):
        if not self._closed:
            self._stream.close()
            self._raw.close()
            self._closed = True

    def close(self):
        if self._closed:
            raise ValueError("Diagnostic recorder is already closed")
        if self._frame is not None or self.records != self.expected_frames:
            self.abort("Incomplete diagnostic frame count or boundary")
            raise ValueError("Incomplete diagnostic frame count or boundary")
        self._close_stream()
        self.partial_path.replace(self.path)
        self.manifest.update(status="COMPLETE", records=self.records,
                             data_sha256=file_sha256(self.path))
        self._write_manifest()
        return _detached(self.manifest)

    def abort(self, error):
        self._close_stream()
        self.manifest.update(status="FAILED", records=self.records, error=str(error),
                             partial_trace_filename=self.partial_path.name)
        self._write_manifest()
        return _detached(self.manifest)
