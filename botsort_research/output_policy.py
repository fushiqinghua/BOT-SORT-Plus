"""Causal output coordinates, deliberately separate from Kalman state updates."""
import numpy as np


DETECTION_OUTPUT_VARIANTS = frozenset({
    "native_iou_det", "m2_nll_floor_det", "m2_nll_floor_native_update_det",
})


def output_policy_for_variant(variant):
    return "detection" if variant in DETECTION_OUTPUT_VARIANTS else "kalman"


def apply_output_policy(predicted, detections, policy="kalman"):
    """Copy N x 8 BoxMOT rows; optionally replace xyxy by their matched box.

    The current input detection index is the only link. IDs, scores, classes,
    selected detections, lifecycle and future internal states are unchanged.
    Empty frames cannot receive extrapolated or backfilled rows. No nearest-box
    guessing is allowed: a missing/duplicate/nonintegral index is an error.
    """
    if policy not in ("kalman", "detection"):
        raise ValueError("Output policy must be kalman or detection")
    rows = np.asarray(predicted)
    dets = np.asarray(detections)
    if rows.size == 0:
        return np.empty((0, 8), dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 8 or not np.isfinite(rows).all():
        raise ValueError("Expected finite N x 8 tracker outputs")
    if dets.ndim != 2 or dets.shape[1] != 6 or not np.isfinite(dets).all():
        raise ValueError("Expected finite N x 6 current detections")
    if np.any(rows[:, 2:4] <= rows[:, :2]) or np.any(dets[:, 2:4] <= dets[:, :2]):
        raise ValueError("Tracker/detection boxes must have positive sizes")
    identities = rows[:, 4]
    indices = rows[:, 7]
    if np.any(identities <= 0) or np.any(identities != np.floor(identities)) or len(np.unique(identities)) != len(rows):
        raise ValueError("Output track identities must be positive, integral and unique")
    if np.any(indices != np.floor(indices)) or np.any(indices < 0) or np.any(indices >= len(dets)):
        raise ValueError("Output detection indices must identify current-frame detections")
    indices = indices.astype(np.int64)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("One current detection cannot supply two output tracks")
    selected = dets[indices]
    if not np.allclose(rows[:, 5], selected[:, 4], rtol=0, atol=1e-6) or not np.array_equal(rows[:, 6], selected[:, 5]):
        raise ValueError("Output detection index disagrees with current score/class")
    result = rows.copy()
    if policy == "detection":
        result[:, :4] = selected[:, :4]
    return result
