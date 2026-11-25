"""Offline-only diagnostic data contracts; never imported by the tracker.

The IoU labels here are deliberately conservative diagnostic labels. They are
not the continuity-aware CLEAR assignment and must not be reported as TrackEval
FP/FN/IDSW. Ground truth is read only by these offline helpers.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from botsort_plus.io import sha256


TRACE_SCHEMA = "botsort_motion_diagnostics_v1"


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return number


def read_json(path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON token: {value}")
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream, parse_constant=invalid, parse_float=_finite_float)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def csv_writer(path, fields, stack):
    stream = stack.enter_context(Path(path).open("w", encoding="utf-8-sig", newline=""))
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    return writer


def finite_array(value, shape=None, label="array"):
    result = np.asarray(value, dtype=float)
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise ValueError(f"Invalid finite {label}; expected {shape}, got {result.shape}")
    return result


def xywh(box):
    b = finite_array(box, (4,), "xyxy box")
    if np.any(b[2:] <= b[:2]):
        raise ValueError("Box must have positive width and height")
    return np.r_[(b[:2] + b[2:]) / 2., b[2:] - b[:2]]


def box_iou_matrix(boxes, truth):
    a = np.asarray(boxes, dtype=float).reshape(-1, 4)
    b = np.asarray(truth, dtype=float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=float)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.maximum(hi - lo, 0.).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def diagnostic_assignment(boxes, gt_rows, threshold=.5, ambiguity_margin=.05):
    """Maximum cardinality then maximum IoU, with explicit unmatched choices.

    A selected edge remains ambiguous if another eligible edge in either its
    row or column is within ``ambiguity_margin``. Such edges are never used as
    identity evidence. Exactly-threshold overlaps are included.
    """
    if not 0 < threshold <= 1 or not 0 <= ambiguity_margin < 1:
        raise ValueError("Invalid diagnostic IoU/margin")
    iou = box_iou_matrix(boxes, [row["xyxy"] for row in gt_rows])
    n, m = iou.shape
    statuses = ["no_gt" if not m else "unmatched"] * n
    labels, assigned = {}, {}
    if not n or not m:
        return labels, statuses, iou, assigned
    eligible = iou >= threshold - np.finfo(float).eps
    # n+1 makes a new feasible edge preferable to any total IoU rearrangement.
    dummy = float(n + 1)
    cost = np.full((n, m + n), dummy * (n + m + 2), dtype=float)
    cost[:, :m] = np.where(eligible, 1. - iou, cost[:, :m])
    cost[np.arange(n), m + np.arange(n)] = dummy
    rows, cols = linear_sum_assignment(cost)
    for i, j in zip(rows, cols):
        if j >= m or not eligible[i, j]:
            if eligible[i].any():
                statuses[i] = "ambiguous"
            continue
        i, j = int(i), int(j)
        assigned[i] = j
        row_rivals = iou[i, np.arange(m) != j]
        col_rivals = iou[np.arange(n) != i, j]
        rivals = np.r_[row_rivals[row_rivals >= threshold],
                       col_rivals[col_rivals >= threshold]]
        if rivals.size and float(iou[i, j] - rivals.max()) < ambiguity_margin:
            statuses[i] = "ambiguous"
        else:
            statuses[i] = "matched"
            labels[i] = int(gt_rows[j]["gt_id"])
    return labels, statuses, iou, assigned


def load_gt(path, frames):
    """Read supplied RGB MOT GT; pool all classes, remove zero-marked rows only.

    This matches the repository's DO_PREPROC=False class-agnostic GT convention.
    The optional class/visibility columns are preserved as annotations and do
    not filter observations. Empty GT frames/files are valid, never negatives
    for recovery stability when a particular identity is unannotated.
    """
    result = {frame: [] for frame in range(1, frames + 1)}
    ignored = 0
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if not row or not any(item.strip() for item in row):
                continue
            if len(row) < 7:
                raise ValueError(f"{path}:{line}: expected at least seven GT columns")
            try:
                values = [float(item) for item in row]
            except ValueError as error:
                raise ValueError(f"{path}:{line}: nonnumeric GT") from error
            if not np.isfinite(values).all():
                raise ValueError(f"{path}:{line}: nonfinite GT")
            frame, gid, x, y, w, h, marked = values[:7]
            if frame != int(frame) or not 1 <= frame <= frames:
                raise ValueError(f"{path}:{line}: GT frame outside video")
            if marked == 0:
                ignored += 1
                continue
            if gid != int(gid) or gid < 0 or w <= 0 or h <= 0:
                raise ValueError(f"{path}:{line}: invalid GT identity or box")
            frame, gid = int(frame), int(gid)
            if any(item["gt_id"] == gid for item in result[frame]):
                raise ValueError(f"{path}:{line}: duplicate GT identity in frame")
            result[frame].append({"gt_id": gid, "xyxy": [x, y, x + w, y + h],
                                  "gt_class": values[7] if len(values) > 7 else None,
                                  "gt_visibility": values[8] if len(values) > 8 else None})
    return result, ignored


def iter_trace(path, sequence, frames):
    """Reject truncated/noncontiguous traces, including missing empty frames."""
    path = Path(path)
    manifest = read_json(path.with_suffix("").with_suffix(".json"))
    if (manifest.get("schema") != TRACE_SCHEMA or manifest.get("status") != "COMPLETE"
            or manifest.get("records") != frames or manifest.get("expected_frames") != frames
            or manifest.get("trace_filename") != path.name):
        raise ValueError(f"{sequence}: incomplete diagnostic manifest")
    if manifest.get("sequence") != sequence or manifest.get("data_sha256") != sha256(path):
        raise ValueError(f"{sequence}: diagnostic trace identity/hash mismatch")
    def invalid(value):
        raise ValueError(f"Nonfinite diagnostic JSON: {value}")
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            count += 1
            row = json.loads(line, parse_constant=invalid, parse_float=_finite_float)
            if (row.get("schema") != TRACE_SCHEMA or row.get("sequence") != sequence
                    or row.get("frame") != count or count > frames):
                raise ValueError(f"{sequence}: noncontiguous/foreign trace at frame {count}")
            required = {"detections", "before_prediction", "after_prediction", "after_gmc",
                        "unconfirmed", "after_update", "output", "events", "gmc"}
            if required - set(row):
                raise ValueError(f"{sequence}: missing trace fields: {required - set(row)}")
            detections = row["detections"]
            for index, detection in enumerate(detections):
                if detection.get("det_index") != index:
                    raise ValueError(f"{sequence}: detection row alignment mismatch")
                xywh(detection["xyxy"])
                if not math.isfinite(detection["score"]) or not 0 <= detection["score"] <= 1:
                    raise ValueError(f"{sequence}: invalid detection score")
            stages = ("before_prediction", "after_prediction", "after_gmc")
            if len({tuple(s["track_id"] for s in row[stage]) for stage in stages}) != 1:
                raise ValueError(f"{sequence}: KF/GMC stages disagree on track pool/order")
            for stage in ("before_prediction", "after_prediction", "after_gmc",
                          "unconfirmed", "after_update"):
                ids = set()
                for snapshot in row[stage]:
                    tid = snapshot["track_id"]
                    if isinstance(tid, bool) or not isinstance(tid, int) or tid < 1 or tid in ids:
                        raise ValueError(f"{sequence}: invalid/duplicate snapshot identity")
                    ids.add(tid)
                    if (snapshot["state"] not in (0, 1, 2, 3)
                            or not isinstance(snapshot["last_seen"], int)
                            or not 1 <= snapshot["last_seen"] <= count):
                        raise ValueError(f"{sequence}: invalid snapshot state/last observation")
                    xywh(snapshot["xyxy"])
                    finite_array(snapshot["mean"], (8,), "state mean")
                    finite_array(snapshot["covariance"], (8, 8), "state covariance")
                    if stage in ("after_gmc", "unconfirmed"):
                        finite_array(snapshot["projected_mean"], (4,), "projected mean")
                        finite_array(snapshot["projected_covariance"], (4, 4), "projected covariance")
            output = np.asarray(row["output"], dtype=float).reshape(-1, 8)
            if not np.isfinite(output).all() or len(set(output[:, 4])) != len(output):
                raise ValueError(f"{sequence}: malformed diagnostic output")
            for event in row["events"]:
                if "frame" in event and event["frame"] != count:
                    raise ValueError(f"{sequence}: event frame disagreement")
            yield row
    if count != frames:
        raise ValueError(f"{sequence}: truncated diagnostic frames ({count}/{frames})")


def statistics(values):
    a = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()) if a.size else None,
            "median": float(np.median(a)) if a.size else None,
            "p90": float(np.quantile(a, .9)) if a.size else None,
            "p95": float(np.quantile(a, .95)) if a.size else None}


def size_bin(box):
    side = float(min(xywh(box)[2:]))
    return "lt8" if side < 8 else "8to16" if side < 16 else "16to32" if side < 32 else "ge32"


def detection_tier(score, native):
    high, low = native["track_high_thresh"], native["track_low_thresh"]
    if score > high:
        return "high"
    if low < score < high:
        return "low"
    return "outside_native_score_bands"


def squared_mahalanobis(delta, covariance):
    """Return None for non-positive-definite covariance; do not repair evidence."""
    cov = np.asarray(covariance, dtype=float)
    if not np.allclose(cov, cov.T, atol=1e-7, rtol=1e-6):
        return None
    try:
        chol = np.linalg.cholesky((cov + cov.T) / 2.)
        whitened = np.linalg.solve(chol, np.asarray(delta, dtype=float))
    except np.linalg.LinAlgError:
        return None
    value = float(whitened @ whitened)
    return value if math.isfinite(value) else None
