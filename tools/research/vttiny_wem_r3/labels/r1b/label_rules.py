"""Causal offline GT labels for the read-only W S0 experiment."""
from __future__ import annotations

import numpy as np


GT_IOU_MIN = 0.50
GT_IOU_MARGIN = 0.05
HISTORY_FRAMES = 5
HISTORY_MIN = 2
HISTORY_MAX = 3
IGNORE_FRACTION = 0.50
REGULAR_STAGES = {"high", "low", "unconfirmed", "B_recovery", "native_birth"}


def xywh_to_xyxy(box):
    x, y, width, height = map(float, box)
    return [x, y, x + width, y + height]


def overlap(box, targets):
    """Return aligned IoU and fraction of the query covered by each target."""
    a = np.asarray(box, dtype=float)
    b = np.asarray(targets, dtype=float).reshape(-1, 4)
    inter = np.maximum(np.minimum(a[2:], b[:, 2:]) - np.maximum(a[:2], b[:, :2]), 0).prod(axis=1)
    area = (a[2:] - a[:2]).prod()
    other_area = (b[:, 2:] - b[:, :2]).prod(axis=1)
    union = area + other_area - inter
    return (np.divide(inter, union, out=np.zeros_like(inter), where=union > 0),
            np.divide(inter, area, out=np.zeros_like(inter), where=area > 0))


def make_truth(frame, ignored):
    boxes = {gt_id: xywh_to_xyxy(box) for gt_id, box in frame["gt_boxes"].items()}
    return {"gt_ids": list(boxes), "boxes": boxes, "box_rows": list(boxes.values()),
            "ignored": ignored, "coverage": frame["coverage"], "outputs": frame["outputs"]}


def observe_ground_truth(raw, truth):
    ious, _ = overlap(raw, truth["box_rows"])
    _, ignored_fraction = overlap(raw, truth["ignored"])
    order = np.argsort(-ious, kind="stable")
    best = float(ious[order[0]]) if len(order) else 0.0
    second = float(ious[order[1]]) if len(order) >= 2 else 0.0
    ignored = float(ignored_fraction.max(initial=0))
    gt_id = truth["gt_ids"][int(order[0])] if best > 0 else None
    if ignored >= IGNORE_FRACTION:
        quality = "ignored_region"
    elif best < GT_IOU_MIN:
        quality = "low_or_no_gt_overlap"
    elif best - second < GT_IOU_MARGIN:
        quality = "ambiguous_gt_overlap"
    else:
        quality = "clear_raw_proximity"
    return {"gt_id": gt_id, "best_iou": best, "second_iou": second, "overlap_gap": best - second,
            "ignored_fraction": ignored, "quality": quality,
            "ious": dict(zip(truth["gt_ids"], map(float, ious), strict=True))}


def identity_anchor(history, frame, track_id, past_frames):
    """Bind at least two earlier accepted observations to the same CLEAR GT."""
    if len(history) > HISTORY_MAX:
        raise ValueError("Observer history exceeds frozen scope")
    if [h["frame"] for h in history] != sorted(set(h["frame"] for h in history)):
        raise ValueError("Observer history repeats frames or is out of order")
    supported, notes = [], []
    for item in history:
        time, row, stage = item["frame"], item["det_index"], item["stage"]
        if time >= frame or time < 1 or item.get("track_id", track_id) != track_id:
            raise ValueError("Current, future, or foreign identity in observer history")
        evidence = dict(frame=time, det_index=row, stage=stage)
        notes.append(evidence)
        old = past_frames.get(time)
        if old is None or frame - time > HISTORY_FRAMES:
            evidence["support"] = "outside_history_window"
            continue
        if stage not in REGULAR_STAGES:
            evidence["support"] = "nonregular_source"
            continue
        obs = old["accepted"].get(track_id)
        if obs is None or obs["det_index"] != row or obs["stage"] != stage:
            evidence["support"] = "accepted_index_unverified"
            continue
        if old["output_indices"].get(track_id) != row:
            evidence["support"] = "not_same_raw_as_final_output"
            continue
        if not 0 <= row < len(old["raw"]):
            raise ValueError("Historical observation index outside frozen cache")
        clear_id = old["truth"]["outputs"][track_id]["matched_gt"]
        evidence["clear_gt_id"] = clear_id
        look = observe_ground_truth(old["raw"][row][:4], old["truth"])
        evidence.update(raw_gt_id=look["gt_id"], raw_gt_iou=look["best_iou"], raw_gt_margin=look["overlap_gap"])
        if clear_id is None or look["quality"] != "clear_raw_proximity" or look["gt_id"] != clear_id:
            evidence["support"] = "unconfirmed_or_conflicting_raw"
            continue
        evidence["support"] = "match"
        supported.append(clear_id)
    if len(set(supported)) > 1:
        return None, "conflicting_past_identity", notes
    if len(supported) < HISTORY_MIN:
        return None, "insufficient_strict_past_support", notes
    return supported[0], "anchored_past_raw_clear", notes


def classify_identity(anchor, candidate, review):
    if candidate["quality"] == "ignored_region":
        return "ignored", "official_ignore_overlap"
    if candidate["quality"] == "clear_raw_proximity" and anchor is not None:
        if anchor == candidate["gt_id"]:
            return "same_identity", "prior_raw_clear_anchor_and_current_unique_gt_overlap"
        return "other_identity", "prior_raw_clear_anchor_and_current_other_gt_overlap"
    if (candidate["quality"] == "low_or_no_gt_overlap" and review is not None
            and review.get("human_review_status") == "confirmed_background"):
        return "confirmed_background", "independent_human_review"
    if candidate["quality"] == "clear_raw_proximity":
        return "unknown", "track_identity_unanchored"
    return "unknown", candidate["quality"]
