"""Label-free, frozen-frame W0--W3 association diagnostics.

The input is the complete geometry/ownership-qualified R3 score set for ONE
sequence and frame, including rows beyond the top-k budget.  Ground-truth
fields are deliberately never read.  This module does not update a tracker,
Kalman filter, detection score, or appearance memory.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from numbers import Integral, Real

import numpy as np
from scipy.optimize import linear_sum_assignment


METHODS = ("W0", "W1", "W2", "W3", "W_old")
DEFAULT_PROTOCOL = {
    "budget": 5,
    "native_iou_floor": 0.5,
    "motion_floor": 0.25,
    "appearance_weight": 0.2,
    "relative_margin": 0.05,
}


def edge_key(row):
    """Identity of a raw candidate edge, without any offline labels."""
    return row["sequence"], row["frame"], row["track_id"], row["det_index"]


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _number(value, name, lower=None, upper=None, optional=False):
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if lower is not None and value < lower or upper is not None and value > upper:
        raise ValueError(f"{name} is outside [{lower}, {upper}]")
    return value


def _boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _protocol(protocol):
    if protocol is None:
        return dict(DEFAULT_PROTOCOL)
    if not isinstance(protocol, Mapping) or set(protocol) - set(DEFAULT_PROTOCOL):
        raise ValueError("Unexpected engine protocol fields")
    result = {**DEFAULT_PROTOCOL, **protocol}
    result["budget"] = _integer(result["budget"], "budget", 1)
    for name in ("native_iou_floor", "motion_floor", "appearance_weight"):
        result[name] = _number(result[name], name, 0.0, 1.0)
    result["relative_margin"] = _number(result["relative_margin"], "relative_margin", 0.0, 2.0)
    return result


def _rows(rows, method):
    """Copy only named inference fields; labels cannot affect decisions."""
    result, keys, frame_key, raw_observations = [], set(), None, {}
    for original in rows:
        if not isinstance(original, Mapping):
            raise ValueError("Every score row must be a mapping")
        sequence = original.get("sequence")
        if not isinstance(sequence, str) or not sequence:
            raise ValueError("sequence must be a nonempty string")
        row = {
            "sequence": sequence,
            "frame": _integer(original["frame"], "frame", 1),
            "track_id": _integer(original["track_id"], "track_id"),
            "det_index": _integer(original["det_index"], "det_index"),
            "geometry_group_rank": _integer(original["geometry_group_rank"], "geometry_group_rank", 1),
            "raw_score": _number(original["raw_score"], "raw_score", 0.0, 1.0),
            "normalized_score": _number(original["normalized_score"], "normalized_score", 0.0, 1.0),
            "native_iou": _number(original["native_iou"], "native_iou", 0.0, 1.0),
            "motion": _number(original.get("motion"), "motion", 0.0, 1.0, optional=True),
            "cosine": _number(original.get("cosine"), "cosine", -1.0, 1.0, optional=True),
        }
        box = original["raw_box_xyxy"]
        if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
            raise ValueError("raw_box_xyxy must contain four coordinates")
        row["raw_box_xyxy"] = [_number(v, "raw_box_xyxy") for v in box]
        if row["raw_box_xyxy"][2] <= row["raw_box_xyxy"][0] or row["raw_box_xyxy"][3] <= row["raw_box_xyxy"][1]:
            raise ValueError("raw_box_xyxy has nonpositive extent")
        current_frame = sequence, row["frame"]
        if frame_key is None:
            frame_key = current_frame
        elif current_frame != frame_key:
            raise ValueError("evaluate_frame requires a single sequence/frame")
        key = edge_key(row)
        if key in keys:
            raise ValueError("Repeated raw candidate edge")
        keys.add(key)
        raw_key = row["det_index"]
        raw_value = tuple(row["raw_box_xyxy"]), row["raw_score"], row["normalized_score"]
        if raw_key in raw_observations and raw_observations[raw_key] != raw_value:
            raise ValueError("The same detection index resolves to different observations")
        raw_observations[raw_key] = raw_value
        if method == "W_old":
            row["history_score"] = _number(original.get("history_score"), "history_score", 0.0, 1.0, optional=True)
            for axis in ("row", "column"):
                row[axis + "_unknown"] = _boolean(original[axis + "_unknown"], axis + "_unknown")
                row[axis + "_margin"] = _number(original.get(axis + "_margin"), axis + "_margin", -1.0, 1.0, optional=True)
        if method == "W3":
            rivals = original.get("challengers")
            if not isinstance(rivals, list):
                raise ValueError("W3 requires the full cached challengers list")
            row["challengers"] = []
            identities = set()
            for rival in rivals:
                tid = _integer(rival["track_id"], "challenger track_id")
                if tid == row["track_id"]:
                    raise ValueError("A challenger must be a different identity")
                if tid in identities:
                    raise ValueError("Repeated challenger identity")
                identities.add(tid)
                row["challengers"].append({"track_id": tid,
                    "cosine": _number(rival.get("cosine"), "challenger cosine", -1.0, 1.0, optional=True)})
        result.append(row)
    return result


def _relative(own, competitors):
    """Return evidence availability and a real margin, never a fake 0/inf."""
    if not competitors:
        return "not_applicable", None
    if own is None or any(value is None for value in competitors):
        return "unavailable", None
    return "available", own - max(competitors)


def _decision(row, method, settings, row_alternatives):
    decision = {k: row[k] for k in ("sequence", "frame", "track_id", "det_index", "raw_box_xyxy",
                                   "raw_score", "native_iou", "motion", "cosine", "geometry_group_rank")}
    decision.update(method=method, utility=None, eligible=False, selected=False,
                    reasons=[], appearance_status="not_used", row_margin=None, column_margin=None,
                    row_competition_status="not_evaluated", column_competition_status="not_evaluated",
                    row_competitor_groups=None, column_competitor_identities=None)
    if row["native_iou"] < settings["native_iou_floor"]:
        decision["reasons"].append("native_geometry")
    if row["geometry_group_rank"] > settings["budget"]:
        decision["reasons"].append("top_k")

    if method == "W0":
        decision["utility"] = row["native_iou"]
    elif method == "W_old":
        decision["appearance_status"] = "required_available" if row["cosine"] is not None else "required_missing"
        decision["utility"] = row["history_score"]
        if row["history_score"] is None:
            decision["reasons"].append("old_history_unavailable")
        if row["cosine"] is None or row["cosine"] < 0.85:
            decision["reasons"].append("old_absolute_cosine")
        if row["motion"] is None or row["motion"] < 0.25:
            decision["reasons"].append("old_motion")
        if row["history_score"] is not None and row["history_score"] < 0.55:
            decision["reasons"].append("old_joint_utility")
        for axis in ("row", "column"):
            margin = row[axis + "_margin"]
            missing = row[axis + "_unknown"]
            decision[axis + "_margin"] = margin
            decision[axis + "_competition_status"] = "unavailable" if missing else "not_applicable" if margin is None else "available"
            if missing:
                decision["reasons"].append("old_" + axis + "_competition_unavailable")
            if margin is not None and margin < 0.05:
                decision["reasons"].append("old_" + axis + "_margin")
    else:
        motion = row["motion"]
        if motion is None:
            decision["reasons"].append("motion_unavailable")
        elif motion < settings["motion_floor"]:
            decision["reasons"].append("motion_floor")
        if motion is not None:
            decision["utility"] = (row["native_iou"] + motion) / 2.0
        if method in ("W2", "W3"):
            cosine = row["cosine"]
            decision["appearance_status"] = "missing_fallback" if cosine is None else "available_soft"
            if cosine is not None and motion is not None:
                weight = settings["appearance_weight"]
                decision["utility"] = (1.0 - weight) * (row["native_iou"] + motion) / 2.0 + weight * (cosine + 1.0) / 2.0
        if method == "W3":
            competitors = defaultdict(list)
            for other in row_alternatives:
                if other["raw_box_xyxy"] != row["raw_box_xyxy"]:
                    competitors[tuple(other["raw_box_xyxy"])].append(other["cosine"])
            # All rows of every other group participate.  Any missing value
            # means that group's comparative evidence is incomplete.
            row_values = [None if any(v is None for v in values) else max(values)
                          for values in competitors.values()]
            column_values = [other["cosine"] for other in row["challengers"]]
            decision["row_competitor_groups"] = len(competitors)
            decision["column_competitor_identities"] = len(row["challengers"])
            for axis, values in (("row", row_values), ("column", column_values)):
                status, margin = _relative(row["cosine"], values)
                decision[axis + "_competition_status"] = status
                decision[axis + "_margin"] = margin
                if status == "unavailable":
                    decision["reasons"].append(axis + "_appearance_unavailable")
                elif margin is not None and margin < settings["relative_margin"]:
                    decision["reasons"].append(axis + "_relative_margin")
    decision["eligible"] = not decision["reasons"] and decision["utility"] is not None
    decision["decision"] = "unassigned" if decision["eligible"] else "rejected"
    return decision


def _assign(rows, decisions):
    viable = [(index, row) for index, row in enumerate(decisions) if row["eligible"]]
    if not viable:
        return
    tracks = sorted({row["track_id"] for _, row in viable})
    groups = sorted({tuple(row["raw_box_xyxy"]) for _, row in viable})
    rr, cc = {tid: i for i, tid in enumerate(tracks)}, {box: i for i, box in enumerate(groups)}
    # Zero-utility dummy columns preserve the R3 option to leave a track
    # unmatched.  Only real viable slots can be returned as selections.
    matrix = np.zeros((len(tracks), len(groups) + len(tracks)), dtype=np.float64)
    chosen = {}
    for index, decision in viable:
        row, value = rows[index], decision["utility"]
        slot = rr[row["track_id"]], cc[tuple(row["raw_box_xyxy"])]
        order = value, row["raw_score"], -row["det_index"]
        if slot not in chosen or order > chosen[slot][0]:
            chosen[slot] = order, index
            matrix[slot] = value + row["normalized_score"] * 1e-10 + 1e-12 / (row["det_index"] + 1)
    for i, j in zip(*linear_sum_assignment(matrix, maximize=True)):
        if (i, j) in chosen:
            decision = decisions[chosen[i, j][1]]
            decision["selected"] = True
            decision["decision"] = "selected"


def evaluate_frame(rows, method, protocol=None):
    """Score and assign ONE complete cached frame without reading labels.

    One result is returned for every input edge, in its original order.  Rows
    outside top-k are retained as rejected decisions and still participate in
    W3's pre-budget comparison.  ``eligible`` means pre-assignment admissible;
    an eligible edge may remain ``unassigned`` due to the one-to-one constraint.
    W_old uses the original fixed .85/.25/.55/.05 settings and budget 5.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown W experiment: {method!r}")
    settings = _protocol(protocol)
    if method == "W_old":
        settings = dict(DEFAULT_PROTOCOL)
    rows = _rows(rows, method)
    by_track = defaultdict(list)
    for row in rows:
        if row["native_iou"] >= settings["native_iou_floor"]:
            by_track[row["track_id"]].append(row)
    decisions = [_decision(row, method, settings, by_track[row["track_id"]]) for row in rows]
    _assign(rows, decisions)
    return decisions
