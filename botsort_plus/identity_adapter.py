"""C2 integration with the pinned native tracker, without a global monkey patch.

Association acceptance and appearance writes are separate decisions.  A rejected
write suppresses the *input* to STrack.update_features, not the Kalman update.
The native low-score stage still has no native feature writer.  The core's frame
snapshot is read-only even though accepted observations are committed in order.
"""
from __future__ import annotations

import numpy as np

from .appearance import MemoryBank

from boxmot.trackers.botsort.basetrack import TrackState
from boxmot.utils.matching import fuse_score, iou_distance, linear_assignment


def pair_decisions(tracker, tracks, detections, geometry, stage, threshold):
    return tracker.identity_reliability.evaluate(
        [int(t.id) for t in tracks], [int(d.det_ind) for d in detections],
        geometry, tracker._packets, stage, threshold)


def _recent_rgb_agrees(tracker, track_id, det_index, decision, reactivate):
    """Record each accepted match; only adjacent eligible high-score matches can agree."""
    memory = tracker.memories.get(track_id)
    if memory is None:
        memory = tracker.memories[track_id] = MemoryBank(tracker.plus.appearance)
    previous = memory.recent_rgb_observation
    memory.recent_rgb_observation = None
    feature = tracker._packets[det_index].get("rgb")
    if (reactivate or decision.stage != "high" or decision.geometry_cost > .50
            or (decision.geometry_margin is not None and decision.geometry_margin < .10)
            or feature is None or feature.quality < memory.config.min_update_quality):
        return False
    vector = np.asarray(feature.vector, dtype=float)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-12:
        return False
    vector = vector / norm
    frame = tracker.frame_count
    memory.recent_rgb_observation = (frame, vector)
    return (previous is not None and previous[0] == frame - 1
            and 1. - float(vector @ previous[1]) <= .40)


def _low_warmup_status(memory, decision, cfg, reactivate):
    """Two adjacent accepted, unambiguous low observations of the same track."""
    previous = memory.last_stable_low_frame
    memory.last_stable_low_frame = None
    if reactivate or decision.stage != "low":
        return "outside_low"
    if decision.reason != "warmup_requires_high":
        return decision.reason
    if decision.quality < max(cfg.min_quality, memory.config.min_update_quality):
        return "low_quality"
    if decision.geometry_cost > cfg.bootstrap_max_iou_cost:
        return "poor_geometry"
    if decision.competitor_count and (decision.geometry_margin is None
                                      or decision.geometry_margin < cfg.geometry_margin):
        return "ambiguous_geometry"
    memory.last_stable_low_frame = decision.frame
    return "eligible" if previous == decision.frame - 1 else "first_qualified_frame"


def apply_match(tracker, track, detection, decision, *, reactivate=False,
                reference_cost=None, applied_cost=None, base_cost=None):
    """Apply the accepted track update and route appearance writes according to policy."""
    cfg = tracker.plus.identity
    gate_applied = cfg.memory_enabled and not cfg.observe_only
    permit_bank = not gate_applied or decision.allow_write
    block_write = not permit_bank
    recent_agrees = False
    low_status = None
    if cfg.recent_confirmed_bank and gate_applied:
        recent_agrees = _recent_rgb_agrees(tracker, track.id, int(detection.det_ind), decision, reactivate)
        permit_bank = decision.allow_write or (decision.reason == "history_conflict" and recent_agrees)
        if cfg.stable_low_warmup:
            low_status = _low_warmup_status(tracker.memories[track.id], decision, cfg, reactivate)
            permit_bank = permit_bank or low_status == "eligible"
        block_write = False  # Keep BoT-SORT's native feature updates.
    native_feature = detection.curr_feat
    # STrack mutates native smooth_feat/curr_feat/features inside update().
    # Passing no feature prevents all three writes; restore the detection object
    # even if native code raises.  No caller-owned cached feature is modified.
    if block_write:
        detection.curr_feat = None
    try:
        if reactivate:
            track.re_activate(detection, tracker.frame_count, new_id=False)
        else:
            track.update(detection, tracker.frame_count)
    finally:
        if block_write:
            detection.curr_feat = native_feature
    if low_status == "eligible":
        updated = tracker._write_memory(track, int(detection.det_ind), rgb_only=True)
    else:
        updated = [] if not permit_bank else tracker._write_memory(track, int(detection.det_ind))
    private_write = tracker.identity_reliability.commit(
        int(track.id), tracker._packets[int(detection.det_ind)], decision, tracker.frame_count)
    tracker.last_events.append({
        "event": "identity_decision", **decision.to_dict(),
        "observe_only": cfg.observe_only,
        "association_applied": bool(cfg.association_enabled and not cfg.observe_only
                                    and decision.stage in ("high", "unconfirmed")),
        "memory_gate_applied": gate_applied,
        **({"recent_rgb_agrees": recent_agrees, "bank_write_permitted": permit_bank}
           if cfg.recent_confirmed_bank and gate_applied else {}),
        "native_feature_written": bool(native_feature is not None and not block_write),
        **({"low_warmup_status": low_status} if low_status is not None else {}),
        "custom_modalities_written": updated,
        "private_anchor_written": bool(private_write),
        "reference_cost": None if reference_cost is None else float(reference_cost),
        "applied_cost": None if applied_cost is None else float(applied_cost),
        "base_cost": None if base_cost is None else float(base_cost),
    })


def associate(tracker, tracks, detections, threshold, stage, activated, refound,
              unconfirmed=False):
    """Calibrate only the existing appearance discount; preserve spatial gates."""
    geometry = iou_distance(tracks, detections)
    base = (fuse_score(geometry, detections)
            if unconfirmed or tracker.fuse_first_associate else geometry)
    reference = tracker._cost(tracks, detections, unconfirmed=unconfirmed)
    decisions = pair_decisions(tracker, tracks, detections, geometry, stage,
                               tracker.proximity_thresh)
    proposed = tracker.identity_reliability.calibrate(base, reference, decisions)
    cfg = tracker.plus.identity
    applied = cfg.association_enabled and not cfg.observe_only
    costs = proposed if applied else reference
    if stage=="high" and tracker.plus.raw_ir.native:
        costs=tracker.raw_ir_observer.native(costs,geometry,tracks,detections,threshold,tracker.proximity_thresh)
    tracker.last_events.append({
        "event": "identity_association", "frame": tracker.frame_count, "stage": stage,
        "pair_count": int(reference.size), "applied": bool(applied),
        "proposed_changed_pairs": int(np.count_nonzero(np.abs(proposed - reference) > 1e-12)),
        "proposed_max_cost_increase": float(np.max(proposed - reference)) if reference.size else 0.,
        "cold_reference_pairs": sum(d.reliability is None for row in decisions for d in row),
    })
    matches, u_tracks, u_detections = linear_assignment(costs, thresh=threshold)
    for i, j in matches:
        i, j = int(i), int(j)
        track, det = tracks[i], detections[j]
        reactivate = track.state != TrackState.Tracked
        apply_match(tracker, track, det, decisions[i][j], reactivate=reactivate,
                    reference_cost=reference[i, j], applied_cost=costs[i, j],
                    base_cost=base[i, j])
        (refound if reactivate else activated).append(track)
        tracker.last_events.append({"event": "match", "stage": stage,
                                   "track_id": int(track.id), "det_index": int(det.det_ind),
                                   "cost": float(costs[i, j])})
    return list(u_tracks), list(u_detections)


def seed_identity(tracker, track, det_index):
    """A quality-qualified birth is provisional, not mature identity evidence."""
    cfg = tracker.plus.identity
    accepted = tracker.identity_reliability.seed(
        int(track.id), tracker._packets[det_index], tracker.frame_count)
    blocked = cfg.memory_enabled and not cfg.observe_only and not accepted
    if blocked and not cfg.recent_confirmed_bank:
        # STrack was constructed before activation and already contains a seed.
        # Do not leave an unqualified native seed while claiming to block memory.
        track.smooth_feat = None
        track.curr_feat = None
        track.features.clear()
    updated = [] if blocked else tracker._write_memory(track, det_index)
    tracker.last_events.append({
        "event": "identity_seed", "frame": tracker.frame_count,
        "track_id": int(track.id), "det_index": int(det_index),
        "provisional": True, "private_anchor_written": bool(accepted),
        "would_allow_write": bool(accepted), "observe_only": cfg.observe_only,
        "memory_gate_applied": bool(cfg.memory_enabled and not cfg.observe_only),
        "native_feature_present": track.smooth_feat is not None,
        "custom_modalities_written": updated,
    })
