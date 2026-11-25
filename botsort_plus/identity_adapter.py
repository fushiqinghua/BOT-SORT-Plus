"""C2 integration with the pinned native tracker, without a global monkey patch.

Association acceptance and appearance writes are separate decisions.  A rejected
write suppresses the *input* to STrack.update_features, not the Kalman update.
The native low-score stage still has no native feature writer.  The core's frame
snapshot is read-only even though accepted observations are committed in order.
"""
from __future__ import annotations

import numpy as np

from boxmot.trackers.botsort.basetrack import TrackState
from boxmot.utils.matching import fuse_score, iou_distance, linear_assignment


def pair_decisions(tracker, tracks, detections, geometry, stage, threshold):
    return tracker.identity_reliability.evaluate(
        [int(t.id) for t in tracks], [int(d.det_ind) for d in detections],
        geometry, tracker._packets, stage, threshold)


def apply_match(tracker, track, detection, decision, *, reactivate=False,
                reference_cost=None, applied_cost=None, base_cost=None):
    """Keep geometry/lifecycle updates, conditionally permit BOTH existing writers."""
    cfg = tracker.plus.identity
    gate_applied = cfg.memory_enabled and not cfg.observe_only
    block_write = gate_applied and not decision.allow_write
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
    updated = [] if block_write else tracker._write_memory(track, int(detection.det_ind))
    private_write = tracker.identity_reliability.commit(
        int(track.id), tracker._packets[int(detection.det_ind)], decision, tracker.frame_count)
    tracker.last_events.append({
        "event": "identity_decision", **decision.to_dict(),
        "observe_only": cfg.observe_only,
        "association_applied": bool(cfg.association_enabled and not cfg.observe_only
                                    and decision.stage in ("high", "unconfirmed")),
        "memory_gate_applied": gate_applied,
        "native_feature_written": bool(native_feature is not None and not block_write),
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
    if blocked:
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
