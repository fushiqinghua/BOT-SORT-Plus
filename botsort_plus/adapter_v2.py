# Native update order derives from BoxMOT 16.0.11, AGPL-3.0.
"""Explicit v2 integration: B recovery, native birth, then private A2 admission.

The v0.1 adapter remains selectable. This file owns the separately identified
v2 foundation: native lost-list enumeration, A-only shadow isolation and weak
birth transactions. It does not change CandidatePool or RecoveryMatcher.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np

from boxmot.trackers.botsort.basetrack import TrackState
from boxmot.trackers.botsort.botsort_track import STrack
from boxmot.trackers.botsort.botsort_utils import (
    joint_stracks, sub_stracks, remove_duplicate_stracks,
)
from boxmot.utils.matching import iou_distance, linear_assignment

from .appearance import appearance_distance
from .geometry import box_iou
from .types import Observation, Prediction, LostTrackView


UTILS_HASH = "99902964de7c01878e5410ba89b2b1da0821db3db0efb5fa6d0c2693d4af4052"


def verify_v2_upstream():
    import boxmot
    relative = "trackers/botsort/botsort_utils.py"
    source = Path(boxmot.__file__).resolve().parent / relative
    if hashlib.sha256(source.read_bytes()).hexdigest() != UTILS_HASH:
        raise RuntimeError("A2 requires the pinned BoxMOT 16.0.11 duplicate-removal source")
    return {relative: UTILS_HASH}


def _forget(tracker, ids):
    ids = list(ids)
    tracker.recovery_matcher.forget(ids)
    tracker.birth_policy.forget(ids)
    tracker.birth_identity.forget(ids)


def _consume(tracker, candidate_id, reason):
    if tracker.candidate_pool.get(candidate_id) is None:
        return
    tracker.candidate_pool.remove(candidate_id, reason)
    tracker.last_events.append(dict(tracker.candidate_pool.events[-1]))
    _forget(tracker, [candidate_id])


def _register_origin(tracker, track, origin, candidate=None):
    tracker.track_origins.setdefault(int(track.id), {
        "origin": origin, "birth_frame": int(track.start_frame),
        "candidate_id": int(candidate.candidate_id) if candidate is not None else None,
        "candidate_first_frame": int(candidate.first_frame) if candidate is not None else None,
        "first_emitted": None, "last_emitted": None, "output_frames": 0,
        "recovery_count": 0, "native_high_match_count": 0,
    })


def _record_output(tracker, output):
    # Diagnostics only. Origin never changes association or B eligibility.
    for event in list(tracker.last_events):
        tid = event.get("track_id")
        info = tracker.track_origins.get(tid)
        if info is None:
            continue
        event["track_origin"] = info["origin"]
        if event["event"] == "recovered":
            info["recovery_count"] += 1
        elif event["event"] == "match" and event.get("stage") in ("high", "unconfirmed"):
            info["native_high_match_count"] += 1
    for row in output:
        tid = int(row[4])
        info = tracker.track_origins[tid]
        if info["first_emitted"] is None:
            info["first_emitted"] = tracker.frame_count
            tracker.last_events.append({
                "event": "first_emitted", "frame": tracker.frame_count,
                "track_id": tid, "track_origin": info["origin"],
                "candidate_id": info["candidate_id"], "birth_frame": info["birth_frame"],
                "candidate_first_frame": info["candidate_first_frame"],
            })
        info["last_emitted"] = tracker.frame_count
        info["output_frames"] += 1


def _projected_lost(tracker, activated, refound, lost):
    """Read-only reconstruction of the native lists just before final dedup.

    Native _prepare_output subtracts the *previous* removed list, which may
    leave newly Removed objects for one frame. Keep that exact behavior here;
    it is distinct from strict Lost-state eligibility for B and the A guard.
    """
    active = [t for t in tracker.active_tracks if t.state == TrackState.Tracked]
    active = joint_stracks(joint_stracks(active, activated), refound)
    result = sub_stracks(tracker.lost_stracks, active) + list(lost)
    return sub_stracks(result, tracker.removed_stracks)


def update_v2(tracker, dets, img, dense):
    tracker.frame_count += 1
    frame = tracker.frame_count
    plus = tracker.plus
    if plus.raw_ir.enabled and (not hasattr(tracker,"raw_ir_observer") or tracker.raw_ir_observer.frame != frame):
        raise ValueError("Expected current aligned raw IR frame before tracker update")
    if plus.identity.enabled:
        # Freeze past-only identity evidence before any stage can write memory.
        tracker.identity_reliability.begin_frame(frame)
    activated, refound, lost, removed = [], [], [], []
    all_dets, high, _, low = tracker._split_detections(dets, dense)
    detections = [tracker._detection(row) for row in high]
    unconfirmed, active = tracker._separate_tracks()

    # Foundation fix: the native association sees the exact native list.
    # Eligibility for explicit recovery is filtered separately below.
    native_lost = list(tracker.lost_stracks)
    pool = joint_stracks(active, native_lost)
    diagnostics = getattr(tracker, "diagnostic_recorder", None)
    if diagnostics is not None:
        diagnostics.begin(tracker, frame, all_dets, pool, unconfirmed, img)
    STrack.multi_predict(pool)
    if diagnostics is not None:
        diagnostics.predicted(pool)
    warp = tracker.cmc.apply(img, all_dets)
    if warp is None or np.shape(warp) != (2, 3) or not np.isfinite(warp).all():
        raise ValueError("GMC returned an invalid transform")
    STrack.multi_gmc(pool, warp)
    STrack.multi_gmc(unconfirmed, warp)
    if diagnostics is not None:
        diagnostics.compensated(tracker, pool, unconfirmed, warp)
    needs_identity_history = plus.birth.enabled and plus.birth_identity.enabled
    if plus.recovery.enabled or needs_identity_history:
        history_window = max(plus.recovery.window if plus.recovery.enabled else 0,
                             plus.birth_identity.window if needs_identity_history else 0)
        for track in pool:
            _, cov = tracker.kalman_filter.project(track.mean, track.covariance)
            history = tracker.prediction_history.setdefault(track.id, {})
            history[frame] = Prediction(frame, track.xyxy.copy(), cov[:2, :2].copy())
            for past in list(history):
                if past <= frame - history_window:
                    del history[past]

    unmatched_pool, unmatched_high = tracker._associate(
        pool, detections, tracker.match_thresh, "high", activated, refound)
    low_tracks = [pool[i] for i in unmatched_pool if pool[i].state == TrackState.Tracked]
    low_detections = [STrack(row, max_obs=tracker.max_obs) for row in low]
    second_cost = iou_distance(low_tracks, low_detections)
    matches, unmatched_low_tracks, unmatched_low = linear_assignment(second_cost, thresh=.5)
    if plus.identity.enabled:
        from .identity_adapter import pair_decisions, apply_match, seed_identity
        low_decisions = pair_decisions(tracker, low_tracks, low_detections,
                                      second_cost, "low", .5)
    for i, j in matches:
        track, detection = low_tracks[int(i)], low_detections[int(j)]
        if plus.identity.enabled:
            apply_match(tracker, track, detection, low_decisions[int(i)][int(j)],
                        reference_cost=second_cost[i, j], applied_cost=second_cost[i, j],
                        base_cost=second_cost[i, j])
        else:
            track.update(detection, frame)
            tracker._write_memory(track, int(detection.det_ind))
        activated.append(track)
        tracker.last_events.append({"event": "match", "stage": "low", "track_id": int(track.id),
                                    "det_index": int(detection.det_ind),
                                    "cost": float(second_cost[i, j])})
    for i in unmatched_low_tracks:
        track = low_tracks[int(i)]
        track.mark_lost()
        lost.append(track)

    remaining_high = [detections[i] for i in unmatched_high]
    unmatched_unc, remaining_indices = tracker._associate(
        unconfirmed, remaining_high, .7, "unconfirmed", activated, refound, True)
    for i in unmatched_unc:
        track = unconfirmed[i]
        track.mark_removed()
        removed.append(track)
    birth_high = [remaining_high[i] for i in remaining_indices]

    current_candidates, reserved, consumed = [], set(), set()
    suppressed_births = set()
    if plus.birth.enabled or plus.recovery.enabled:
        residual = [low_detections[int(i)] for i in unmatched_low] + birth_high
        observations = [Observation(frame, int(d.det_ind), all_dets[int(d.det_ind), :4],
                                    float(d.conf), tracker._packets[int(d.det_ind)]) for d in residual]
        before = set(tracker.candidate_pool.candidates)
        occupied = [t.xyxy.copy() for t in joint_stracks(activated, refound)]
        occupied += [all_dets[int(t.det_ind), :4].copy() for t in joint_stracks(activated, refound)]
        current_candidates = tracker.candidate_pool.update(
            frame, observations, warp=warp, occupied_boxes=occupied)
        tracker.last_events.extend(dict(e) for e in tracker.candidate_pool.events)
        _forget(tracker, before - set(tracker.candidate_pool.candidates))
        # Candidate observation alone has NO authority over native births in
        # A-only/shadow mode. Preserve existing B's explicit suppression.
        if plus.recovery.enabled:
            suppressed_births = {int(e["det_index"]) for e in tracker.candidate_pool.events
                                 if e["event"] in ("candidate_observation_suppressed",
                                                   "candidate_seed_deferred")}

    def app_cost(memory, packet):
        if not plus.appearance.enabled:
            packet = {k: v for k, v in packet.items() if k == "rgb"}
        return appearance_distance(memory, packet, plus.appearance)

    if plus.recovery.enabled:
        eligible = {t.id: t for t in joint_stracks(native_lost, lost)
                    if t.state == TrackState.Lost and frame - t.end_frame <= tracker.max_time_lost}
        views = [LostTrackView(t.id, t.end_frame, tracker.prediction_history.get(t.id, {}),
                               tracker.memories.get(t.id)) for t in eligible.values()]
        decision = tracker.recovery_matcher.match(frame, views, current_candidates, app_cost)
        reserved = decision.reserved_candidate_ids
        tracker.last_events.extend({"event": "recovery_check", **item} for item in decision.diagnostics)
        for match in decision.matches:
            candidate = tracker.candidate_pool.get(match.candidate_id)
            track = eligible[match.track_id]
            detection = tracker._detection(all_dets[match.det_index], trusted_only=True)
            if plus.identity.enabled:
                recovery_decision = pair_decisions(
                    tracker, [track], [detection], iou_distance([track], [detection]),
                    "recovery", 1.)[0][0]
                apply_match(tracker, track, detection, recovery_decision, reactivate=True,
                            reference_cost=match.cost, applied_cost=match.cost)
            else:
                track.re_activate(detection, frame, new_id=False)
                tracker._write_memory(track, match.det_index)
            refound.append(track)
            lost = [t for t in lost if t.id != track.id]
            consumed.add(match.det_index)
            _consume(tracker, match.candidate_id, "recovered")
            _register_origin(tracker, track, "native")
            tracker.last_events.append({"event": "recovered", **asdict(match)})

    # Native high-score initialization precedes A admission. A neither blocks
    # this branch nor steals its observations. B's previous reservations apply.
    candidates_by_index = {c.latest.det_index: c for c in current_candidates
                           if c.candidate_id in tracker.candidate_pool.candidates}
    for track in birth_high:
        index = int(track.det_ind)
        candidate = candidates_by_index.get(index)
        if index in consumed or track.conf < tracker.new_track_thresh:
            continue
        if index in suppressed_births:
            tracker.last_events.append({"event": "suppressed_native_birth", "det_index": index,
                                        "reason": "existing_B_duplicate_or_candidate_ambiguity"})
            continue
        if candidate is not None and candidate.candidate_id in reserved:
            tracker.last_events.append({"event": "deferred_birth", "det_index": index,
                                        "candidate_id": candidate.candidate_id,
                                        "reason": "existing_B_reservation"})
            continue
        track.activate(tracker.kalman_filter, frame)
        activated.append(track)
        consumed.add(index)
        if plus.identity.enabled:
            seed_identity(tracker, track, index)
        else:
            tracker._write_memory(track, index)
        _register_origin(tracker, track, "native")
        if candidate is not None:
            _consume(tracker, candidate.candidate_id, "native_birth")
        tracker.last_events.append({"event": "native_birth", "track_id": int(track.id),
                                    "det_index": index})

    proposals = []
    if plus.birth.enabled:
        identity_tracks = {t.id: t for t in joint_stracks(native_lost, lost)
                           if t.state == TrackState.Lost
                           and 1 <= frame - t.end_frame <= tracker.max_time_lost}
        identity_views = [LostTrackView(t.id, t.end_frame, tracker.prediction_history.get(t.id, {}),
                                        tracker.memories.get(t.id)) for t in identity_tracks.values()]
        occupied = [t.xyxy.copy() for t in joint_stracks(activated, refound)]
        occupied += [all_dets[int(t.det_ind), :4].copy() for t in joint_stracks(activated, refound)]
        projected_lost = _projected_lost(tracker, activated, refound, lost)
        for candidate in sorted(tracker.candidate_pool.candidates.values(), key=lambda c: c.candidate_id):
            # Includes missed frames: verification deadlines count video time.
            evidence = tracker.birth_policy.evaluate(frame, candidate)
            # Read-only access to B's clock. RecoveryMatcher itself is unchanged.
            b_start = tracker.recovery_matcher._first_reserved.get(candidate.candidate_id)
            identity = tracker.birth_identity.inspect(frame, candidate, identity_views,
                                                       app_cost, reservation_start=b_start)
            tracker.last_events.append({"event": "birth_evidence", **asdict(evidence)})
            tracker.last_events.append({"event": "birth_identity", "candidate_id": candidate.candidate_id,
                                        **asdict(identity)})
            if not evidence.ready:
                continue
            index = int(candidate.latest.det_index)
            if candidate.last_frame != frame or index in consumed:
                continue
            if all_dets[index, 4] >= tracker.new_track_thresh:
                continue
            if candidate.candidate_id in reserved:
                tracker.last_events.append({"event": "birth_deferred", "candidate_id": candidate.candidate_id,
                                            "reason": "existing_B_reservation"})
                continue
            if identity.status != "CLEAR_NEW":
                tracker.last_events.append({"event": "birth_deferred" if identity.status == "DEFER" else "birth_rejected",
                                            "candidate_id": candidate.candidate_id,
                                            "reason": identity.reason, "stage": "identity"})
                continue
            if any(box_iou(candidate.latest.box, box) >= plus.birth.duplicate_iou for box in occupied):
                tracker.last_events.append({"event": "birth_rejected", "candidate_id": candidate.candidate_id,
                                            "reason": "native_observation_occupied", "stage": "proposal"})
                continue
            tracker.last_events.append({"event": "birth_proposed", "candidate_id": candidate.candidate_id,
                                        "det_index": index, "mode": plus.birth_policy.mode,
                                        "shadow": plus.birth_policy.observe_only})
            # Private STrack: do not call activate() or advance BaseTrack's ID
            # counter until the exact native duplicate rule accepts the proposal.
            track = tracker._detection(all_dets[index], trusted_only=True)
            track.frame_id = frame
            track.start_frame = frame
            retained, _ = remove_duplicate_stracks([track], projected_lost)
            if not retained:
                tracker.last_events.append({"event": "birth_rejected", "candidate_id": candidate.candidate_id,
                                            "det_index": index, "reason": "native_duplicate_preflight",
                                            "stage": "proposal"})
                continue
            if plus.birth_policy.observe_only:
                tracker.last_events.append({"event": "birth_shadow_ready", "candidate_id": candidate.candidate_id,
                                            "det_index": index})
                continue
            track.activate(tracker.kalman_filter, frame)
            track.is_activated = True
            activated.append(track)
            consumed.add(index)
            occupied.append(candidate.latest.box.copy())
            proposals.append((candidate, track, evidence))

    tracker._update_track_states(removed)
    output = tracker._prepare_output(activated, refound, lost, removed).reshape(-1, 8)
    output_ids = {int(row[4]) for row in output}
    active_ids = {int(t.id) for t in tracker.active_tracks if t.state == TrackState.Tracked}
    for candidate, track, evidence in proposals:
        index = int(candidate.latest.det_index)
        if int(track.id) not in active_ids or int(track.id) not in output_ids:
            # Defensive second check: preserve private history and formal memory
            # if native finalization nevertheless rejects the new track.
            tracker.last_events.append({"event": "birth_rejected", "candidate_id": candidate.candidate_id,
                                        "track_id": int(track.id), "reason": "native_finalization",
                                        "stage": "commit"})
            continue
        tracker._write_memory(track, index)
        _register_origin(tracker, track, "weak_birth", candidate)
        _consume(tracker, candidate.candidate_id, "born_v2")
        commit = {"track_id": int(track.id), "candidate_id": candidate.candidate_id,
                  "det_index": index, "mode": plus.birth_policy.mode,
                  "candidate_first_frame": int(candidate.first_frame), "birth_frame": frame,
                  "confirmation_delay_frames": frame - candidate.first_frame,
                  "support": dict(evidence.stats), "identity_status": "CLEAR_NEW"}
        tracker.last_events.append({"event": "birth_committed", **commit})
        # Compatibility alias for earlier event readers. Never add these counts
        # to birth_committed: they identify the same successful transaction.
        tracker.last_events.append({"event": "weak_birth", **commit, "alias_of": "birth_committed",
                                    "evidence": float(candidate.evidence),
                                    "evidence_role": ("legacy_confirmation_and_pool_priority"
                                                      if plus.birth_policy.mode == "legacy"
                                                      else "legacy_pool_priority_only"),
                                    "hits": candidate.hit_count(frame, plus.birth.window)})

    # Origin is diagnostic only, including tracks inserted by native code/tests.
    for track in tracker.active_tracks:
        _register_origin(tracker, track, "native")
    _record_output(tracker, output)
    live = {t.id for t in tracker.active_tracks + tracker.lost_stracks if t.state != TrackState.Removed}
    tracker.memories = {tid: value for tid, value in tracker.memories.items() if tid in live}
    if plus.identity.enabled:
        tracker.identity_reliability.retain(live)
    tracker.prediction_history = {tid: value for tid, value in tracker.prediction_history.items() if tid in live}
    tracker.last_events.insert(0, {"event": "frame", "frame": frame, "detections": len(dets),
                                  "active": len(tracker.active_tracks), "lost": len(tracker.lost_stracks),
                                  "candidates": len(tracker.candidate_pool.candidates),
                                  "adapter_mode": "v2"})
    if plus.recovery.output_prediction_frames:
        from .prediction_output import extend
        output = extend(tracker, output, img.shape, all_dets, live)
    if plus.raw_ir.enabled:
        observer=tracker.raw_ir_observer
        output=observer.check_output(output)
        observer.finish(output)
        rejected={e["track_id"] for e in observer.audit if e["event"]=="ir_projection_check" and not e["kept"]}
        for e in tracker.last_events:
            if e["event"]=="motion_output" and e["track_id"] in rejected:
                e["event"]="ir_rejected_projection"
        tracker.last_events.extend(observer.audit)
    if diagnostics is not None:
        diagnostics.finish(tracker, output, removed)
    return output
