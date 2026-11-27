"""Causal, read-only J3 probes for the pinned B/GMC experiment.

The reference tracker alone owns public IDs, state, memory and reservations.
All counterfactuals are conditional on a shared observed history. They are not
complete alternative tracking runs, and no probe consumes GT or future frames.
"""
from __future__ import annotations

from collections import Counter
from copy import copy, deepcopy
from dataclasses import asdict, is_dataclass
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "v04_joint_probe_v1"


def json_value(value):
    """Strict JSON: missing/infinite pair costs are null, never fake zeros."""
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_value(v) for v in value]
    return value


def members(candidate):
    return [(int(o.frame), int(o.det_index)) for o in candidate.observations]


def pool_snapshot(pool):
    return {int(cid): {"members": members(c), "state_frame": int(c.state_frame),
                      "box": c.last_box.copy(), "anchor": c._anchor_box.copy(),
                      "velocity": c.velocity.copy(), "evidence": float(c.evidence)}
            for cid, c in pool.candidates.items()}


def decision_pairs(decision):
    return sorted((int(m.track_id), int(m.candidate_id), int(m.det_index), float(m.cost))
                  for m in decision.matches)


def _same(a, b):
    return json_value(a) == json_value(b)


def _advance(tracks, warp):
    """Invoke the exact native state operators on detached objects only."""
    if not tracks:
        return
    kind = type(tracks[0])
    kind.multi_predict(tracks)
    kind.multi_gmc(tracks, np.asarray(warp).copy())


def prediction_disagreement(left, right, pixel_floor):
    """Diagnostic scale uses the reference prediction, never GT."""
    a, b = np.asarray(left, float), np.asarray(right, float)
    wh = np.maximum(a[2:] - a[:2], 0.)
    scale = max(float(pixel_floor), float(np.sqrt(np.prod(wh))))
    return {"center_pixels": float(np.linalg.norm((a[:2] + a[2:] - b[:2] - b[2:]) / 2)),
            "center_normalized": float(np.linalg.norm((a[:2] + a[2:] - b[:2] - b[2:]) / 2) / scale),
            "size_normalized": float(np.linalg.norm((a[2:] - a[:2]) - (b[2:] - b[:2])) / scale),
            "normalization_scale": scale}


def common_identity_support(history, predictions, memory, geometry, appearance, recovery):
    """Same score for active/Lost identities; excludes Lost eligibility and age.

    This is a diagnostic score, not an alternative RecoveryMatcher probability.
    Missing any same-frame prediction leaves the score unknown. Appearance
    absence stays explicit and changes the denominator just as declared below.
    """
    from botsort_plus.appearance import appearance_distance
    from botsort_plus.geometry import geometry_cost
    if not history:
        return {"available": False, "reason": "empty_history", "score": None}
    g, a = [], []
    for observation in history:
        pred = predictions.get(int(observation.frame))
        if pred is None or pred.frame != observation.frame:
            return {"available": False, "reason": "missing_same_frame_prediction", "score": None}
        value = geometry_cost(pred.box, observation.box, geometry, center_cov=pred.center_cov)
        if not np.isfinite(value):
            return {"available": False, "reason": "geometry_gate", "score": None}
        g.append(float(value))
        packet = observation.features if appearance.enabled else {
            k: v for k, v in observation.features.items() if k == "rgb"}
        distance = appearance_distance(memory, packet, appearance)
        if distance is not None:
            a.append(float(distance))
    geom, app = float(np.mean(g)), float(np.mean(a)) if a else None
    weight = recovery.geometry_weight + (recovery.appearance_weight if a else 0.)
    if weight <= 0:
        return {"available": False, "reason": "zero_common_weight", "score": None}
    score = (recovery.geometry_weight * geom +
             (recovery.appearance_weight * app if a else 0.)) / weight
    return {"available": True, "reason": "scored", "score": float(score),
            "geometry": geom, "appearance": app, "appearance_hits": len(a),
            "history_hits": len(g), "time_weight_excluded": True}


def _plain_matcher(tracker):
    """Never deepcopy a bound gate closure that could mutate its live owner."""
    from botsort_plus.recovery import RecoveryMatcher
    source = tracker.recovery_matcher
    result = RecoveryMatcher(source.config, source.geometry_config)
    result._first_reserved = deepcopy(source._first_reserved)
    result._last_frame = int(source._last_frame)
    return result


def _app(tracker):
    from botsort_plus.appearance import appearance_distance
    config = tracker.plus.appearance
    def cost(memory, packet):
        if not config.enabled:
            packet = {k: v for k, v in packet.items() if k == "rgb"}
        return appearance_distance(memory, packet, config)
    return cost


def native_counterfactual(tracker, saved, warp):
    """Full native high/low/unconfirmed allocation, followed by private B.

    Birth/final output is deliberately excluded: no public ID is allocated.
    This is a same-prior, single-frame intervention, not a closed-loop method.
    """
    from boxmot.trackers.botsort.basetrack import TrackState
    from boxmot.trackers.botsort.botsort_track import STrack
    from boxmot.trackers.botsort.botsort_utils import joint_stracks
    from boxmot.utils.matching import iou_distance, linear_assignment
    from botsort_plus.types import Observation, Prediction, LostTrackView

    private = copy(tracker)
    private.memories = deepcopy(saved["memories"])
    private._packets = deepcopy(saved["packets"])
    private.last_events = []
    private.prediction_history = deepcopy(saved["prediction_history"])
    private.candidate_pool = deepcopy(saved["candidate_pool"])
    private.recovery_matcher = _plain_matcher(tracker)
    # Capture the pre-native matcher state, even when called later in the frame.
    private.recovery_matcher._first_reserved = deepcopy(saved["reserved"])
    private.recovery_matcher._last_frame = saved["matcher_frame"]
    pool, unconfirmed, high = deepcopy((saved["pool"], saved["unconfirmed"], saved["high"]))
    by_id = {int(t.id): t for t in pool}
    native_lost = [by_id[i] for i in saved["native_lost_ids"] if i in by_id]
    frame = int(tracker.frame_count)
    _advance(pool, warp)
    if unconfirmed:
        STrack.multi_gmc(unconfirmed, np.asarray(warp).copy())
    for track in pool:
        _, cov = private.kalman_filter.project(track.mean.copy(), track.covariance.copy())
        history = private.prediction_history.setdefault(int(track.id), {})
        history[frame] = Prediction(frame, track.xyxy.copy(), cov[:2, :2].copy())
        for old in list(history):
            if old <= frame - private.plus.recovery.window:
                del history[old]
    activated, refound, lost = [], [], []
    stages = []

    def associate(tracks, detections, threshold, stage, unconfirmed=False):
        # Root permits this only for the plain reference _cost implementation.
        costs = private._cost(tracks, detections, unconfirmed=unconfirmed)
        pairs, ut, ud = linear_assignment(costs, thresh=threshold)
        stages.append({"stage": stage, "track_ids": [int(t.id) for t in tracks],
                       "det_indices": [int(d.det_ind) for d in detections],
                       "costs": costs.copy(), "threshold": threshold,
                       "pairs": [(int(tracks[i].id), int(detections[j].det_ind)) for i, j in pairs]})
        for i, j in pairs:
            track, detection = tracks[int(i)], detections[int(j)]
            if track.state == TrackState.Tracked:
                track.update(detection, frame)
                activated.append(track)
            else:
                track.re_activate(detection, frame, new_id=False)
                refound.append(track)
            private._write_memory(track, int(detection.det_ind))
        return list(ut), list(ud)

    up, uh = associate(pool, high, private.match_thresh, "high")
    low_tracks = [pool[i] for i in up if pool[i].state == TrackState.Tracked]
    low = [STrack(row.copy(), max_obs=private.max_obs) for row in saved["low_rows"]]
    costs = iou_distance(low_tracks, low)
    pairs, ut, ud = linear_assignment(costs, thresh=.5)
    stages.append({"stage": "low", "track_ids": [int(t.id) for t in low_tracks],
                   "det_indices": [int(d.det_ind) for d in low], "costs": costs.copy(),
                   "threshold": .5,
                   "pairs": [(int(low_tracks[i].id), int(low[j].det_ind)) for i, j in pairs]})
    for i, j in pairs:
        track, detection = low_tracks[int(i)], low[int(j)]
        track.update(detection, frame)
        private._write_memory(track, int(detection.det_ind))
        activated.append(track)
    for i in ut:
        track = low_tracks[int(i)]
        track.mark_lost()
        lost.append(track)
    remaining = [high[i] for i in uh]
    uu, remaining_indices = associate(unconfirmed, remaining, .7, "unconfirmed", True)
    for i in uu:
        unconfirmed[int(i)].mark_removed()
    birth_high = [remaining[i] for i in remaining_indices]
    residual = [low[int(i)] for i in ud] + birth_high
    observations = [Observation(frame, int(d.det_ind), saved["all_dets"][int(d.det_ind), :4],
                                float(d.conf), private._packets[int(d.det_ind)]) for d in residual]
    occupied_tracks = joint_stracks(activated, refound)
    occupied = [t.xyxy.copy() for t in occupied_tracks]
    occupied += [saved["all_dets"][int(t.det_ind), :4].copy() for t in occupied_tracks]
    before = set(private.candidate_pool.candidates)
    current = private.candidate_pool.update(frame, observations, warp=warp, occupied_boxes=occupied)
    private.recovery_matcher.forget(before - set(private.candidate_pool.candidates))
    eligible = {int(t.id): t for t in joint_stracks(native_lost, lost)
                if t.state == TrackState.Lost and frame - t.end_frame <= private.max_time_lost}
    views = [LostTrackView(t.id, t.end_frame, private.prediction_history.get(t.id, {}),
                          private.memories.get(t.id)) for t in eligible.values()]
    decision = private.recovery_matcher.match(frame, views, current, _app(private))
    return {"stages": stages, "residual_indices": [int(o.det_index) for o in observations],
            "occupied": occupied, "eligible_lost_ids": list(eligible),
            "candidate_members": {int(c.candidate_id): members(c) for c in current},
            "recovery_pairs": decision_pairs(decision), "recovery_diagnostics": decision.diagnostics,
            "reserved_candidate_ids": sorted(decision.reserved_candidate_ids)}


class JointProbe:
    """Observer attached through explicit adapter boundaries; all I/O is new."""
    def __init__(self, output_path, *, mode="full", native_probe=True, candidate_probe=True):
        if mode not in ("full", "signals"):
            raise ValueError("probe mode must be full or signals")
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if self.path.suffix == ".gz" else open
        self.stream = opener(self.path, "xt", encoding="utf-8")
        self.native_probe = bool(native_probe) and mode == "full"
        self.candidate_probe = bool(candidate_probe) and mode == "full"
        self.mode = mode
        self.counts = Counter()
        self.p_states = {"native": {}, "selected": {}}
        self.p_history = {"native": {}, "selected": {}}
        self.anchors = {}
        self.c_pools = None
        self.frame = 0
        self.closed = False
        self.error = None
        self._emit("contract", schema=SCHEMA, mode=mode,
                   prediction_scope="common actual observation posterior; forward until next actual observation",
                   native_scope="single-frame same-prior native allocation and private B; no births or outputs",
                   candidate_scope="same actual residual/occupied schedule; actual consumption by observation key",
                   identity_scope="same geometry/appearance score; excludes age and Lost-only eligibility",
                   missing_cost="null", reads_gt=False, affects_reference=False)

    def _emit(self, event, **fields):
        self.counts[event] += 1
        self.stream.write(json.dumps(json_value({"event": event, "frame": self.frame, **fields}),
                                     ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")

    def _check(self, label, condition):
        self.counts["parity_checks"] += 1
        if not condition:
            self.counts["parity_failures"] += 1
            self._emit("parity_failure", check=label)
            self.stream.flush()
            raise RuntimeError(f"J3 read-only reference parity failed: {label} at frame {self.frame}")

    def point(self, stage, tracker, **kwargs):
        if self.closed:
            raise RuntimeError("J3 probe is closed")
        method = getattr(self, "_" + stage, None)
        if method is None:
            raise ValueError(f"Unknown J3 boundary: {stage}")
        try:
            return method(tracker, **kwargs)
        except Exception as error:
            self.error = {"stage": stage, "type": type(error).__name__, "message": str(error)}
            self._emit("probe_error", **self.error)
            self.stream.flush()
            raise

    def _begin(self, tracker, pool, unconfirmed, high_detections, low_rows, all_dets):
        if int(tracker.frame_count) != self.frame + 1:
            raise ValueError("J3 requires every frame from frame 1, including empty frames")
        self.frame = int(tracker.frame_count)
        plus = tracker.plus
        if (plus.adapter_mode != "v2" or not plus.recovery.enabled or plus.birth.enabled or
                plus.appearance.enabled or plus.identity.enabled or plus.raw_ir.enabled or
                plus.recovery.output_prediction_frames):
            raise ValueError("J3 supports frozen v2 B only; disable A/C/C2/IR/extra outputs")
        self.extra_gate = bool(getattr(tracker, "joint_evidence_gate", False) or
                               getattr(tracker, "joint_evidence_both", False))
        self.n_available = self.native_probe and not self.extra_gate and not bool(
            getattr(tracker, "joint_dual_low", False))
        self.saved = {"pool": deepcopy(pool), "unconfirmed": deepcopy(unconfirmed),
                      "high": deepcopy(high_detections), "low_rows": np.asarray(low_rows).copy(),
                      "all_dets": np.asarray(all_dets).copy(), "packets": deepcopy(tracker._packets),
                      "memories": deepcopy(tracker.memories),
                      "prediction_history": deepcopy(tracker.prediction_history),
                      "candidate_pool": deepcopy(tracker.candidate_pool),
                      "native_lost_ids": [int(t.id) for t in tracker.lost_stracks],
                      "reserved": deepcopy(tracker.recovery_matcher._first_reserved),
                      "matcher_frame": int(tracker.recovery_matcher._last_frame)}
        self.n_results = {}
        self.current = []
        self.c_current = {}
        self.c_scores = {}
        self.p_scores = {}
        self.live_current_keys = {}
        for track in pool:
            tid = int(track.id)
            for branch in self.p_states:
                if tid not in self.p_states[branch]:
                    self.p_states[branch][tid] = deepcopy(track)
                    # A late attached identity is anchored here, never fabricated backwards.
                    self.anchors[tid] = self.frame - 1
                self.p_states[branch][tid].state = track.state
        if self.candidate_probe and self.c_pools is None:
            self.c_pools = {name: deepcopy(tracker.candidate_pool) for name in ("native", "selected")}

    def _compensated(self, tracker, pool, unconfirmed, warp, native_warp=None):
        from botsort_plus.types import Prediction
        native = native_warp if native_warp is not None else getattr(tracker, "joint_native_warp", None)
        if native is None:
            raise ValueError("J3 requires the actual same-frame native warp, not an inverse approximation")
        self.warps = {"native": np.asarray(native).copy(), "selected": np.asarray(warp).copy()}
        for transform in self.warps.values():
            if transform.shape != (2, 3) or not np.isfinite(transform).all():
                raise ValueError("J3 received an invalid warp")
        ids = [int(t.id) for t in pool]
        for branch in self.p_states:
            tracks = [self.p_states[branch][tid] for tid in ids]
            _advance(tracks, self.warps[branch])
            for track in tracks:
                _, cov = tracker.kalman_filter.project(track.mean.copy(), track.covariance.copy())
                history = self.p_history[branch].setdefault(int(track.id), {})
                history[self.frame] = Prediction(self.frame, track.xyxy.copy(), cov[:2, :2].copy())
                for old in list(history):
                    if old <= self.frame - tracker.plus.recovery.window:
                        del history[old]
        for track in pool:
            tid = int(track.id)
            selected = self.p_states["selected"][tid]
            self._check("P_selected_mean", np.array_equal(selected.mean, track.mean))
            self._check("P_selected_covariance", np.array_equal(selected.covariance, track.covariance))
            diff = prediction_disagreement(track.xyxy, self.p_states["native"][tid].xyxy,
                                           tracker.plus.geometry.pixel_floor)
            self._emit("prediction_state", track_id=tid, anchor_frame=self.anchors[tid],
                       last_seen=int(track.end_frame), state=int(track.state),
                       selected_box=track.xyxy.copy(), native_box=self.p_states["native"][tid].xyxy.copy(),
                       **diff)
        if self.n_available:
            for branch, transform in self.warps.items():
                result = native_counterfactual(tracker, self.saved, transform)
                self.n_results[branch] = result
                self._emit("native_counterfactual", branch=branch, **result)
        elif self.native_probe:
            self._emit("native_counterfactual_unavailable", reason=(
                "extra_evidence_gate" if self.extra_gate else "dual_low_enabled"))

    def _candidates_before(self, tracker, observations, occupied, warp):
        self.residual_keys = [(int(o.frame), int(o.det_index)) for o in observations]
        self._emit("candidate_input", residual_keys=self.residual_keys,
                   occupied_boxes=[np.asarray(b).copy() for b in occupied])
        if self.n_available:
            actual = sorted((e["stage"], int(e["track_id"]), int(e["det_index"]))
                            for e in tracker.last_events if e.get("event") == "match" and
                            e.get("stage") in ("high", "low", "unconfirmed"))
            expected = sorted((s["stage"], tid, did) for s in self.n_results["selected"]["stages"]
                              for tid, did in s["pairs"])
            self._check("N_selected_native_matches", expected == actual)
            self._check("N_selected_residual", self.n_results["selected"]["residual_indices"] ==
                        [int(o.det_index) for o in observations])
            self._check("N_selected_occupied", _same(self.n_results["selected"]["occupied"], occupied))
        if self.c_pools is not None:
            for branch, candidate_pool in self.c_pools.items():
                self.c_current[branch] = candidate_pool.update(
                    self.frame, deepcopy(observations), warp=self.warps[branch].copy(),
                    occupied_boxes=deepcopy(occupied))
                self._emit("candidate_counterfactual", branch=branch,
                           candidates={int(c.candidate_id): members(c) for c in self.c_current[branch]},
                           decisions=deepcopy(candidate_pool.events))

    def _candidates_after(self, tracker, current_candidates):
        self.current = deepcopy(current_candidates)
        self.live_current_keys = {int(c.candidate_id): (int(c.latest.frame), int(c.latest.det_index))
                                  for c in current_candidates}
        if self.c_pools is not None:
            self._check("C_selected_pool", _same(pool_snapshot(self.c_pools["selected"]),
                                                pool_snapshot(tracker.candidate_pool)))

    def _recovery_before(self, tracker, views, current_candidates, app_cost):
        from botsort_plus.types import LostTrackView
        # Memory is frozen before any native association; legal Lost views have
        # not received a current observation. Explicitly check this assumption.
        self.views = deepcopy(views)
        self.frozen_matcher = _plain_matcher(tracker)
        if not self.extra_gate:
            reference = self.frozen_matcher.match(self.frame, deepcopy(views), deepcopy(current_candidates), _app(tracker))
            self.p_scores["reference"] = reference
            for branch in ("native", "selected"):
                branch_views = [LostTrackView(int(v.track_id), int(v.last_seen),
                                             deepcopy(self.p_history[branch].get(int(v.track_id), {})),
                                             deepcopy(self.saved["memories"].get(int(v.track_id)))) for v in views]
                decision = _plain_matcher(tracker).match(self.frame, branch_views,
                                                        deepcopy(current_candidates), _app(tracker))
                self.p_scores[branch] = decision
                self._emit("prediction_recovery", branch=branch, pairs=decision_pairs(decision),
                           diagnostics=decision.diagnostics,
                           reserved_candidate_ids=sorted(decision.reserved_candidate_ids))
            self._check("P_selected_recovery", _same(decision_pairs(reference),
                                                     decision_pairs(self.p_scores["selected"])))
            if self.c_pools is not None:
                for branch, candidates in self.c_current.items():
                    decision = _plain_matcher(tracker).match(self.frame, deepcopy(views),
                                                            deepcopy(candidates), _app(tracker))
                    self.c_scores[branch] = decision
                    self._emit("candidate_recovery", branch=branch, pairs=decision_pairs(decision),
                               diagnostics=decision.diagnostics,
                               reservation_interpretation="not a cross-branch budget comparison")
                self._check("C_selected_recovery", _same(decision_pairs(reference),
                                                         decision_pairs(self.c_scores["selected"])))
        else:
            self._emit("recovery_counterfactual_unavailable", reason="extra_evidence_gate_passive_only")

    def _recovery_after(self, tracker, decision):
        from botsort_plus.recovery import RecoveryMatcher
        self._emit("actual_recovery", pairs=decision_pairs(decision), diagnostics=deepcopy(decision.diagnostics),
                   current_candidates={int(c.candidate_id): members(c) for c in self.current},
                   reserved_candidate_ids=sorted(decision.reserved_candidate_ids))
        if "reference" in self.p_scores:
            self._check("B_reference_decision", _same(decision_pairs(decision),
                                                     decision_pairs(self.p_scores["reference"])))
            self._check("B_reference_diagnostics", _same(decision.diagnostics,
                                                          self.p_scores["reference"].diagnostics))
        if self.n_available:
            self._check("N_selected_B", _same(decision_pairs(decision),
                                               self.n_results["selected"]["recovery_pairs"]))
        active_ids = {int(e["track_id"]) for e in tracker.last_events if e.get("event") == "match"}
        lost_ids = {int(v.track_id) for v in self.views}
        nominated = {int(d["candidate_id"]) for d in decision.diagnostics
                     if d.get("reason") in ("accepted", "plausible", "ambiguous")}
        for candidate in self.current:
            if int(candidate.candidate_id) not in nominated:
                continue
            history = RecoveryMatcher._history(candidate, self.frame, tracker.plus.recovery.window)
            scores = []
            for tid in sorted(active_ids | lost_ids):
                support = common_identity_support(
                    history, self.p_history["selected"].get(tid, {}), self.saved["memories"].get(tid),
                    tracker.plus.geometry, tracker.plus.appearance, tracker.plus.recovery)
                scores.append({"track_id": tid, "role": "active" if tid in active_ids else "lost", **support})
            self._emit("identity_competition", candidate_id=int(candidate.candidate_id),
                       current_det_index=int(candidate.latest.det_index), history=members(candidate),
                       scores=scores, memory_scope="before_native_association")

    def _end(self, tracker, output):
        # External consumption follows actual accepted observation keys, not
        # private candidate IDs, which can differ after counterfactual linking.
        consumed = {self.live_current_keys[int(e["candidate_id"])] for e in tracker.last_events
                    if e.get("event") == "candidate_removed" and
                    e.get("reason") in ("recovered", "native_birth") and
                    int(e["candidate_id"]) in self.live_current_keys}
        if self.c_pools is not None:
            for branch, candidate_pool in self.c_pools.items():
                for cid, candidate in list(candidate_pool.candidates.items()):
                    if (int(candidate.latest.frame), int(candidate.latest.det_index)) in consumed:
                        candidate_pool.discard(cid, "frozen_reference_consumption")
            self._check("C_selected_end_pool", _same(pool_snapshot(self.c_pools["selected"]),
                                                    pool_snapshot(tracker.candidate_pool)))
        tracks = {int(t.id): t for t in tracker.active_tracks + tracker.lost_stracks}
        for tid, track in tracks.items():
            if int(track.end_frame) == self.frame:
                self.anchors[tid] = self.frame
                for branch in self.p_states:
                    self.p_states[branch][tid] = deepcopy(track)
            else:
                for branch in self.p_states:
                    if tid in self.p_states[branch]:
                        self.p_states[branch][tid].state = track.state
        for branch in self.p_states:
            for tid in list(self.p_states[branch]):
                if tid not in tracks:
                    del self.p_states[branch][tid]
                    self.p_history[branch].pop(tid, None)
        self.anchors = {tid: frame for tid, frame in self.anchors.items() if tid in tracks}
        self._emit("frame_end", outputs=np.asarray(output).copy(),
                   reference_events=deepcopy(tracker.last_events),
                   frozen_consumption_keys=sorted(consumed))

    def close(self):
        if not self.closed:
            self.stream.flush()
            self.stream.close()
            self.closed = True
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        summary = {"schema": SCHEMA, "frames": self.frame, "counts": dict(self.counts),
                   "data_sha256": digest.hexdigest(),
                   "status": "PASS" if self.counts["parity_failures"] == 0 and self.error is None else "FAIL",
                   "scope": "probe internal parity; MOT byte parity is checked by batch runner",
                   "mode": self.mode, "gt_used": False, "error": self.error}
        summary_path = self.path.with_name(self.path.name + ".summary.json")
        if not summary_path.exists():
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return summary
