"""W0 anchor-support and state-feedback ablations; inference never reads GT.

The published R4.1 off/W0 implementations are delegated unchanged.  The
emit-only arm runs that same W0 proposal/measurement transaction on detached
track lists, publishes its output AFTER native adapter finalization, and
keeps the canonical tracker on the off trajectory.  Its states/ stream is
therefore explicitly based on the native output; the additional R4.2 audit
records both native and published-output state digests.
"""
from __future__ import annotations

from collections import Counter
from copy import copy, deepcopy
import hashlib

import numpy as np

from capture import current_assignments, encode, fingerprint, track_state
from score_features import groups_and_margins, overlap
from tools.research.wem_r31.engine import edge_key, evaluate_frame
from tools.research.wem_r41.online import (
    ClosedLoop as R41ClosedLoop, audit_row, box, compare_decisions,
    decision_bundle, transact,
)


RELAXED = "W0_relaxed_anchor"
EMIT_ONLY = "W0_emit_only"
ALL_MODES = ("off", "W0", RELAXED, EMIT_ONLY)
RGB_MODES = frozenset()
_BYPASS_FIELDS = ("anchor_not_ready", "anchor_prediction_mismatch", "anchor_box_mismatch")


def canonical_snapshot(tracker, gmc):
    """Include all existing R0 state fields and diagnostic origin bookkeeping."""
    result = track_state(tracker, np.empty((0, 8)), gmc)
    result["track_origins"] = getattr(tracker, "track_origins", None)
    return encode(result)


def detached_finalization(tracker, activated, refound, lost, removed):
    """Preserve shared identity between copied lists, without copying GPU models.

The pinned finalizer works on these lists.  Copy their track graph too, so
even an in-place list/track operation cannot affect the official tracker.
Other structures are read-only in the unchanged W0 transaction; the caller
checks the complete canonical snapshot around the preview.
"""
    proxy = copy(tracker)
    copied = deepcopy((tracker.active_tracks, tracker.lost_stracks, tracker.removed_stracks,
                       activated, refound, lost, removed))
    proxy.active_tracks, proxy.lost_stracks, proxy.removed_stracks = copied[:3]
    return proxy, copied[3:]


class ClosedLoop(R41ClosedLoop):
    def __init__(self, path, seq, method, output, protocol, mode, settings, encoder):
        if mode not in ALL_MODES:
            raise ValueError(f"Unknown R4.2 mode: {mode}")
        super().__init__(path, seq, method, output, protocol,
                         mode if mode in ("off", "W0") else "W0", settings, encoder)
        self.r42_mode = mode
        self.extra_log = None
        if mode in (RELAXED, EMIT_ONLY):
            self.variant = self.mode = mode
            self.policy = "W0"
            self.extra_log = self._audit_stream(output, "wem_r42_audit")
            self.runtime_counts.update(weak_outputs_emitted=0, weak_state_updates=0)
            if mode == RELAXED:
                self.runtime_counts.update({"exposure_" + name: 0 for name in (
                    "anchor_not_ready_bypassed_tracks", "anchor_prediction_bypassed_tracks",
                    "anchor_box_bypassed_edges", "anchor_support_bypassed_edges",
                    "anchor_guard_added_selected", "anchor_guard_removed_selected",
                    "anchor_guard_changed_track_frames", "anchor_guard_added_eligible",
                    "anchor_guard_removed_eligible",
                )})
                self.runtime_counts.update(anchor_support_bypassed_selected=0,
                                           anchor_support_bypassed_committed=0)

    def write_event(self, event, **fields):
        # Keep off/W0 event bytes identical to the already archived runner.
        if self.r42_mode in (RELAXED, EMIT_ONLY) and event == "committed":
            state_committed = self.r42_mode != EMIT_ONLY
            fields.update(state_committed=state_committed, output_only=not state_committed)
            self.runtime_counts["weak_outputs_emitted"] += 1
            self.runtime_counts["weak_state_updates"] += int(state_committed)
            self._frame_weak_outputs += 1
        return super().write_event(event, **fields)

    def finish_before_output(self, tracker, activated, refound, lost, removed, reserved, consumed, all_dets):
        mode = self.r42_mode
        if mode in ("off", "W0"):
            return super().finish_before_output(tracker, activated, refound, lost, removed,
                                                reserved, consumed, all_dets)
        self._frame_weak_outputs = 0
        if mode == RELAXED:
            return self._relaxed_finish(tracker, activated, refound, lost, removed,
                                        reserved, consumed, all_dets)

        before = canonical_snapshot(tracker, self.gmc)
        proxy, private_lists = detached_finalization(tracker, activated, refound, lost, removed)
        # Reuse the exact strict W0 qualifications, scoring, assignment, native
        # re_activate(new_id=False), and duplicate-removal preflight.
        emitted = super().finish_before_output(proxy, *private_lists, reserved, consumed, all_dets)
        if canonical_snapshot(tracker, self.gmc) != before:
            raise ValueError("Emit-only private preview changed canonical tracking state")
        native = tracker._prepare_output(activated, refound, lost, removed).reshape(-1, 8)
        self._pending_native = native.copy()
        self._pending_emitted = emitted.copy()
        self._pending_frame = self.frame
        # Adapter bookkeeping and memory/history cleanup must see native output.
        return native

    def publish(self, output):
        """Call after super().update(), before observer.finish()/output_index."""
        if self.r42_mode != EMIT_ONLY:
            return output
        if getattr(self, "_pending_frame", None) != self.frame:
            raise ValueError("Emit-only publish did not follow this frame's native finalization")
        if getattr(self, "_published_frame", None) == self.frame:
            raise ValueError("Emit-only output was published twice")
        if not np.array_equal(output, self._pending_native):
            raise ValueError("Post-finalizer output changed; emit-only protocol is not applicable")
        self._published_frame = self.frame
        return self._pending_emitted

    def finish(self, tracker, output):
        if self.r42_mode in ("off", "W0"):
            return super().finish(tracker, output)
        native = output
        basis = "published_output"
        if self.r42_mode == EMIT_ONLY:
            if getattr(self, "_published_frame", None) != self.frame:
                raise ValueError("Emit-only output must be published before finish")
            native = self._pending_native
            basis = "canonical_native_output"
        canonical = hashlib.sha256(encode(track_state(tracker, native, self.gmc))).hexdigest()
        emitted = hashlib.sha256(encode(track_state(tracker, output, self.gmc))).hexdigest()
        self.extra_log.write(encode(dict(
            schema="wem_r42_frame_audit_v1", sequence=self.seq, frame=self.frame,
            online_arm=self.r42_mode, state_chain_output_basis=basis,
            canonical_state_sha256=canonical, emitted_state_sha256=emitted,
            canonical_output=fingerprint(native), emitted_output=fingerprint(output),
            canonical_output_rows=len(native), emitted_output_rows=len(output),
            weak_outputs_emitted=self._frame_weak_outputs,
            weak_state_updates=0 if self.r42_mode == EMIT_ONLY else self._frame_weak_outputs,
        )) + b"\n")
        # This existing stream can now be compared byte-for-byte with R0/off
        # for emit-only, despite a different published MOT output.
        return super().finish(tracker, native)

    def close(self):
        if self.extra_log is not None:
            self.extra_log.close()
        return super().close()

    def _relaxed_finish(self, tracker, activated, refound, lost, removed, reserved, consumed, all_dets):
        """Remove the declared support bundle, retaining freshness and budget.

Removed: ready, anchor-versus-formal IoU, and weak-versus-anchor IoU.
Retained: an existing ordinary anchor observed exactly at t-1, unspent
budget, native/B priority, selected/backup geometry, top-5 and one-to-one,
finalization compatibility, original ID/KF and appearance restrictions.
"""
        if tracker.plus.birth.enabled or tracker.plus.identity.enabled:
            raise ValueError("R4.2 requires the frozen B or BG foundation")
        accepted = [dict(**self.detections[idx], track_id=tid, frame=self.frame, stage=stage)
                    for tid, idx, stage in current_assignments(tracker.last_events)]
        owned = {item["det_index"] for item in accepted} | set(consumed)
        protected = {int(candidate.latest.det_index)
                     for cid, candidate in tracker.candidate_pool.candidates.items()
                     if cid in reserved and int(candidate.latest.frame) == self.frame}
        priors = {item["track_id"]: item for item in self.post_gmc}
        lost_ids = {int(track.id) for track in lost if int(track.state) == 2}
        accepted_ids = {item["track_id"] for item in accepted}
        missing = [item["track_id"] for item in self.start_tracks
                   if item["collection"] == "active_tracks" and item["confirmed"] and item["state"] == 1
                   and item["track_id"] in lost_ids and item["track_id"] not in accepted_ids]
        exposure = Counter(missing_confirmed_tracks=len(missing))
        eligible, histories, flags_by_track = [], [], {}
        start = {item["track_id"]: item for item in self.start_tracks}
        for tid in missing:
            anchor = self.anchors.get(tid)
            if anchor is None or anchor.spent or anchor.observed != self.frame - 1:
                exposure["anchor_ineligible"] += 1
                continue
            flags = dict(anchor_not_ready=not bool(anchor.ready),
                         anchor_prediction_mismatch=overlap(box(anchor.mean), np.asarray(priors[tid]["selected_xyxy"])) < .5)
            exposure["anchor_not_ready_bypassed_tracks"] += int(flags["anchor_not_ready"])
            exposure["anchor_prediction_bypassed_tracks"] += int(flags["anchor_prediction_mismatch"])
            flags_by_track[tid] = flags
            eligible.append(tid)
            histories.append(dict(track_id=tid, observed=anchor.observed,
                                  history_frames=[int(row["frame"]) for row in start[tid].get("raw_history", ())],
                                  ready=bool(anchor.ready), spent=bool(anchor.spent)))
        exposure["anchor_eligible_tracks"] = len(eligible)
        edges, flags_by_edge = [], {}
        for tid in eligible:
            prior, anchor_box = priors[tid], box(self.anchors[tid].mean)
            for candidate in self.detections:
                idx = candidate["det_index"]
                if candidate["band"] != "S0" or not .05 <= candidate["raw_score"] <= .1:
                    continue
                exposure["s0_edges_before_ownership"] += 1
                if idx in owned or idx in protected:
                    exposure["ownership_reject"] += 1
                    continue
                raw = np.asarray(candidate["raw_xyxy"])
                geom = overlap(raw, np.asarray(prior["selected_xyxy"]))
                if geom < .5 or (self.gmc["backup_guard_active"] and
                                overlap(raw, np.asarray(prior["backup_xyxy"])) < .5):
                    exposure["geometry_reject"] += 1
                    continue
                mismatch = overlap(raw, anchor_box) < .5
                exposure["anchor_box_bypassed_edges"] += int(mismatch)
                if any(overlap(raw, np.asarray(self.detections[index]["raw_xyxy"])) >= .90
                       for index in owned | protected):
                    exposure["near_owner_reject"] += 1
                    continue
                flags = dict(flags_by_track[tid], anchor_box_mismatch=mismatch)
                flags["anchor_support_bypassed"] = any(flags.values())
                flags_by_edge[tid, idx] = flags
                exposure["anchor_support_bypassed_edges"] += int(flags["anchor_support_bypassed"])
                edges.append(dict(track_id=tid, det_index=idx, native_low_feasible=True,
                                  gmc_guard_feasible=True, selected_iou=geom))
        exposure["common_candidate_edges"] = len(edges)
        shot = dict(frame=self.frame, sequence=self.seq, reference=self.method, image=self.image,
                    detections=self.detections, s0_edges=edges, post_gmc=self.post_gmc,
                    accepted=accepted, start_tracks=self.start_tracks,
                    consumed=sorted(owned), reserved=sorted(protected))
        scores = self._scores(shot)
        for row in scores:
            row.update(flags_by_edge[row["track_id"], row["det_index"]])
        decisions = decision_bundle(scores, False)
        # A same-current-state strict-anchor counterfactual. Rerank AFTER
        # filtering, because strict W0 would never have seen bypassed edges.
        strict_scores = deepcopy([row for row in scores if not row["anchor_support_bypassed"]])
        groups_and_margins(strict_scores)
        strict = evaluate_frame(strict_scores, "W0")
        change = compare_decisions(strict, decisions["W0"])
        for key in ("added_selected", "removed_selected", "changed_track_frames", "added_eligible", "removed_eligible"):
            exposure["anchor_guard_" + key] += change[key]
        self._audit_frame(scores, decisions, exposure, histories)
        by_key = {edge_key(row): row for row in scores}
        selected = [dict(by_key[edge_key(row)], utility=row["utility"], decision_policy="W0")
                    for row in decisions["W0"] if row["selected"]]
        self.runtime_counts["anchor_support_bypassed_selected"] += sum(row["anchor_support_bypassed"] for row in selected)
        for row in selected:
            self.write_event("proposed", **audit_row(row), **{key: row[key] for key in (*_BYPASS_FIELDS, "anchor_support_bypassed")})
        output, committed, rejected = transact(tracker, activated, refound, lost, removed, selected, all_dets, measurement=True)
        for row in rejected:
            self.write_event("rejected_finalization", **audit_row(row))
        for row in committed:
            tid = row["track_id"]
            self.anchors[tid].spent = True
            emitted, = output[output[:, 4] == tid]
            self.runtime_counts["anchor_support_bypassed_committed"] += int(row["anchor_support_bypassed"])
            self.write_event("committed", **audit_row(row), prediction=False,
                             output_box=emitted[:4].tolist(),
                             **{key: row[key] for key in (*_BYPASS_FIELDS, "anchor_support_bypassed")})
        self._update_anchors(tracker, accepted)
        return output
