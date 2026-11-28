"""Online W0--W3 policies with the unchanged R4 outer continuation rules.

No labels are read. Decisions use each arm's current state. Counterfactual
decisions are read-only comparisons within that state; only the selected
policy commits. The runner adds the legacy R3 observer/scorer to sys.path.
"""
from __future__ import annotations

from collections import Counter, deque
import gzip

import numpy as np

from capture import current_assignments, encode
from score_features import center, evidence, frame_features, groups_and_margins, overlap
from tools.research.vttiny_wem_r4.online import (
    Anchor, ClosedLoop as R4ClosedLoop, box, transact,
)
from tools.research.wem_r31.engine import edge_key, evaluate_frame


POLICIES = ("W0", "W1", "W2", "W3")
CONTROL_MODES = ("off", "shadow", "force_reject")
RGB_MODES = frozenset(("W2", "W3", "shadow", "force_reject"))
ALL_MODES = (*CONTROL_MODES, *POLICIES)
SHADOW_POLICY = "W3"


def motion_context(history, frame, warps):
    """The R3 GMC-aligned trend without loading or extracting RGB vectors."""
    observations = [item for item in history if 1 <= frame - item["frame"] <= 5]
    if len({item["frame"] for item in observations}) != len(observations):
        raise ValueError("Repeated history frame")
    trend = None
    if len(observations) >= 2:
        points, times = [], []
        for item in observations:
            time = item["frame"]
            pos = center(np.asarray(item["raw_xyxy"], dtype=np.float64))
            for current in range(time + 1, frame + 1):
                warp = warps[current]
                pos = warp[:, :2] @ pos + warp[:, 2]
            points.append(pos)
            times.append(time - frame)
        trend = np.linalg.lstsq(
            np.column_stack((np.ones(len(times)), times)), np.stack(points), rcond=None,
        )[0][0]
    return trend, observations


def history_only_decisions(rows):
    """Read-only W_H: history available, original ranks and IoU utility."""
    decisions = {
        edge_key(row): row for row in evaluate_frame(
            [row for row in rows if row.get("motion") is not None], "W0"
        )
    }
    result = []
    for row in evaluate_frame(rows, "W0"):
        if edge_key(row) in decisions:
            row = decisions[edge_key(row)]
        else:
            row.update(eligible=False, selected=False, decision="rejected")
            row["reasons"].append("motion_unavailable")
        row["method"] = "W_H"
        result.append(row)
    return result


def compare_decisions(before, after):
    old = {edge_key(row) for row in before if row["selected"]}
    new = {edge_key(row) for row in after if row["selected"]}
    old_allowed = {edge_key(row) for row in before if row["eligible"]}
    new_allowed = {edge_key(row) for row in after if row["eligible"]}
    return dict(
        added_selected=len(new - old), removed_selected=len(old - new),
        changed_track_frames=len({key[:3] for key in old ^ new}),
        added_eligible=len(new_allowed - old_allowed),
        removed_eligible=len(old_allowed - new_allowed),
        same_selected=old == new, same_eligible=old_allowed == new_allowed,
    )


def decision_bundle(rows, with_rgb):
    """Alternative policies on one arm's same pre-update frame."""
    result = {"W0": evaluate_frame(rows, "W0"),
              "W_H": history_only_decisions(rows),
              "W1": evaluate_frame(rows, "W1")}
    if with_rgb:
        result.update(W2=evaluate_frame(rows, "W2"), W3=evaluate_frame(rows, "W3"))
        a = {edge_key(row) for row in result["W1"] if row["eligible"]}
        b = {edge_key(row) for row in result["W2"] if row["eligible"]}
        c = {edge_key(row) for row in result["W3"] if row["eligible"]}
        if a != b or not c <= b:
            raise ValueError("W1/W2/W3 qualification invariant differs")
    return result


def motion_scores(shot, cache_frame, warps):
    """Score the complete common pool without images, feature cache or GPU."""
    frame = shot["frame"]
    raw = np.asarray(cache_frame["detections"], dtype=np.float32).reshape(-1, 6).astype(np.float64)
    if int(cache_frame["frame"]) != frame or len(raw) != len(shot["detections"]):
        raise ValueError("Frozen detector frame differs")
    prior = {item["track_id"]: np.asarray(item["selected_xyxy"], dtype=np.float64)
             for item in shot["post_gmc"]}
    start = {item["track_id"]: item for item in shot["start_tracks"]
             if item["collection"] != "removed_stracks"}
    contexts, result = {}, []
    for edge in shot["s0_edges"]:
        tid, idx = edge["track_id"], edge["det_index"]
        candidate = shot["detections"][idx]
        if (candidate["band"] != "S0" or candidate["raw_xyxy"] != raw[idx, :4].tolist()
                or candidate["raw_score"] != raw[idx, 4]):
            raise ValueError("Weak observation is not its frozen detector row")
        if tid not in contexts:
            contexts[tid] = motion_context(start[tid].get("raw_history", ()), frame, warps)
        trend, observations = contexts[tid]
        parts = evidence(candidate, None, edge["selected_iou"], prior[tid], [], trend)
        result.append(dict(reference=shot["reference"], sequence=shot["sequence"], frame=frame,
                           track_id=tid, det_index=idx, raw_box_xyxy=candidate["raw_xyxy"],
                           raw_score=candidate["raw_score"], challengers=[],
                           previous_regular_observations=len(observations),
                           history_frames=[int(item["frame"]) for item in observations],
                           rgb_evaluated=False, column_unknown=False, column_margin=None,
                           **parts))
    groups_and_margins(result)
    return result


class ClosedLoop(R4ClosedLoop):
    """off/shadow/force_reject are no-commit controls; W0--W3 update tracks."""

    def __init__(self, path, seq, method, output, protocol, mode, settings, encoder):
        if mode not in ALL_MODES:
            raise ValueError(f"Unknown R4.1 mode: {mode}")
        # 'plain' avoids RGB loading for W0/W1. Lifecycle and native finalizer
        # are the archived R4 implementation; the policy is explicit below.
        legacy_mode = "off" if mode == "off" else "w_full" if mode in RGB_MODES else "plain"
        super().__init__(path, seq, method, output, protocol, legacy_mode, settings, encoder)
        self.variant = self.mode = mode
        self.policy = SHADOW_POLICY if mode in ("shadow", "force_reject") else mode
        self.with_rgb = mode in RGB_MODES
        self._raw_audit_streams = []
        self.score_log = self._audit_stream(output, "wem_scores")
        self.decision_log = self._audit_stream(output, "wem_decisions")
        self.frame_log = self._audit_stream(output, "wem_frame_audit")

    def _audit_stream(self, output, kind):
        raw = self.file(output, kind, ".jsonl.gz").open("xb")
        self._raw_audit_streams.append(raw)
        return gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=2, mtime=0)

    def _scores(self, shot):
        base = motion_scores(shot, self.rows[self.frame - 1], self.warps)
        if not self.with_rgb or not base:
            return base
        if self.encoder is None:
            raise ValueError("RGB policy requires the frozen RGB encoder")
        from botsort_plus.io import resolve_image

        source = str(resolve_image(self.image, self.images_root))
        rgb_shot = dict(shot, image=source)
        cache_frame = dict(self.rows[self.frame - 1], image=source)
        rgb_rows, vectors, _ = frame_features(
            rgb_shot, cache_frame, self.features, self.encoder, self.warps, self.current_image,
        )
        self.runtime_counts["rgb_crops"] += len(vectors)
        self.runtime_counts["rgb_valid_crops"] += sum(bool(row[-1]) for row in vectors)
        geometry = {edge_key(row): row for row in base}
        if len(rgb_rows) != len(base) or {edge_key(row) for row in rgb_rows} != set(geometry):
            raise ValueError("RGB extraction changed the shared weak candidate pool")
        for row in rgb_rows:
            before = geometry[edge_key(row)]
            for name in ("raw_box_xyxy", "raw_score", "native_iou", "normalized_score",
                         "motion", "geometry_group_rank", "previous_regular_observations"):
                if row[name] != before[name]:
                    raise ValueError(f"RGB extraction changed common score field: {name}")
            row.update(reference=self.method, history_frames=before["history_frames"], rgb_evaluated=True)
        return rgb_rows

    def _audit_frame(self, scores, decisions, exposure, histories):
        # Shadow and force-reject logs intentionally identify the same policy,
        # enabling byte parity while the arm name is carried by its directory.
        audit_mode = "read_only_W3" if self.variant in ("shadow", "force_reject") else self.variant
        for row in scores:
            self.score_log.write(encode(dict(row, online_arm=audit_mode,
                                             state_scope="current_arm_pre_update")) + b"\n")
        for rows in decisions.values():
            for row in rows:
                self.decision_log.write(encode(dict(row, online_arm=audit_mode,
                                                   state_scope="current_arm_pre_update")) + b"\n")
        exposure.update(scored_edges=len(scores),
                        motion_unavailable=sum(row["motion"] is None for row in scores),
                        motion_below_floor=sum(row["motion"] is not None and row["motion"] < .25 for row in scores),
                        insufficient_history=sum(row["previous_regular_observations"] < 2 for row in scores),
                        outside_top5=sum(row["geometry_group_rank"] > 5 for row in scores))
        pairs = [("W0", "W_H"), ("W_H", "W1"), ("W0", "W1")]
        if self.with_rgb:
            pairs += [("W1", "W2"), ("W2", "W3")]
        contrasts = {f"{a}_to_{b}": compare_decisions(decisions[a], decisions[b]) for a, b in pairs}
        self.frame_log.write(encode(dict(
            schema="wem_r41_frame_audit_v1", sequence=self.seq, frame=self.frame,
            online_arm=audit_mode, policy=self.policy, reference=self.method,
            rgb_evaluated=self.with_rgb, counts=exposure, anchor_histories=histories,
            comparisons=contrasts, counterfactuals_commit=False,
        )) + b"\n")
        for name, value in exposure.items():
            self.runtime_counts["exposure_" + name] += value
        for name, comparison in contrasts.items():
            for key in ("added_selected", "removed_selected", "changed_track_frames",
                        "added_eligible", "removed_eligible"):
                self.runtime_counts[f"compare_{name}_{key}"] += comparison[key]

    def finish_before_output(self, tracker, activated, refound, lost, removed, reserved, consumed, all_dets):
        if not self.enabled:
            return tracker._prepare_output(activated, refound, lost, removed).reshape(-1, 8)
        if tracker.plus.birth.enabled or tracker.plus.identity.enabled:
            raise ValueError("R4.1 requires the frozen B or BG foundation")
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
        eligible, histories = [], []
        start = {item["track_id"]: item for item in self.start_tracks}
        for tid in missing:
            anchor = self.anchors.get(tid)
            if anchor is None or not anchor.ready or anchor.spent or anchor.observed != self.frame - 1:
                exposure["anchor_ineligible"] += 1
                continue
            if overlap(box(anchor.mean), np.asarray(priors[tid]["selected_xyxy"])) < .5:
                exposure["anchor_diverged"] += 1
                continue
            eligible.append(tid)
            records = start[tid].get("raw_history", ())
            histories.append(dict(track_id=tid, observed=anchor.observed,
                                  history_frames=[int(row["frame"]) for row in records],
                                  ready=bool(anchor.ready), spent=bool(anchor.spent)))
        exposure["anchor_eligible_tracks"] = len(eligible)
        edges = []
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
                if overlap(raw, anchor_box) < .5:
                    exposure["anchor_box_reject"] += 1
                    continue
                if any(overlap(raw, np.asarray(self.detections[index]["raw_xyxy"])) >= .90
                       for index in owned | protected):
                    exposure["near_owner_reject"] += 1
                    continue
                edges.append(dict(track_id=tid, det_index=idx, native_low_feasible=True,
                                  gmc_guard_feasible=True, selected_iou=geom))
        exposure["common_candidate_edges"] = len(edges)
        shot = dict(frame=self.frame, sequence=self.seq, reference=self.method, image=self.image,
                    detections=self.detections, s0_edges=edges, post_gmc=self.post_gmc,
                    accepted=accepted, start_tracks=self.start_tracks,
                    consumed=sorted(owned), reserved=sorted(protected))
        scores = self._scores(shot)
        decisions = decision_bundle(scores, self.with_rgb)
        self._audit_frame(scores, decisions, exposure, histories)
        if any(row["motion"] is None for row in scores):
            raise ValueError("R4 anchor-qualified candidate lacks the required two-frame motion history")
        by_key = {edge_key(row): row for row in scores}
        selected = [dict(by_key[edge_key(row)], utility=row["utility"], decision_policy=self.policy)
                    for row in decisions[self.policy] if row["selected"]]
        for row in selected:
            self.write_event("proposed", **audit_row(row))
        if self.variant in ("shadow", "force_reject"):
            self.runtime_counts["policy_rejected"] += len(selected)
            selected = []
        output, committed, rejected = transact(
            tracker, activated, refound, lost, removed, selected, all_dets, measurement=True,
        )
        for row in rejected:
            self.write_event("rejected_finalization", **audit_row(row))
        for row in committed:
            tid = row["track_id"]
            self.anchors[tid].spent = True
            emitted, = output[output[:, 4] == tid]
            self.write_event("committed", **audit_row(row), prediction=False,
                             output_box=emitted[:4].tolist())
        self._update_anchors(tracker, accepted)
        return output

    def _update_anchors(self, tracker, accepted):
        # Same lifecycle as R4: weak observations never enter the regular
        # history, recharge an episode or create independent anchor support.
        current = {int(track.id) for track in tracker.active_tracks}
        kf = tracker.kalman_filter
        for record in accepted:
            tid = record["track_id"]
            if tid not in current:
                continue
            raw = np.asarray(record["raw_xyxy"], dtype=np.float64)
            measurement = np.r_[(raw[:2] + raw[2:]) * .5, raw[2:] - raw[:2]]
            previous = self.anchors.get(tid)
            if (previous is None or previous.spent or previous.observed != self.frame - 1
                    or overlap(box(previous.mean), raw) < .5):
                self.anchors[tid] = Anchor(*kf.initiate(measurement), self.frame)
                self.history[tid] = deque(maxlen=3)
            else:
                previous.mean, previous.covariance = kf.update(previous.mean, previous.covariance, measurement)
                previous.observed, previous.ready = self.frame, True
            self.history[tid].append(record)
        self.anchors = {tid: anchor for tid, anchor in self.anchors.items()
                        if self.frame - anchor.observed <= 2}
        self.history = {tid: rows for tid, rows in self.history.items() if tid in self.anchors}

    def close(self):
        for stream in (self.score_log, self.decision_log, self.frame_log):
            stream.close()
        for stream in self._raw_audit_streams:
            stream.close()
        super().close()


def audit_row(row):
    names = ("track_id", "det_index", "raw_score", "native_iou", "motion", "cosine",
             "history_score", "utility", "decision_policy", "previous_regular_observations",
             "history_frames", "geometry_group_rank", "rgb_evaluated")
    return {name: row.get(name) for name in names}
