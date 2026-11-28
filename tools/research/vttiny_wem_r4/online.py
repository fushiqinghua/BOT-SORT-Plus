"""Exploratory S0 continuation with private anchor and native-finalization preflight."""
from collections import Counter, defaultdict, deque
from copy import copy, deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from capture import Observer, current_assignments, encode, fingerprint, track_state
from score_features import frame_features, overlap
from admission import assign, score as admission_score


SETTING = dict(method="history_competition", cosine_floor=.75, margin=.05)
RGB_MODES = {"shadow", "force_reject", "w_coast", "w_full"}


def box(mean):
    return np.r_[mean[:2] - mean[2:4] / 2, mean[:2] + mean[2:4] / 2]


@dataclass
class Anchor:
    mean: np.ndarray
    covariance: np.ndarray
    observed: int
    ready: bool = False
    spent: bool = False

    def advance(self, kf, warp):
        from boxmot.trackers.botsort.botsort_track import STrack
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        STrack.multi_gmc([self], warp)


def native_preview(tracker, activated, refound, lost, removed):
    proxy = copy(tracker)
    proxy.removed_stracks = list(tracker.removed_stracks)
    result = proxy._prepare_output(activated, refound, lost, removed).reshape(-1, 8)
    return proxy, result


def transact(tracker, activated, refound, lost, removed, proposed, all_dets, measurement):
    """Test the original finalizer on private track copies and commit only safe proposals."""
    from boxmot.trackers.botsort.basetrack import TrackState
    from boxmot.trackers.botsort.botsort_track import STrack

    accepted, rejected = [], []
    if not proposed:
        return tracker._prepare_output(activated, refound, lost, removed).reshape(-1, 8), accepted, rejected
    old, previous = native_preview(tracker, activated, refound, lost, removed)
    by_lost = {int(t.id): t for t in lost if t.state == TrackState.Lost}
    replacements = []
    discarded = set()
    for item in proposed:
        tid, det = item["track_id"], item["det_index"]
        original = by_lost[tid]
        private = deepcopy(original)
        if measurement:
            weak = STrack(all_dets[det], max_obs=tracker.max_obs)
            private.re_activate(weak, tracker.frame_count, new_id=False)
        else:
            private.state = TrackState.Tracked
            private.det_ind = -1  # Explicit prediction output, no detector row or measurement.
            if any(fingerprint(getattr(private, key)) != fingerprint(getattr(original, key))
                   for key in ("mean", "covariance")) or private.frame_id != original.frame_id:
                raise ValueError("Coast changed filter state or observation clock")
        if any(fingerprint(getattr(private, key)) != fingerprint(getattr(original, key))
               for key in ("curr_feat", "smooth_feat")) or private.id != original.id:
            raise ValueError("Weak output changed appearance or identity")
        trial, output = native_preview(tracker, activated, [*refound, *replacements, private],
                                       [t for t in lost if t.id not in discarded | {tid}], removed)
        kept = output[:, 4] == tid
        equal_output = np.array_equal(previous[previous[:, 4] != tid], output[~kept])
        equal_lists = all([t.id for t in getattr(old, collection) if t.id != tid] ==
                          [t.id for t in getattr(trial, collection) if t.id != tid]
                          for collection in ("active_tracks", "lost_stracks", "removed_stracks"))
        if sum(kept) != 1 or not equal_output or not equal_lists:
            rejected.append(item)
            continue
        replacements.append(private)
        discarded.add(tid)
        old, previous = trial, output
        accepted.append(item)
    tracker.active_tracks, tracker.lost_stracks, tracker.removed_stracks = (
        old.active_tracks, old.lost_stracks, old.removed_stracks)
    return previous, accepted, rejected


class ClosedLoop(Observer):
    def __init__(self, path, seq, method, output, protocol, mode, settings, encoder):
        super().__init__(path, seq, method, output, protocol, "off" if mode == "off" else "force_reject")
        self.enabled = mode != "off"
        self.variant = mode
        self.mode = mode
        self.anchors = {}
        self.warps = {}
        self.encoder = encoder
        self.images_root = Path(settings["images_root"])
        self.events = self.file(output, "wem_events", ".jsonl").open("xb")
        self.runtime_counts = Counter()
        if mode in RGB_MODES:
            from botsort_plus.io import FeatureCache, read_detection_cache
            feature_path = Path(settings["features_dir"]) / f"{seq}.npz"
            self.features = FeatureCache(feature_path, read_detection_cache(path),
                                         expected_rgb_sha=protocol["rgb_encoder_sha256"]).arrays

    def start(self, tracker, detections, image, features):
        super().start(tracker, detections, image, features)
        if self.enabled:
            self.current_image = image

    def compensated(self, tracker, pool, unconfirmed, warp):
        super().compensated(tracker, pool, unconfirmed, warp)
        if not self.enabled:
            return
        self.warps[self.frame] = warp.copy()
        self.warps.pop(self.frame - 6, None)
        for anchor in self.anchors.values():
            anchor.advance(tracker.kalman_filter, warp)

    def write_event(self, event, **fields):
        self.runtime_counts[event] += 1
        self.events.write(encode(dict(event=event, frame=self.frame, **fields)) + b"\n")

    def finish_before_output(self, tracker, activated, refound, lost, removed, reserved, consumed, all_dets):
        if not self.enabled:
            return tracker._prepare_output(activated, refound, lost, removed).reshape(-1, 8)
        if tracker.plus.birth.enabled or tracker.plus.identity.enabled:
            raise ValueError("W exploration requires the archived B or BG foundation")
        accepted = [dict(**self.detections[idx], track_id=tid, frame=self.frame, stage=stage)
                    for tid, idx, stage in current_assignments(tracker.last_events)]
        owned = {item["det_index"] for item in accepted} | set(consumed)
        protected = {int(c.latest.det_index) for cid, c in tracker.candidate_pool.candidates.items()
                     if cid in reserved and int(c.latest.frame) == self.frame}
        priors = {t["track_id"]: t for t in self.post_gmc}
        lost_ids = {int(t.id) for t in lost if int(t.state) == 2}
        missing = [t["track_id"] for t in self.start_tracks
                   if t["collection"] == "active_tracks" and t["confirmed"] and t["state"] == 1
                   and t["track_id"] in lost_ids and t["track_id"] not in {a["track_id"] for a in accepted}]
        eligible = []
        for tid in missing:
            anchor = self.anchors.get(tid)
            if anchor is None or not anchor.ready or anchor.spent or anchor.observed != self.frame - 1:
                self.runtime_counts["anchor_ineligible"] += 1
                continue
            if overlap(box(anchor.mean), np.asarray(priors[tid]["selected_xyxy"])) < .5:
                self.runtime_counts["anchor_diverged"] += 1
                continue
            eligible.append(tid)

        if self.variant == "coast":
            # Reject forecasts overlapping a current accepted observation or another forecast.
            used = [np.asarray(self.detections[idx]["raw_xyxy"]) for idx in owned | protected]
            selected = []
            for tid in sorted(eligible):
                location = np.asarray(priors[tid]["selected_xyxy"])
                if any(overlap(location, other) >= .5 for other in used):
                    self.runtime_counts["forecast_overlap"] += 1
                    continue
                used.append(location)
                selected.append(dict(track_id=tid, det_index=None, raw_score=None, native_iou=None))
        else:
            edges = []
            for tid in eligible:
                prior = priors[tid]
                anchor_box = box(self.anchors[tid].mean)
                for candidate in self.detections:
                    idx = candidate["det_index"]
                    if candidate["band"] != "S0" or not .05 <= candidate["raw_score"] <= .1:
                        continue
                    if idx in owned or idx in protected:
                        self.runtime_counts["ownership_reject"] += 1
                        continue
                    raw = np.asarray(candidate["raw_xyxy"])
                    geom = overlap(raw, np.asarray(prior["selected_xyxy"]))
                    if geom < .5 or (self.gmc["backup_guard_active"] and
                                      overlap(raw, np.asarray(prior["backup_xyxy"])) < .5):
                        self.runtime_counts["geometry_reject"] += 1
                        continue
                    if overlap(raw, anchor_box) < .5:
                        self.runtime_counts["anchor_box_reject"] += 1
                        continue
                    edges.append(dict(track_id=tid, det_index=idx, native_low_feasible=True,
                                      gmc_guard_feasible=True, selected_iou=geom))
            self.runtime_counts["eligible_s0_edges"] += len(edges)
            if self.variant == "plain":
                scored = []
                by_track = defaultdict(list)
                for edge in edges:
                    candidate = self.detections[edge["det_index"]]
                    raw = np.asarray(candidate["raw_xyxy"])
                    if any(overlap(raw, np.asarray(self.detections[idx]["raw_xyxy"])) >= .90 for idx in owned | protected):
                        self.runtime_counts["near_owner_reject"] += 1
                        continue
                    by_track[edge["track_id"]].append(dict(track_id=edge["track_id"], **candidate,
                        native_iou=edge["selected_iou"], normalized_score=(candidate["raw_score"]-.05)/.05))
                for group in by_track.values():
                    order = sorted(group, key=lambda item:(-item["native_iou"], -item["raw_score"], item["det_index"]))
                    top = list(dict.fromkeys(tuple(item["raw_xyxy"]) for item in order))[:5]
                    scored.extend(dict(row, raw_box_xyxy=row["raw_xyxy"]) for row in order if tuple(row["raw_xyxy"]) in top)
                selected = assign(scored, dict(method="geometry", floor=.5))
            else:
                from botsort_plus.io import resolve_image

                source = str(resolve_image(self.image, self.images_root)) if edges else self.image
                shot = dict(frame=self.frame, sequence=self.seq, reference=self.method, image=source,
                            detections=self.detections, s0_edges=edges, post_gmc=self.post_gmc, accepted=accepted,
                            start_tracks=self.start_tracks, consumed=sorted(owned), reserved=sorted(protected))
                # R3 scoring reuses the original frozen embeddings for history and only encodes current S0 rows.
                cache_frame = dict(self.rows[self.frame - 1], image=source)
                scored, vectors, _ = frame_features(shot, cache_frame, self.features,
                                                     self.encoder, self.warps, self.current_image)
                self.runtime_counts["rgb_crops"] += len(vectors)
                self.runtime_counts["scored_edges"] += len(scored)
                self.runtime_counts["competition_or_quality_reject"] += sum(
                    admission_score(row, SETTING) is None for row in scored)
                selected = assign([row for row in scored if row["geometry_group_rank"] <= 5], SETTING)

        for row in selected:
            self.write_event("proposed", **audit_row(row))
        if self.variant in ("shadow", "force_reject"):
            self.runtime_counts["policy_rejected"] += len(selected)
            selected = []
        output, committed, rejected = transact(tracker, activated, refound, lost, removed, selected,
                                                all_dets, self.variant in ("plain", "w_full"))
        for row in rejected:
            self.write_event("rejected_finalization", **audit_row(row))
        for row in committed:
            tid = row["track_id"]
            self.anchors[tid].spent = True
            emitted, = output[output[:, 4] == tid]
            self.write_event("committed", **audit_row(row), prediction=self.variant in ("coast", "w_coast"),
                             output_box=emitted[:4].tolist())
        current = {int(t.id) for t in tracker.active_tracks}
        kf = tracker.kalman_filter
        for record in accepted:
            tid = record["track_id"]
            if tid not in current:
                continue
            raw = np.asarray(record["raw_xyxy"], dtype=np.float64)
            z = np.r_[(raw[:2]+raw[2:])*.5, raw[2:]-raw[:2]]
            previous = self.anchors.get(tid)
            if (previous is None or previous.spent or previous.observed != self.frame - 1 or
                    overlap(box(previous.mean), raw) < .5):
                self.anchors[tid] = Anchor(*kf.initiate(z), self.frame)
                self.history[tid] = deque(maxlen=3)
            else:
                previous.mean, previous.covariance = kf.update(previous.mean, previous.covariance, z)
                previous.observed, previous.ready = self.frame, True
            self.history[tid].append(record)
        self.anchors = {tid:anchor for tid, anchor in self.anchors.items() if self.frame-anchor.observed<=2}
        self.history = {tid:rows for tid, rows in self.history.items() if tid in self.anchors}
        return output

    def finish(self, tracker, output):
        import hashlib
        if self.frame != tracker.frame_count:
            raise ValueError("Frozen frame order mismatch")
        digest = hashlib.sha256(encode(track_state(tracker, output, self.gmc))).hexdigest()
        self.states.write(f"{self.frame} {digest}\n".encode())
        self.previous = self.frame
        self.counts["frames"] += 1

    def close(self):
        self.events.close()
        self.counts.update(self.runtime_counts)
        super().close()


def audit_row(row):
    return {name: row.get(name) for name in ("track_id", "det_index", "raw_score", "native_iou", "cosine", "history_score")}
