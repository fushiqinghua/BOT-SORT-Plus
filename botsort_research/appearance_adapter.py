# Association overrides adapted from Mikel Broström's BoxMOT, AGPL-3.0;
# the upstream license is retained at provenance/BOXMOT_LICENSE.
"""Cache-only native RGB ReID and explicit, bounded Lost-track recovery.

Native ReID inherits the reviewed runtime's association and lifecycle exactly.
Experimental recovery first allocates observations to active/unconfirmed tracks,
then gives old Lost tracks exclusive access to the remaining high-score rows.
No low-score feature is fabricated, no buffer is extended, and no past output
is rewritten. All changes are local to the returned tracker instance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .appearance import appearance_threshold


APPEARANCE_VARIANTS = ("native_reid", "reid_fixed", "reid_gap", "reid_recovery_fixed", "reid_recovery")
DEFAULT_APPEARANCE_CONFIG = {
    "recovery_max_center_distance_px": 64.0,
    "recovery_max_size_ratio": 4.0,
    "recovery_ambiguity_margin": .05,
    "gallery_min_score": .6,
    "gallery_min_side_px": 4.0,
}


def _validated_config(config=None):
    config = dict(config or {})
    if set(config) - set(DEFAULT_APPEARANCE_CONFIG):
        raise ValueError(f"Unknown appearance parameters: {sorted(set(config)-set(DEFAULT_APPEARANCE_CONFIG))}")
    config = {**DEFAULT_APPEARANCE_CONFIG, **config}
    if any(not np.isfinite(value) for value in config.values()):
        raise ValueError("Appearance parameters must be finite")
    if config["recovery_max_center_distance_px"] <= 0 or config["gallery_min_side_px"] <= 0:
        raise ValueError("Recovery radius and gallery minimum side must be positive")
    if config["recovery_max_size_ratio"] < 1:
        raise ValueError("Recovery size ratio must be at least one")
    if not 0 < config["recovery_ambiguity_margin"] <= 2:
        raise ValueError("Recovery ambiguity margin must be in (0,2]")
    if not .5 < config["gallery_min_score"] <= 1:
        raise ValueError("Trusted gallery score must be above .5 and at most 1")
    return config


def conservative_recovery_matches(distances, allowed, thresholds, margin=.05):
    """Mutual unique nearest neighbors, with row and column ambiguity margins.

    Competitors include every geometrically eligible pair, even pairs that fail
    their own acceptance threshold. This avoids optimistic ambiguity filtering.
    A threshold of None disables that track's appearance acceptance.
    """
    distances = np.asarray(distances, dtype=float)
    allowed = np.asarray(allowed, dtype=bool)
    if distances.ndim != 2 or allowed.shape != distances.shape or len(thresholds) != len(distances):
        raise ValueError("Recovery distance/gate/threshold shapes disagree")
    if not np.isfinite(distances).all() or np.any((distances < 0) | (distances > 2)):
        raise ValueError("Recovery cosine distances must be finite in [0,2]")
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("Recovery margin must be positive")
    n, m = distances.shape
    if not n or not m:
        return np.empty((0, 2), dtype=int)
    gated = np.where(allowed, distances, np.inf)
    pairs = []
    for i in range(n):
        threshold = thresholds[i]
        if threshold is None:
            continue
        if not np.isfinite(threshold) or not 0 <= threshold <= 2:
            raise ValueError("Recovery thresholds must be null or finite in [0,2]")
        j = int(np.argmin(gated[i]))
        value = gated[i, j]
        if not np.isfinite(value) or value > threshold or int(np.argmin(gated[:, j])) != i:
            continue
        row_alternatives = np.delete(gated[i], j)
        column_alternatives = np.delete(gated[:, j], i)
        row_margin = float(row_alternatives.min(initial=np.inf)-value)
        column_margin = float(column_alternatives.min(initial=np.inf)-value)
        if row_margin >= margin and column_margin >= margin:
            pairs.append((i, j))
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


class CacheOnlyEncoder:
    def get_features(self, *args, **kwargs):
        raise RuntimeError("M3 requires aligned cached embeddings; online feature extraction is disabled")


def build_appearance_tracker(variant, tracker_config, appearance_config=None, calibration=None,
                             trace_callback=None):
    """Build a reviewed-runtime tracker with explicit cached RGB features.

    `native_reid` is the .4/.5 frozen control supplied by tracker_config. Other
    variants need an offline model whose provenance the replay runner verifies
    before installing its no-GT guard. Direct callers own that verification.
    """
    if variant not in APPEARANCE_VARIANTS:
        raise ValueError(f"Unknown appearance variant: {variant}")
    params = _validated_config(appearance_config)
    cfg = dict(tracker_config)
    if cfg.get("per_class", False) or cfg.get("fuse_first_associate", False):
        raise ValueError("M3 keeps pooled classes and fuse_first_associate=false")
    if cfg.get("track_high_thresh", .5) != .5 or cfg.get("track_low_thresh", .1) != .1:
        raise ValueError("High-only RGB cache requires the frozen .5/.1 score thresholds")
    if variant != "native_reid":
        if calibration is None:
            raise ValueError(f"{variant} needs an offline calibrated appearance model")
        appearance_threshold(calibration, 0, conditioned=False)
        for gap in (0, 1, 4, 11):
            appearance_threshold(calibration, gap)
    import torch
    from boxmot.trackers.botsort.botsort import BotSort
    from boxmot.trackers.botsort.botsort_track import STrack
    from boxmot.trackers.botsort.basetrack import TrackState
    from boxmot.utils.matching import embedding_distance, fuse_score, iou_distance, linear_assignment
    from .adapter import verify_runtime
    runtime = verify_runtime()
    recovery = variant in ("reid_recovery", "reid_recovery_fixed")

    class FeatureTrack(STrack):
        """Ordinary EMA plus a separately protected, last trusted RGB vector."""
        def _remember(self, new_track, frame_id):
            if new_track.curr_feat is None:
                return
            self.appearance_frame = int(frame_id)
            eligible = (new_track.conf >= params["gallery_min_score"] and
                        np.min(new_track.xywh[2:]) >= params["gallery_min_side_px"])
            if eligible and not getattr(self, "_freeze_trusted_gallery", False):
                self.trusted_feature = new_track.curr_feat.copy()
                self.trusted_feature_frame = int(frame_id)

        def update(self, new_track, frame_id):
            super().update(new_track, frame_id)
            self._remember(new_track, frame_id)

        def re_activate(self, new_track, frame_id, new_id=False):
            super().re_activate(new_track, frame_id, new_id=new_id)
            self._remember(new_track, frame_id)

    class CachedBotSort(BotSort):
        def update(self, dets, img, embs=None):
            dets = np.asarray(dets)
            if dets.ndim != 2 or dets.shape[1] != 6 or not np.isfinite(dets).all():
                raise ValueError("Cached appearance expects finite N x 6 detections")
            if embs is None:
                raise ValueError("Cached appearance replay requires an embedding array, including empty frames")
            embs = np.asarray(embs)
            if embs.ndim != 2 or embs.shape[0] != len(dets) or embs.shape[1] < 1:
                raise ValueError("Embedding array must be N x D and aligned with detections")
            high = dets[:, 4] > self.track_high_thresh
            selected = embs[high]
            if not np.isfinite(selected).all() or np.any(np.linalg.norm(selected, axis=1) <= 1e-12):
                raise ValueError("Every high-score detection requires a finite, nonzero feature")
            # Copies protect caller caches from STrack's in-place normalization.
            return super().update(dets, img, embs=embs.copy())

    if variant == "native_reid":
        Tracker = CachedBotSort
    else:
        class CalibratedBotSort(CachedBotSort):
            def _create_detections(self, dets_first, features_high):
                detections = []
                for det, feature in zip(dets_first, features_high):
                    track = FeatureTrack(det, feature, max_obs=self.max_obs)
                    track.appearance_frame = int(self.frame_count)
                    track.trusted_feature = None
                    track.trusted_feature_frame = None
                    track._remember(track, self.frame_count)
                    detections.append(track)
                return detections

            def _gap(self, track, trusted=False):
                last = getattr(track, "trusted_feature_frame" if trusted else "appearance_frame", None)
                return max(0, self.frame_count-last-1) if last is not None else 31

            def _thresholds(self, tracks, *, trusted=False, conditioned=False):
                return [appearance_threshold(calibration, self._gap(track, trusted), conditioned=conditioned)
                        for track in tracks]

            def _first_association(self, dets, dets_first, active_tracks, unconfirmed, img,
                                   detections, activated_stracks, refind_stracks, strack_pool):
                STrack.multi_predict(strack_pool)
                warp = self.cmc.apply(img, dets)
                STrack.multi_gmc(strack_pool, warp)
                STrack.multi_gmc(unconfirmed, warp)
                self._recovery_refind = refind_stracks
                self._recovery_candidates = [t for t in strack_pool if t.state == TrackState.Lost
                    and self.frame_count-t.end_frame <= self.max_time_lost]
                pool_indices = [i for i, t in enumerate(strack_pool) if not recovery or t.state == TrackState.Tracked]
                candidates = [strack_pool[i] for i in pool_indices]
                iou = iou_distance(candidates, detections)
                raw = embedding_distance(candidates, detections)
                thresholds = self._thresholds(candidates, conditioned=variant == "reid_gap")
                appearance = raw.copy()
                for i, threshold in enumerate(thresholds):
                    if threshold is None:
                        appearance[i, :] = 1.0
                    else:
                        appearance[i, appearance[i] > threshold] = 1.0
                appearance[iou > self.proximity_thresh] = 1.0
                cost = np.minimum(iou, appearance)
                matches, unmatched, u_detection = linear_assignment(cost, thresh=self.match_thresh)
                mapped = np.asarray([(pool_indices[i], j) for i, j in matches], dtype=int).reshape(-1, 2)
                matched_pool = {int(i) for i, _ in mapped}
                u_track = tuple(i for i in range(len(strack_pool)) if i not in matched_pool)
                trace_matches = []
                for (i, j), (pool_i, _) in zip(matches, mapped):
                    track, detection = strack_pool[pool_i], detections[j]
                    was_lost = track.state == TrackState.Lost
                    trace_matches.append({"track_id": int(track.id), "det_index": int(detection.det_ind),
                        "raw_cosine_distance": float(raw[i, j]), "iou_distance": float(iou[i, j]),
                        "appearance_threshold": thresholds[i], "feature_gap": self._gap(track),
                        "was_lost": bool(was_lost)})
                    if was_lost:
                        track.re_activate(detection, self.frame_count, new_id=False)
                        refind_stracks.append(track)
                    else:
                        track.update(detection, self.frame_count)
                        activated_stracks.append(track)
                if trace_callback:
                    trace_callback({"frame": int(self.frame_count), "stage": "appearance_first",
                                    "variant": variant, "matches": trace_matches,
                                    "tracks": len(candidates), "detections": len(detections),
                                    "old_lost_reserved_for_recovery": len(self._recovery_candidates) if recovery else 0})
                return mapped, u_track, u_detection

            def _handle_unconfirmed_tracks(self, u_detection, detections, activated_stracks,
                                           removed_stracks, unconfirmed):
                # Native code divides cosine by two in this stage. Convert the
                # fitted RAW cosine cutoff, then restore the public parameter.
                saved = self.appearance_thresh
                threshold = appearance_threshold(calibration, 0, conditioned=False)
                self.appearance_thresh = threshold/2 if threshold is not None else -1.0
                try:
                    matches, u_unconfirmed, remaining = super()._handle_unconfirmed_tracks(
                        u_detection, detections, activated_stracks, removed_stracks, unconfirmed)
                finally:
                    self.appearance_thresh = saved
                if not recovery:
                    return matches, u_unconfirmed, remaining
                # `remaining` indexes the high-score subset selected by first
                # association's u_detection, the same indexing used by births.
                selected = [detections[int(u_detection[int(j)])] for j in remaining]
                tracks = [track for track in self._recovery_candidates
                          if track.state == TrackState.Lost and track.trusted_feature is not None]
                distances = np.ones((len(tracks), len(selected)), dtype=float)
                allowed = np.zeros_like(distances, dtype=bool)
                thresholds = self._thresholds(tracks, trusted=True, conditioned=variant == "reid_recovery")
                for i, track in enumerate(tracks):
                    predicted = track.mean[:4]
                    if not np.isfinite(predicted).all() or np.any(predicted[2:] <= 0):
                        continue
                    for j, detection in enumerate(selected):
                        z = detection.xywh
                        vector = detection.curr_feat
                        distances[i, j] = float(np.clip(1-np.dot(track.trusted_feature, vector), 0, 2))
                        ratios = z[2:] / predicted[2:]
                        allowed[i, j] = (np.linalg.norm(z[:2]-predicted[:2]) <= params["recovery_max_center_distance_px"]
                            and np.max(np.maximum(ratios, 1/ratios)) <= params["recovery_max_size_ratio"])
                recovered = conservative_recovery_matches(distances, allowed, thresholds,
                                                           params["recovery_ambiguity_margin"])
                attempt_details = []
                accepted_rows = set(recovered[:, 0].tolist())
                if trace_callback:
                    gated = np.where(allowed, distances, np.inf)
                    for i, track in enumerate(tracks):
                        valid = np.flatnonzero(allowed[i])
                        best = int(valid[np.argmin(distances[i, valid])]) if len(valid) else None
                        best_distance = float(distances[i, best]) if best is not None else None
                        if not len(valid):
                            reason = "no_geometric_candidate"
                        elif thresholds[i] is None:
                            reason = "appearance_disabled"
                        elif best_distance > thresholds[i]:
                            reason = "above_appearance_threshold"
                        elif i not in accepted_rows:
                            reason = "not_mutual_unique_nearest"
                        else:
                            reason = "accepted"
                        row_margin, column_margin = None, None
                        if best is not None:
                            row_value = float(np.delete(gated[i], best).min(initial=np.inf) - best_distance)
                            column_value = float(np.delete(gated[:, best], i).min(initial=np.inf) - best_distance)
                            row_margin = row_value if np.isfinite(row_value) else None
                            column_margin = column_value if np.isfinite(column_value) else None
                        attempt_details.append({"track_id": int(track.id), "feature_gap": self._gap(track, trusted=True),
                            "lost_age": int(self.frame_count-track.end_frame), "appearance_threshold": thresholds[i],
                            "geometry_candidates": int(len(valid)), "best_raw_cosine_distance": best_distance,
                            "best_detection_index": int(selected[best].det_ind) if best is not None else None,
                            "row_margin": row_margin, "column_margin": column_margin, "decision": reason})
                trace_matches = []
                for i, j in recovered:
                    track, detection = tracks[i], selected[j]
                    trace_matches.append({"track_id": int(track.id), "det_index": int(detection.det_ind),
                        "raw_cosine_distance": float(distances[i, j]), "appearance_threshold": thresholds[i],
                        "feature_gap": self._gap(track, trusted=True),
                        "lost_age": int(self.frame_count-track.end_frame)})
                    # A recovery decision cannot immediately replace its own
                    # trusted evidence; wait for a later high-quality match.
                    track._freeze_trusted_gallery = True
                    try:
                        track.re_activate(detection, self.frame_count, new_id=False)
                    finally:
                        track._freeze_trusted_gallery = False
                    self._recovery_refind.append(track)
                consumed = {int(j) for _, j in recovered}
                remaining = np.asarray([int(value) for j, value in enumerate(remaining) if j not in consumed], dtype=int)
                if trace_callback:
                    trace_callback({"frame": int(self.frame_count), "stage": "appearance_recovery",
                        "variant": variant, "tracks": len(tracks), "detections": len(selected),
                        "geometry_valid_pairs": int(allowed.sum()), "matches": trace_matches,
                        "attempts": attempt_details,
                        "lost_without_trusted_feature": sum(track.trusted_feature is None for track in self._recovery_candidates),
                        "unmatched_lost_ids": [int(track.id) for i, track in enumerate(tracks) if i not in set(recovered[:, 0])],
                        "remaining_high_detection_indices": [int(detections[int(u_detection[j])].det_ind) for j in remaining],
                        "low_score_recovery": "disabled_missing_features"})
                return matches, u_unconfirmed, remaining
        Tracker = CalibratedBotSort

    # The reviewed constructor's with_reid branch only instantiates the model.
    # Constructing with false and enabling the cached path avoids global patches
    # and downloads while retaining native matching behavior.
    cfg["with_reid"] = False
    tracker = Tracker(reid_weights=Path("unused_cached_rgb.pt"), device=torch.device("cpu"), half=False, **cfg)
    tracker.with_reid = True
    tracker.model = CacheOnlyEncoder()
    tracker.research_runtime_info = runtime
    tracker.research_appearance = {"variant": variant, "parameters": params,
                                  "feature_scope": "high_score_rgb_only", "max_time_lost": tracker.max_time_lost,
                                  "unconfirmed_threshold_policy": "native_half_cosine" if variant == "native_reid" else "calibrated_raw_cosine_converted_to_half"}
    return tracker
