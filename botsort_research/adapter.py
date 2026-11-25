# Association-stage overrides are adapted from Mikel Broström's BoxMOT,
# AGPL-3.0; the upstream license is retained at provenance/BOXMOT_LICENSE.
"""Small, version-checked adapter over the frozen BoxMOT BoT-SORT lifecycle.

Importing this module does not load PyTorch, OpenCV or a ReID model. M2 changes
the first/second association and, for floor/calibrated R, measurement updates.
The original unconfirmed-track association and lifecycle remain inherited.
"""
import hashlib
import importlib.metadata
import inspect
from pathlib import Path

import numpy as np

from .association import assign_with_dummies, gaussian_cost, nwd_config, nwd_cost, stage_config
from .noise import NoiseModel, correct_with_covariance


SUPPORTED_VARIANTS = (
    "native_iou", "m2_nll_baseline_r", "m2_nll_floor", "m2_nll_calibrated",
    "native_iou_det", "m2_nll_floor_det", "m2_nll_floor_native_update",
    "m2_nll_floor_native_update_det", "iou_floor_update",
    "m2_nll_first_native_update", "m2_nll_second_native_update",
    "iou_relaxed", "nwd_native_update",
)

# SHA256 of UTF-8 source, normalized newlines, trailing whitespace stripped.
# Source: mikel-brostrom/boxmot tag v16.0.11. Its __version__ is stale (16.0.10).
REVIEWED_SOURCE_HASHES = {
    "botsort.py": "97af655524c80c54104124d2ec78881d79454a25fa5b86edc37617d5d408ba41",
    "botsort_track.py": "2a4596f3c839e403c006d37212c6a52cd3c49f6b102e9c95578819acf8a1238e",
    "xywh_kf.py": "2187c8f54cf4765a6b36b88df1e5f3c024b0fa6e860d480f2947288ac2279f05",
    "base_kalman_filter.py": "c7a079a40cb0b7bb802f63c8cda61c9731b1c78046d222e98bbd43b281c545d6",
}


def runtime_info():
    """Inspect the actually imported runtime, not an unrelated checkout path."""
    import boxmot
    from boxmot.trackers.botsort.botsort import BotSort
    from boxmot.trackers.botsort.botsort_track import STrack
    from boxmot.motion.kalman_filters.aabb.xywh_kf import KalmanFilterXYWH
    from boxmot.motion.kalman_filters.aabb.base_kalman_filter import BaseKalmanFilter

    try:
        version = importlib.metadata.version("boxmot")
    except importlib.metadata.PackageNotFoundError:
        version = None
    sources = {}
    for obj in (BotSort, STrack, KalmanFilterXYWH, BaseKalmanFilter):
        path = Path(inspect.getsourcefile(obj)).resolve()
        sources[path.name] = {
            "path": str(path),
            "sha256_normalized": hashlib.sha256(path.read_text(encoding="utf-8").rstrip().encode()).hexdigest(),
        }
    return {"distribution_version": version, "package_version": getattr(boxmot, "__version__", None),
            "sources": sources, "reviewed_tag": "v16.0.11"}


def verify_runtime():
    """Fail explicitly if private APIs or reviewed source semantics changed."""
    from boxmot.trackers.botsort.botsort import BotSort
    info = runtime_info()
    bad = [name for name, expected in REVIEWED_SOURCE_HASHES.items()
           if info["sources"].get(name, {}).get("sha256_normalized") != expected]
    if bad:
        raise RuntimeError(
            "M2 requires reviewed BoxMOT v16.0.11 sources. Different core files: "
            + ", ".join(bad)
            + ". Use the frozen tracking runtime or install boxmot==16.0.11. "
            "Do not bypass this guard without reviewing and regression-testing the adapter."
        )
    signature = tuple(inspect.signature(BotSort._first_association).parameters)
    expected = ("self", "dets", "dets_first", "active_tracks", "unconfirmed", "img", "detections",
                "activated_stracks", "refind_stracks", "strack_pool")
    if signature != expected:
        raise RuntimeError("Unsupported BoxMOT _first_association signature")
    return info


def build_tracker(variant, tracker_config, m2_config=None, trace_callback=None):
    """Build a CPU association tracker; images supply the original SOF CMC.

    native_iou returns an untouched native BotSort instance. Callers can set an
    identity CMC object for the explicitly named M1 no-CMC diagnostic control.
    Calibration metadata lives in the caller's run manifest, never in tracking.
    """
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unknown tracker variant {variant!r}; choose {SUPPORTED_VARIANTS}")
    cfg = dict(tracker_config)
    if cfg.get("with_reid", False):
        raise ValueError("This M2 experiment requires with_reid=false; raw cosine costs cannot be mixed with NLL")
    cfg["with_reid"] = False
    if cfg.get("per_class", False):
        raise ValueError("The frozen VT-Tiny-MOT protocol pools classes; per_class must be false")
    if cfg.get("fuse_first_associate", False):
        raise ValueError("fuse_first_associate must be false in this controlled M2 experiment")
    import torch
    from boxmot.trackers.botsort.botsort import BotSort
    from boxmot.trackers.botsort.basetrack import TrackState
    from boxmot.trackers.botsort.botsort_track import STrack
    from boxmot.motion.kalman_filters.aabb.xywh_kf import KalmanFilterXYWH
    from boxmot.utils.matching import iou_distance, linear_assignment

    info = verify_runtime()
    kwargs = dict(reid_weights=Path("unused.pt"), device=torch.device("cpu"), half=False, **cfg)
    if variant in ("native_iou", "native_iou_det", "iou_relaxed"):
        tracker = BotSort(**kwargs)
        tracker.research_runtime_info = info
        tracker.research_update_mode = "baseline"
        return tracker

    m2_config = dict(m2_config or {})
    if set(m2_config) - {"noise", "association", "nwd"}:
        raise ValueError("M2 config accepts only noise, association and nwd sections")
    noise_cfg = dict(m2_config.get("noise", {}))
    requested_mode = {"m2_nll_baseline_r": "baseline", "m2_nll_calibrated": "calibrated"}.get(variant, "floor")
    if requested_mode == "calibrated" and noise_cfg.get("mode") != "calibrated":
        raise ValueError("m2_nll_calibrated requires a fitted noise model with mode='calibrated'")
    noise_cfg["mode"] = requested_mode
    noise = NoiseModel.from_dict(noise_cfg)
    assoc_cfg = m2_config.get("association", {})
    if set(assoc_cfg) - {"first", "second"}:
        raise ValueError("Association config accepts first and second stages only")
    stages = {name: stage_config(assoc_cfg.get(name)) for name in ("first", "second")}
    adaptive = variant in ("m2_nll_floor", "m2_nll_floor_det", "m2_nll_calibrated", "iou_floor_update")
    update_mode = requested_mode if adaptive else "baseline"
    baseline_noise = NoiseModel(mode="baseline")
    methods = {"first": "nll", "second": "nll"}
    if variant == "iou_floor_update":
        methods = {"first": "iou", "second": "iou"}
    elif variant == "m2_nll_first_native_update":
        methods["second"] = "iou"
    elif variant == "m2_nll_second_native_update":
        methods["first"] = "iou"
    elif variant == "nwd_native_update":
        methods = {"first": "nwd", "second": "nwd"}
    nwd = nwd_config(m2_config.get("nwd"))

    class ExplicitRKalman(KalmanFilterXYWH):
        """One instance per track; R is supplied by the matched-detection context."""
        pending_R = None

        def update(self, mean, covariance, measurement, confidence=0.0):
            if self.pending_R is None:
                raise RuntimeError("A measurement update occurred without its selected detection R")
            return correct_with_covariance(mean, covariance, measurement, self.pending_R)

    class MeasurementTrack(STrack):
        def activate(self, kalman_filter, frame_id):
            # Original initialization and shared STrack.multi_predict keep P0/Q.
            super().activate(ExplicitRKalman(), frame_id)

        def _correct(self, new_track, method, *args, **kw):
            selected = getattr(self, "_m2_selected_R", None)
            if selected is None:
                # Original unconfirmed association has no Gaussian cost; it
                # nevertheless uses this detection's R for the state update.
                selected = noise.covariance(new_track.xywh, new_track.conf, self.mean[:4])
            self.kalman_filter.pending_R = selected
            try:
                return method(new_track, *args, **kw)
            finally:
                self.kalman_filter.pending_R = None
                if hasattr(self, "_m2_selected_R"):
                    del self._m2_selected_R

        def update(self, new_track, frame_id):
            return self._correct(new_track, super().update, frame_id)

        def re_activate(self, new_track, frame_id, new_id=False):
            return self._correct(new_track, super().re_activate, frame_id, new_id=new_id)

    Track = MeasurementTrack if adaptive else STrack

    class M2BotSort(BotSort):
        def _create_detections(self, dets_first, features_high):
            return [Track(det, max_obs=self.max_obs) for det in dets_first]

        def _associate(self, tracks, detections, stage):
            method = methods[stage]
            details = None
            if method == "nll":
                cost, details = gaussian_cost(
                    [t.mean for t in tracks], [t.covariance for t in tracks],
                    [d.xywh for d in detections], [d.conf for d in detections], noise, stages[stage])
                threshold = stages[stage]["miss_cost"] + stages[stage]["unassigned_cost"]
                matches, u_track, u_det = assign_with_dummies(
                    cost, stages[stage]["miss_cost"], stages[stage]["unassigned_cost"])
                reason = details["gate_reason"]
            else:
                if method == "iou":
                    cost = iou_distance(tracks, detections)
                    reason = np.zeros(cost.shape, dtype=np.uint8)
                    threshold = self.match_thresh if stage == "first" else 0.5
                else:
                    cost, reason = nwd_cost([t.mean[:4] for t in tracks], [d.xywh for d in detections],
                                            normalizer_px=nwd["normalizer_px"], config=stages[stage])
                    threshold = nwd[stage + "_threshold"]
                # Preserve the native LAP implementation and boundary semantics
                # for IoU; NWD uses the same cost-limit assignment as its control.
                matches, u_track, u_det = linear_assignment(cost, thresh=threshold)
                matches = np.asarray(matches, dtype=int).reshape(-1, 2)
            if adaptive:
                for i, j in matches:
                    # Round1 keeps the exact association R. The IoU/update-only
                    # control obtains R from this detection without changing IoU.
                    tracks[i]._m2_selected_R = (details["measurement_covariance"][i, j].copy()
                        if details is not None else noise.covariance(detections[j].xywh, detections[j].conf, tracks[i].mean[:4]))
            if trace_callback is not None:
                matched = []
                for i, j in matches:
                    track, det = tracks[i], detections[j]
                    update_r = (noise if adaptive else baseline_noise).covariance(det.xywh, det.conf, track.mean[:4])
                    matched.append({"track_id": int(track.id), "det_index": int(det.det_ind),
                                    "cost": float(cost[i, j]),
                                    "mahalanobis2": float(details["mahalanobis2"][i, j]) if details is not None else None,
                                    "logdet": float(details["logdet"][i, j]) if details is not None else None,
                                    "score": float(det.conf), "was_lost": bool(track.state == TrackState.Lost),
                                    "association_R_diag": np.diag(details["measurement_covariance"][i, j]).tolist() if details is not None else None,
                                    "update_R_diag": np.diag(update_r).tolist(), "update_mode": update_mode,
                                    "predicted_xywh": track.mean[:4].tolist(), "measurement_xywh": det.xywh.tolist()})
                trace_callback({
                    "frame": int(self.frame_count), "stage": stage, "association_method": method,
                    "tracks": len(tracks), "detections": len(detections),
                    "valid_pairs": int(np.isfinite(cost).sum()),
                    "rejected_chi2": int(((reason & 1) != 0).sum()),
                    "rejected_center": int(((reason & 2) != 0).sum()),
                    "rejected_size": int(((reason & 4) != 0).sum()),
                    "rejected_invalid": int(((reason & 8) != 0).sum()),
                    "rejected_cost": int((np.isfinite(cost) & ((cost >= threshold) if method == "nll" else (cost > threshold))).sum()),
                    "unmatched_track_ids": [int(tracks[i].id) for i in u_track],
                    "unmatched_detection_indices": [int(detections[j].det_ind) for j in u_det],
                    "matches": matched,
                })
            return matches, u_track, u_det

        def _first_association(self, dets, dets_first, active_tracks, unconfirmed, img,
                               detections, activated_stracks, refind_stracks, strack_pool):
            STrack.multi_predict(strack_pool)
            warp = self.cmc.apply(img, dets)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)
            matches, u_track, u_detection = self._associate(strack_pool, detections, "first")
            for i, j in matches:
                track, det = strack_pool[i], detections[j]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_count)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_count, new_id=False)
                    refind_stracks.append(track)
            return matches, u_track, u_detection

        def _second_association(self, dets_second, activated_stracks, lost_stracks,
                                refind_stracks, u_track_first, strack_pool):
            detections = [Track(det, max_obs=self.max_obs) for det in dets_second]
            remaining = [strack_pool[i] for i in u_track_first if strack_pool[i].state == TrackState.Tracked]
            matches, u_track, u_detection = self._associate(remaining, detections, "second")
            for i, j in matches:
                track = remaining[i]
                track.update(detections[j], self.frame_count)
                activated_stracks.append(track)
            for i in u_track:
                track = remaining[i]
                if track.state != TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)
            return matches, u_track, u_detection

    tracker = M2BotSort(**kwargs)
    tracker.research_runtime_info = info
    tracker.research_noise = noise.to_dict()
    tracker.research_association = stages
    tracker.research_association_methods = methods
    tracker.research_update_mode = update_mode
    tracker.research_nwd = nwd if variant == "nwd_native_update" else None
    return tracker
