# BoT-SORT update flow adapted from BoxMOT 16.0.11 (Mikel Broström), AGPL-3.0.
# See docs/botsort_plus/PROVENANCE.md. New A/B/C mechanisms are isolated modules.
"""A real BoxMOT adapter, not a post-hoc ID renaming or future-frame smoother.

The disabled path calls the original BotSort.update. The enabled path preserves
its high/low/unconfirmed association order, then runs B before A/new-ID birth.
B rescues residual observations only; it does not undo a native accepted match.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

from boxmot.trackers.botsort.botsort import BotSort
from boxmot.trackers.botsort.botsort_track import STrack
from boxmot.trackers.botsort.basetrack import TrackState
from boxmot.trackers.botsort.botsort_utils import joint_stracks
from boxmot.utils.matching import iou_distance, fuse_score, linear_assignment

from .appearance import MemoryBank, appearance_distance, appearance_details
from .candidates import CandidatePool
from .config import PlusConfig
from .recovery import RecoveryMatcher
from .types import Feature, Observation, Prediction, LostTrackView


UPSTREAM_HASHES = {
    "trackers/botsort/botsort.py": "d8b69ec680b59e915b079994756c27b4bdf81ecd214a2866b97b31122fcdc93d",
    "trackers/botsort/botsort_track.py": "bcd42eaa3e9d06716fa71749b8983bf7c16d2a96b5829bb97f5c549e0f0cb0bc",
    "trackers/basetracker.py": "f5eb9838128d999944c444733b18948ae895c3682977eee1e4eeb07046ca04a1",
    "motion/kalman_filters/aabb/xywh_kf.py": "a12ea28981ac7044bd1c7a60d3701b54754bc88fbd7770eda3fecd4a02a735a2",
    "utils/matching.py": "c56c0d03e78dbe10e17761b2e95fd4f9064f1cd864d47ac0e45243a907d42096",
}


def verify_upstream():
    import boxmot
    root = Path(boxmot.__file__).resolve().parent
    actual = {}
    for relative, expected in UPSTREAM_HASHES.items():
        path = root / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"Unsupported BoxMOT source: {path}. Expected the 16.0.11 wheel source; "
                               "inspect the difference before adapting this patch.")
        actual[relative] = digest
    return actual


class CacheOnlyEncoder:
    def get_features(self, *args, **kwargs):
        raise RuntimeError("Provide real precomputed RGB features; uncached extraction is disabled")


class BoTSORTPlus(BotSort):
    def __init__(self, plus_config=None, **kwargs):
        self.plus = plus_config or PlusConfig()
        self.plus.validate_v2()
        self.upstream_hashes = verify_upstream()
        if kwargs.get("per_class",False) or kwargs.get("is_obb",False):
            raise ValueError("This patch implements the baseline's pooled axis-aligned protocol only")
        kwargs.update(self.plus.native_overrides)
        with_reid = kwargs.pop("with_reid", True)
        # No global monkey-patch and no model download: every run consumes the
        # same explicit feature cache, including the completely disabled run.
        super().__init__(with_reid=False, **kwargs)
        self.with_reid = with_reid
        self.model = CacheOnlyEncoder()
        if self.plus.appearance.enabled and not self.with_reid:
            raise ValueError("C requires with_reid=True")
        if self.plus.identity.enabled and not self.with_reid:
            raise ValueError("Identity reliability (C2) requires with_reid=True")
        if self.plus.recovery.enabled and self.plus.recovery.max_age > self.max_time_lost:
            raise ValueError("B max_age exceeds the actual FPS-scaled native buffer; "
                             "extend the native buffer explicitly and run the buffer-only control")
        self.candidate_pool = CandidatePool(self.plus.birth, self.plus.geometry)
        self.recovery_matcher = RecoveryMatcher(self.plus.recovery, self.plus.geometry)
        self.memories = {}
        self.prediction_history = {}
        self.last_events = []
        self._packets = []
        self._rgb_dim = None
        if self.plus.adapter_mode == "v2":
            from .adapter_v2 import verify_v2_upstream
            self.upstream_hashes.update(verify_v2_upstream())
            from .birth_policy import BirthPolicy
            from .birth_identity import BirthIdentityGuard
            if (self.plus.birth.enabled and self.plus.birth_identity.enabled and
                    self.plus.birth_identity.max_age > self.max_time_lost):
                raise ValueError("A2 identity max_age exceeds FPS-scaled native retention")
            self.birth_policy = BirthPolicy(self.plus.birth_policy)
            self.birth_identity = BirthIdentityGuard(self.plus.birth_identity, self.plus.geometry)
            self.track_origins = {}
        if self.plus.identity.enabled:
            from .identity_reliability import IdentityReliability
            self.identity_reliability = IdentityReliability(self.plus.identity)

    def _features(self, dets, embs, features):
        n = len(dets)
        if features is not None and len(features) != n:
            raise ValueError("Feature packets must align with ALL detection rows")
        packets = [dict(x) for x in features] if features is not None else [{} for _ in range(n)]
        if embs is not None:
            embs = np.asarray(embs, dtype=np.float32).copy()
            if embs.ndim != 2 or len(embs) != n or not np.isfinite(embs).all():
                raise ValueError("Invalid RGB embedding matrix")
            for i, vector in enumerate(embs):
                if np.linalg.norm(vector) > 1e-12 and "rgb" not in packets[i]:
                    packets[i]["rgb"] = Feature(vector.copy(), 1.0)
        # Validate without writing to identity memory. Quality-zero embeddings
        # can still be real and needed for an unchanged native baseline.
        for packet in packets:
            appearance_details(None, packet, self.plus.appearance)
        vectors = []
        for i, packet in enumerate(packets):
            f = packet.get("rgb")
            v = None if f is None else np.asarray(f.vector, np.float32).copy()
            if v is not None and np.linalg.norm(v) > 1e-12:
                self._rgb_dim = len(v) if self._rgb_dim is None else self._rgb_dim
                if len(v) != self._rgb_dim:
                    raise ValueError("RGB embedding dimension changed")
            else:
                v = None
            if self.with_reid and float(dets[i,4]) > self.track_high_thresh and v is None:
                raise ValueError(f"High-score detection {i} has no real RGB ReID feature")
            vectors.append(v)
        dense = np.zeros((n, self._rgb_dim or 0), dtype=np.float32)
        for i,v in enumerate(vectors):
            if v is not None:
                dense[i] = v
        self._packets = packets
        return dense

    def _trusted_rgb(self, det_index):
        feature = self._packets[det_index].get("rgb")
        if feature is None or feature.quality < self.plus.appearance.min_update_quality:
            return None
        vector = np.asarray(feature.vector, np.float32)
        return vector.copy() if np.linalg.norm(vector) > 1e-12 else None

    def _detection(self, row, trusted_only=False):
        index = int(row[6])
        feature = self._packets[index].get("rgb") if self.with_reid else None
        vector = None if feature is None else np.asarray(feature.vector,np.float32).copy()
        if vector is not None and np.linalg.norm(vector) <= 1e-12:
            vector = None
        if trusted_only or self.plus.appearance.enabled:
            vector = self._trusted_rgb(index) if self.with_reid else None
        return STrack(row, vector, max_obs=self.max_obs)

    def _write_memory(self, track, det_index):
        memory = self.memories.setdefault(track.id, MemoryBank(self.plus.appearance))
        packet = self._packets[det_index]
        if not self.plus.appearance.enabled:
            packet = {k:v for k,v in packet.items() if k == "rgb"}
        updated = memory.update(packet)
        if updated:
            self.last_events.append({"event":"memory_update", "track_id":int(track.id),
                                     "det_index":int(det_index), "modalities":updated})
        return updated

    def _cost(self, tracks, detections, unconfirmed=False):
        ious = iou_distance(tracks,detections)
        mask = ious > self.proximity_thresh
        base = fuse_score(ious,detections) if unconfirmed or self.fuse_first_associate else ious
        if not self.with_reid or base.size == 0:
            return base
        appearance = np.ones_like(base)
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                if mask[i,j]:
                    continue
                if self.plus.appearance.enabled:
                    distance = appearance_distance(self.memories.get(track.id),
                                                   self._packets[int(det.det_ind)], self.plus.appearance)
                    threshold = self.plus.appearance.max_distance
                else:
                    a,b = track.smooth_feat, det.curr_feat
                    distance = None
                    if a is not None and b is not None and np.linalg.norm(a)>0 and np.linalg.norm(b)>0:
                        distance = max(0., float(cdist(np.asarray(a).reshape(1,-1),
                                                      np.asarray(b).reshape(1,-1),"cosine")[0,0]))
                    threshold = self.appearance_thresh
                if distance is not None:
                    value = distance / 2 if unconfirmed else distance
                    gate_value = distance if self.plus.appearance.enabled else value
                    if gate_value <= threshold:
                        appearance[i,j] = value
        return np.minimum(base,appearance)

    def _associate(self, tracks, detections, threshold, stage, activated, refound, unconfirmed=False):
        if self.plus.identity.enabled:
            from .identity_adapter import associate
            return associate(self, tracks, detections, threshold, stage,
                             activated, refound, unconfirmed)
        costs = self._cost(tracks,detections,unconfirmed=unconfirmed)
        matches, u_tracks, u_detections = linear_assignment(costs, thresh=threshold)
        for i,j in matches:
            track, det = tracks[int(i)],detections[int(j)]
            detail = (appearance_details(self.memories.get(track.id),self._packets[int(det.det_ind)],
                                         self.plus.appearance) if self.plus.appearance.enabled else None)
            # All pair costs were calculated BEFORE ANY memory update.
            if track.state == TrackState.Tracked:
                track.update(det,self.frame_count); activated.append(track)
            else:
                track.re_activate(det,self.frame_count,new_id=False); refound.append(track)
            self._write_memory(track,int(det.det_ind))
            self.last_events.append({"event":"match", "stage":stage, "track_id":int(track.id),
                                     "det_index":int(det.det_ind), "cost":float(costs[i,j]),
                                     **({"appearance":detail} if detail is not None else {})})
        return list(u_tracks),list(u_detections)

    def update(self, dets, img, embs=None, features=None):
        dets = np.asarray(dets,dtype=np.float32).copy().reshape(-1,6)
        self.check_inputs(dets,img,embs)
        if not np.isfinite(dets).all() or np.any(dets[:,2:4] <= dets[:,:2]):
            raise ValueError("Detections must be finite positive xyxy boxes")
        if np.any((dets[:,4]<0)|(dets[:,4]>1)) or np.any(dets[:,5] != 0):
            raise ValueError("Expected confidence in [0,1] and class-agnostic class 0")
        dense = self._features(dets,embs,features)
        self.last_events = []
        if not self.plus.enabled:
            return super().update(dets,img,embs=dense if self.with_reid else None)
        if self.plus.adapter_mode == "v2":
            from .adapter_v2 import update_v2
            return update_v2(self, dets, img, dense)
        self.frame_count += 1
        frame = self.frame_count
        activated,refound,lost,removed = [],[],[],[]
        all_dets,high,_,low = self._split_detections(dets,dense)
        detections = [self._detection(r) for r in high]
        unconfirmed, active = self._separate_tracks()
        # An expired/removed identity may never be resurrected by B.
        old_lost = [t for t in self.lost_stracks if t.state == TrackState.Lost]
        pool = joint_stracks(active,old_lost)
        STrack.multi_predict(pool)
        warp = self.cmc.apply(img,all_dets)
        if warp is None or np.shape(warp)!=(2,3) or not np.isfinite(warp).all():
            raise ValueError("GMC returned an invalid transform")
        STrack.multi_gmc(pool,warp); STrack.multi_gmc(unconfirmed,warp)
        if self.plus.recovery.enabled:
            for track in pool:
                _,cov = self.kalman_filter.project(track.mean,track.covariance)
                history = self.prediction_history.setdefault(track.id,{})
                history[frame] = Prediction(frame,track.xyxy.copy(),cov[:2,:2].copy())
                for old_frame in list(history):
                    if old_frame <= frame - self.plus.recovery.window:
                        del history[old_frame]
        unmatched_pool, unmatched_high = self._associate(pool,detections,self.match_thresh,
                                                       "high",activated,refound)
        # Preserve native low-score geometry-only second association.
        low_tracks = [pool[i] for i in unmatched_pool if pool[i].state == TrackState.Tracked]
        low_detections = [STrack(r,max_obs=self.max_obs) for r in low]
        second_cost = iou_distance(low_tracks,low_detections)
        second_matches,unmatched_second_tracks,unmatched_low = linear_assignment(second_cost,thresh=.5)
        for i,j in second_matches:
            track, det = low_tracks[int(i)],low_detections[int(j)]
            track.update(det,frame); activated.append(track)
            # C/B memory can receive a real, quality-qualified low-score crop;
            # the native smooth_feat is untouched in this native second stage.
            self._write_memory(track,int(det.det_ind))
            self.last_events.append({"event":"match","stage":"low","track_id":int(track.id),
                                     "det_index":int(det.det_ind),"cost":float(second_cost[i,j])})
        for i in unmatched_second_tracks:
            track=low_tracks[int(i)]; track.mark_lost(); lost.append(track)
        remaining_high = [detections[i] for i in unmatched_high]
        unmatched_unc, remaining_indices = self._associate(unconfirmed,remaining_high,.7,
                                                         "unconfirmed",activated,refound,True)
        for i in unmatched_unc:
            track=unconfirmed[i]; track.mark_removed(); removed.append(track)
        birth_high = [remaining_high[i] for i in remaining_indices]
        current_candidates=[]; reserved=set(); consumed=set(); suppressed_births=set()
        if self.plus.birth.enabled or self.plus.recovery.enabled:
            residual = [low_detections[int(i)] for i in unmatched_low] + birth_high
            observations = [Observation(frame,int(d.det_ind),all_dets[int(d.det_ind),:4],float(d.conf),
                                        self._packets[int(d.det_ind)]) for d in residual]
            before=set(self.candidate_pool.candidates)
            occupied=[t.xyxy.copy() for t in joint_stracks(activated,refound)]
            occupied += [all_dets[int(t.det_ind),:4].copy() for t in joint_stracks(activated,refound)]
            current_candidates=self.candidate_pool.update(frame,observations,warp=warp,occupied_boxes=occupied)
            suppressed_births={int(e["det_index"]) for e in self.candidate_pool.events
                               if e["event"]=="candidate_observation_suppressed" or
                               (self.plus.recovery.enabled and e["event"]=="candidate_seed_deferred")}
            self.last_events.extend(self.candidate_pool.events)
            self.recovery_matcher.forget(before-set(self.candidate_pool.candidates))
        if self.plus.recovery.enabled:
            eligible = {t.id:t for t in joint_stracks(old_lost,lost)
                        if t.state==TrackState.Lost and frame-t.end_frame<=self.max_time_lost}
            views=[LostTrackView(t.id,t.end_frame,self.prediction_history.get(t.id,{}),
                                self.memories.get(t.id)) for t in eligible.values()]
            def app_cost(memory,packet):
                packet = packet if self.plus.appearance.enabled else {k:v for k,v in packet.items() if k=="rgb"}
                return appearance_distance(memory,packet,self.plus.appearance)
            decision=self.recovery_matcher.match(frame,views,current_candidates,app_cost)
            reserved=decision.reserved_candidate_ids
            self.last_events.extend({"event":"recovery_check",**d} for d in decision.diagnostics)
            for match in decision.matches:
                candidate=self.candidate_pool.get(match.candidate_id)
                index=match.det_index; track=eligible[match.track_id]
                det=self._detection(all_dets[index],trusted_only=True)
                track.re_activate(det,frame,new_id=False); refound.append(track)
                lost=[t for t in lost if t.id!=track.id]
                consumed.add(index)
                self._write_memory(track,index)
                self.candidate_pool.remove(match.candidate_id,"recovered")
                self.recovery_matcher.forget([match.candidate_id])
                self.last_events.append({"event":"recovered",**asdict(match)})
        # A acts only after B has matched or explicitly reserved the candidate.
        if self.plus.birth.enabled:
            for candidate in current_candidates:
                if candidate.candidate_id not in self.candidate_pool.candidates:
                    continue
                if candidate.candidate_id in reserved or not candidate.ready(frame):
                    continue
                index=candidate.latest.det_index
                if all_dets[index,4]>=self.new_track_thresh:
                    continue  # retain native high-score initialization below
                track=self._detection(all_dets[index],trusted_only=True)
                track.activate(self.kalman_filter,frame)
                track.is_activated=True  # temporal candidate already supplied confirmation
                activated.append(track); consumed.add(index)
                self._write_memory(track,index)
                self.candidate_pool.remove(candidate.candidate_id,"born")
                self.recovery_matcher.forget([candidate.candidate_id])
                self.last_events.append({"event":"weak_birth","track_id":int(track.id),
                                         "candidate_id":candidate.candidate_id,"det_index":index,
                                         "evidence":float(candidate.evidence),
                                         "hits":candidate.hit_count(frame,self.plus.birth.window)})
        candidates_by_index={c.latest.det_index:c for c in current_candidates
                             if c.candidate_id in self.candidate_pool.candidates}
        for track in birth_high:
            index=int(track.det_ind); candidate=candidates_by_index.get(index)
            if index in consumed or track.conf<self.new_track_thresh:
                continue
            if index in suppressed_births:
                self.last_events.append({"event":"suppressed_native_birth","det_index":index,
                                         "reason":"duplicate_or_candidate_ambiguity"})
                continue
            if candidate is not None and candidate.candidate_id in reserved:
                self.last_events.append({"event":"deferred_birth","det_index":index,
                                         "candidate_id":candidate.candidate_id})
                continue
            track.activate(self.kalman_filter,frame); activated.append(track)
            self._write_memory(track,index)
            if candidate is not None:
                self.candidate_pool.remove(candidate.candidate_id,"native_birth")
                self.recovery_matcher.forget([candidate.candidate_id])
            self.last_events.append({"event":"native_birth","track_id":int(track.id),"det_index":index})
        self._update_track_states(removed)
        output=self._prepare_output(activated,refound,lost,removed)
        live={t.id for t in self.active_tracks+self.lost_stracks if t.state!=TrackState.Removed}
        self.memories={k:v for k,v in self.memories.items() if k in live}
        self.prediction_history={k:v for k,v in self.prediction_history.items() if k in live}
        self.last_events.insert(0,{"event":"frame","frame":frame,"detections":len(dets),
                                   "active":len(self.active_tracks),"lost":len(self.lost_stracks),
                                   "candidates":len(self.candidate_pool.candidates)})
        return output.reshape(-1,8)

    def finish_sequence_events(self):
        """Read-only origin/lifetime summaries; never used for online decisions."""
        if self.plus.adapter_mode != "v2":
            return []
        return [{"event": "track_lifetime", "frame": self.frame_count,
                 "track_id": int(tid), **dict(info)}
                for tid, info in sorted(self.track_origins.items())]
