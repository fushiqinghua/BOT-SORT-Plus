"""Read-only W R0 frame capture on the pinned VT-Tiny-MOT worker."""
from collections import Counter, deque
import gzip
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


WORKER_BLOB = "1611bb7f5527cd7ca9fd539b9535ab763325a416"
ADAPTER_BLOB = "6818f98fb9b02b32c33e89e603e68a0f06928f79"
SCHEMA = "wem_r0_snapshot_v1"
STATE_NAMES = {0: "New", 1: "Tracked", 2: "Lost", 3: "Removed"}


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one pinned source boundary: {old}")
    return source.replace(old, new)


def fingerprint(array):
    if array is None:
        return None
    value = np.ascontiguousarray(array)
    return [str(value.dtype), list(value.shape), hashlib.sha256(value.tobytes()).hexdigest()]


def brief(track):
    state = int(track.state)
    return {"track_id": int(track.id), "state": state, "state_name": STATE_NAMES.get(state, "Unknown"),
            "confirmed": bool(track.is_activated), "last_frame": int(track.frame_id),
            "start_frame": int(track.start_frame), "last_det_index": int(track.det_ind)}


def full_track(track):
    return [brief(track), float(track.conf), int(track.tracklet_len), fingerprint(track.mean),
            fingerprint(track.covariance), fingerprint(track.smooth_feat), fingerprint(track.curr_feat)]


def current_assignments(events):
    for row in events:
        event = row.get("event")
        if event == "match":
            stage = row["stage"]
        elif event == "recovered":
            stage = "B_recovery"
        elif event == "native_birth":
            stage = event
        else:
            continue
        yield int(row["track_id"]), int(row["det_index"]), stage


def projected_backup(track, pre_mean, pre_covariance, warp, selected, guard_active):
    from boxmot.trackers.botsort.botsort_track import STrack
    # This exactly reproduces the native affine operator on copies. Shared KF
    # time prediction has already occurred for the original pool.
    detached = SimpleNamespace(mean=pre_mean.copy(), covariance=pre_covariance.copy())
    STrack.multi_gmc([detached], warp.copy())
    z = detached.mean[:4]
    xyxy = np.hstack((z[:2] - z[2:] / 2, z[:2] + z[2:] / 2))
    if guard_active and not np.allclose(xyxy, track.backup_boxes_for_audit, rtol=0, atol=1e-10):
        raise ValueError("Detached backup differs from the original dual-low geometry")
    if np.array_equal(warp, selected) and (
            not np.array_equal(detached.mean, track.mean) or
            not np.array_equal(detached.covariance, track.covariance)):
        raise ValueError("Equal warp produced a different detached prior")
    return xyxy.tolist(), detached.mean.tolist(), detached.covariance.tolist()


def track_state(tracker, output, gmc):
    from botsort_plus.adapter_v2 import STrack

    tracks = {name: [full_track(t) for t in getattr(tracker, name)]
              for name in ("active_tracks", "lost_stracks", "removed_stracks")}
    candidates = {}
    for cid, row in tracker.candidate_pool.candidates.items():
        candidates[cid] = [row.state_frame, row.first_frame, float(row.evidence),
                           fingerprint(row.last_box), fingerprint(row.velocity), fingerprint(row._anchor_box),
                           [[o.frame, o.det_index, float(o.score), fingerprint(o.box),
                             {m: [fingerprint(p.vector), float(p.quality)] for m, p in o.features.items()}]
                            for o in row.observations]]
    memory = {tid: dict(counts=row.update_counts, qualities=row.qualities,
                        prototypes={name: fingerprint(data) for name, data in row.prototypes.items()})
              for tid, row in tracker.memories.items()}
    predictions = {tid: {f: [fingerprint(p.box), fingerprint(p.center_cov)] for f, p in rows.items()}
                   for tid, rows in tracker.prediction_history.items()}
    return dict(frame=int(tracker.frame_count), output=fingerprint(output), tracks=tracks, gmc=gmc,
                candidate_pool=candidates, memory=memory, predictions=predictions,
                candidate_pool_clock=[tracker.candidate_pool._next_id, tracker.candidate_pool._frame],
                b_reservations=tracker.recovery_matcher._first_reserved,
                b_frame=tracker.recovery_matcher._last_frame, events=tracker.last_events,
                public_count=int(getattr(STrack, "_count", 0)))


class Observer:
    def __init__(self, path, seq, method, output, protocol, mode):
        from botsort_plus.io import detection_fingerprint, read_detection_cache

        self.mode, self.seq, self.method = mode, seq, method
        self.enabled = mode != "off"
        self.expected = protocol["sequences"][seq]
        self.previous = 0
        self.history = {}
        self.counts = Counter()
        self.snapshot = self.ownership = None
        self.snapshot_hash = hashlib.sha256()
        self.ownership_hash = hashlib.sha256()
        self.states = self.file(output, "states", ".sha256").open("xb")
        self.manifest = self.file(output, "capture_receipts", ".json")
        self.source = dict(detection_sha256=sha(path), detector_sha256=protocol["detector_sha256"],
                           detection_fingerprint=protocol["detection_fingerprints"][seq],
                           features=protocol["feature_hashes"][seq],
                           rgb_encoder_sha256=protocol["rgb_encoder_sha256"],
                           source_protocol_sha256=protocol["source_protocol_sha256"])
        if self.enabled:
            payload = read_detection_cache(path)
            if detection_fingerprint(payload) != self.source["detection_fingerprint"]:
                raise ValueError("Observer detector rows differ from the frozen protocol")
            self.rows = payload["frames"]
        if mode == "on":
            self.snapshot_path = self.file(output, "snapshots", ".jsonl.gz")
            self.ownership_path = self.file(output, "candidate_ownership", ".jsonl.gz")
            self.snapshot = gzip.open(self.snapshot_path, "xb", compresslevel=2)
            self.ownership = gzip.open(self.ownership_path, "xb", compresslevel=2)

    def file(self, root, kind, suffix):
        folder = root / kind / self.method
        folder.mkdir(parents=True, exist_ok=True)
        return folder / (self.seq + suffix)

    def start(self, tracker, detections, image, features):
        self.frame = int(tracker.frame_count) + 1
        if self.frame != self.previous + 1:
            raise ValueError("Nonconsecutive reference frame")
        self.gmc_passes = 0
        self.formal_predict_calls = 0
        if not self.enabled:
            return
        data = self.rows[self.frame - 1]
        raw = np.asarray(data["detections"], dtype=np.float32).reshape(-1, 6)
        if int(data["frame"]) != self.frame or not np.array_equal(raw[:, :5], detections[:, :5]):
            raise ValueError(f"{self.seq}/{self.frame}: frozen detector row order changed")
        if len(features) != len(raw):
            raise ValueError("Original feature packets and detection rows are not aligned")
        self.image = str(data["image"])
        self.image_shape = list(image.shape)
        self.thresholds = dict(low=float(tracker.track_low_thresh), high=float(tracker.track_high_thresh),
                               low_iou=.5)
        self.detections = []
        for i, row in enumerate(raw):
            score = float(row[4])
            band = "high" if score > self.thresholds["high"] else (
                "low" if self.thresholds["low"] < score < self.thresholds["high"] else (
                    "S0" if score <= self.thresholds["low"] else "high_equal_excluded"))
            self.detections.append(dict(det_index=i, raw_xyxy=row[:4].tolist(), raw_score=score,
                                        detector_class=int(row[5]), band=band, rgb_valid="rgb" in features[i]))
        self.start_tracks = []
        for collection in ("active_tracks", "lost_stracks", "removed_stracks"):
            for track in getattr(tracker, collection):
                row = brief(track)
                row["collection"] = collection
                if collection != "removed_stracks":
                    row["raw_history"] = list(self.history.get(row["track_id"], ()))
                self.start_tracks.append(row)
        self.counts["empty_detector_frames"] += int(not len(raw))

    def pre_gmc(self, pool, unconfirmed, high, low):
        self.formal_predict_calls += 1
        if not self.enabled:
            return
        self.pool_ids = {int(track.id) for track in pool}
        self.pre_gmc_state = {int(track.id): (track.mean.copy(), track.covariance.copy())
                              for track in [*pool, *unconfirmed]}
        if [int(row[6]) for row in high] != [d["det_index"] for d in self.detections if d["band"] == "high"]:
            raise ValueError("Observer high score band diverged from native split")
        if [int(row[6]) for row in low] != [d["det_index"] for d in self.detections if d["band"] == "low"]:
            raise ValueError("Observer low score band diverged from native split")

    def compensated(self, tracker, pool, unconfirmed, warp):
        self.gmc_passes += 1
        self.gmc = dict(native=tracker.joint_native_warp.tolist(), selected=warp.tolist(),
                        backup=tracker.joint_backup_warp.tolist(),
                        backup_guard_active=bool(tracker.joint_dual_low and tracker.matched_camera),
                        predict_calls=self.formal_predict_calls, gmc_calls=self.gmc_passes)
        if self.formal_predict_calls != 1 or self.gmc_passes != 1:
            raise ValueError("A formal update was advanced more than once")
        if not self.enabled:
            return
        backup = tracker.joint_backup_warp
        self.post_gmc = []
        for track in [*pool, *unconfirmed]:
            tid = int(track.id)
            mean, cov = self.pre_gmc_state[tid]
            guard = self.gmc["backup_guard_active"]
            expected_box = tracker.backup_boxes.get(tid) if guard else None
            # Read the guard's independent existing box without writing to the
            # formal track. Its projection is defined for all candidate tracks.
            detached = SimpleNamespace(mean=track.mean, covariance=track.covariance,
                                       backup_boxes_for_audit=expected_box)
            xyxy, bmean, bcov = projected_backup(detached, mean, cov, backup, warp, expected_box is not None)
            self.post_gmc.append(dict(**brief(track), selected_xyxy=track.xyxy.tolist(),
                                      selected_mean=track.mean.tolist(), selected_covariance=track.covariance.tolist(),
                                      backup_xyxy=xyxy, backup_mean=bmean, backup_covariance=bcov,
                                      formal_prediction_count=int(tid in self.pool_ids), formal_gmc_count=1))

    def before_finalize(self, tracker, activated, refound, lost, removed, reserved, consumed):
        if not self.enabled:
            return
        self.pre_finalize = {name: [brief(track) for track in tracks]
                             for name, tracks in (("activated", activated), ("refound", refound),
                                                   ("lost", lost), ("removed", removed))}
        self.pre_consumed = sorted(consumed)
        self.candidates_current = []
        self.candidates_history_only = []
        self.reservations = sorted(reserved)
        for cid, candidate in tracker.candidate_pool.candidates.items():
            latest = candidate.latest
            item = dict(candidate_id=int(cid), last_observation_frame=int(latest.frame),
                        last_det_index=int(latest.det_index), reserved=cid in reserved,
                        prior_reservation_start=tracker.recovery_matcher._first_reserved.get(cid),
                        last_box=candidate.last_box.tolist(), last_raw_box=latest.box.tolist(),
                        observation_history=[[int(o.frame), int(o.det_index)] for o in candidate.observations])
            if int(latest.frame) == self.frame:
                row = self.detections[int(latest.det_index)]
                self.candidates_current.append(dict(**item, det_index=int(latest.det_index),
                                                    raw_xyxy=row["raw_xyxy"], raw_score=row["raw_score"]))
            else:
                self.candidates_history_only.append(item)

    def finish(self, tracker, output):
        if self.frame != int(tracker.frame_count):
            raise ValueError("Observer frame and reference tracker frame diverged")
        signature = hashlib.sha256(encode(track_state(tracker, output, self.gmc))).hexdigest()
        self.states.write(f"{self.frame} {signature}\n".encode())
        self.previous = self.frame
        self.counts["frames"] += 1
        if not self.enabled:
            return
        from experiments.vttiny_gmc.native_botsort_gmc_0030_fp_v1.guard import iou
        from boxmot.utils.matching import iou_distance

        accepted = []
        by_id = {}
        consumed = set()
        for tid, idx, stage in current_assignments(tracker.last_events):
            if tid in by_id or idx in consumed or not 0 <= idx < len(self.detections):
                raise ValueError("Reference observation index not unique in current frame")
            obs = dict(**self.detections[idx], frame=self.frame, track_id=tid, stage=stage)
            accepted.append(obs)
            by_id[tid] = obs
            consumed.add(idx)
            self.counts["accepted_" + stage] += 1
        reserved_det_indices = sorted(row["det_index"] for row in self.candidates_current if row["reserved"])
        if not set(self.pre_consumed) <= consumed or consumed & set(reserved_det_indices):
            raise ValueError("Final original assignments and current B reservations conflict")
        alive_tracks = {int(t.id): t for t in tracker.active_tracks}
        outputs = []
        for row in output:
            tid, idx = int(row[4]), int(row[7])
            if tid not in by_id or int(by_id[tid]["det_index"]) != idx or tid not in alive_tracks:
                raise ValueError("Final output must resolve to a same-frame official detector row")
            track = alive_tracks[tid]
            outputs.append(dict(track_id=tid, det_index=idx, xyxy=row[:4].tolist(),
                                score=float(row[5]), posterior_mean=track.mean.tolist(),
                                posterior_covariance=track.covariance.tolist()))
        priors = {row["track_id"]: row for row in self.post_gmc}
        risk = [row["track_id"] for row in self.start_tracks
                if row["collection"] == "active_tracks" and row["confirmed"] and row["state"] == 1
                and row["track_id"] not in by_id]
        s0 = [row for row in self.detections if row["band"] == "S0"]
        costs = iou_distance([np.asarray(priors[tid]["selected_xyxy"]) for tid in risk],
                             [np.asarray(row["raw_xyxy"]) for row in s0])
        edges = []
        for i, tid in enumerate(risk):
            prior = priors[tid]
            for j, row in enumerate(s0):
                box = np.asarray(row["raw_xyxy"])
                cost = float(costs[i, j])
                backup = iou(np.asarray(prior["backup_xyxy"]), box)
                edges.append(dict(track_id=tid, det_index=row["det_index"], selected_iou=1.-cost,
                                  backup_iou=backup, native_low_feasible=cost <= .5,
                                  gmc_guard_feasible=not self.gmc["backup_guard_active"] or backup >= .5,
                                  independent_anchor_feasible=None))
        final = [dict(**brief(t), collection=name)
                 for name in ("active_tracks", "lost_stracks", "removed_stracks") for t in getattr(tracker, name)]
        ownership = dict(sequence=self.seq, reference=self.method, frame=self.frame, accepted=accepted,
                         consumed=sorted(consumed), consumed_b_and_birth=self.pre_consumed,
                         reserved=reserved_det_indices, reserved_candidate_ids=self.reservations,
                         candidates_current=self.candidates_current, candidates_history_only=self.candidates_history_only)
        frame = dict(schema=SCHEMA, image=self.image, image_shape=self.image_shape,
                     thresholds=self.thresholds, detections=self.detections, gmc=self.gmc,
                     start_tracks=self.start_tracks, post_gmc=self.post_gmc, pre_finalize=self.pre_finalize,
                     outputs=outputs, final_tracks=final, s0_edges=edges, at_risk_track_ids=risk,
                     anchor_status="NOT_IMPLEMENTED", **ownership)
        for record, digest, stream in ((frame, self.snapshot_hash, self.snapshot),
                                       (ownership, self.ownership_hash, self.ownership)):
            data = encode(record) + b"\n"
            digest.update(data)
            if stream is not None:
                stream.write(data)
        if self.mode == "force_reject":
            self.counts["forced_rejections"] += len(edges)
        for obs in accepted:
            self.history.setdefault(obs["track_id"], deque(maxlen=3)).append(obs)
        live = set(tracker.memories)
        self.history = {tid: rows for tid, rows in self.history.items() if tid in live}
        self.counts.update(s0=len(s0), at_risk_tracks=len(risk), s0_edges=len(edges),
                           reserved_current=len(reserved_det_indices), output_rows=len(outputs),
                           candidates_history_only=len(self.candidates_history_only),
                           geometry_feasible_edges=sum(1 for e in edges if e["native_low_feasible"]
                                                       and e["gmc_guard_feasible"]))

    def close(self):
        self.states.close()
        if self.previous != self.expected:
            raise ValueError("Observer did not cover the full reference sequence")
        result = dict(schema=SCHEMA, method=self.method, sequence=self.seq, mode=self.mode,
                      source=self.source, counts=self.counts, snapshot_applied=False,
                      gt_open_allowed=False, snapshots_decoded_sha256=self.snapshot_hash.hexdigest(),
                      ownership_decoded_sha256=self.ownership_hash.hexdigest())
        if self.snapshot is not None:
            self.snapshot.close()
            self.ownership.close()
            result.update(snapshots_sha256=sha(self.snapshot_path), ownership_sha256=sha(self.ownership_path))
        self.manifest.write_bytes(encode(result) + b"\n")


def overlays():
    adapter = importlib.import_module("botsort_plus.adapter_v2")
    original = pinned_source(adapter, ADAPTER_BLOB).decode("utf-8")
    source = replace_once(original, "    STrack.multi_predict(pool)\n",
                          "    STrack.multi_predict(pool)\n    tracker.wem.pre_gmc(pool, unconfirmed, high, low)\n")
    source = replace_once(source, "    STrack.multi_gmc(unconfirmed, warp)\n",
                          "    STrack.multi_gmc(unconfirmed, warp)\n"
                          "    tracker.wem.compensated(tracker, pool, unconfirmed, warp)\n")
    source = replace_once(source, "    tracker._update_track_states(removed)\n",
                          "    tracker.wem.before_finalize(tracker, activated, refound, lost, removed, reserved, consumed)\n"
                          "    tracker._update_track_states(removed)\n")
    exec(compile(source, adapter.__file__, "exec"), adapter.__dict__)

    worker = importlib.import_module("experiments.vttiny_gmc.round3_joint.worker")
    source = pinned_source(worker, WORKER_BLOB).decode()
    source = replace_once(source, "                self.audit_closed = False\n",
                          "                self.audit_closed = False\n"
                          "                self.wem = _WemObserver(paths[self.seq], self.seq, method, output, protocol, _wem_mode)\n")
    source = replace_once(source, "                result = super().update(detections, image, *a, **kw)\n",
                          "                self.wem.start(self, detections, image, kw['features'])\n"
                          "                result = super().update(detections, image, *a, **kw)\n"
                          "                self.wem.finish(self, result)\n")
    source = replace_once(source, "            def finish_sequence_events(self):\n",
                          "            def finish_sequence_events(self):\n                self.wem.close()\n")
    worker.__dict__.update(_WemObserver=Observer)
    exec(compile(source, worker.__file__, "exec"), worker.__dict__)
    return worker


def pinned_source(module, expected):
    raw = Path(module.__file__).read_bytes().replace(b"\r\n", b"\n")
    blob = b"blob " + str(len(raw)).encode() + b"\0" + raw
    if hashlib.sha1(blob).hexdigest() != expected:
        raise ValueError(f"Pinned source provenance differs: {module.__file__}")
    return raw
