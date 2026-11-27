"""Causal observation-only capture around the pinned Round 3 tracker."""
from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
import gzip
import hashlib
import importlib
import json
from pathlib import Path

WORKER_BLOB = "1611bb7f5527cd7ca9fd539b9535ab763325a416"
METHODS = ("B_early_moderate", "evidence_both")
MAX_HISTORY, MAX_AGE, MIN_HISTORY = 3, 12, 2
CUT = 0.5


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def roi_for(box, shape, *, history=False):
    """Continuous box edges mapped by symmetric letterbox to continuous output edges."""
    height, width = shape[:2]
    x1, y1, x2, y2 = box
    factor, minimum, size = (1.5, 12, 64) if history else (2.5, 24, 96)
    if not (x2 > x1 and y2 > y1):
        return {"eligible": False, "reason": "invalid_box", "output_size": size}
    rw, rh = max(minimum, factor * (x2 - x1)), max(minimum, factor * (y2 - y1))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    request = [cx - rw / 2, cy - rh / 2, cx + rw / 2, cy + rh / 2]
    bounds = [width, height, width, height]
    valid = [min(bound, max(0, value)) for bound, value in zip(bounds, request)]
    fraction = (valid[2] - valid[0]) * (valid[3] - valid[1]) / (rw * rh)
    reason = "roi_axis_exceeds_128" if max(rw, rh) > 128 else (
        "roi_valid_fraction_below_half" if fraction < 0.5 else None)
    scale = size / max(rw, rh)
    return dict(requested_xyxy=request, valid_xyxy=valid, valid_fraction=fraction,
                scale=scale, padding_xy=[(size-scale*rw)/2, (size-scale*rh)/2],
                output_size=size, eligible=reason is None, reason=reason)


def accepted_pairs(events):
    """Only original events following a real tracking state update."""
    for event in events:
        kind = event.get("event")
        if kind == "match":
            stage = event["stage"]
        elif kind in ("native_birth", "recovered"):
            stage = "B_recovery" if kind == "recovered" else kind
        else:
            continue
        yield int(event["track_id"]), int(event["det_index"]), stage


def array_state(value):
    import numpy as np
    if value is None:
        return None
    value = np.ascontiguousarray(value)
    return [str(value.dtype), list(value.shape), hashlib.sha256(value.tobytes()).hexdigest()]


def state_digest(tracker, output):
    """Hash public IDs, full KF means/covariances, trusted memory, and B history."""
    from botsort_plus.adapter_v2 import STrack

    tracks = {}
    for state in ("active_tracks", "lost_stracks", "removed_stracks"):
        tracks[state] = [
            [int(t.id), int(t.state), bool(t.is_activated), int(t.frame_id), int(t.start_frame),
             int(t.det_ind), float(t.conf), array_state(t.mean), array_state(t.covariance),
             array_state(t.smooth_feat)] for t in getattr(tracker, state)
        ]
    memory = {
        str(tid): {"update_counts": bank.update_counts, "qualities": bank.qualities,
                   "prototypes": {k: array_state(v) for k, v in bank.prototypes.items()}}
        for tid, bank in tracker.memories.items()
    }
    candidates = {
        str(cid): [[o.frame, o.det_index, float(o.score), o.box.tolist()]
                   for o in candidate.observations]
        for cid, candidate in tracker.candidate_pool.candidates.items()
    }
    return hashlib.sha256(encode(dict(frame=tracker.frame_count, output=array_state(output),
        public_count=int(getattr(STrack, "_count", 0)), tracks=tracks, memory=memory,
        candidates=candidates, reservations=tracker.recovery_matcher._first_reserved))).hexdigest()


class Observer:
    def __init__(self, path, feature, source_video, sequence, method, output, protocol, enabled):
        from botsort_plus.io import detection_fingerprint, read_detection_cache

        self.enabled = enabled
        self.sequence, self.method, self.source_video = sequence, method, source_video
        self.history = {}
        self.previous = 0
        self.chain = hashlib.sha256()
        self.counts = Counter()
        self.feature_offset = 0
        self.sequence_length = protocol["sequences"][sequence]
        self.source = dict(
            detection_fingerprint=protocol["detection_fingerprints"][sequence],
            detection_file_sha256=file_hash(path),
            features=protocol["feature_hashes"][sequence],
            configuration_sha256=hashlib.sha256(encode(protocol["config"])).hexdigest(),
            baseline_source_protocol_sha256=protocol["source_protocol_sha256"],
            detector_sha256=protocol["detector_sha256"],
            rgb_encoder_sha256=protocol["rgb_encoder_sha256"],
        )
        if self.source["features"]["features"] != file_hash(feature):
            raise ValueError(f"{sequence}: feature changed since preflight")
        self.fingerprint = hashlib.sha256(encode(dict(
            method=method, sequences=protocol["sequences"], native=protocol["native_config"],
            code=protocol["joint_batch"]["source_sha256"], collector=file_hash(__file__),
            detection=protocol["detection_fingerprints"], features=protocol["feature_hashes"],
            configuration=self.source["configuration_sha256"],
            camera=protocol["joint_batch"]["camera"], gmc_mode=protocol["joint_batch"]["gmc_mode"],
        ))).hexdigest()
        directory = Path(output) / "tco_states" / method
        directory.mkdir(parents=True, exist_ok=True)
        self.state_path = directory / f"{sequence}.sha256"
        self.states = self.state_path.open("x", encoding="utf-8", newline="\n")
        self.capture = None
        if enabled:
            cache = read_detection_cache(path)
            if detection_fingerprint(cache) != self.source["detection_fingerprint"]:
                raise ValueError(f"{sequence}: observer detector rows differ from frozen cache")
            self.rows = cache["frames"]
            directory = Path(output) / "tco_capture" / method
            directory.mkdir(parents=True, exist_ok=True)
            self.capture_path = directory / f"{sequence}.jsonl.gz"
            self.capture = gzip.open(self.capture_path, "xt", encoding="utf-8",
                                     newline="\n", compresslevel=2)

    def begin(self, tracker, detections, image, *, features):
        import numpy as np
        self.frame = int(tracker.frame_count) + 1
        if self.frame != self.previous + 1:
            raise ValueError(f"{self.sequence}: nonconsecutive frame")
        if not self.enabled:
            return
        row = self.rows[self.frame - 1]
        self.raw = np.asarray(row["detections"], np.float32).reshape(-1, 6)
        if row["frame"] != self.frame or not np.array_equal(self.raw[:, :5], detections[:, :5]):
            raise ValueError(f"{self.sequence}/{self.frame}: detector row order mismatch")
        if len(features) != len(self.raw):
            raise ValueError("Original feature packets and detector rows must align")
        self.features = features
        self.image = str(row["image"])
        self.shape = list(image.shape)
        self.image_sha = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
        self.guard = None
        self.thresholds = [float(tracker.track_low_thresh), float(tracker.track_high_thresh)]

    def detection(self, idx, *, history):
        source = self.raw[idx]
        box = source[:4].tolist()
        return dict(det_index=idx, raw_xyxy=box, raw_score=float(source[4]),
                    raw_class=int(source[5]), feature_index=self.feature_offset + idx,
                    rgb_available="rgb" in self.features[idx],
                    roi=roi_for(box, self.shape, history=history))

    def low_matrix(self, tracks, rows, before, after):
        """Read the actual low-stage matrix immediately before its native solver."""
        import numpy as np
        ids = [int(track.id) for track in tracks]
        indices = [int(row[6]) for row in rows]
        expected = [i for i, raw in enumerate(self.raw) if
                    self.thresholds[0] < float(raw[4]) < self.thresholds[1]]
        if indices != expected or len(ids) != len(set(ids)):
            raise ValueError(f"{self.sequence}/{self.frame}: native low-stage partition mismatch")
        if np.shape(before) != (len(ids), len(indices)) or np.shape(after) != np.shape(before):
            raise ValueError("Low-stage matrices differ from node dimensions")
        if not (np.isfinite(before).all() and np.isfinite(after).all()):
            raise ValueError("Low-stage matrices contain nonfinite costs")
        track_rows = []
        for track in tracks:
            tid = int(track.id)
            old = [obs for obs in self.history.get(tid, ())
                   if self.frame - MAX_AGE <= obs["frame"] < self.frame]
            track_rows.append(dict(track_id=tid, pred_xyxy=track.xyxy.tolist(),
                track_state=int(track.state), confirmed=bool(track.is_activated),
                last_seen=int(track.end_frame), history=old,
                history_eligible=sum(obs["roi"]["eligible"] for obs in old) >= MIN_HISTORY))
        self.guard = dict(low_tracks=track_rows,
                          low_dets=[self.detection(i, history=False) for i in indices],
                          original_cost=before.tolist(), guarded_cost=after.tolist())

    def end(self, tracker, output):
        if int(tracker.frame_count) != self.frame:
            raise ValueError("Round 3 frame count changed unexpectedly")
        signature = state_digest(tracker, output)
        self.chain.update(bytes.fromhex(signature))
        self.states.write(f"{self.frame} {signature}\n")
        self.previous = self.frame
        if not self.enabled:
            return
        if self.guard is None:
            raise ValueError("The original low-stage guard was not observed")
        observations, matches = [], []
        assigned_ids, assigned_dets = set(), set()
        row_ids = {row["track_id"]: n for n, row in enumerate(self.guard["low_tracks"])}
        col_ids = {row["det_index"]: n for n, row in enumerate(self.guard["low_dets"])}
        for tid, det, stage in accepted_pairs(tracker.last_events):
            if tid in assigned_ids or det in assigned_dets or not 0 <= det < len(self.raw):
                raise ValueError(f"{self.sequence}/{self.frame}: duplicate or missing observation row")
            assigned_ids.add(tid)
            assigned_dets.add(det)
            if stage == "low":
                if self.guard["guarded_cost"][row_ids[tid]][col_ids[det]] > CUT:
                    raise ValueError("Native low-stage solver accepted an edge above threshold")
                matches.append([tid, det])
            observations.append(dict(**self.detection(det, history=True), frame=self.frame,
                track_id=tid, source_stage=stage, frame_image=self.image,
                image_sha256=self.image_sha, image_shape=self.shape))
        frame_data = dict(**self.guard, run_fingerprint=self.fingerprint, method=self.method,
            sequence=self.sequence, source_video=self.source_video, frame=self.frame,
            frame_image=self.image, image_sha256=self.image_sha, image_shape=self.shape,
            assignments=matches, observations=observations,
            outputs=[[int(row[4]), int(row[7])] for row in output],
            cost_threshold=CUT, score_thresholds=self.thresholds)
        self.capture.write(encode(frame_data).decode() + "\n")
        # Keep observations of actual updates, even when this frame was not emitted.
        for row in observations:
            self.history.setdefault(row["track_id"], deque(maxlen=MAX_HISTORY)).append(row)
        live = set(tracker.memories)
        self.history = {tid: rows for tid, rows in self.history.items()
                        if tid in live and rows[-1]["frame"] >= self.frame - MAX_AGE}
        self.feature_offset += len(self.raw)
        self.counts.update(frames=1, low_tracks=len(row_ids), low_detections=len(col_ids),
                           candidate_pairs=len(row_ids) * len(col_ids), assigned_low=len(matches),
                           observations=len(observations))

    def close(self):
        if self.previous != self.sequence_length:
            raise ValueError(f"{self.sequence}: incomplete observation stream")
        self.states.close()
        if self.capture is not None:
            self.capture.close()
            self.capture_path.with_suffix("").with_suffix(".json").write_bytes(encode(dict(
                run_fingerprint=self.fingerprint, sequence=self.sequence, method=self.method,
                source_video=self.source_video, **self.source, counts=self.counts,
                capture_sha256=file_hash(self.capture_path),
                state_digest_chain_sha256=self.chain.hexdigest(),
                state_file_sha256=file_hash(self.state_path),
                history=dict(max_observations=MAX_HISTORY, max_age=MAX_AGE, minimum=MIN_HISTORY),
                prediction_label_access=False)) + b"\n")


def one_replace(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Pinned worker requires exactly one observation boundary: {old}")
    return text.replace(old, new, 1)


def instrument(text):
    """Insert observation calls without modifying the pinned checkout."""
    text = one_replace(text, "                self.audit_closed = False\n",
                       "                self.audit_closed = False\n"
                       "                self.tco = _TcoObserver(paths[self.seq], args.features_dir / (self.seq + '.npz'),\n"
                       "                    _tco_groups.get(self.seq), self.seq, method, output, protocol, _tco_enabled)\n")
    text = one_replace(text, "            def guard_low_cost(self, cost, tracks, detections):",
                       "            def _native_guard_low_cost(self, cost, tracks, detections):")
    text = one_replace(text, "            def warp(self, image, detections):",
                       "            def guard_low_cost(self, cost, tracks, detections):\n"
                       "                before = cost.copy() if self.tco.enabled else None\n"
                       "                self._native_guard_low_cost(cost, tracks, detections)\n"
                       "                if self.tco.enabled:\n"
                       "                    self.tco.low_matrix(tracks, detections, before, cost)\n\n"
                       "            def warp(self, image, detections):")
    text = one_replace(text, "                result = super().update(detections, image, *a, **kw)",
                       "                self.tco.begin(self, detections, image, features=kw.get('features'))\n"
                       "                result = super().update(detections, image, *a, **kw)\n"
                       "                self.tco.end(self, result)")
    return one_replace(text, "            def finish_sequence_events(self):",
                       "            def finish_sequence_events(self):\n"
                       "                self.tco.close()")


def read_groups(path):
    if path is None:
        return {}
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as source:
            return {row["sequence"]: row["original_video"] for row in csv.DictReader(source)}
    result = json.loads(path.read_bytes())
    if not isinstance(result, dict) or not all(isinstance(value, str) for value in result.values()):
        raise ValueError("Source video groups must map sequence keys to explicit video keys")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--observer", choices=("off", "on"), required=True)
    parser.add_argument("--groups", type=Path)
    args = parser.parse_args()
    module = importlib.import_module("experiments.vttiny_gmc.round3_joint.worker")
    source = Path(module.__file__).read_bytes().replace(b"\r\n", b"\n")
    stamp = b"blob " + str(len(source)).encode() + b"\0" + source
    if hashlib.sha1(stamp).hexdigest() != WORKER_BLOB:
        raise ValueError("The Round 3 worker differs from the TCO branch's pinned worker blob")
    groups = read_groups(args.groups)
    if groups and args.sequences and not set(args.sequences) <= groups.keys():
        raise ValueError("Some requested videos are absent from the explicit source grouping")
    module.__dict__.update(_TcoObserver=Observer, _tco_groups=groups,
                           _tco_enabled=args.observer == "on")
    overlay = instrument(source.decode())
    exec(compile(overlay, str(module.__file__), "exec"), module.__dict__)
    return module.run(json.loads(args.settings.read_bytes()), args.method,
                      args.output_dir, args.sequences, probe_mode="off")


if __name__ == "__main__":
    raise SystemExit(main())
