"""Replay the exact frozen native high and B costs from detached shadow inputs."""
from __future__ import annotations

from dataclasses import asdict
import math
from types import SimpleNamespace

import numpy as np

from boxmot.utils.matching import fuse_score, iou_distance, linear_assignment
from botsort_plus.appearance import MemoryBank, appearance_distance
from botsort_plus.recovery import RecoveryMatcher
from botsort_plus.types import LostTrackView, Observation, Prediction


def numbers(rows, shape):
    return np.asarray(rows, dtype=float).reshape(shape)


def verify_value(a, b, context):
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            raise ValueError((context, a.keys(), b.keys()))
        for key in a:
            verify_value(a[key], b[key], (context, key))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise ValueError((context, len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            verify_value(x, y, (context, i))
    elif type(a) is float and type(b) is float:
        if not math.isclose(a, b, rel_tol=0., abs_tol=1e-10):
            raise ValueError((context, a, b))
    elif a != b:
        raise ValueError((context, a, b))


def verify_matrix(actual, expected, context):
    if actual.shape != expected.shape or not np.allclose(
            actual, expected, rtol=0., atol=1e-10, equal_nan=True):
        if actual.shape == expected.shape and actual.size:
            wrong = np.argwhere(~np.isclose(actual, expected, rtol=0., atol=1e-10, equal_nan=True))
            row, col = wrong[0]
            detail = (row, col, actual[row, col], expected[row, col])
        else:
            detail = (actual.shape, expected.shape)
        raise ValueError((context, detail))


def moved_box(box, center):
    original = np.asarray(box, dtype=float)
    if center is None:
        return original
    shift = np.asarray(center) - (original[:2] + original[2:]) * .5
    return original + np.tile(shift, 2)


def high_result(frame, native, centers=None):
    native_tracks, native_dets = native["tracks"], native["detections"]
    if centers is None:
        centers = {}
    tracks = [SimpleNamespace(xyxy=moved_box(track[3:], centers.get(track[0]))) for track in native_tracks]
    detections = [SimpleNamespace(xyxy=np.asarray(d[2:], dtype=float), conf=d[1]) for d in native_dets]
    dims = (len(tracks), len(detections))
    ious = iou_distance(tracks, detections)
    assert ious.shape == dims
    raw_app = numbers(native["appearance_raw"], dims)
    near = ious <= native["proximity_thresh"]
    app_ok = np.isfinite(raw_app) & (raw_app <= native["appearance_thresh"])
    base = fuse_score(ious, detections) if native["fuse_score"] else ious
    final = np.minimum(base, np.where(near & app_ok, raw_app, 1.))
    result, _, _ = linear_assignment(final, thresh=native["threshold"])
    matches = [[int(i), int(j)] for i, j in result]
    if not centers:
        verify_matrix(ious, numbers(native["iou_distance"], dims), (frame, "high_iou"))
        verify_matrix(final, numbers(native["cost"], dims), (frame, "high_final_cost"))
        verify_value(matches, native["matches"], (frame, "high_pairs"))
    return {"pairs": {(native_tracks[i][0], native_dets[j][0]): float(final[i,j]) for i,j in matches},
            "iou_distance": ious, "appearance_raw": raw_app, "cost": final, "near": near,
            "row": {t[0]:i for i,t in enumerate(native_tracks)},
            "column": {d[0]:j for j,d in enumerate(native_dets)}}


class MatrixCapture:
    def __init__(self,dims):
        self.raw = np.full(dims, np.inf, dtype=float)
        self.eligible = np.full(dims, np.inf, dtype=float)

    def b_cost(self,raw,eligible):
        self.raw=raw.copy()
        self.eligible=eligible.copy()


class Replay:
    def __init__(self, config, packets):
        self.config = config
        self.packets = packets
        self.memories = {}
        self.candidates = []
        self.shadow = None
        self.cache = {}

    def frame(self, payload):
        b = payload["b"]
        self.shadow = b
        self.cache.clear()
        for key, row in b["memory_updates"].items():
            if key in self.memories:
                raise ValueError(f"Duplicate memory snapshot {key}")
            memory = MemoryBank(self.config.appearance)
            memory.prototypes = {name: np.asarray(vec,dtype=float) for name,vec in row["prototypes"].items()}
            memory.qualities = row["qualities"]
            memory.update_counts = row["update_counts"]
            self.memories[key] = memory
        now=payload["frame"]
        self.candidates=[]
        for c in b["candidates"]:
            history=[]
            for frame,index,score,xyxy in c["obs"]:
                if not (now-self.config.recovery.window < frame <= now):
                    raise ValueError((now, frame, "B observation outside causal window"))
                packet = self.packets(frame)[index]
                history.append(Observation(frame,index,np.asarray(xyxy),score,
                    {k:v for k,v in packet.items() if k=="rgb"}))
            self.candidates.append(SimpleNamespace(candidate_id=c["id"], observations=history,
                                                   last_frame=c["last_frame"]))

    def app(self, memory, features):
        key = id(memory),id(features)
        if key not in self.cache:
            self.cache[key] = appearance_distance(memory,features,self.config.appearance)
        return self.cache[key]

    def recovery(self, frame, centers=None):
        b = self.shadow
        lookup = centers or {}
        tracks=[]
        for snapshot in b["tracks"]:
            tid = snapshot["id"]
            preds = {f: Prediction(f, moved_box(box, lookup.get((f,tid))), np.asarray(cov,dtype=float))
                     for f,box,cov in snapshot["predictions"]}
            key = snapshot["memory"]
            tracks.append(LostTrackView(tid,snapshot["last_seen"],preds,
                                        self.memories[key] if key is not None else None))
        dims = len(tracks),len(self.candidates)
        probe = MatrixCapture(dims)
        matcher = RecoveryMatcher(self.config.recovery,self.config.geometry)
        matcher._first_reserved = {int(k):v for k,v in b["first_reserved"].items()}
        matcher.e1_shadow = probe
        decision = matcher.match(frame, tracks, self.candidates, self.app)
        if not centers:
            verify_value([asdict(x) for x in decision.matches],b["matches"],(frame,"B_matches"))
            verify_value(sorted(decision.reserved_candidate_ids),b["reserved"],(frame,"B_reserved"))
            verify_value(decision.diagnostics,b["diagnostics"],(frame,"B_reasons"))
            for name,values in (("raw",probe.raw),("eligible",probe.eligible)):
                expected=numbers(b[name],dims)
                expected=np.where(np.isnan(expected),np.inf,expected)
                verify_matrix(values,expected,(frame,"B_"+name))
        return {
            "pairs":{(match.track_id,match.det_index):match.cost for match in decision.matches},
            "raw":probe.raw, "eligible":probe.eligible,
            "diagnostics":{(row["track_id"],row["candidate_id"]):row for row in decision.diagnostics},
            "columns":{c.observations[-1].det_index:(i,c.candidate_id) for i,c in enumerate(self.candidates)},
            "rows":{track.track_id:i for i,track in enumerate(tracks)},
        }
