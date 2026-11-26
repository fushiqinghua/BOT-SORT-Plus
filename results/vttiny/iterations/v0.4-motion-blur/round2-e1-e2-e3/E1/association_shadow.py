"""Read-only association telemetry for the frozen v0.4 native+B pipeline."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from boxmot.utils.matching import iou_distance


def matrix(array):
    raw = np.asarray(array, dtype=float)
    return [[float(x) if np.isfinite(x) else None for x in row] for row in raw]


def pairs(rows):
    return [[int(row), int(col)] for row, col in rows]


def snapshot_memory(memory):
    if memory is None:
        return None
    return {"prototypes": {name: np.asarray(value, dtype=float).tolist()
                           for name, value in memory.prototypes.items()},
            "qualities": {name: float(value) for name, value in memory.qualities.items()},
            "update_counts": {name: int(value) for name, value in memory.update_counts.items()}}


class ShadowRecorder:
    def __init__(self, path, sequence, frames):
        self.path = Path(path)
        self.partial = self.path.with_suffix(".partial")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() or self.partial.exists():
            raise FileExistsError(self.path)
        self.stream = gzip.open(self.partial, "xt", compresslevel=2, encoding="utf-8", newline="\n")
        self.sequence, self.frames = sequence, frames
        self.count = 0
        self.frame = None
        self.memory_versions = {}

    def begin(self, tracker, frame):
        cfg = tracker.plus
        if (self.frame is not None or frame != self.count + 1 or cfg.adapter_mode != "v2"
                or not cfg.recovery.enabled or not tracker.with_reid or cfg.identity.enabled
                or cfg.appearance.enabled or cfg.birth.enabled or cfg.raw_ir.enabled
                or cfg.recovery.output_prediction_frames):
            raise ValueError("Unexpected frozen E1 protocol or duplicate frame")
        self.frame = {"sequence": self.sequence, "frame": int(frame)}

    def native(self, tracker, tracks, dets, threshold, costs, matches):
        geometric = iou_distance(tracks, dets)
        raw_app = np.full(geometric.shape, np.nan, dtype=float)
        row_idx = [i for i, t in enumerate(tracks)
                   if t.smooth_feat is not None and np.linalg.norm(t.smooth_feat)>0]
        col_idx = [j for j, d in enumerate(dets)
                   if d.curr_feat is not None and np.linalg.norm(d.curr_feat)>0]
        if row_idx and col_idx:
            left = np.stack([np.asarray(tracks[i].smooth_feat) for i in row_idx])
            right = np.stack([np.asarray(dets[j].curr_feat) for j in col_idx])
            raw_app[np.ix_(row_idx, col_idx)] = np.maximum(0., cdist(left, right, "cosine"))
        if not np.isfinite(costs).all():
            raise ValueError("Non-finite original native high cost")
        self.frame["high"] = {
            "tracks": [[int(t.id), int(t.state), int(t.end_frame), *np.asarray(t.xyxy).tolist()]
                       for t in tracks],
            "detections": [[int(d.det_ind), float(d.conf), *np.asarray(d.xyxy).tolist()] for d in dets],
            "threshold": float(threshold), "fuse_score": bool(tracker.fuse_first_associate),
            "proximity_thresh": float(tracker.proximity_thresh),
            "appearance_thresh": float(tracker.appearance_thresh),
            "iou_distance": matrix(geometric), "appearance_raw": matrix(raw_app),
            "cost": matrix(costs), "matches": pairs(matches),
        }

    def candidate(self, observation, retained, old, costs, matched):
        self.frame["candidate"] = {
            "residual": [int(o.det_index) for o in observation],
            "columns": [int(o.det_index) for o in retained],
            "rows": [int(c.candidate_id) for c in old],
            "link_cost": matrix(costs), "links": pairs(matched),
        }

    def before_b(self, matcher, views, candidates):
        new_memories, b_tracks = {}, []
        for view in views:
            memory_key = None
            if view.memory is not None:
                updates = tuple(sorted(view.memory.update_counts.items()))
                memory_key = ":".join([str(view.track_id), *(str(n)+str(v) for n,v in updates)])
                if self.memory_versions.get(view.track_id) != updates:
                    self.memory_versions[view.track_id] = updates
                    new_memories[memory_key] = snapshot_memory(view.memory)
            predictions = [[int(frame), np.asarray(p.box).tolist(), np.asarray(p.center_cov).tolist()]
                           for frame,p in sorted(view.predictions.items())]
            b_tracks.append({"id":int(view.track_id),"last_seen":int(view.last_seen),
                             "memory":memory_key,"predictions":predictions})
        b_candidates = []
        for c in candidates:
            history = matcher._history(c, self.frame["frame"], matcher.config.window)
            b_candidates.append({"id":int(c.candidate_id),"last_frame":int(c.last_frame),
                                 "obs":[[int(o.frame), int(o.det_index), float(o.score), o.box.tolist()]
                                        for o in history]})
        self.frame["b"] = {"tracks": b_tracks, "candidates": b_candidates,
                           "first_reserved": {str(k):int(v) for k,v in matcher._first_reserved.items()},
                           "memory_updates":new_memories}

    def b_cost(self, raw, eligible):
        self.frame["b"].update(raw=matrix(raw), eligible=matrix(eligible))

    def after_b(self, decision):
        b = self.frame["b"]
        if not b["tracks"] or not b["candidates"]:
            b.update(raw=[[] for _ in b["tracks"]], eligible=[[] for _ in b["tracks"]])
        b.update(matches=[asdict(x) for x in decision.matches],
                 reserved=sorted(decision.reserved_candidate_ids),
                 diagnostics=decision.diagnostics)

    def finish(self):
        if set(self.frame) != {"sequence", "frame", "high", "candidate", "b"}:
            raise ValueError("Shadow missing native, candidate or recovery boundary")
        self.stream.write(json.dumps(self.frame, allow_nan=False, separators=(",", ":"))+"\n")
        self.frame = None
        self.count += 1

    def close(self):
        self.stream.close()
        if self.count != self.frames or self.frame is not None:
            raise RuntimeError("Incomplete association shadow")
        self.partial.replace(self.path)

    def abort(self):
        if not self.stream.closed:
            self.stream.close()
