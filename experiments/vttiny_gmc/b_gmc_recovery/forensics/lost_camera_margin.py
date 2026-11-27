"""Require extra evidence on a marginal long-gap reacquisition affected by a camera choice."""
import numpy as np
from boxmot.trackers.botsort.basetrack import TrackState


class LostCameraMargin:
    def __init__(self, tracker, *, recovery=True, high=True, require_observation=False):
        self.tracker = tracker
        self.recovery = recovery
        self.high = high
        self.require_observation = require_observation
        self.unsupported_rejections = 0
        self.last_change = -1
        self.recovery_rejections = 0
        self.high_rejections = 0

    def observe(self, frame, native, used):
        if not np.array_equal(native, used):
            self.last_change = frame

    def exposed(self, frame, last_seen):
        return frame - last_seen > self.tracker.plus.recovery.window and self.last_change >= last_seen

    def recovery_pair(self, pair):
        def assess(frame, view, candidate, history, appearance):
            info = pair(frame, view, candidate, history, appearance)
            cfg = self.tracker.plus.recovery
            if self.recovery and info["reason"] == "plausible" and self.exposed(frame, view.last_seen):
                unsupported = (self.require_observation and info["appearance_hits"] == 0
                               and all(o.score <= self.tracker.track_high_thresh for o in history))
                if unsupported:
                    self.unsupported_rejections += 1
                if unsupported or info["cost"] > cfg.max_cost - cfg.ambiguity_margin:
                    self.recovery_rejections += 1
                    info = {**info, "reason": "weak_observations" if unsupported else "uncertain_camera_recovery"}
            return info
        return assess

    def high_cost(self, cost, tracks):
        if not self.high:
            return
        tracker = self.tracker
        limit = tracker.match_thresh - tracker.plus.recovery.ambiguity_margin
        for i, track in enumerate(tracks):
            if track.state == TrackState.Lost and self.exposed(tracker.frame_count, track.end_frame):
                marginal = (cost[i] > limit) & (cost[i] <= tracker.match_thresh)
                self.high_rejections += int(marginal.sum())
                cost[i, marginal] = 1.
