"""Causal centroid alternatives from frozen observations and GMC transforms."""
from dataclasses import dataclass

import numpy as np

OBSERVATION_EVENTS = {"match", "recovered", "native_birth"}
MIN_GAP = 2
MAX_GAP = 30
VELOCITY_DECAY = 0.8
DISPLACEMENT_CAP_PX = 48.0


@dataclass
class Anchor:
    """Last real measurement transported to the current image coordinates."""

    last_seen: int
    center: np.ndarray
    velocity: np.ndarray | None
    displacement: np.ndarray


class MotionTransport:
    """Produce P1 and P2 centers without changing any recorded track state."""

    def __init__(self):
        self.frame = 0
        self.anchors = {}
        self.last_unavailable = {}

    def advance(self, row: dict) -> dict[int, tuple[np.ndarray | None, np.ndarray | None]]:
        """Advance exactly one frozen frame and return centers for eligible Lost tracks.

        An absent ID is outside the Lost 2–30 cohort. A None center denotes that
        the alternative must fall back to P0. Reported vectors are copies.
        """
        frame = int(row["frame"])
        if frame != self.frame + 1:
            raise ValueError(f"Expected consecutive frames, got {self.frame} -> {frame}")
        self.frame = frame
        self.last_unavailable = {}

        warp = np.asarray(row.get("gmc", {}).get("warp"), dtype=np.float64)
        warp_available = warp.shape == (2, 3) and np.isfinite(warp).all()
        if warp_available:
            linear, shift = warp[:, :2], warp[:, 2]
            expired = []
            for track_id, anchor in self.anchors.items():
                if frame - anchor.last_seen > MAX_GAP:
                    expired.append(track_id)
                    continue
                anchor.center = linear @ anchor.center + shift
                if anchor.velocity is not None:
                    anchor.velocity = VELOCITY_DECAY * (linear @ anchor.velocity)
                    step = linear @ anchor.displacement + anchor.velocity
                    norm = float(np.linalg.norm(step))
                    if norm > DISPLACEMENT_CAP_PX:
                        step *= DISPLACEMENT_CAP_PX / norm
                    anchor.displacement = step
            for track_id in expired:
                del self.anchors[track_id]
        else:
            self.anchors.clear()

        centers = {}
        for state in row["after_gmc"]:
            track_id = int(state["track_id"])
            last_seen = int(state["last_seen"])
            if int(state["state"]) != 2 or not MIN_GAP <= frame - last_seen <= MAX_GAP:
                continue
            anchor = self.anchors.get(track_id)
            if not warp_available or anchor is None or anchor.last_seen != last_seen:
                centers[track_id] = (None, None)
                self.last_unavailable[track_id] = "missing_gmc" if not warp_available else "missing_or_mismatched_anchor"
                continue
            p1 = anchor.center.copy()
            p2 = anchor.center + anchor.displacement if anchor.velocity is not None else None
            if p2 is None:
                self.last_unavailable[track_id] = "missing_velocity"
            centers[track_id] = (p1, p2)

        # Current acceptance is only available for future frames.
        measurements = {}
        for event in row["events"]:
            if event["event"] not in OBSERVATION_EVENTS:
                continue
            track_id = int(event["track_id"])
            index = int(event["det_index"])
            if track_id in measurements:
                raise ValueError(f"Multiple accepted measurements for {track_id} at frame {frame}")
            measurements[track_id] = index
        dets = {int(det["det_index"]): det for det in row["detections"]}
        post = {int(state["track_id"]): state for state in row["after_update"]}
        for track_id, index in measurements.items():
            snapshot = post.get(track_id)
            detection = dets.get(index)
            self.anchors.pop(track_id, None)
            if (snapshot is None or detection is None or int(snapshot["last_seen"]) != frame
                    or int(snapshot["last_observation_det_index"]) != index):
                continue
            xyxy = np.asarray(detection["xyxy"], dtype=np.float64)
            if xyxy.shape != (4,) or not np.isfinite(xyxy).all() or not (xyxy[2:] > xyxy[:2]).all():
                continue
            center = (xyxy[:2] + xyxy[2:]) / 2
            mean = np.asarray(snapshot.get("mean"), dtype=np.float64)
            velocity = mean[4:6].copy() if mean.shape == (8,) and np.isfinite(mean[4:6]).all() else None
            self.anchors[track_id] = Anchor(frame, center, velocity, np.zeros(2, dtype=np.float64))
        return centers
