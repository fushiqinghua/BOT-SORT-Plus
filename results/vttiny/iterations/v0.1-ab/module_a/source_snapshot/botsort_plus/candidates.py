"""A: causal, scale-aware accumulation of tentative observation tracks.

This module never assigns a public track ID and never rewrites earlier outputs.
The BoT-SORT adapter decides whether a ready candidate restores an old ID (B)
or starts a new ID (A). Both paths consume the candidate once.

``enabled`` is an adapter switch, not a switch inside ``CandidatePool``: the
same pool is also required for B-only experiments. Thresholds are initial
research settings, not values validated on VT-Tiny-MOT.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable
import math

import numpy as np

from .assignment import assign_with_rejection
from .geometry import GeometryConfig, geometry_cost, warp_box
from .types import Observation


@dataclass
class BirthConfig:
    enabled: bool = False
    window: int = 5
    min_hits: int = 3
    max_misses: int = 2
    decay: float = 0.8
    score_weight: float = 0.5
    evidence_threshold: float = 1.2
    miss_penalty: float = 0.4
    link_max_cost: float = 0.85
    ambiguity_margin: float = 0.08
    duplicate_iou: float = 0.7
    max_candidates: int = 256
    max_age: int = 20
    score_low: float = 0.1
    score_high: float = 0.5
    mode: str = "adaptive"
    velocity_alpha: float = 0.5

    def __post_init__(self):
        self.validate()

    def validate(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("birth.enabled must be bool")
        for name in ("window", "min_hits", "max_candidates", "max_age"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"birth.{name} must be a positive integer")
        if (isinstance(self.max_misses, bool) or
                not isinstance(self.max_misses, int) or self.max_misses < 0):
            raise ValueError("birth.max_misses must be a nonnegative integer")
        if self.min_hits > self.window or self.window > self.max_age:
            raise ValueError("birth requires min_hits <= window <= max_age")
        for name in ("decay", "score_weight", "velocity_alpha"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"birth.{name} must lie in [0, 1]")
        if not 0 < self.decay:
            raise ValueError("birth.decay must be positive")
        for name in ("evidence_threshold", "link_max_cost"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"birth.{name} must be positive and finite")
        for name in ("miss_penalty", "ambiguity_margin"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"birth.{name} must be nonnegative and finite")
        if not math.isfinite(self.duplicate_iou) or not 0 < self.duplicate_iou <= 1:
            raise ValueError("birth.duplicate_iou must lie in (0, 1]")
        if not (0 <= self.score_low < self.score_high <= 1):
            raise ValueError("birth requires 0 <= score_low < score_high <= 1")
        if self.mode not in ("adaptive", "fixed_hits"):
            raise ValueError("birth.mode must be adaptive or fixed_hits")
        return self


def _center(box: np.ndarray) -> np.ndarray:
    return 0.5 * (box[:2] + box[2:])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    overlap = np.maximum(0.0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(overlap))
    union = float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection)
    return intersection / union if union > 0 else 0.0


@dataclass
class Candidate:
    candidate_id: int
    observations: deque[Observation]
    config: BirthConfig
    evidence: float
    last_box: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    state_frame: int = 0
    # The last observed box is transported through GMC even during misses. This
    # separates object displacement from camera displacement in velocity fitting.
    _anchor_box: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float), repr=False)
    _birth_frame: int = 0

    @property
    def latest(self) -> Observation:
        return self.observations[-1]

    @property
    def last_frame(self) -> int:
        return self.latest.frame

    @property
    def first_frame(self) -> int:
        return self._birth_frame

    @property
    def misses(self) -> int:
        return self.state_frame - self.last_frame

    def hit_count(self, frame: int, window: int | None = None) -> int:
        window = self.config.window if window is None else window
        if window < 1:
            raise ValueError("hit-count window must be positive")
        return sum(frame - window < item.frame <= frame for item in self.observations)

    def ready(self, frame: int) -> bool:
        """Confirmation is causal and only possible on an observed current frame."""
        return bool(
            self.last_frame == frame
            and self.hit_count(frame) >= self.config.min_hits
            and (self.config.mode == "fixed_hits" or self.evidence >= self.config.evidence_threshold)
        )


class CandidatePool:
    """One-to-one weak tracklets with bounded lifetime and duplicate suppression.

    Call ``update`` once for EVERY video frame, including empty frames. Detections
    already used by native BoT-SORT must be omitted by the caller; their current
    boxes can additionally be supplied through ``occupied_boxes``. A candidate
    that overlaps an occupied box is invalidated before accumulating evidence.

    ``Observation.box`` remains in its original image frame. Working predictions
    and their velocity are transformed with the current GMC warp exactly once.
    B can therefore compare historical observations with KF predictions archived
    in the *same* historical frames, without mixing coordinate systems.
    """

    def __init__(self, config: BirthConfig, geometry_config: GeometryConfig):
        self.config = config.validate()
        self.geometry_config = geometry_config
        self.candidates: dict[int, Candidate] = {}
        self.events: list[dict] = []
        self._next_id = 1
        self._frame: int | None = None

    def get(self, candidate_id: int) -> Candidate | None:
        return self.candidates.get(int(candidate_id))

    def remove(self, candidate_id: int, reason: str = "consumed") -> Candidate:
        candidate = self.candidates.pop(int(candidate_id))
        self.events.append({"event": "candidate_removed", "frame": self._frame,
                            "candidate_id": candidate.candidate_id, "reason": reason})
        return candidate

    def discard(self, candidate_id: int, reason: str = "discarded") -> Candidate | None:
        if int(candidate_id) in self.candidates:
            return self.remove(candidate_id, reason)
        return None

    def _bonus(self, observation: Observation, geometry: float) -> float:
        score = np.clip((observation.score - self.config.score_low) /
                        (self.config.score_high - self.config.score_low), 0.0, 1.0)
        return float(self.config.score_weight * score +
                     (1.0 - self.config.score_weight) * math.exp(-geometry))

    def _new(self, observation: Observation) -> Candidate:
        candidate = Candidate(
            candidate_id=self._next_id,
            observations=deque([observation], maxlen=self.config.max_age),
            config=self.config,
            evidence=self._bonus(observation, 0.0),
            last_box=observation.box.copy(),
            state_frame=observation.frame,
            _anchor_box=observation.box.copy(),
            _birth_frame=observation.frame,
        )
        self._next_id += 1
        self.candidates[candidate.candidate_id] = candidate
        self.events.append({"event": "candidate_seed", "frame": observation.frame,
                            "candidate_id": candidate.candidate_id,
                            "det_index": observation.det_index, "evidence": candidate.evidence})
        return candidate

    def update(self, frame: int, observations: Iterable[Observation],
               warp: np.ndarray | None = None,
               occupied_boxes: Iterable[np.ndarray] = ()) -> list[Candidate]:
        if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)) or frame < 1:
            raise ValueError("candidate frame must be a positive integer")
        if self._frame is not None and frame != self._frame + 1:
            raise ValueError("CandidatePool requires consecutive frames, including empty ones")
        observations = list(observations)
        if any(item.frame != frame for item in observations):
            raise ValueError("candidate observations must belong to the current frame")
        if len({item.det_index for item in observations}) != len(observations):
            raise ValueError("a detection index may occur only once in a frame")
        warp = np.eye(2, 3, dtype=float) if warp is None else np.asarray(warp, dtype=float)
        if warp.shape != (2, 3) or not np.isfinite(warp).all():
            raise ValueError("GMC warp must be a finite 2 x 3 matrix")
        occupied = []
        for box in occupied_boxes:
            box = np.asarray(box, dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
                raise ValueError("occupied boxes must be finite positive-area xyxy boxes")
            occupied.append(box)

        self.events = []
        self._frame = int(frame)
        # Transport/predict once. Only last_box and _anchor_box change coordinates;
        # observations retain their original image-frame coordinates.
        for candidate in list(self.candidates.values()):
            if (frame - candidate.first_frame >= self.config.max_age or
                    frame - candidate.last_frame > self.config.max_misses + 1):
                self.remove(candidate.candidate_id, "expired")
                continue
            candidate.last_box = warp_box(candidate.last_box, warp)
            candidate._anchor_box = warp_box(candidate._anchor_box, warp)
            candidate.velocity = warp[:, :2] @ candidate.velocity
            candidate.last_box += np.tile(candidate.velocity, 2)
            candidate.state_frame = frame
            candidate.evidence *= self.config.decay
            if any(_iou(candidate.last_box, box) >= self.config.duplicate_iou for box in occupied):
                self.remove(candidate.candidate_id, "occupied_track")

        # Suppress duplicate candidate evidence before matching or seeding. This
        # is local birth-pool suppression only; the detector cache is unchanged.
        retained: list[Observation] = []
        for observation in sorted(observations, key=lambda item: (-item.score, item.det_index)):
            if any(_iou(observation.box, box) >= self.config.duplicate_iou for box in occupied):
                self.events.append({"event": "candidate_observation_suppressed", "frame": frame,
                                    "det_index": observation.det_index, "reason": "occupied_track"})
            elif any(_iou(observation.box, item.box) >= self.config.duplicate_iou for item in retained):
                self.events.append({"event": "candidate_observation_suppressed", "frame": frame,
                                    "det_index": observation.det_index, "reason": "duplicate_observation"})
            else:
                retained.append(observation)
        retained.sort(key=lambda item: item.det_index)
        old = sorted(self.candidates.values(), key=lambda item: item.candidate_id)
        costs = np.full((len(old), len(retained)), np.inf, dtype=float)
        for row, candidate in enumerate(old):
            for col, observation in enumerate(retained):
                costs[row, col] = geometry_cost(candidate.last_box, observation.box, self.geometry_config)
        pairs, unmatched_rows, unmatched_cols = assign_with_rejection(
            costs, max_cost=self.config.link_max_cost, margin=self.config.ambiguity_margin)
        current: list[Candidate] = []
        for row, col in pairs:
            candidate, observation = old[int(row)], retained[int(col)]
            elapsed = frame - candidate.last_frame
            measured_velocity = (_center(observation.box) - _center(candidate._anchor_box)) / elapsed
            alpha = self.config.velocity_alpha
            candidate.velocity = alpha * candidate.velocity + (1.0 - alpha) * measured_velocity
            candidate.evidence += self._bonus(observation, float(costs[row, col]))
            candidate.observations.append(observation)
            candidate.last_box = observation.box.copy()
            candidate._anchor_box = observation.box.copy()
            current.append(candidate)
            self.events.append({"event": "candidate_hit", "frame": frame,
                                "candidate_id": candidate.candidate_id,
                                "det_index": observation.det_index, "geometry_cost": float(costs[row, col]),
                                "evidence": candidate.evidence, "hits": candidate.hit_count(frame),
                                "ready": candidate.ready(frame)})
        for row in unmatched_rows:
            candidate = old[int(row)]
            # Rectification is deliberate: absence must not accumulate an
            # arbitrarily negative debt that can never be repaid after a gap.
            candidate.evidence = max(0.0, candidate.evidence - self.config.miss_penalty)
            self.events.append({"event": "candidate_miss", "frame": frame,
                                "candidate_id": candidate.candidate_id, "evidence": candidate.evidence})
            if candidate.misses > self.config.max_misses:
                self.remove(candidate.candidate_id, "max_misses")
        for col in unmatched_cols:
            col = int(col)
            observation = retained[col]
            # An observation rejected by one-to-one assignment or ambiguity may
            # not start a parallel explanation of the same candidate this frame.
            # Once no plausible link remains, it may seed an independent track.
            plausible = np.isfinite(costs[:, col]) & (costs[:, col] <= self.config.link_max_cost)
            if np.any(plausible):
                self.events.append({"event": "candidate_seed_deferred", "frame": frame,
                                    "det_index": observation.det_index, "reason": "competing_links"})
                continue
            current.append(self._new(observation))

        # Highest evidence first, then recent observation and deterministic ID.
        # Bounded memory also protects long sequences containing stable clutter.
        if len(self.candidates) > self.config.max_candidates:
            ranked = sorted(self.candidates.values(),
                            key=lambda item: (-item.evidence, -item.last_frame, item.candidate_id))
            for candidate in ranked[self.config.max_candidates:]:
                self.remove(candidate.candidate_id, "capacity")
        return sorted((item for item in current if item.candidate_id in self.candidates),
                      key=lambda item: item.candidate_id)
