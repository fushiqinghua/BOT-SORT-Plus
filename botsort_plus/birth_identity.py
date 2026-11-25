"""A v0.2: read-only admission of a *new* identity near retained lost tracks.

This is deliberately not another recovery matcher.  It never chooses an old
identity, reactivates a track, consumes a detection, or writes appearance memory.
B alone may perform recovery.  Failure or expiry of B's reservation is not
evidence that the candidate represents a new target.

``CLEAR_NEW`` means that no unresolved old-ID explanation remains under the
configured, bounded gates; the caller must still require fresh birth evidence,
a current observation, and successful native duplicate checks before committing.
All settings are development hypotheses, not benchmark-calibrated parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Callable, Iterable

import numpy as np

from .geometry import GeometryConfig, geometry_cost
from .types import Feature, LostTrackView, Observation, Prediction


AppearanceCost = Callable[[Any, dict[str, Feature]], float | None]


@dataclass(frozen=True)
class BirthIdentityConfig:
    enabled: bool = True
    window: int = 5
    max_age: int = 30
    wait_frames: int = 5
    appearance_max_cost: float = 0.65

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("birth_identity.enabled must be boolean")
        for name in ("window", "max_age", "wait_frames"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, np.integer))
                    or value < 1):
                raise ValueError(f"birth_identity.{name} must be a positive integer")
        value = self.appearance_max_cost
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not np.isfinite(value) or value <= 0):
            raise ValueError("birth_identity.appearance_max_cost must be positive and finite")


@dataclass(frozen=True)
class BirthIdentityDecision:
    status: str
    reason: str
    conflicting_track_ids: list[int] = field(default_factory=list)
    first_conflict_frame: int | None = None
    elapsed: int = 0
    diagnostics: list[dict] = field(default_factory=list)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class BirthIdentityGuard:
    """Bounded, non-renewable new-ID deferral keyed by private candidate ID.

    Call before A's birth, including while A is still collecting evidence, so
    the first observed conflict starts the budget.  ``reservation_start`` is
    B's original first-reservation frame, if any; the earlier clock wins.
    Skipped/missing frames and changes of competing old ID never restart it.

    A conflict at elapsed >= ``wait_frames`` yields REJECT, never CLEAR_NEW.
    A later genuinely conflict-free current observation may yield CLEAR_NEW,
    but the original clock remains: renewed conflict is still out of budget.
    Only ``forget`` after the candidate leaves the shared pool releases state.

    Views must already be restricted to physically retained, Lost-state tracks.
    This module additionally checks age.  The appearance callback must return
    None for absent/untrusted evidence, and must be read-only.  Only a trusted
    *current* appearance contradiction excludes a geometrically plausible old
    ID; old contradictions alone cannot certify the current candidate as new.
    """

    def __init__(self, config: BirthIdentityConfig, geometry_config: GeometryConfig):
        self.config = config
        self.geometry_config = geometry_config
        self._first_conflict: dict[int, int] = {}
        self._last_frame = 0

    def forget(self, candidate_ids: Iterable[int]) -> None:
        for candidate_id in candidate_ids:
            self._first_conflict.pop(int(candidate_id), None)

    @staticmethod
    def _history(candidate: Any, frame: int, window: int) -> list[Observation]:
        observations = list(candidate.observations)
        frames = [_positive_integer(o.frame, "observation frame") for o in observations]
        if frames != sorted(set(frames)):
            raise ValueError("Candidate observations must have unique increasing frames")
        if any(t > frame for t in frames):
            raise ValueError("Future observations are forbidden in birth identity decisions")
        if not observations or candidate.last_frame != observations[-1].frame:
            raise ValueError("Candidate last_frame does not match its observation history")
        return [o for o in observations if frame - window < o.frame <= frame]

    @staticmethod
    def _prediction(pred: Prediction, frame: int) -> None:
        if pred.frame != frame:
            raise ValueError("Prediction must belong to its exact historical frame")
        box = np.asarray(pred.box, dtype=float)
        if (box.shape != (4,) or not np.isfinite(box).all()
                or np.any(box[2:] <= box[:2])):
            raise ValueError("Prediction must have a finite positive-area xyxy box")
        cov = np.asarray(pred.center_cov, dtype=float)
        if cov.shape != (2, 2) or not np.isfinite(cov).all():
            raise ValueError("Prediction covariance must be a finite 2x2 matrix")
        if np.linalg.eigvalsh((cov + cov.T) / 2).min() < -1e-6:
            raise ValueError("Prediction covariance must be positive semidefinite")

    @staticmethod
    def _appearance(callback: AppearanceCost | None, memory: Any,
                    observation: Observation) -> float | None:
        if callback is None:
            return None
        value = callback(memory, observation.features)
        if value is None:
            return None
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0):
            raise ValueError("Appearance callback must return finite nonnegative distance or None")
        return float(value)

    def _pair(self, frame: int, candidate: Any, view: LostTrackView,
              history: list[Observation], appearance_cost: AppearanceCost | None) -> dict:
        age = frame - int(view.last_seen)
        info = {
            "track_id": int(view.track_id), "candidate_id": int(candidate.candidate_id),
            "age": age, "conflict": False, "current_geometry": None,
            "current_appearance": None, "history_geometry": [],
            "history_appearance_mean": None, "appearance_hits": 0,
            "history_before_last_seen": [],
        }
        if not 1 <= age <= self.config.max_age:
            return {**info, "reason": "outside_retention"}

        current = history[-1]
        prediction = view.predictions.get(frame)
        # Without the actual current prediction we cannot use a stale box to
        # certify spatial separation.  Fail closed, without extrapolating.
        if prediction is None:
            return {**info, "conflict": True, "reason": "unknown_current_prediction"}
        self._prediction(prediction, frame)
        distance = geometry_cost(prediction.box, current.box, self.geometry_config,
                                 center_cov=prediction.center_cov)
        if not np.isfinite(distance):
            return {**info, "reason": "current_geometry_excludes"}
        info["current_geometry"] = float(distance)
        current_app = self._appearance(appearance_cost, view.memory, current)
        info["current_appearance"] = current_app

        geometric_history = []
        app_history = []
        incomplete = False
        inconsistent = False
        for observation in history:
            historical = view.predictions.get(observation.frame)
            record = {"frame": int(observation.frame), "cost": None}
            if observation.frame <= view.last_seen:
                info["history_before_last_seen"].append(int(observation.frame))
            if historical is None:
                incomplete = True
                record["reason"] = "missing_prediction"
            else:
                self._prediction(historical, observation.frame)
                value = geometry_cost(historical.box, observation.box, self.geometry_config,
                                      center_cov=historical.center_cov)
                if np.isfinite(value):
                    record.update(cost=float(value), reason="compatible")
                else:
                    # A bad past pair is insufficient to clear a near current
                    # identity; association/history could itself be uncertain.
                    inconsistent = True
                    record["reason"] = "outside_geometry"
            geometric_history.append(record)
            app = (current_app if observation.frame == frame else
                   self._appearance(appearance_cost, view.memory, observation))
            if app is not None:
                app_history.append(app)
        info["history_geometry"] = geometric_history
        info["history_appearance_mean"] = float(np.mean(app_history)) if app_history else None
        info["appearance_hits"] = len(app_history)

        if current_app is not None and current_app > self.config.appearance_max_cost:
            return {**info, "reason": "current_appearance_excludes"}
        if incomplete:
            reason = "unknown_history_prediction"
        elif inconsistent or info["history_before_last_seen"]:
            reason = "current_compatible_history_uncertain"
        elif current_app is None:
            reason = "compatible_geometry_appearance_unknown"
        else:
            reason = "compatible_old_identity"
        return {**info, "conflict": True, "reason": reason}

    def inspect(self, frame: int, candidate: Any, lost_views: Iterable[LostTrackView],
                appearance_cost: AppearanceCost | None = None,
                reservation_start: int | None = None) -> BirthIdentityDecision:
        frame = _positive_integer(frame, "Birth identity frame")
        if frame < self._last_frame:
            raise ValueError("Birth identity frames must be nondecreasing")
        candidate_id = _positive_integer(candidate.candidate_id, "Candidate ID")
        history = self._history(candidate, frame, self.config.window)
        views = list(lost_views)
        ids = [_positive_integer(v.track_id, "Lost track ID") for v in views]
        if len(ids) != len(set(ids)):
            raise ValueError("Lost track IDs must be unique")
        for view in views:
            _positive_integer(view.last_seen, "Lost track last_seen")
            if view.last_seen > frame:
                raise ValueError("Future last_seen is forbidden")
            for key, prediction in view.predictions.items():
                key = _positive_integer(key, "Prediction frame")
                if key > frame or prediction.frame > frame:
                    raise ValueError("Future predictions are forbidden in birth identity decisions")
                if prediction.frame != key:
                    raise ValueError("Prediction must belong to its exact historical frame")
        if reservation_start is not None:
            reservation_start = _positive_integer(reservation_start, "B reservation_start")
            if reservation_start > frame:
                raise ValueError("B reservation_start cannot be in the future")

        # Do not change state until all input and callback checks have passed.
        has_current = bool(history and history[-1].frame == frame)
        diagnostics = []
        if self.config.enabled and has_current:
            diagnostics = [self._pair(frame, candidate, view, history, appearance_cost)
                           for view in views]
        conflicting = sorted(d["track_id"] for d in diagnostics if d["conflict"])
        first = self._first_conflict.get(candidate_id)
        if self.config.enabled:
            if reservation_start is not None:
                first = reservation_start if first is None else min(first, reservation_start)
            if conflicting and first is None:
                first = frame
            if first is not None:
                self._first_conflict[candidate_id] = first
        self._last_frame = frame
        elapsed = 0 if first is None else frame - first

        if not has_current:
            status = ("REJECT" if first is not None and elapsed >= self.config.wait_frames
                      else "DEFER")
            reason = "no_current_observation"
        elif not self.config.enabled:
            status, reason = "CLEAR_NEW", "identity_guard_disabled"
        elif not conflicting:
            status, reason = "CLEAR_NEW", "no_unresolved_old_identity"
        elif elapsed >= self.config.wait_frames:
            status, reason = "REJECT", "identity_wait_exhausted"
        else:
            status, reason = "DEFER", "unresolved_old_identity"
        return BirthIdentityDecision(status, reason, conflicting, first, elapsed, diagnostics)
