"""B: causal, short-track-verified identity recovery.

This module only decides whether an *unclaimed* current observation can inherit a
lost track's ID.  It never updates a Kalman filter or an appearance prototype.
The integration must read trusted memory first, call ``match``, then update the
accepted native track.  It must never replay an earlier output frame.

Prediction records must be exported by the native filter at their actual frame
times, after camera-motion compensation.  We do not extrapolate a current
prediction backwards to create fictitious evidence for a short trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Callable, Iterable, Any

import numpy as np

from .assignment import assign_with_rejection
from .geometry import GeometryConfig, geometry_cost
from .types import Feature, LostTrackView, Observation


# Per-frame maxima for early recovery, ordered as (geometry, appearance).
EARLY_EVIDENCE_LIMITS = {"strong": (0.30, 0.48), "moderate": (0.42, 0.55)}


@dataclass(frozen=True)
class RecoveryConfig:
    enabled: bool = False
    min_hits: int = 3
    window: int = 5
    min_age: int = 2
    max_age: int = 30
    max_cost: float = 0.65
    ambiguity_margin: float = 0.10
    geometry_weight: float = 0.65
    appearance_weight: float = 0.25
    time_weight: float = 0.10
    appearance_max_cost: float = 0.65
    require_appearance: bool = False
    reservation_frames: int = 5
    output_prediction_frames: int = 0

    early_evidence: str = "off"

    def __post_init__(self):
        for name in ("enabled", "require_appearance"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        for name in ("min_hits", "window", "min_age", "max_age", "reservation_frames"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.early_evidence not in ("off", *EARLY_EVIDENCE_LIMITS):
            raise ValueError("recovery.early_evidence must be off, strong or moderate")
        if type(self.output_prediction_frames) is not int or self.output_prediction_frames not in (0,1,2):
            raise ValueError("output_prediction_frames must be 0, 1 or 2")
        if self.output_prediction_frames and not self.enabled:
            raise ValueError("Online prediction output requires recovery motion history")
        if self.min_hits > self.window:
            raise ValueError("min_hits must fit in the trailing window")
        if self.min_age > self.max_age:
            raise ValueError("min_age must not exceed max_age")
        if self.reservation_frames < self.min_hits:
            raise ValueError("reservation_frames must allow min_hits observations")
        for name in ("max_cost", "ambiguity_margin", "geometry_weight", "appearance_weight",
                     "time_weight", "appearance_max_cost"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.max_cost <= 0 or self.appearance_max_cost <= 0 or self.geometry_weight <= 0:
            raise ValueError("Geometry and acceptance thresholds must be positive")
        if self.require_appearance and self.appearance_weight <= 0:
            raise ValueError("require_appearance needs a positive appearance_weight")


@dataclass(frozen=True)
class RecoveryMatch:
    track_id: int
    candidate_id: int
    det_index: int
    cost: float
    geometry_cost: float
    appearance_cost: float | None
    age: int
    hits: int


@dataclass
class RecoveryDecision:
    matches: list[RecoveryMatch] = field(default_factory=list)
    reserved_candidate_ids: set[int] = field(default_factory=set)
    diagnostics: list[dict] = field(default_factory=list)


AppearanceCost = Callable[[Any, dict[str, Feature]], float | None]


class RecoveryMatcher:
    """Joint assignment with explicit reject choices and bounded birth deferral.

    ``age`` always means ``current_frame - last_detection_frame``.  Thus an age
    of 2 corresponds to one missing frame between two observations.  The runner
    must keep ``max_age`` within the native BoT-SORT retention period unless a
    longer-buffer ablation is explicitly requested.

    A reservation delays *new-ID creation*, never a previously accepted output.
    Its budget starts at the first plausible old-ID link and cannot be renewed
    during that candidate's lifetime.  A persistent ambiguous candidate can
    therefore leave the recovery path and undergo normal birth confirmation.
    """

    def __init__(self, config: RecoveryConfig, geometry_config: GeometryConfig):
        self.config = config
        self.geometry_config = geometry_config
        self._first_reserved: dict[int, int] = {}
        self._last_frame = 0

    def forget(self, candidate_ids: Iterable[int]) -> None:
        """Free reservation bookkeeping only after candidates leave the pool."""
        for candidate_id in candidate_ids:
            self._first_reserved.pop(int(candidate_id), None)

    @staticmethod
    def _history(candidate: Any, frame: int, window: int) -> list[Observation]:
        observations = list(candidate.observations)
        frames = [o.frame for o in observations]
        if any(t > frame for t in frames):
            raise ValueError("Future observations are forbidden in online recovery")
        if frames != sorted(set(frames)):
            raise ValueError("Candidate observations must have unique increasing frames")
        if not observations or candidate.last_frame != observations[-1].frame:
            raise ValueError("Candidate last_frame does not match its observation history")
        return [o for o in observations if frame - window < o.frame <= frame]

    def _pair(self, frame: int, view: LostTrackView, candidate: Any,
              history: list[Observation], appearance_cost: AppearanceCost | None) -> dict:
        cfg = self.config
        age = frame - view.last_seen
        info = {"track_id": int(view.track_id), "candidate_id": int(candidate.candidate_id),
                "age": int(age), "hits": len(history), "cost": None}

        def reject(reason: str) -> dict:
            return {**info, "reason": reason}

        if not cfg.min_age <= age <= cfg.max_age:
            return reject("outside_retention")
        if not history or history[-1].frame != frame:
            return reject("no_current_observation")
        # A candidate that already coexisted with this observed identity cannot
        # use that shared period as recovery evidence.
        if any(o.frame <= view.last_seen for o in history):
            return reject("overlaps_last_observation")
        geom: list[float] = []
        appearance: list[float] = []
        current_app = None
        for obs in history:
            pred = view.predictions.get(obs.frame)
            if pred is None or pred.frame != obs.frame:
                return reject("missing_same_frame_prediction")
            distance = geometry_cost(pred.box, obs.box, self.geometry_config,
                                     center_cov=pred.center_cov)
            if not np.isfinite(distance):
                return reject("geometry_gate")
            geom.append(float(distance))
            # Missing/untrustworthy evidence is None, never distance zero.
            if appearance_cost is not None and cfg.appearance_weight > 0:
                value = appearance_cost(view.memory, obs.features)
                if value is not None:
                    if not np.isfinite(value) or value < 0:
                        raise ValueError("Appearance callback must return finite nonnegative distance or None")
                    appearance.append(float(value))
                    if obs.frame == frame:
                        current_app = float(value)
        app_mean = float(np.mean(appearance)) if appearance else None
        if cfg.require_appearance and app_mean is None:
            return reject("appearance_required")
        # A reliable current contradictory feature must not be hidden by a low
        # geometric cost or by averaging previous compatible observations.
        if (app_mean is not None and app_mean > cfg.appearance_max_cost) or (
                current_app is not None and current_app > cfg.appearance_max_cost):
            return reject("appearance_contradiction")
        geom_mean = float(np.mean(geom))
        weights = cfg.geometry_weight + cfg.time_weight
        total = cfg.geometry_weight * geom_mean + cfg.time_weight * (age / cfg.max_age)
        if app_mean is not None:
            weights += cfg.appearance_weight
            total += cfg.appearance_weight * app_mean
        cost = float(total / weights)
        info.update(cost=cost, geometry_cost=geom_mean, appearance_cost=app_mean,
                    appearance_hits=len(appearance), ready=len(history) >= cfg.min_hits)
        # The two observations must support a single old identity. Compare
        # each to the pre-recovery memory and its own same-frame prediction.
        if (cfg.early_evidence != "off" and cfg.min_hits == 3 and len(history) == 2
                and history[0].frame == frame - 1 and len(appearance) == 2):
            max_geometry, max_appearance = EARLY_EVIDENCE_LIMITS[cfg.early_evidence]
            worst_geometry, worst_appearance = max(geom), max(appearance)
            if worst_geometry <= max_geometry and worst_appearance <= max_appearance:
                info.update(two_hit_evidence=True, two_hit_max_geometry=worst_geometry,
                            two_hit_max_appearance=worst_appearance)
        return {**info, "reason": "plausible" if cost < cfg.max_cost else "cost_reject"}

    def match(self, frame: int, lost_views: Iterable[LostTrackView], candidates: Iterable[Any],
              appearance_cost: AppearanceCost | None = None) -> RecoveryDecision:
        if (isinstance(frame, bool) or not isinstance(frame, (int, np.integer)) or
                frame < 1 or frame < self._last_frame):
            raise ValueError("Recovery frames must be positive and nondecreasing")
        self._last_frame = frame
        decision = RecoveryDecision()
        if not self.config.enabled:
            return decision
        views = list(lost_views)
        candidates = list(candidates)
        ids = [int(c.candidate_id) for c in candidates]
        track_ids = [int(v.track_id) for v in views]
        if len(set(ids)) != len(ids) or len(set(track_ids)) != len(track_ids):
            raise ValueError("Candidate and lost-track IDs must each be unique")
        # A caller may supply only candidates observed in this frame. Absence
        # therefore cannot reset a reservation budget. ``forget`` is available
        # for explicit pool removals; IDs must not be reused within a sequence.
        histories = [self._history(c, frame, self.config.window) for c in candidates]
        if not views or not candidates:
            return decision

        raw_costs = np.full((len(views), len(candidates)), np.inf, dtype=float)
        pair_info: dict[tuple[int, int], dict] = {}
        for i, view in enumerate(views):
            for j, candidate in enumerate(candidates):
                info = self._pair(frame, view, candidate, histories[j], appearance_cost)
                pair_info[i, j] = info
                decision.diagnostics.append(info)
                if info["reason"] == "plausible":
                    raw_costs[i, j] = info["cost"]

        observer=getattr(self,"raw_ir_observer",None)
        if observer is not None:
            raw_costs=observer.recovery(raw_costs,pair_info,views,histories,ids,self.config.max_cost)

        # Reserve unresolved plausible candidates, including those with too few
        # hits. This prevents the usual high-score birth branch from stealing a
        # potential old identity before short-trajectory verification finishes.
        for j, cid in enumerate(ids):
            if np.isfinite(raw_costs[:, j]).any():
                first = self._first_reserved.setdefault(cid, frame)
                if frame - first < self.config.reservation_frames:
                    decision.reserved_candidate_ids.add(cid)

        eligible = raw_costs.copy()
        for j, history in enumerate(histories):
            if len(history) >= self.config.min_hits:
                continue
            if self.config.early_evidence == "off":
                eligible[:, j] = np.inf
                continue
            # Remove uncertified edges from assignment, retaining raw costs
            # for the rival and reservation checks, including immature rivals.
            for i in range(len(views)):
                if not pair_info[i, j].get("two_hit_evidence", False):
                    eligible[i, j] = np.inf
        # Include not-yet-confirmed alternatives in ambiguity checks. Otherwise
        # deleting their columns can manufacture a large best/second-best gap.
        margin = self.config.ambiguity_margin
        if margin > 0:
            for i, j in np.argwhere(np.isfinite(eligible)):
                value = raw_costs[i, j]
                row_rivals = np.delete(raw_costs[i, :], j)
                col_rivals = np.delete(raw_costs[:, j], i)
                if (np.any(row_rivals <= value + margin) or
                        np.any(col_rivals <= value + margin)):
                    eligible[i, j] = np.inf
                    pair_info[i, j]["reason"] = "ambiguous"
        pairs, _, _ = assign_with_rejection(eligible, self.config.max_cost, margin=0)
        for i, j in pairs:
            info = pair_info[i, j]
            obs = histories[j][-1]
            decision.matches.append(RecoveryMatch(
                track_id=int(views[i].track_id), candidate_id=ids[j], det_index=int(obs.det_index),
                cost=float(info["cost"]), geometry_cost=float(info["geometry_cost"]),
                appearance_cost=info["appearance_cost"], age=int(info["age"]), hits=int(info["hits"]),
            ))
            info["reason"] = "accepted"
            decision.reserved_candidate_ids.discard(ids[j])
        return decision
