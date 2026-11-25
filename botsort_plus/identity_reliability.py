"""C2: causal identity evidence for appearance use and memory writes.

This module consumes RGB embeddings already produced by the frozen encoder. It
does not learn a model, estimate an existence probability, or use ground truth.
The geometric association and accepted track lifecycle belong to the adapter.

Every pair decision uses a detached snapshot from ``begin_frame``. Committing a
match changes tomorrow's anchor, never today's competitors. The private anchor
is deliberately separate from both native ``smooth_feat`` and B's MemoryBank;
the adapter must enforce ``allow_write`` on their existing write paths.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Mapping, Sequence

import numpy as np

from .types import Feature


@dataclass(frozen=True)
class IdentityConfig:
    enabled: bool = False
    observe_only: bool = False
    association_enabled: bool = True
    memory_enabled: bool = True
    min_quality: float = .35
    min_history: int = 3
    history_max_distance: float = .4
    appearance_margin: float = .10
    geometry_margin: float = .10
    bootstrap_max_iou_cost: float = .30
    write_max_iou_cost: float = .50
    ema: float = .95
    recent_confirmed_bank: bool = False
    stable_low_warmup: bool = False

    def __post_init__(self):
        for name in ("enabled", "observe_only", "association_enabled", "memory_enabled", "recent_confirmed_bank", "stable_low_warmup"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"identity.{name} must be bool")
        if self.stable_low_warmup and not (self.enabled and self.memory_enabled and self.recent_confirmed_bank and not self.observe_only):
            raise ValueError("stable_low_warmup requires active recent-confirmed identity memory")
        if self.enabled and not (self.association_enabled or self.memory_enabled):
            raise ValueError("Enabled identity module needs at least one intervention")
        if isinstance(self.min_history, bool) or not isinstance(self.min_history, Integral) or self.min_history < 1:
            raise ValueError("identity.min_history must be a positive integer")
        for name in ("min_quality", "history_max_distance", "appearance_margin", "geometry_margin",
                     "bootstrap_max_iou_cost", "write_max_iou_cost", "ema"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"identity.{name} must be a finite number")
        if not 0 <= self.min_quality <= 1:
            raise ValueError("identity.min_quality must be in [0,1]")
        if not 0 < self.history_max_distance <= 2 or not 0 < self.appearance_margin <= 2:
            raise ValueError("Identity cosine thresholds must be in (0,2]")
        if not 0 < self.geometry_margin <= 1:
            raise ValueError("identity.geometry_margin must be in (0,1]")
        if not 0 <= self.bootstrap_max_iou_cost <= self.write_max_iou_cost <= 1:
            raise ValueError("Require 0 <= bootstrap_max_iou_cost <= write_max_iou_cost <= 1")
        if not 0 <= self.ema < 1:
            raise ValueError("identity.ema must be in [0,1)")


@dataclass(frozen=True)
class IdentityAnchor:
    vector: np.ndarray
    quality: float
    updates: int
    last_frame: int


@dataclass(frozen=True)
class IdentityDecision:
    frame: int
    stage: str
    track_id: int
    det_index: int
    reliability: float | None
    allow_write: bool
    reason: str
    distance: float | None
    appearance_margin: float | None
    geometry_margin: float | None
    competitor_count: int
    competition_known: bool
    history_updates: int
    quality: float
    geometry_cost: float

    def to_dict(self):
        return asdict(self)


def _positive_id(value, label, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _rgb(packet: Mapping[str, Feature]):
    if not isinstance(packet, Mapping):
        raise TypeError("An identity feature packet must be a mapping")
    feature = packet.get("rgb")
    if feature is None:
        return None, 0.
    if not isinstance(feature, Feature):
        raise TypeError("Identity RGB value must be Feature")
    vector = np.asarray(feature.vector, dtype=float)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("Identity RGB embedding must be a nonempty finite vector")
    quality = feature.quality
    if isinstance(quality, bool) or not isinstance(quality, Real) or not np.isfinite(quality) or not 0 <= quality <= 1:
        raise ValueError("Identity RGB quality must be finite in [0,1]")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm):
        raise ValueError("Identity RGB norm is not finite")
    return (None if norm <= 1e-12 else vector.copy() / norm), float(quality)


def _anchor(vector, quality, updates, frame):
    detached = np.asarray(vector, dtype=float).copy()
    detached.setflags(write=False)
    return IdentityAnchor(detached, float(quality), int(updates), int(frame))


def _minimum_excluding_each(values, axis):
    """Per-cell minimum over the other entries, with exact tie handling.

    The first two order statistics suffice: excluding any occurrence of a tied
    minimum leaves the same minimum. ``+inf`` denotes a non-feasible entry and
    a length-zero/one axis has no other entry. ``np.partition`` is linear in the
    axis length, so computing both directions costs O(number of pairs).
    """
    if values.shape[axis] < 2:
        return np.full(values.shape, np.inf)
    partition = np.partition(values, 1, axis=axis)
    first = np.expand_dims(np.take(partition, 0, axis=axis), axis=axis)
    second = np.expand_dims(np.take(partition, 1, axis=axis), axis=axis)
    return np.where(values == first, second, first)


class IdentityReliability:
    """Bounded per-ID RGB history with immutable, frame-start read snapshots.

    ``packets`` in evaluate is the complete detection-aligned packet sequence;
    ``det_indices`` contains its global row indices, not local assignment indices.
    The caller supplies the ORIGINAL geometric feasible set. Never prefilter
    ambiguous or feature-missing rivals before calling evaluate.

    A high-quality birth is a provisional one-sample seed. Before min_history
    writes, association falls back to the existing reference costs. Such seeds
    can mature only through geometrically unique, historically consistent high
    or unconfirmed matches. Low matches cannot create/certify a new anchor.
    """

    def __init__(self, config: IdentityConfig):
        self.config = config
        self.states: dict[int, IdentityAnchor] = {}
        self._snapshot: dict[int, IdentityAnchor] = {}
        self._frame = 0
        self._written: set[int] = set()
        self._dimension: int | None = None
        self._observations: dict[int, tuple[np.ndarray | None, float]] = {}

    def begin_frame(self, frame: int):
        frame = _positive_id(frame, "frame")
        if frame <= self._frame:
            raise ValueError("Identity frames must increase strictly")
        self._frame = frame
        self._written.clear()
        self._observations.clear()
        self._snapshot = {tid: _anchor(s.vector, s.quality, s.updates, s.last_frame)
                          for tid, s in self.states.items()}

    def _check_frame(self, frame):
        frame = _positive_id(frame, "frame")
        if frame != self._frame:
            raise ValueError("Identity decision/write must belong to the active frame")

    def _check_dimension(self, vector):
        if vector is not None and self._dimension is not None and len(vector) != self._dimension:
            raise ValueError("Identity RGB embedding dimension changed")

    def evaluate(self, track_ids, det_indices, geometry_cost, packets: Sequence[Mapping[str, Feature]],
                 stage: str, feasible_threshold: float):
        if self._frame < 1:
            raise ValueError("Call begin_frame before evaluating identity evidence")
        if stage not in ("high", "low", "unconfirmed", "recovery", "recovered"):
            raise ValueError("Unknown identity association stage")
        ids = [_positive_id(x, "track_id") for x in track_ids]
        indices = [_positive_id(x, "det_index", 0) for x in det_indices]
        if len(ids) != len(set(ids)) or len(indices) != len(set(indices)):
            raise ValueError("Identity association rows and columns must be unique")
        geometry = np.asarray(geometry_cost, dtype=float).copy()
        if geometry.shape != (len(ids), len(indices)) or not np.isfinite(geometry).all():
            raise ValueError("Identity geometry must be a finite matrix of matching dimensions")
        if np.any((geometry < 0) | (geometry > 1)):
            raise ValueError("Identity geometry must be an IoU cost in [0,1]")
        if isinstance(feasible_threshold, bool) or not isinstance(feasible_threshold, Real) or not np.isfinite(feasible_threshold) or not 0 <= feasible_threshold <= 1:
            raise ValueError("Identity feasible threshold must be in [0,1]")
        if any(index >= len(packets) for index in indices):
            raise ValueError("Identity detection index is outside the packet sequence")
        features = [_rgb(packets[index]) for index in indices]
        for vector, _ in features:
            self._check_dimension(vector)
        dimensions = {len(vector) for vector, _ in features if vector is not None}
        if len(dimensions) > 1:
            raise ValueError("Identity RGB dimensions differ within a frame")
        # Bind a later accepted write to the SAME observation used for its
        # decision. Global row indices must not silently name another packet
        # when the adapter moves from high to low/unconfirmed/recovery stages.
        for index, (vector, quality) in zip(indices, features):
            if index in self._observations:
                previous, previous_quality = self._observations[index]
                if (quality != previous_quality or (vector is None) != (previous is None)
                        or (vector is not None and not np.array_equal(vector, previous))):
                    raise ValueError("Identity observation changed within a frame")
        for index, (vector, quality) in zip(indices, features):
            self._observations[index] = (None if vector is None else vector.copy(), quality)
        feasible = geometry <= feasible_threshold
        distances = np.full(geometry.shape, np.nan)
        known = np.zeros(geometry.shape, dtype=bool)
        cfg = self.config
        states = [self._snapshot.get(tid) for tid in ids]
        anchor_rows = [i for i, state in enumerate(states) if state is not None]
        feature_columns = [j for j, (vector, _) in enumerate(features) if vector is not None]
        if anchor_rows and feature_columns:
            anchors = np.stack([states[i].vector for i in anchor_rows])
            observations = np.stack([features[j][0] for j in feature_columns])
            if anchors.shape[1] != observations.shape[1]:
                raise ValueError("Identity RGB embedding dimension changed")
            selected = np.ix_(anchor_rows, feature_columns)
            distances[selected] = np.clip(1. - anchors @ observations.T, 0., 2.)
            anchor_known = np.array([states[i].updates >= cfg.min_history
                                     and states[i].quality >= cfg.min_quality
                                     and states[i].quality > 0 for i in anchor_rows])
            observation_known = np.array([features[j][1] >= cfg.min_quality
                                          and features[j][1] > 0 for j in feature_columns])
            known[selected] = anchor_known[:, None] & observation_known[None, :]

        # Competitors are drawn from the ORIGINAL geometry feasible set, even
        # when their embeddings are unknown. Aggregate counts and the first two
        # minima once per row/column instead of rescanning them for every pair.
        # Thus competitor analysis is O(N*M), rather than O(N*M*(N+M)).
        competitor_counts = (feasible.sum(axis=1)[:, None] + feasible.sum(axis=0)[None, :]
                             - 2 * feasible.astype(int))
        unknown = feasible & ~known
        unknown_counts = (unknown.sum(axis=1)[:, None] + unknown.sum(axis=0)[None, :]
                          - 2 * unknown.astype(int))
        known_competition = (competitor_counts > 0) & (unknown_counts == 0)
        geometric_choices = np.where(feasible, geometry, np.inf)
        geometric_alternative = np.minimum(_minimum_excluding_each(geometric_choices, axis=1),
                                           _minimum_excluding_each(geometric_choices, axis=0))
        geometric_gaps = geometric_alternative - geometry
        appearance_choices = np.where(feasible & np.isfinite(distances), distances, np.inf)
        appearance_alternative = np.minimum(_minimum_excluding_each(appearance_choices, axis=1),
                                            _minimum_excluding_each(appearance_choices, axis=0))
        appearance_gaps = appearance_alternative - distances
        decisions = []
        for i, tid in enumerate(ids):
            row = []
            state = states[i]
            count = 0 if state is None else state.updates
            for j, index in enumerate(indices):
                vector, quality = features[j]
                competitors = int(competitor_counts[i, j])
                distance = float(distances[i, j]) if np.isfinite(distances[i, j]) else None
                geom_margin = float(geometric_gaps[i, j]) if competitors else None
                competition_known = bool(known_competition[i, j])
                app_margin = (float(appearance_gaps[i, j])
                              if distance is not None and competition_known else None)
                good_quality = vector is not None and quality >= cfg.min_quality and quality > 0
                history_ok = distance is not None and distance <= cfg.history_max_distance
                unique_geometry = geom_margin is None or geom_margin >= cfg.geometry_margin
                strong_geometry = geometry[i, j] <= cfg.bootstrap_max_iou_cost and unique_geometry
                r, allow, reason = 0., False, "outside_geometry"
                if feasible[i, j]:
                    if not good_quality:
                        reason = "missing_feature" if vector is None else "low_quality"
                    elif count < cfg.min_history:
                        # Explicit reference fallback, not a fabricated confidence 1.
                        r = None
                        if stage not in ("high", "unconfirmed"):
                            reason = "warmup_requires_high"
                        elif state is not None and not history_ok:
                            reason = "history_conflict"
                        elif not strong_geometry:
                            reason = "warmup_geometry_ambiguous"
                        else:
                            allow, reason = True, "warmup_geometry_verified"
                    else:
                        historical_support = (float(np.clip(1. - distance / cfg.history_max_distance, 0., 1.))
                                              if distance is not None else 0.)
                        if competitors:
                            margin_support = (float(np.clip(app_margin / cfg.appearance_margin, 0., 1.))
                                              if app_margin is not None else 0.)
                        else:
                            # No rival is not infinite separation: use a second
                            # absolute historical-agreement factor instead.
                            margin_support = historical_support
                        r = float(min(state.quality, quality) * historical_support * margin_support)
                        if not history_ok:
                            reason = "history_conflict"
                        elif geometry[i, j] > cfg.write_max_iou_cost:
                            reason = "write_geometry_weak"
                        elif competitors and not competition_known:
                            reason = "competition_unknown"
                        elif competitors and app_margin < cfg.appearance_margin:
                            reason = "appearance_ambiguous"
                        elif not competitors and not strong_geometry:
                            reason = "singleton_geometry_weak"
                        else:
                            allow, reason = True, "identity_verified"
                    # B accepts/recovers its identity as before. Its first
                    # reactivation is not sufficient to overwrite memory.
                if stage in ("recovery", "recovered"):
                    allow, reason = False, "recovered_quarantine"
                row.append(IdentityDecision(self._frame, stage, tid, index, r, allow, reason,
                                            distance, app_margin, geom_margin, competitors,
                                            competition_known, count, quality, float(geometry[i, j])))
            decisions.append(row)
        return decisions

    @staticmethod
    def calibrate(base_cost, reference_cost, decisions):
        """Only attenuate the reference appearance bonus; never broaden gates."""
        base = np.asarray(base_cost, dtype=float)
        reference = np.asarray(reference_cost, dtype=float)
        if base.shape != reference.shape or base.ndim != 2 or not np.isfinite(base).all() or not np.isfinite(reference).all():
            raise ValueError("Identity calibration expects finite equally shaped cost matrices")
        if np.any(base < 0) or np.any(reference < 0) or np.any(reference > base + 1e-12):
            raise ValueError("Reference must be nonnegative and no greater than geometric cost")
        if len(decisions) != len(base) or any(len(row) != base.shape[1] for row in decisions):
            raise ValueError("Identity decisions do not align with costs")
        reference = np.minimum(reference, base)
        result = reference.copy()
        for i, row in enumerate(decisions):
            for j, decision in enumerate(row):
                r = decision.reliability
                if r is not None:
                    if not np.isfinite(r) or not 0 <= r <= 1:
                        raise ValueError("Identity reliability must be in [0,1] or None")
                    result[i, j] = base[i, j] - r * max(0., base[i, j] - reference[i, j])
        return result

    def seed(self, track_id, packet, frame):
        """One provisional native-birth seed; no claimed identity confirmation."""
        self._check_frame(frame)
        tid = _positive_id(track_id, "track_id")
        vector, quality = _rgb(packet)
        self._check_dimension(vector)
        if tid in self._written or tid in self.states:
            raise ValueError("Identity seed would overwrite an existing identity/write")
        self._written.add(tid)
        if vector is None or quality < self.config.min_quality or quality <= 0:
            return False
        self._dimension = len(vector)
        self.states[tid] = _anchor(vector, quality, 1, frame)
        return True

    def commit(self, track_id, packet, decision: IdentityDecision, frame):
        self._check_frame(frame)
        tid = _positive_id(track_id, "track_id")
        if decision.frame != frame or decision.track_id != tid:
            raise ValueError("Identity write does not match the evaluated identity/frame")
        vector, quality = _rgb(packet)
        self._check_dimension(vector)
        evaluated = self._observations.get(decision.det_index)
        if evaluated is None:
            raise ValueError("Identity write has no evaluated observation")
        expected_vector, expected_quality = evaluated
        if (quality != expected_quality or (vector is None) != (expected_vector is None)
                or (vector is not None and not np.array_equal(vector, expected_vector))):
            raise ValueError("Identity write feature differs from its evaluated observation")
        if tid in self._written:
            raise ValueError("An identity may be committed only once per frame")
        self._written.add(tid)
        if not decision.allow_write:
            return False
        if vector is None or quality < self.config.min_quality or quality <= 0:
            raise ValueError("Eligible identity write no longer has its valid RGB feature")
        state = self._snapshot.get(tid)
        if state is None:
            updated, q, count = vector, quality, 1
        else:
            updated = self.config.ema * state.vector + (1 - self.config.ema) * vector
            norm = float(np.linalg.norm(updated))
            if norm <= 1e-12:
                return False
            updated /= norm
            q = self.config.ema * state.quality + (1 - self.config.ema) * quality
            count = state.updates + 1
        self._dimension = len(vector)
        self.states[tid] = _anchor(updated, q, count, frame)
        return True

    def retain(self, live_ids):
        live = {_positive_id(x, "track_id") for x in live_ids}
        self.states = {tid: state for tid, state in self.states.items() if tid in live}
        self._snapshot = {tid: state for tid, state in self._snapshot.items() if tid in live}
