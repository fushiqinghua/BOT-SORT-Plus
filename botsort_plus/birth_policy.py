"""A v0.2: private, causal evidence confirmation before public-ID creation.

Geometry links observations in the shared CandidatePool. This policy does not
use geometric evidence as existence support, except in the explicit ``legacy``
ablation. It never edits a candidate, consumes a detection, allocates an ID or
writes an identity memory. B can therefore continue using the same history when
A's confirmation fails.

Call ``evaluate`` for EVERY live candidate on EVERY video frame, including
misses. Repeating an unchanged candidate/frame is idempotent. After confirmation
has first succeeded, readiness is re-evaluated using recent observations; an
old confirmation is not a permanent birth permit during identity deferral.

The score threshold is an initial research setting, not a calibrated existence
probability. Successive video observations are new in time, not statistically
independent evidence. ``observe_only`` is enforced by the integration layer:
this policy still reports its real decisions for a useful shadow audit.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from typing import Any, Iterable


@dataclass(frozen=True)
class BirthPolicyConfig:
    mode: str = "two_stage"
    collect_window: int = 5
    collect_hits: int = 3
    mean_score: float = 0.18
    verify_window: int = 3
    verify_hits: int = 2
    fixed_window: int = 8
    fixed_hits: int = 5
    observe_only: bool = False

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.mode not in ("legacy", "score_window", "fixed_window", "two_stage"):
            raise ValueError("birth_policy.mode must be legacy, score_window, fixed_window or two_stage")
        for name in ("collect_window", "collect_hits", "verify_window", "verify_hits",
                     "fixed_window", "fixed_hits"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"birth_policy.{name} must be a positive integer")
        if self.collect_hits > self.collect_window:
            raise ValueError("birth_policy.collect_hits must fit in collect_window")
        if self.verify_hits > self.verify_window:
            raise ValueError("birth_policy.verify_hits must fit in verify_window")
        if self.fixed_hits > self.fixed_window:
            raise ValueError("birth_policy.fixed_hits must fit in fixed_window")
        if (isinstance(self.mean_score, bool) or not isinstance(self.mean_score, Real)
                or not math.isfinite(self.mean_score) or not 0 <= self.mean_score <= 1):
            raise ValueError("birth_policy.mean_score must be finite and lie in [0, 1]")
        if not isinstance(self.observe_only, bool):
            raise ValueError("birth_policy.observe_only must be bool")
        return self


@dataclass(frozen=True)
class BirthEvidenceDecision:
    candidate_id: int
    frame: int
    mode: str
    ready: bool
    phase: str
    reason: str
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class _CandidateState:
    epoch_start: int = 0
    armed_frame: int | None = None
    verified_at: int | None = None
    armed_frames: tuple[int, ...] = ()
    armed_mean_score: float | None = None
    last_frame: int | None = None
    last_signature: tuple | None = None
    last_decision: BirthEvidenceDecision | None = None


class BirthPolicy:
    """Only A's private evidence state; shared candidate and B state stay intact.

    A failed first verification advances this policy's ``epoch_start`` to the
    deadline. Its next collection must use observations strictly after that
    frame. The original candidate observations and its overall TTL are untouched.

    A successful first verification remains remembered only to choose the phase.
    Every later ready decision additionally needs a current detection and enough
    supported observations in the trailing ``verify_window`` after ``armed_frame``.
    """

    def __init__(self, config: BirthPolicyConfig):
        self.config = config.validate()
        self.states: dict[int, _CandidateState] = {}
        self._last_frame = 0

    def forget(self, candidate_ids: Iterable[int]) -> None:
        """Release private bookkeeping when a candidate leaves the shared pool."""
        for candidate_id in candidate_ids:
            self.states.pop(int(candidate_id), None)

    @staticmethod
    def _history(candidate: Any, frame: int) -> list[tuple[int, float]]:
        history = []
        for observation in candidate.observations:
            when, score = observation.frame, observation.score
            if isinstance(when, bool) or not isinstance(when, Integral) or when < 1:
                raise ValueError("Candidate observations must have positive integer frames")
            if when > frame:
                raise ValueError("Future observations are forbidden in online birth confirmation")
            if (isinstance(score, bool) or not isinstance(score, Real)
                    or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError("Candidate observation scores must be finite and lie in [0, 1]")
            history.append((int(when), float(score)))
        frames = [item[0] for item in history]
        if frames != sorted(set(frames)):
            raise ValueError("Candidate observations must have unique increasing frames")
        if not history:
            raise ValueError("A candidate must contain at least one real observation")
        if candidate.last_frame != frames[-1]:
            raise ValueError("Candidate last_frame does not match its observation history")
        return history

    @staticmethod
    def _mean(history: list[tuple[int, float]]) -> float | None:
        return math.fsum(s for _, s in history) / len(history) if history else None

    @staticmethod
    def _window(history: list[tuple[int, float]], frame: int, window: int,
                floor: int = 0) -> list[tuple[int, float]]:
        return [(f, s) for f, s in history if max(frame - window, floor) < f <= frame]

    def evaluate(self, frame: int, candidate: Any) -> BirthEvidenceDecision:
        if (isinstance(frame, bool) or not isinstance(frame, Integral)
                or frame < 1 or frame < self._last_frame):
            raise ValueError("Birth-policy frames must be positive and nondecreasing")
        cid = candidate.candidate_id
        if isinstance(cid, bool) or not isinstance(cid, Integral) or cid < 1:
            raise ValueError("Candidate IDs must be positive integers")
        frame, cid = int(frame), int(cid)
        history = self._history(candidate, frame)
        # Do not access legacy evidence at all in the new confirmation modes.
        legacy_evidence = None
        if self.config.mode == "legacy":
            legacy_evidence = float(candidate.evidence)
            if not math.isfinite(legacy_evidence) or legacy_evidence < 0:
                raise ValueError("Legacy candidate evidence must be finite and nonnegative")
        signature = (tuple(history), legacy_evidence)
        state = self.states.setdefault(cid, _CandidateState())
        if state.last_frame == frame:
            if state.last_signature != signature:
                raise ValueError("Candidate evidence changed after its decision in the same frame")
            return deepcopy(state.last_decision)
        if state.last_frame is not None and frame != state.last_frame + 1:
            raise ValueError("Evaluate every live candidate on consecutive frames, including misses")
        self._last_frame = frame
        cfg = self.config
        current_hit = history[-1][0] == frame
        collection_window = cfg.fixed_window if cfg.mode == "fixed_window" else cfg.collect_window
        collection = self._window(history, frame, collection_window, state.epoch_start)
        collect_mean = self._mean(collection)
        verification = (self._window(history, frame, cfg.verify_window, state.armed_frame)
                        if state.armed_frame is not None else [])
        verify_mean = self._mean(verification)
        stats = {
            "epoch_start": state.epoch_start,
            "armed_frame": state.armed_frame,
            "verification_deadline": (state.armed_frame + int(cfg.verify_window)
                                      if state.armed_frame is not None else None),
            "verified_at": state.verified_at,
            "armed_frames": list(state.armed_frames),
            "armed_mean_score": state.armed_mean_score,
            "collect_frames": [f for f, _ in collection],
            "collect_hits": len(collection),
            "collect_mean_score": collect_mean,
            "verify_frames": [f for f, _ in verification],
            "verify_hits": len(verification),
            "verify_mean_score": verify_mean,
            "current_hit": current_hit,
            "threshold": float(cfg.mean_score),
        }
        ready, phase, reason = False, "COLLECTING", "insufficient_hits"
        if cfg.mode == "legacy":
            ready = current_hit and bool(candidate.ready(frame))
            phase = "READY" if ready else "COLLECTING"
            reason = "legacy_ready" if ready else "legacy_not_ready"
            stats["legacy_evidence"] = legacy_evidence
        elif cfg.mode in ("score_window", "fixed_window"):
            required_hits = cfg.fixed_hits if cfg.mode == "fixed_window" else cfg.collect_hits
            if not current_hit:
                reason = "no_current_observation"
            elif len(collection) < required_hits:
                reason = "insufficient_hits"
            elif collect_mean < cfg.mean_score:
                reason = "insufficient_score_support"
            else:
                ready, phase, reason = True, "READY", "window_supported"
        elif state.armed_frame is None:
            if not current_hit:
                reason = "no_current_observation"
            elif len(collection) < cfg.collect_hits:
                reason = "insufficient_hits"
            elif collect_mean < cfg.mean_score:
                reason = "insufficient_score_support"
            else:
                state.armed_frame = frame
                state.armed_frames = tuple(f for f, _ in collection)
                state.armed_mean_score = collect_mean
                phase, reason = "VERIFYING", "armed_waiting_for_new_observations"
                stats.update(armed_frame=frame,
                             verification_deadline=frame + int(cfg.verify_window),
                             armed_frames=list(state.armed_frames),
                             armed_mean_score=collect_mean)
        else:
            supported = (current_hit and len(verification) >= cfg.verify_hits
                         and verify_mean >= cfg.mean_score)
            if supported:
                if state.verified_at is None:
                    state.verified_at = frame
                ready, phase, reason = True, "READY", "fresh_verification_supported"
                stats["verified_at"] = state.verified_at
            elif state.verified_at is not None:
                phase = "READY_STALE"
                reason = ("no_current_observation" if not current_hit else
                          "insufficient_recent_hits" if len(verification) < cfg.verify_hits else
                          "insufficient_recent_score_support")
            elif frame >= state.armed_frame + cfg.verify_window:
                # Keep the failed attempt's arm/verification details in this
                # decision for audit, while exposing its new epoch separately.
                deadline = state.armed_frame + int(cfg.verify_window)
                state.epoch_start = deadline
                state.armed_frame = None
                state.armed_frames = ()
                state.armed_mean_score = None
                state.verified_at = None
                phase, reason = "COLLECTING", "verification_timeout"
                stats["next_epoch_start"] = deadline
            else:
                phase = "VERIFYING"
                reason = ("no_current_observation" if not current_hit else
                          "insufficient_new_hits" if len(verification) < cfg.verify_hits else
                          "insufficient_new_score_support")
        decision = BirthEvidenceDecision(cid, frame, cfg.mode, bool(ready), phase, reason, stats)
        state.last_frame = frame
        state.last_signature = signature
        state.last_decision = deepcopy(decision)
        return decision
