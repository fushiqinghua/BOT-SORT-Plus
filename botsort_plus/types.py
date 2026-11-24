"""Small, explicit data contracts shared by A (birth), B (recovery) and C (appearance)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class Feature:
    """A real embedding and its observation quality, never a zero placeholder."""
    vector: np.ndarray
    quality: float = 1.0


@dataclass
class Observation:
    frame: int
    det_index: int
    box: np.ndarray
    score: float
    features: dict[str, Feature] = field(default_factory=dict)

    def __post_init__(self):
        self.box = np.asarray(self.box, dtype=float).copy()
        if self.box.shape != (4,) or not np.isfinite(self.box).all():
            raise ValueError("Observation.box must be a finite xyxy vector")
        if np.any(self.box[2:] <= self.box[:2]):
            raise ValueError("Observation must have positive width and height")
        if self.frame < 1 or self.det_index < 0 or not 0 <= self.score <= 1:
            raise ValueError("Invalid frame, detection index or confidence")


@dataclass
class Prediction:
    """KF prediction in the image coordinates of exactly this frame."""
    frame: int
    box: np.ndarray
    center_cov: np.ndarray


@dataclass
class LostTrackView:
    track_id: int
    last_seen: int
    predictions: dict[int, Prediction]
    memory: Any = None
