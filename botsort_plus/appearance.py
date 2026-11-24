"""C: quality-weighted RGB/thermal appearance with explicit trusted-memory writes.

This is an inference mechanism, not a trained thermal encoder. RGB and thermal
vectors are compared only to their own modality's historical prototype, so their
dimensions need not agree. The integration layer decides whether C is enabled;
the same functions also support B with RGB alone.

``distance_scale`` is a positive divisor for an observed cosine distance. Its
default of 1 is an initialization, *not* a claim of learned/calibrated distances.
The returned distance is a cost, not a probability. Missing/zero features produce
``None`` when neither modality is usable; they never produce a perfect match.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import math

import numpy as np

from .types import Feature


MODALITIES = ("rgb", "thermal")
_NORM_EPS = 1e-12


@dataclass
class AppearanceConfig:
    enabled: bool = False
    weighting: str = "quality"
    min_observation_quality: float = 0.15
    min_update_quality: float = 0.35
    ema: float = 0.9
    weight_rgb: float = 1.0
    weight_thermal: float = 1.0
    rgb_distance_scale: float = 1.0
    thermal_distance_scale: float = 1.0
    max_distance: float = 0.6

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("appearance.enabled must be bool")
        if self.weighting not in ("quality", "fixed"):
            raise ValueError("appearance.weighting must be quality or fixed")
        for name in ("min_observation_quality", "min_update_quality", "ema"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"appearance.{name} must be finite in [0, 1]")
        if self.ema >= 1:
            raise ValueError("appearance.ema must be less than 1")
        if self.min_update_quality < self.min_observation_quality:
            raise ValueError("Memory-update quality must not be lower than match quality")
        for name in ("weight_rgb", "weight_thermal"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"appearance.{name} must be finite and nonnegative")
        if self.weight_rgb + self.weight_thermal <= 0:
            raise ValueError("At least one appearance modality must have positive weight")
        for name in ("rgb_distance_scale", "thermal_distance_scale", "max_distance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"appearance.{name} must be finite and positive")

    def validate(self):
        """Also usable by a configuration loader after construction."""
        self.__post_init__()
        return self


def _validated_features(features: Mapping[str, Feature]) -> dict[str, tuple[np.ndarray | None, float]]:
    """Validate a complete observation before any memory mutation.

    A zero vector is the known old-cache placeholder and is treated as missing.
    Malformed/nonfinite vectors are data-contract errors and are not hidden.
    Input arrays are copied; the caller's cache is never normalized in-place.
    """
    unknown = set(features) - set(MODALITIES)
    if unknown:
        raise ValueError(f"Unknown appearance modalities: {sorted(unknown)}")
    result = {}
    for modality, feature in features.items():
        if not isinstance(feature, Feature):
            raise TypeError(f"{modality} feature must be botsort_plus.types.Feature")
        vector = np.asarray(feature.vector, dtype=float)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError(f"{modality} embedding must be a nonempty 1-D vector")
        if not np.isfinite(vector).all():
            raise ValueError(f"{modality} embedding contains NaN or infinity")
        quality = float(feature.quality)
        if not math.isfinite(quality) or not 0 <= quality <= 1:
            raise ValueError(f"{modality} quality must be finite in [0, 1]")
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm):
            raise ValueError(f"{modality} embedding norm is not finite")
        result[modality] = (None if norm <= _NORM_EPS else vector.copy() / norm, quality)
    return result


class MemoryBank:
    """Separate trusted prototypes; call update ONLY after accepting an identity.

    Association functions below are read-only. A low-quality crop may update the
    track's geometry in the caller without changing this memory. A failed update
    validation leaves all modalities unchanged.
    """

    def __init__(self, config: AppearanceConfig | None = None):
        self.config = config if config is not None else AppearanceConfig()
        self.config.validate()
        self.prototypes: dict[str, np.ndarray] = {}
        self.qualities: dict[str, float] = {}
        self.update_counts: dict[str, int] = {}

    def update(self, features: Mapping[str, Feature]) -> list[str]:
        validated = _validated_features(features)
        # Validate dimensions even for low-quality observations, before writes.
        for modality, (vector, _) in validated.items():
            if vector is not None and modality in self.prototypes:
                if vector.shape != self.prototypes[modality].shape:
                    raise ValueError(f"{modality} embedding dimension changed: "
                                     f"{self.prototypes[modality].shape} -> {vector.shape}")
        pending = {}
        for modality in MODALITIES:
            if modality not in validated:
                continue
            vector, quality = validated[modality]
            if vector is None or quality < self.config.min_update_quality or quality <= 0:
                continue
            if modality not in self.prototypes:
                pending[modality] = (vector, quality)
                continue
            alpha = self.config.ema
            merged = alpha * self.prototypes[modality] + (1 - alpha) * vector
            norm = float(np.linalg.norm(merged))
            # Antipodal vectors with EMA=.5 contain no direction. Keep trusted
            # memory rather than replacing it with a zero/NaN prototype.
            if norm <= _NORM_EPS:
                continue
            merged_quality = alpha * self.qualities[modality] + (1 - alpha) * quality
            pending[modality] = (merged / norm, float(merged_quality))
        for modality, (vector, quality) in pending.items():
            self.prototypes[modality] = vector.copy()
            self.qualities[modality] = quality
            self.update_counts[modality] = self.update_counts.get(modality, 0) + 1
        return list(pending)

    def as_features(self) -> dict[str, Feature]:
        """A detached snapshot, suitable for logging or passing between modules."""
        return {m: Feature(v.copy(), self.qualities[m]) for m, v in self.prototypes.items()}


def appearance_details(
    memory: MemoryBank | None,
    features: Mapping[str, Feature],
    config: AppearanceConfig | None = None,
) -> dict:
    """Read-only pair cost and diagnostics. Rejection is the caller's decision.

    Pair quality is ``min(historical_quality, observation_quality)``. This
    conservative bottleneck prevents a sharp current crop from hiding a poor
    historical template (and vice versa). Weights are then normalized across the
    usable modalities. The ``weighting='fixed'`` ablation removes this multiplier
    but retains the same observation-validity and trusted-memory update gates;
    it isolates dynamic fusion weights, not the whole quality mechanism.
    ``max_distance`` is reported as a gate flag; a bad match
    remains a large cost and is not converted to missing evidence.
    """
    cfg = config if config is not None else (memory.config if memory is not None else AppearanceConfig())
    cfg.validate()
    validated = _validated_features(features)
    result = {"distance": None, "available": False, "within_gate": False,
              "total_weight": 0.0, "used_modalities": [], "modalities": {}}
    weighted_distance = 0.0
    for modality in MODALITIES:
        row = {"available": False, "reason": None, "weight": 0.0,
               "normalized_weight": 0.0, "raw_distance": None, "distance": None,
               "observation_quality": None, "memory_quality": None}
        result["modalities"][modality] = row
        if modality not in validated:
            row["reason"] = "missing_observation"
            continue
        vector, quality = validated[modality]
        row["observation_quality"] = quality
        if vector is None:
            row["reason"] = "zero_placeholder"
            continue
        if memory is None or modality not in memory.prototypes:
            row["reason"] = "missing_memory"
            continue
        prototype = memory.prototypes[modality]
        if vector.shape != prototype.shape:
            raise ValueError(f"{modality} embedding dimension changed: {prototype.shape} -> {vector.shape}")
        previous_quality = float(memory.qualities[modality])
        row["memory_quality"] = previous_quality
        if min(quality, previous_quality) < cfg.min_observation_quality or min(quality, previous_quality) <= 0:
            row["reason"] = "low_quality"
            continue
        static_weight = getattr(cfg, f"weight_{modality}")
        if static_weight <= 0:
            row["reason"] = "disabled_modality"
            continue
        weight = static_weight * (min(quality, previous_quality) if cfg.weighting == "quality" else 1.0)
        raw_distance = float(np.clip(1.0 - np.dot(prototype, vector), 0.0, 2.0))
        distance = raw_distance / getattr(cfg, f"{modality}_distance_scale")
        row.update(available=True, weight=float(weight), raw_distance=raw_distance,
                   distance=float(distance), reason="usable")
        weighted_distance += weight * distance
        result["total_weight"] += weight
        result["used_modalities"].append(modality)
    if result["total_weight"] > 0:
        result["distance"] = float(weighted_distance / result["total_weight"])
        result["available"] = True
        result["within_gate"] = result["distance"] <= cfg.max_distance
        for row in result["modalities"].values():
            row["normalized_weight"] = row["weight"] / result["total_weight"]
    return result


def appearance_distance(
    memory: MemoryBank | None,
    features: Mapping[str, Feature],
    config: AppearanceConfig | None = None,
) -> float | None:
    """Return a finite weighted distance, or None when no appearance is usable."""
    return appearance_details(memory, features, config)["distance"]


def estimate_quality(image: np.ndarray, bbox: np.ndarray, modality: str = "rgb") -> float:
    """Conservative, unlearned crop-quality proxy computed at native resolution.

    This measures visible pixel support, contrast and adjacent-pixel gradients;
    it does not estimate semantic discriminativeness, registration correctness,
    or actual ReID accuracy. Calibration is necessary before using its thresholds
    in an experiment. RGB input follows OpenCV's BGR order. Grayscale thermal,
    uint8/uint16 images, and explicitly normalized float [0,1] inputs are
    supported. Never resize a 4-pixel crop to 256 pixels before this function.

    The modality argument validates provenance but deliberately introduces no
    unvalidated claim that thermal is intrinsically more/less reliable than RGB.
    """
    if modality not in MODALITIES:
        raise ValueError(f"Unknown appearance modality: {modality}")
    image = np.asarray(image)
    if image.ndim not in (2, 3) or min(image.shape[:2], default=0) <= 0:
        raise ValueError("Quality estimation needs a nonempty HxW or HxWxC image")
    if image.ndim == 3 and image.shape[2] not in (1, 3):
        raise ValueError("Quality estimation supports one or three image channels")
    bbox = np.asarray(bbox, dtype=float)
    if bbox.shape != (4,) or not np.isfinite(bbox).all() or np.any(bbox[2:] <= bbox[:2]):
        raise ValueError("Quality bbox must be finite xyxy with positive dimensions")
    h, w = image.shape[:2]
    x1, y1 = np.maximum(bbox[:2], (0, 0))
    x2, y2 = np.minimum(bbox[2:], (w, h))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    visible_area = float((x2 - x1) * (y2 - y1))
    requested_area = float(np.prod(bbox[2:] - bbox[:2]))
    clipped_fraction = min(1.0, visible_area / requested_area)
    # Use continuous native pixel support, not the potentially rounded-up crop.
    size_quality = min(1.0, math.sqrt(visible_area) / 16.0)
    crop = image[int(math.floor(y1)):int(math.ceil(y2)),
                 int(math.floor(x1)):int(math.ceil(x2))].astype(float)
    # Check the requested crop, not the full image once per detection. The latter
    # would scale as image_pixels × number_of_objects in feature export.
    if not np.isfinite(crop).all():
        raise ValueError("Image crop contains NaN or infinity")
    if np.issubdtype(image.dtype, np.integer):
        if np.min(crop) < 0:
            raise ValueError("Integer images must have nonnegative intensity")
        crop /= float(np.iinfo(image.dtype).max)
    elif np.issubdtype(image.dtype, np.floating):
        if np.min(crop) < 0 or np.max(crop) > 1:
            raise ValueError("Float images must be explicitly normalized to [0,1]")
    else:
        raise ValueError("Quality estimation requires integer or float image pixels")
    if crop.ndim == 3:
        crop = crop[..., 0] if crop.shape[2] == 1 else crop @ np.array([0.114, 0.587, 0.299])
    if min(crop.shape) < 2:
        return 0.0
    contrast = min(1.0, float(np.std(crop)) / 0.12)
    dx, dy = np.diff(crop, axis=1), np.diff(crop, axis=0)
    gradient_rms = math.sqrt((float(np.mean(dx ** 2)) + float(np.mean(dy ** 2))) / 2.0)
    sharpness = min(1.0, gradient_rms / 0.12)
    return float(np.clip(size_quality * clipped_fraction * math.sqrt(contrast * sharpness), 0, 1))
