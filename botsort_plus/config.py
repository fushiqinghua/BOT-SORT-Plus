"""Strict, serializable experiment settings; unknown/misspelled options fail."""
from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path
from .appearance import AppearanceConfig
from .candidates import BirthConfig
from .geometry import GeometryConfig
from .recovery import RecoveryConfig


@dataclass
class PlusConfig:
    name: str = "baseline"
    birth: BirthConfig = field(default_factory=BirthConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    appearance: AppearanceConfig = field(default_factory=AppearanceConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    native_overrides: dict = field(default_factory=dict)

    @property
    def enabled(self):
        return self.birth.enabled or self.recovery.enabled or self.appearance.enabled

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown experiment options: {sorted(unknown)}")
        data = dict(data)
        for key, kind in (("birth",BirthConfig), ("recovery",RecoveryConfig),
                          ("appearance",AppearanceConfig), ("geometry",GeometryConfig)):
            if key in data:
                extra = set(data[key]) - {f.name for f in fields(kind)}
                if extra:
                    raise ValueError(f"Unknown {key} options: {sorted(extra)}")
                data[key] = kind(**data[key])
        result = cls(**data)
        if not result.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in result.name):
            raise ValueError("Experiment name must be a plain filename component")
        if not isinstance(result.native_overrides, dict):
            raise ValueError("native_overrides must be an object")
        extra = set(result.native_overrides) - {"new_track_thresh", "track_high_thresh", "track_buffer"}
        if extra:
            raise ValueError(f"Only explicit threshold/buffer controls are supported: {extra}")
        for key,value in result.native_overrides.items():
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(f"native_overrides.{key} must be a finite number")
            if key=="track_buffer":
                if not isinstance(value,int) or value<1:
                    raise ValueError("track_buffer must be a positive integer")
            elif not 0 <= value <= 1:
                raise ValueError(f"native_overrides.{key} must lie in [0,1]")
        if result.recovery.enabled and result.recovery.window > result.birth.max_age:
            raise ValueError("Candidate pool max_age must cover B's verification window")
        return result

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
