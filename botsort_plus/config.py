"""Strict, serializable experiment settings; unknown/misspelled options fail."""
from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path
from .appearance import AppearanceConfig
from .candidates import BirthConfig
from .geometry import GeometryConfig
from .recovery import RecoveryConfig
from .birth_policy import BirthPolicyConfig
from .birth_identity import BirthIdentityConfig
from .identity_reliability import IdentityConfig


@dataclass
class RawInfraredConfig:
    presence: str | None = None
    native: str | None = None
    recovery: str | None = None
    shift: bool = False

    def __post_init__(self):
        if (self.presence not in (None, "shadow", "guard", "sensitive", "position") or
                self.native not in (None, "shadow", "rerank") or
                self.recovery not in (None, "shadow", "rerank", "delay") or type(self.shift) is not bool):
            raise ValueError("Unknown raw infrared policy")

    @property
    def enabled(self):
        return bool(self.presence or self.native or self.recovery)


@dataclass
class PlusConfig:
    name: str = "baseline"
    raw_ir: RawInfraredConfig = field(default_factory=RawInfraredConfig)
    birth: BirthConfig = field(default_factory=BirthConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    appearance: AppearanceConfig = field(default_factory=AppearanceConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    native_overrides: dict = field(default_factory=dict)
    # Old configurations retain the exact v0.1 adapter path.  "v2" opts into
    # native list parity, isolated observation mode, and transactional birth.
    adapter_mode: str = "legacy"
    birth_policy: BirthPolicyConfig = field(default_factory=BirthPolicyConfig)
    birth_identity: BirthIdentityConfig = field(default_factory=BirthIdentityConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)

    @property
    def enabled(self):
        return (self.birth.enabled or self.recovery.enabled or self.appearance.enabled
                or self.identity.enabled)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("Experiment configuration must be an object")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown experiment options: {sorted(unknown)}")
        data = dict(data)
        for key, kind in (("birth",BirthConfig), ("recovery",RecoveryConfig),
                          ("appearance",AppearanceConfig), ("geometry",GeometryConfig),
                          ("birth_policy",BirthPolicyConfig), ("birth_identity",BirthIdentityConfig),
                          ("identity",IdentityConfig), ("raw_ir",RawInfraredConfig)):
            if key in data:
                if not isinstance(data[key], dict):
                    raise ValueError(f"{key} configuration must be an object")
                extra = set(data[key]) - {f.name for f in fields(kind)}
                if extra:
                    raise ValueError(f"Unknown {key} options: {sorted(extra)}")
                data[key] = kind(**data[key])
        result = cls(**data)
        if result.recovery.output_prediction_frames and result.adapter_mode != "v2":
            raise ValueError("Online prediction output requires the v2 adapter")
        result.validate_v2()
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

    def validate_v2(self):
        if self.raw_ir.enabled and (self.adapter_mode != "v2" or not self.recovery.enabled):
            raise ValueError("Raw infrared evidence needs the v2 recovery pipeline")
        if self.raw_ir.presence and not self.recovery.output_prediction_frames:
            raise ValueError("Presence checks require a projection output")
        if self.raw_ir.native and not self.identity.enabled:
            raise ValueError("Raw-IR native reranking requires the C2 high association path")
        if self.adapter_mode not in ("legacy", "v2"):
            raise ValueError("adapter_mode must be legacy or v2")
        if self.identity.enabled:
            if self.adapter_mode != "v2":
                raise ValueError("Identity reliability (C2) requires adapter_mode=v2")
            if self.birth.enabled or self.appearance.enabled:
                raise ValueError("C2 currently requires A and legacy C disabled; use separate ablations")
        if self.adapter_mode == "legacy" and (
                self.birth_policy != BirthPolicyConfig() or
                self.birth_identity != BirthIdentityConfig()):
            raise ValueError("Custom birth_policy/birth_identity require adapter_mode=v2")
        if self.adapter_mode == "v2" and self.birth.enabled:
            policy = self.birth_policy
            needed = (policy.fixed_window if policy.mode == "fixed_window"
                      else max(policy.collect_window, policy.verify_window))
            if needed > self.birth.max_age:
                raise ValueError("Candidate max_age must cover A2 confirmation windows")
        return self

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
