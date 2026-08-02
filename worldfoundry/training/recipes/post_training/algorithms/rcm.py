"""Behavior-only recipe contracts for native rCM and Causal-rCM."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi

from ..common import strict_mapping


def _integer(value: object, *, field_name: str, allow_zero: bool) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < (0 if allow_zero else 1):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{field_name} must be a {qualifier} integer")
    return int(value)


def _float(value: object, *, field_name: str) -> float:
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    return resolved


@dataclass(frozen=True, slots=True)
class RCMAlgorithmSpec:
    """Bidirectional rCM choices consumed by the native objective and engine."""

    teacher_checkpoint: str = "default"
    fake_score_checkpoint: str | None = "default"
    consistency_mode: str = "continuous"
    tangent_warmup_steps: int = 0
    student_update_frequency: int = 5
    teacher_guidance_scale: float = 5.0
    consistency_loss_scale: float = 100.0
    dmd_loss_scale: float = 1.0
    max_rollout_steps: int = 4
    generator_time_mean: float = -0.8
    generator_time_std: float = 1.6
    score_time_mean: float = 0.0
    score_time_std: float = 1.6
    tangent_normalization_constant: float = 0.1
    dcm_total_steps: int = 48
    dcm_skipping_interval_steps: int = 1
    dcm_timestep_shift: float = 5.0
    fixed_rollout_timesteps: tuple[float, ...] = ()
    type: str = "rcm"

    def __post_init__(self) -> None:
        algorithm_type = str(self.type).strip().lower().replace("_", "-")
        if algorithm_type != "rcm":
            raise ValueError("rCM algorithm type must be 'rcm'")
        mode = str(self.consistency_mode).strip().lower().replace("_", "-")
        if mode not in {"continuous", "discrete"}:
            raise ValueError("rCM consistency_mode must be continuous or discrete")
        object.__setattr__(self, "type", algorithm_type)
        object.__setattr__(self, "consistency_mode", mode)
        for name, allow_zero in (
            ("tangent_warmup_steps", True),
            ("student_update_frequency", False),
            ("max_rollout_steps", False),
            ("dcm_total_steps", False),
            ("dcm_skipping_interval_steps", False),
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), field_name=name, allow_zero=allow_zero),
            )
        if self.dcm_skipping_interval_steps >= self.dcm_total_steps:
            raise ValueError("dcm_skipping_interval_steps must be smaller than dcm_total_steps")
        for name in (
            "teacher_guidance_scale",
            "consistency_loss_scale",
            "dmd_loss_scale",
            "generator_time_mean",
            "generator_time_std",
            "score_time_mean",
            "score_time_std",
            "tangent_normalization_constant",
            "dcm_timestep_shift",
        ):
            object.__setattr__(self, name, _float(getattr(self, name), field_name=name))
        if self.consistency_loss_scale < 0 or self.dmd_loss_scale < 0:
            raise ValueError("rCM loss scales must be non-negative")
        if self.consistency_loss_scale == 0 and self.dmd_loss_scale == 0:
            raise ValueError("at least one rCM loss must be enabled")
        if self.consistency_loss_scale == 0 and self.tangent_warmup_steps != 0:
            raise ValueError("tangent warmup must be zero without consistency loss")
        teacher_checkpoint = str(self.teacher_checkpoint).strip()
        if not teacher_checkpoint:
            raise ValueError("teacher_checkpoint must be non-empty")
        fake_score_checkpoint = (
            None
            if self.fake_score_checkpoint is None
            else str(self.fake_score_checkpoint).strip()
        )
        if fake_score_checkpoint == "":
            raise ValueError("fake_score_checkpoint must be non-empty when supplied")
        if (self.dmd_loss_scale > 0) != (fake_score_checkpoint is not None):
            raise ValueError(
                "fake_score_checkpoint must be supplied exactly when rCM DMD executes"
            )
        if self.generator_time_std <= 0 or self.score_time_std <= 0:
            raise ValueError("rCM time standard deviations must be positive")
        if self.tangent_normalization_constant <= 0 or self.dcm_timestep_shift <= 0:
            raise ValueError("rCM normalization and timestep shift must be positive")
        rollout = tuple(_float(value, field_name="fixed_rollout_timesteps") for value in self.fixed_rollout_timesteps)
        if rollout and len(rollout) != self.max_rollout_steps - 1:
            raise ValueError(
                "fixed_rollout_timesteps must be empty or cover every multi-step rollout"
            )
        if any(not 0 < value < pi / 2 for value in rollout):
            raise ValueError("fixed rollout times must be in (0,pi/2)")
        if any(left <= right for left, right in zip(rollout, rollout[1:])):
            raise ValueError("fixed rollout times must be strictly descending")
        object.__setattr__(self, "fixed_rollout_timesteps", rollout)
        object.__setattr__(self, "teacher_checkpoint", teacher_checkpoint)
        object.__setattr__(self, "fake_score_checkpoint", fake_score_checkpoint)


@dataclass(frozen=True, slots=True)
class CausalRCMAlgorithmSpec:
    """Causal teacher-forcing CM plus self-forcing DMD behavior."""

    causal_teacher_checkpoint: str | None = "default"
    bidirectional_teacher_checkpoint: str | None = "default"
    fake_score_checkpoint: str | None = "default"
    consistency_mode: str = "discrete"
    tangent_warmup_steps: int = 0
    student_update_frequency: int = 5
    causal_teacher_guidance_scale: float = 3.0
    bidirectional_teacher_guidance_scale: float = 5.0
    consistency_loss_scale: float = 100.0
    dmd_loss_scale: float = 1.0
    max_rollout_steps: int = 4
    generator_time_mean: float = -0.8
    generator_time_std: float = 1.6
    score_timestep_shift: float = 5.0
    tangent_normalization_constant: float = 0.1
    dcm_total_steps: int = 48
    dcm_skipping_interval_steps: int = 1
    dcm_timestep_shift: float = 3.0
    first_chunk_frames: int = 1
    chunk_frames: int = 1
    spatial_patch_area: int = 4
    rollout_timesteps: tuple[float, ...] = (15 / 16, 5 / 6, 5 / 8)
    type: str = "causal-rcm"

    def __post_init__(self) -> None:
        algorithm_type = str(self.type).strip().lower().replace("_", "-")
        if algorithm_type != "causal-rcm":
            raise ValueError("Causal-rCM algorithm type must be 'causal-rcm'")
        mode = str(self.consistency_mode).strip().lower().replace("_", "-")
        if mode not in {"continuous", "discrete"}:
            raise ValueError("Causal-rCM consistency_mode must be continuous or discrete")
        object.__setattr__(self, "type", algorithm_type)
        object.__setattr__(self, "consistency_mode", mode)
        for name, allow_zero in (
            ("tangent_warmup_steps", True),
            ("student_update_frequency", False),
            ("max_rollout_steps", False),
            ("dcm_total_steps", False),
            ("dcm_skipping_interval_steps", False),
            ("first_chunk_frames", False),
            ("chunk_frames", False),
            ("spatial_patch_area", False),
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), field_name=name, allow_zero=allow_zero),
            )
        if self.dcm_skipping_interval_steps >= self.dcm_total_steps:
            raise ValueError("dcm_skipping_interval_steps must be smaller than dcm_total_steps")
        for name in (
            "causal_teacher_guidance_scale",
            "bidirectional_teacher_guidance_scale",
            "consistency_loss_scale",
            "dmd_loss_scale",
            "generator_time_mean",
            "generator_time_std",
            "score_timestep_shift",
            "tangent_normalization_constant",
            "dcm_timestep_shift",
        ):
            object.__setattr__(self, name, _float(getattr(self, name), field_name=name))
        if self.consistency_loss_scale < 0 or self.dmd_loss_scale < 0:
            raise ValueError("Causal-rCM loss scales must be non-negative")
        if self.consistency_loss_scale == 0 and self.dmd_loss_scale == 0:
            raise ValueError("at least one Causal-rCM loss must be enabled")
        if self.consistency_loss_scale == 0 and self.tangent_warmup_steps != 0:
            raise ValueError("tangent warmup must be zero without consistency loss")
        checkpoints: dict[str, str | None] = {}
        for name in (
            "causal_teacher_checkpoint",
            "bidirectional_teacher_checkpoint",
            "fake_score_checkpoint",
        ):
            raw = getattr(self, name)
            value = None if raw is None else str(raw).strip()
            if value == "":
                raise ValueError(f"{name} must be non-empty when supplied")
            checkpoints[name] = value
        if (self.consistency_loss_scale > 0) != (
            checkpoints["causal_teacher_checkpoint"] is not None
        ):
            raise ValueError(
                "causal_teacher_checkpoint must be supplied exactly when consistency executes"
            )
        dmd_enabled = self.dmd_loss_scale > 0
        for name in ("bidirectional_teacher_checkpoint", "fake_score_checkpoint"):
            if dmd_enabled != (checkpoints[name] is not None):
                raise ValueError(f"{name} must be supplied exactly when Causal-rCM DMD executes")
        if self.generator_time_std <= 0:
            raise ValueError("generator_time_std must be positive")
        if (
            self.score_timestep_shift <= 0
            or self.tangent_normalization_constant <= 0
            or self.dcm_timestep_shift <= 0
        ):
            raise ValueError("Causal-rCM time shifts and normalization must be positive")
        rollout = tuple(_float(value, field_name="rollout_timesteps") for value in self.rollout_timesteps)
        if len(rollout) != self.max_rollout_steps - 1:
            raise ValueError("rollout_timesteps must exactly cover max_rollout_steps - 1")
        if any(not 0 < value < 1 for value in rollout):
            raise ValueError("causal rollout times must be in (0,1)")
        if any(left <= right for left, right in zip(rollout, rollout[1:])):
            raise ValueError("causal rollout times must be strictly descending")
        object.__setattr__(self, "rollout_timesteps", rollout)
        for name, value in checkpoints.items():
            object.__setattr__(self, name, value)


RCM_ALGORITHM_FIELDS = frozenset(RCMAlgorithmSpec.__dataclass_fields__)
CAUSAL_RCM_ALGORITHM_FIELDS = frozenset(CausalRCMAlgorithmSpec.__dataclass_fields__)


def parse_rcm_algorithm(value: object) -> RCMAlgorithmSpec:
    return RCMAlgorithmSpec(
        **strict_mapping(
            value,
            field_name="algorithm",
            allowed=set(RCM_ALGORITHM_FIELDS),
        )
    )


def parse_causal_rcm_algorithm(value: object) -> CausalRCMAlgorithmSpec:
    return CausalRCMAlgorithmSpec(
        **strict_mapping(
            value,
            field_name="algorithm",
            allowed=set(CAUSAL_RCM_ALGORITHM_FIELDS),
        )
    )


__all__ = [
    "CausalRCMAlgorithmSpec",
    "RCMAlgorithmSpec",
    "parse_causal_rcm_algorithm",
    "parse_rcm_algorithm",
]
