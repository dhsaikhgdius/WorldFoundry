"""Execution configuration for native rCM objectives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, pi
from typing import Literal

from worldfoundry.core.io.integrity import canonical_sha256


@dataclass(frozen=True, slots=True)
class RCMConfig:
    """Every field directly controls the rCM execution path."""

    consistency_mode: Literal["continuous", "discrete"] = "continuous"
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

    def __post_init__(self) -> None:
        if self.consistency_mode not in {"continuous", "discrete"}:
            raise ValueError("consistency_mode must be continuous or discrete")
        for name, value, allow_zero in (
            ("tangent_warmup_steps", self.tangent_warmup_steps, True),
            ("student_update_frequency", self.student_update_frequency, False),
            ("max_rollout_steps", self.max_rollout_steps, False),
            ("dcm_total_steps", self.dcm_total_steps, False),
            ("dcm_skipping_interval_steps", self.dcm_skipping_interval_steps, False),
        ):
            if isinstance(value, bool) or int(value) != value or int(value) < (0 if allow_zero else 1):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be a {qualifier} integer")
        if self.dcm_skipping_interval_steps >= self.dcm_total_steps:
            raise ValueError("dcm_skipping_interval_steps must be smaller than dcm_total_steps")
        for name, value in (
            ("teacher_guidance_scale", self.teacher_guidance_scale),
            ("consistency_loss_scale", self.consistency_loss_scale),
            ("dmd_loss_scale", self.dmd_loss_scale),
            ("generator_time_mean", self.generator_time_mean),
            ("generator_time_std", self.generator_time_std),
            ("score_time_mean", self.score_time_mean),
            ("score_time_std", self.score_time_std),
            ("tangent_normalization_constant", self.tangent_normalization_constant),
            ("dcm_timestep_shift", self.dcm_timestep_shift),
        ):
            if not isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.consistency_loss_scale < 0 or self.dmd_loss_scale < 0:
            raise ValueError("loss scales must be non-negative")
        if self.consistency_loss_scale == 0 and self.dmd_loss_scale == 0:
            raise ValueError("at least one rCM loss must be enabled")
        if self.consistency_loss_scale == 0 and self.tangent_warmup_steps != 0:
            raise ValueError("tangent_warmup_steps must be zero when consistency loss is disabled")
        if self.generator_time_std <= 0 or self.score_time_std <= 0:
            raise ValueError("time distribution standard deviations must be positive")
        if self.tangent_normalization_constant <= 0:
            raise ValueError("tangent_normalization_constant must be positive")
        if self.dcm_timestep_shift <= 0:
            raise ValueError("dcm_timestep_shift must be positive")
        rollout = tuple(float(value) for value in self.fixed_rollout_timesteps)
        if rollout and len(rollout) != self.max_rollout_steps - 1:
            raise ValueError(
                "fixed_rollout_timesteps must be empty or cover every multi-step rollout"
            )
        if any(not isfinite(value) or not 0 < value < pi / 2 for value in rollout):
            raise ValueError("fixed rollout times must be finite and in (0,pi/2)")
        if any(left <= right for left, right in zip(rollout, rollout[1:])):
            raise ValueError("fixed rollout times must be strictly descending")
        object.__setattr__(self, "fixed_rollout_timesteps", rollout)

    @property
    def dmd_enabled(self) -> bool:
        return self.dmd_loss_scale > 0

    @property
    def digest(self) -> str:
        return canonical_sha256({"schema": "worldfoundry-rcm-config", **asdict(self)})


__all__ = ["RCMConfig"]
