"""Pure recipe contract for distribution-matching distillation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..common import positive_int, strict_mapping
from .auxiliary_optimizers import (
    AuxiliaryOptimizerRule,
    forbids_auxiliary,
    requires_auxiliary,
)

DMD_ALGORITHM_FIELDS = {
    "type",
    "student_timesteps",
    "student_sigmas",
    "real_score_checkpoint",
    "fake_score_checkpoint",
    "num_train_timesteps",
    "score_min_sigma",
    "score_max_sigma",
    "score_flow_shift",
    "teacher_guidance_scale",
    "generator_update_interval",
    "student_scheduler_cadence",
    "normalization_epsilon",
    "shared_score_timestep",
}


def _normalize_few_step_schedule(
    timesteps: tuple[float, ...],
    sigmas: tuple[float, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate schedule data without importing the execution plane."""

    resolved_timesteps = tuple(float(value) for value in timesteps)
    resolved_sigmas = tuple(float(value) for value in sigmas)
    if not resolved_timesteps or len(resolved_timesteps) != len(resolved_sigmas):
        raise ValueError("few-step timesteps and sigmas must be non-empty and equal length")
    if any(not isfinite(value) or value < 0 for value in resolved_timesteps):
        raise ValueError("few-step timesteps must be finite and non-negative")
    if any(not isfinite(value) or not 0 < value <= 1 for value in resolved_sigmas):
        raise ValueError("few-step sigmas must be finite and in (0,1]")
    if any(left <= right for left, right in zip(resolved_timesteps, resolved_timesteps[1:])):
        raise ValueError("few-step timesteps must be strictly descending")
    if any(left <= right for left, right in zip(resolved_sigmas, resolved_sigmas[1:])):
        raise ValueError("few-step sigmas must be strictly descending")
    return resolved_timesteps, resolved_sigmas


@dataclass(frozen=True, slots=True)
class DMDAlgorithmSpec:
    """Three-role, two-optimizer few-step DMD configuration."""

    student_timesteps: tuple[float, ...]
    student_sigmas: tuple[float, ...]
    real_score_checkpoint: str
    fake_score_checkpoint: str
    num_train_timesteps: int = 1000
    score_min_sigma: float = 0.02
    score_max_sigma: float = 0.98
    score_flow_shift: float = 1.0
    teacher_guidance_scale: float = 3.5
    generator_update_interval: int = 5
    student_scheduler_cadence: str = "iteration"
    normalization_epsilon: float = 0.0
    shared_score_timestep: bool = True
    type: str = "dmd"

    def __post_init__(self) -> None:
        if str(self.type).lower().replace("_", "-") != "dmd":
            raise ValueError("DMD algorithm type must be 'dmd'")
        timesteps, sigmas = _normalize_few_step_schedule(
            self.student_timesteps,
            self.student_sigmas,
        )
        for name, value in (
            ("real_score_checkpoint", self.real_score_checkpoint),
            ("fake_score_checkpoint", self.fake_score_checkpoint),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty checkpoint reference")
        if self.student_scheduler_cadence not in {"iteration", "generator-update"}:
            raise ValueError("student_scheduler_cadence must be 'iteration' or 'generator-update'")
        if not isinstance(self.shared_score_timestep, bool):
            raise TypeError("shared_score_timestep must be a bool")
        if not 0 <= float(self.score_min_sigma) < float(self.score_max_sigma) <= 1:
            raise ValueError("DMD score sigma bounds must satisfy 0 <= min < max <= 1")
        for name, value in (
            ("score_flow_shift", self.score_flow_shift),
            ("teacher_guidance_scale", self.teacher_guidance_scale),
            ("normalization_epsilon", self.normalization_epsilon),
        ):
            if not isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if float(self.score_flow_shift) <= 0 or float(self.normalization_epsilon) < 0:
            raise ValueError("score_flow_shift must be positive and normalization_epsilon non-negative")
        object.__setattr__(self, "student_timesteps", timesteps)
        object.__setattr__(self, "student_sigmas", sigmas)
        object.__setattr__(
            self,
            "num_train_timesteps",
            positive_int(
                self.num_train_timesteps,
                field_name="algorithm.num_train_timesteps",
            ),
        )
        object.__setattr__(
            self,
            "generator_update_interval",
            positive_int(
                self.generator_update_interval,
                field_name="algorithm.generator_update_interval",
            ),
        )

    def auxiliary_optimizer_rules(self) -> tuple[AuxiliaryOptimizerRule, ...]:
        return (
            requires_auxiliary("fake_score_optimizer", "DMD requires fake_score_optimizer"),
            forbids_auxiliary(
                "guidance_optimizer",
                "discriminator_optimizer",
                message=f"{self.type} only accepts fake_score_optimizer",
            ),
        )


def parse_dmd_algorithm(value: object) -> DMDAlgorithmSpec:
    """Parse a strict DMD algorithm section."""

    return DMDAlgorithmSpec(
        **strict_mapping(
            value,
            field_name="algorithm",
            allowed=DMD_ALGORITHM_FIELDS,
        )
    )


__all__ = ["DMDAlgorithmSpec"]
