"""Behavior-bearing configuration for native SGMD."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

from worldfoundry.core.io.integrity import canonical_sha256
from worldfoundry.training.recipes.post_training.algorithms.sgmd import (
    SGMDAlgorithmSpec,
)


def _positive_float(value: object, *, field_name: str) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return resolved


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


def shifted_flow_sigma(value: float, shift: float) -> float:
    """Apply the rational flow-time shift used by the SGMD schedulers."""

    sigma = float(value)
    resolved_shift = _positive_float(shift, field_name="flow shift")
    if not isfinite(sigma) or not 0.0 <= sigma <= 1.0:
        raise ValueError("flow sigma must be finite and in [0,1]")
    return resolved_shift * sigma / (1.0 + (resolved_shift - 1.0) * sigma)


@dataclass(frozen=True, slots=True)
class SGMDConfig:
    """SGMD math and rollout choices, defaulted to the released Wan profile."""

    student_timesteps: tuple[float, ...] = (1000.0, 750.0, 500.0, 250.0)
    num_train_timesteps: int = 1000
    student_flow_shift: float = 5.0
    score_min_sigma: float = 0.02
    score_max_sigma: float = 0.98
    score_discrete_samples: int = 1000
    score_flow_shift: float = 5.0
    teacher_guidance_scale: float = 3.0
    fake_correction_weight: float = 0.1
    numerical_epsilon: float = 1.0e-8
    diversity_enabled: bool = True
    diversity_weight: float = 0.05
    diversity_teacher_steps: int = 30
    diversity_anchor_step: int = 5
    diversity_teacher_flow_shift: float = 5.0

    def __post_init__(self) -> None:
        train_steps = _positive_int(self.num_train_timesteps, field_name="num_train_timesteps")
        timesteps = tuple(float(value) for value in self.student_timesteps)
        if not timesteps or any(not isfinite(value) for value in timesteps):
            raise ValueError("student_timesteps must contain finite values")
        if any(not 0.0 < value <= train_steps for value in timesteps):
            raise ValueError("student_timesteps must be in (0,num_train_timesteps]")
        if any(left <= right for left, right in zip(timesteps, timesteps[1:])):
            raise ValueError("student_timesteps must be strictly descending")

        student_shift = _positive_float(self.student_flow_shift, field_name="student_flow_shift")
        score_shift = _positive_float(self.score_flow_shift, field_name="score_flow_shift")
        teacher_shift = _positive_float(
            self.diversity_teacher_flow_shift,
            field_name="diversity_teacher_flow_shift",
        )
        minimum = float(self.score_min_sigma)
        maximum = float(self.score_max_sigma)
        if not isfinite(minimum) or not isfinite(maximum) or not 0.0 <= minimum < maximum <= 1.0:
            raise ValueError("score sigma bounds must satisfy 0 <= min < max <= 1")
        discrete = _positive_int(
            self.score_discrete_samples,
            field_name="score_discrete_samples",
        )
        guidance = float(self.teacher_guidance_scale)
        correction = float(self.fake_correction_weight)
        epsilon = _positive_float(self.numerical_epsilon, field_name="numerical_epsilon")
        diversity_weight = float(self.diversity_weight)
        if not isfinite(guidance) or guidance < 0:
            raise ValueError("teacher_guidance_scale must be finite and non-negative")
        if not isfinite(correction) or correction < 0:
            raise ValueError("fake_correction_weight must be finite and non-negative")
        if not isinstance(self.diversity_enabled, bool):
            raise TypeError("diversity_enabled must be bool")
        if not isfinite(diversity_weight) or diversity_weight < 0:
            raise ValueError("diversity_weight must be finite and non-negative")
        teacher_steps = _positive_int(
            self.diversity_teacher_steps,
            field_name="diversity_teacher_steps",
        )
        anchor_step = _positive_int(
            self.diversity_anchor_step,
            field_name="diversity_anchor_step",
        )
        if anchor_step > teacher_steps:
            raise ValueError("diversity_anchor_step cannot exceed diversity_teacher_steps")
        if self.diversity_enabled and len(timesteps) < 2:
            raise ValueError("diversity requires at least two student denoising steps")
        if self.diversity_enabled and diversity_weight == 0:
            raise ValueError("enabled diversity must have positive weight")

        object.__setattr__(self, "student_timesteps", timesteps)
        object.__setattr__(self, "num_train_timesteps", train_steps)
        object.__setattr__(self, "student_flow_shift", student_shift)
        object.__setattr__(self, "score_min_sigma", minimum)
        object.__setattr__(self, "score_max_sigma", maximum)
        object.__setattr__(self, "score_discrete_samples", discrete)
        object.__setattr__(self, "score_flow_shift", score_shift)
        object.__setattr__(self, "teacher_guidance_scale", guidance)
        object.__setattr__(self, "fake_correction_weight", correction)
        object.__setattr__(self, "numerical_epsilon", epsilon)
        object.__setattr__(self, "diversity_weight", diversity_weight)
        object.__setattr__(self, "diversity_teacher_steps", teacher_steps)
        object.__setattr__(self, "diversity_anchor_step", anchor_step)
        object.__setattr__(self, "diversity_teacher_flow_shift", teacher_shift)

    @property
    def student_sigmas(self) -> tuple[float, ...]:
        return tuple(
            shifted_flow_sigma(value / self.num_train_timesteps, self.student_flow_shift)
            for value in self.student_timesteps
        )

    @property
    def teacher_sigmas(self) -> tuple[float, ...]:
        steps = self.diversity_teacher_steps
        return tuple(
            shifted_flow_sigma(1.0 - index / steps, self.diversity_teacher_flow_shift)
            for index in range(steps + 1)
        )

    @property
    def minimum_student_target_index(self) -> int:
        return int(self.diversity_enabled)

    @property
    def digest(self) -> str:
        return canonical_sha256({"schema": "worldfoundry-sgmd-config", **asdict(self)})

    @classmethod
    def from_recipe(cls, spec: SGMDAlgorithmSpec) -> SGMDConfig:
        if not isinstance(spec, SGMDAlgorithmSpec):
            raise TypeError("spec must be SGMDAlgorithmSpec")
        return cls(
            student_timesteps=spec.student_timesteps,
            num_train_timesteps=spec.num_train_timesteps,
            student_flow_shift=spec.student_flow_shift,
            score_min_sigma=spec.score_min_sigma,
            score_max_sigma=spec.score_max_sigma,
            score_discrete_samples=spec.score_discrete_samples,
            score_flow_shift=spec.score_flow_shift,
            teacher_guidance_scale=spec.teacher_guidance_scale,
            fake_correction_weight=spec.fake_correction_weight,
            numerical_epsilon=spec.numerical_epsilon,
            diversity_enabled=True,
            diversity_weight=spec.diversity_weight,
            diversity_teacher_steps=spec.diversity_teacher_steps,
            diversity_anchor_step=spec.diversity_anchor_step,
            diversity_teacher_flow_shift=spec.diversity_teacher_flow_shift,
        )


__all__ = [
    "SGMDConfig",
    "shifted_flow_sigma",
]
