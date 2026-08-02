"""Strict recipe contract for native Self-Guided Matching Distillation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..common import positive_int, strict_mapping


def _finite_float(value: object, *, field_name: str) -> float:
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    return resolved


@dataclass(frozen=True, slots=True)
class SGMDAlgorithmSpec:
    """Every released SGMD schedule and objective choice used at runtime."""

    student_timesteps: tuple[float, ...]
    teacher_checkpoint: str
    fake_score_checkpoint: str
    num_train_timesteps: int = 1000
    student_flow_shift: float = 5.0
    score_min_sigma: float = 0.02
    score_max_sigma: float = 0.98
    score_discrete_samples: int = 1000
    score_flow_shift: float = 5.0
    teacher_guidance_scale: float = 3.0
    fake_correction_weight: float = 0.1
    numerical_epsilon: float = 1.0e-8
    diversity_weight: float = 0.05
    diversity_teacher_steps: int = 30
    diversity_anchor_step: int = 5
    diversity_teacher_flow_shift: float = 5.0
    type: str = "sgmd"

    def __post_init__(self) -> None:
        algorithm_type = str(self.type).strip().lower().replace("_", "-")
        if algorithm_type != "sgmd":
            raise ValueError("SGMD algorithm type must be 'sgmd'")
        train_steps = positive_int(
            self.num_train_timesteps,
            field_name="algorithm.num_train_timesteps",
        )
        if train_steps < 2:
            raise ValueError("algorithm.num_train_timesteps must be at least two")
        timesteps = tuple(float(value) for value in self.student_timesteps)
        if len(timesteps) < 2:
            raise ValueError("SGMD requires at least two student timesteps")
        if any(not isfinite(value) or not 0.0 < value <= train_steps for value in timesteps):
            raise ValueError("student_timesteps must lie in (0,num_train_timesteps]")
        if any(left <= right for left, right in zip(timesteps, timesteps[1:])):
            raise ValueError("student_timesteps must be strictly descending")
        for name in ("teacher_checkpoint", "fake_score_checkpoint"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty checkpoint reference")
            object.__setattr__(self, name, value.strip())
        minimum = _finite_float(self.score_min_sigma, field_name="score_min_sigma")
        maximum = _finite_float(self.score_max_sigma, field_name="score_max_sigma")
        if not 0.0 <= minimum < maximum <= 1.0:
            raise ValueError("score sigma bounds must satisfy 0 <= min < max <= 1")
        positive_values: dict[str, float] = {}
        for name in (
            "student_flow_shift",
            "score_flow_shift",
            "numerical_epsilon",
            "diversity_weight",
            "diversity_teacher_flow_shift",
        ):
            value = _finite_float(getattr(self, name), field_name=name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            positive_values[name] = value
        non_negative_values: dict[str, float] = {}
        for name in ("teacher_guidance_scale", "fake_correction_weight"):
            value = _finite_float(getattr(self, name), field_name=name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            non_negative_values[name] = value
        score_samples = positive_int(
            self.score_discrete_samples,
            field_name="algorithm.score_discrete_samples",
        )
        teacher_steps = positive_int(
            self.diversity_teacher_steps,
            field_name="algorithm.diversity_teacher_steps",
        )
        anchor_step = positive_int(
            self.diversity_anchor_step,
            field_name="algorithm.diversity_anchor_step",
        )
        if anchor_step > teacher_steps:
            raise ValueError("diversity_anchor_step cannot exceed diversity_teacher_steps")
        object.__setattr__(self, "type", algorithm_type)
        object.__setattr__(self, "student_timesteps", timesteps)
        object.__setattr__(self, "num_train_timesteps", train_steps)
        object.__setattr__(self, "score_min_sigma", minimum)
        object.__setattr__(self, "score_max_sigma", maximum)
        object.__setattr__(self, "score_discrete_samples", score_samples)
        object.__setattr__(self, "diversity_teacher_steps", teacher_steps)
        object.__setattr__(self, "diversity_anchor_step", anchor_step)
        for name, value in {**positive_values, **non_negative_values}.items():
            object.__setattr__(self, name, value)


SGMD_ALGORITHM_FIELDS = frozenset(SGMDAlgorithmSpec.__dataclass_fields__)


def parse_sgmd_algorithm(value: object) -> SGMDAlgorithmSpec:
    return SGMDAlgorithmSpec(
        **strict_mapping(
            value,
            field_name="algorithm",
            allowed=set(SGMD_ALGORITHM_FIELDS),
        )
    )


__all__ = ["SGMDAlgorithmSpec", "parse_sgmd_algorithm"]
