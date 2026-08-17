"""Behavior-bearing configuration for Data-Forcing Distillation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

from worldfoundry.training.recipes.post_training.algorithms.dfd import (
    DFDAlgorithmSpec,
)


def _positive_float(value: object, *, field_name: str) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return resolved


def _non_negative_float(value: object, *, field_name: str) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return resolved


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class DFDConfig:
    """Released Wan DFD behavior, including its inherited DMD2 GAN branch."""

    student_timesteps: tuple[float, ...] = (0.999, 0.937, 0.833, 0.624, 0.0)
    score_min_timestep: float = 0.001
    score_max_timestep: float = 0.999
    score_timestep_shift: float = 5.0
    teacher_guidance_scale: float = 5.0
    data_forcing_probability: float = 0.5
    student_update_frequency: int = 5
    normalization_epsilon: float = 1.0e-6
    distribution_matching_weight: float = 1.0
    generator_adversarial_weight: float = 0.03
    fake_score_denoising_weight: float = 1.0
    discriminator_weight: float = 1.0

    def __post_init__(self) -> None:
        timesteps = tuple(float(value) for value in self.student_timesteps)
        if len(timesteps) < 2 or any(not isfinite(value) for value in timesteps):
            raise ValueError("student_timesteps must contain at least one step followed by zero")
        if timesteps[-1] != 0.0:
            raise ValueError("student_timesteps must end at zero")
        if any(not 0.0 < value <= 1.0 for value in timesteps[:-1]):
            raise ValueError("non-terminal student timesteps must be in (0,1]")
        if any(left <= right for left, right in zip(timesteps, timesteps[1:])):
            raise ValueError("student_timesteps must be strictly descending")
        minimum = float(self.score_min_timestep)
        maximum = float(self.score_max_timestep)
        if not isfinite(minimum) or not isfinite(maximum) or not 0.0 <= minimum < maximum <= 1.0:
            raise ValueError("score timestep bounds must satisfy 0 <= min < max <= 1")
        shift = _positive_float(self.score_timestep_shift, field_name="score_timestep_shift")
        if shift < 1.0:
            raise ValueError("score_timestep_shift must be at least one")
        guidance = float(self.teacher_guidance_scale)
        if not isfinite(guidance):
            raise ValueError("teacher_guidance_scale must be finite")
        probability = float(self.data_forcing_probability)
        if not isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("data_forcing_probability must be in [0,1]")
        frequency = _positive_int(
            self.student_update_frequency,
            field_name="student_update_frequency",
        )
        epsilon = _positive_float(
            self.normalization_epsilon,
            field_name="normalization_epsilon",
        )
        dm_weight = _non_negative_float(
            self.distribution_matching_weight,
            field_name="distribution_matching_weight",
        )
        gan_weight = _non_negative_float(
            self.generator_adversarial_weight,
            field_name="generator_adversarial_weight",
        )
        fake_weight = _non_negative_float(
            self.fake_score_denoising_weight,
            field_name="fake_score_denoising_weight",
        )
        discriminator_weight = _non_negative_float(
            self.discriminator_weight,
            field_name="discriminator_weight",
        )
        if dm_weight <= 0:
            raise ValueError("DFD distribution matching must remain enabled")
        if fake_weight <= 0:
            raise ValueError("DFD fake-score denoising must remain enabled")
        if (gan_weight == 0.0) != (discriminator_weight == 0.0):
            raise ValueError(
                "generator_adversarial_weight and discriminator_weight must be enabled together"
            )
        object.__setattr__(self, "student_timesteps", timesteps)
        object.__setattr__(self, "score_min_timestep", minimum)
        object.__setattr__(self, "score_max_timestep", maximum)
        object.__setattr__(self, "score_timestep_shift", shift)
        object.__setattr__(self, "teacher_guidance_scale", guidance)
        object.__setattr__(self, "data_forcing_probability", probability)
        object.__setattr__(self, "student_update_frequency", frequency)
        object.__setattr__(self, "normalization_epsilon", epsilon)
        object.__setattr__(self, "distribution_matching_weight", dm_weight)
        object.__setattr__(self, "generator_adversarial_weight", gan_weight)
        object.__setattr__(self, "fake_score_denoising_weight", fake_weight)
        object.__setattr__(self, "discriminator_weight", discriminator_weight)

    @property
    def trainable_student_timesteps(self) -> tuple[float, ...]:
        return self.student_timesteps[:-1]

    @property
    def adversarial_enabled(self) -> bool:
        return self.generator_adversarial_weight > 0.0

    @classmethod
    def from_recipe(cls, algorithm: DFDAlgorithmSpec) -> DFDConfig:
        if not isinstance(algorithm, DFDAlgorithmSpec):
            raise TypeError("algorithm must be DFDAlgorithmSpec")
        values = asdict(algorithm)
        if values.pop("type") != "dfd":
            raise RuntimeError("validated DFD dispatch tag changed unexpectedly")
        values.pop("teacher_checkpoint")
        values.pop("fake_score_checkpoint")
        values.pop("discriminator_checkpoint")
        return cls(**values)


__all__ = ["DFDConfig"]
