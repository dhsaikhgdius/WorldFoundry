"""Behavior contract for Data-Forcing Distillation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..common import strict_mapping

DFD_ALGORITHM_FIELDS = {
    "type",
    "teacher_checkpoint",
    "fake_score_checkpoint",
    "discriminator_checkpoint",
    "student_timesteps",
    "score_min_timestep",
    "score_max_timestep",
    "score_timestep_shift",
    "teacher_guidance_scale",
    "data_forcing_probability",
    "student_update_frequency",
    "normalization_epsilon",
    "distribution_matching_weight",
    "generator_adversarial_weight",
    "fake_score_denoising_weight",
    "discriminator_weight",
}


def _finite(value: object, *, field_name: str) -> float:
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    return resolved


@dataclass(frozen=True, slots=True)
class DFDAlgorithmSpec:
    """FastGen DFD loss, forcing, and optimizer-cadence choices."""

    teacher_checkpoint: str
    fake_score_checkpoint: str
    discriminator_checkpoint: str | None = None
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
    type: str = "dfd"

    def __post_init__(self) -> None:
        algorithm_type = str(self.type).strip().lower().replace("_", "-")
        if algorithm_type != "dfd":
            raise ValueError("DFD algorithm type must be 'dfd'")
        for name in ("teacher_checkpoint", "fake_score_checkpoint"):
            checkpoint = str(getattr(self, name)).strip()
            if not checkpoint:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, checkpoint)
        discriminator_checkpoint = (
            None
            if self.discriminator_checkpoint is None
            else str(self.discriminator_checkpoint).strip()
        )
        if discriminator_checkpoint == "":
            raise ValueError("discriminator_checkpoint must be non-empty when supplied")
        timesteps = tuple(float(value) for value in self.student_timesteps)
        if len(timesteps) < 2 or any(not isfinite(value) for value in timesteps):
            raise ValueError("student_timesteps must contain trainable steps and zero")
        if timesteps[-1] != 0.0:
            raise ValueError("student_timesteps must end at zero")
        if any(not 0.0 < value <= 1.0 for value in timesteps[:-1]):
            raise ValueError("non-terminal student_timesteps must be in (0,1]")
        if any(left <= right for left, right in zip(timesteps, timesteps[1:])):
            raise ValueError("student_timesteps must be strictly descending")
        minimum = _finite(self.score_min_timestep, field_name="score_min_timestep")
        maximum = _finite(self.score_max_timestep, field_name="score_max_timestep")
        if not 0.0 <= minimum < maximum <= 1.0:
            raise ValueError("score timestep bounds must satisfy 0 <= min < max <= 1")
        shift = _finite(self.score_timestep_shift, field_name="score_timestep_shift")
        if shift < 1.0:
            raise ValueError("score_timestep_shift must be at least one")
        probability = _finite(
            self.data_forcing_probability,
            field_name="data_forcing_probability",
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError("data_forcing_probability must be in [0,1]")
        if (
            isinstance(self.student_update_frequency, bool)
            or not isinstance(self.student_update_frequency, int)
            or self.student_update_frequency <= 0
        ):
            raise ValueError("student_update_frequency must be a positive integer")
        normalized = {
            name: _finite(getattr(self, name), field_name=name)
            for name in (
                "teacher_guidance_scale",
                "normalization_epsilon",
                "distribution_matching_weight",
                "generator_adversarial_weight",
                "fake_score_denoising_weight",
                "discriminator_weight",
            )
        }
        if normalized["normalization_epsilon"] <= 0:
            raise ValueError("normalization_epsilon must be positive")
        for name in (
            "distribution_matching_weight",
            "generator_adversarial_weight",
            "fake_score_denoising_weight",
            "discriminator_weight",
        ):
            if normalized[name] < 0:
                raise ValueError(f"{name} must be non-negative")
        if normalized["distribution_matching_weight"] == 0:
            raise ValueError("DFD distribution matching must remain enabled")
        if normalized["fake_score_denoising_weight"] == 0:
            raise ValueError("DFD fake-score denoising must remain enabled")
        adversarial = normalized["generator_adversarial_weight"] > 0
        if adversarial != (normalized["discriminator_weight"] > 0):
            raise ValueError("DFD generator and discriminator weights must be enabled together")
        if adversarial != (discriminator_checkpoint is not None):
            raise ValueError(
                "DFD discriminator_checkpoint must be supplied exactly when GAN loss is enabled"
            )
        object.__setattr__(self, "type", algorithm_type)
        object.__setattr__(self, "discriminator_checkpoint", discriminator_checkpoint)
        object.__setattr__(self, "student_timesteps", timesteps)
        object.__setattr__(self, "score_min_timestep", minimum)
        object.__setattr__(self, "score_max_timestep", maximum)
        object.__setattr__(self, "score_timestep_shift", shift)
        object.__setattr__(self, "data_forcing_probability", probability)
        for name, value in normalized.items():
            object.__setattr__(self, name, value)

    @property
    def adversarial_enabled(self) -> bool:
        return self.discriminator_checkpoint is not None


def parse_dfd_algorithm(value: object) -> DFDAlgorithmSpec:
    return DFDAlgorithmSpec(
        **strict_mapping(
            value,
            field_name="algorithm",
            allowed=DFD_ALGORITHM_FIELDS,
        )
    )


__all__ = ["DFDAlgorithmSpec"]
