"""Behavior-bearing schedules and hyperparameters for ADD training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Real
from typing import Literal

from worldfoundry.core.io.integrity import canonical_sha256
from worldfoundry.training.recipes.post_training.algorithms.adversarial_diffusion import (
    AdversarialDiffusionAlgorithmSpec,
)

ADDDistillationWeighting = Literal["exponential", "sds"]


def _finite_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number")
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    return resolved


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class ADDNoiseSchedule:
    """A discrete variance-preserving schedule consumed by ADD objectives.

    ``alpha_cumprods`` contains the signal power at every addressable
    timestep.  A score-distillation run additionally supplies the teacher's
    weighted-diffusion loss weights; exponential weighting rejects them so a
    configured value can never become an inert field.
    """

    alpha_cumprods: tuple[float, ...]
    training_loss_weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        values = tuple(_finite_float(value, field_name="alpha_cumprods value") for value in self.alpha_cumprods)
        if len(values) < 2:
            raise ValueError("alpha_cumprods must contain at least two timesteps")
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("alpha_cumprods must be finite and lie in [0,1]")
        if any(left < right for left, right in zip(values, values[1:], strict=False)):
            raise ValueError("alpha_cumprods must be non-increasing")
        if values[0] == values[-1]:
            raise ValueError("alpha_cumprods must describe a non-degenerate process")
        weights = self.training_loss_weights
        if weights is not None:
            resolved = tuple(_finite_float(value, field_name="training_loss_weights value") for value in weights)
            if len(resolved) != len(values):
                raise ValueError("training_loss_weights must align with alpha_cumprods")
            if any(value <= 0.0 for value in resolved):
                raise ValueError("training_loss_weights must be finite and positive")
            weights = resolved
        object.__setattr__(self, "alpha_cumprods", values)
        object.__setattr__(self, "training_loss_weights", weights)

    @property
    def num_timesteps(self) -> int:
        return len(self.alpha_cumprods)

    @property
    def digest(self) -> str:
        return canonical_sha256(
            {
                "kind": "add-noise-schedule",
                "alpha_cumprods": self.alpha_cumprods,
                "training_loss_weights": self.training_loss_weights,
            }
        )


@dataclass(frozen=True, slots=True)
class ADDConfig:
    """The complete model-neutral ADD objective and update cadence."""

    student_timesteps: tuple[int, ...]
    teacher_timestep_min: int
    teacher_timestep_max: int
    feature_resolutions: tuple[int, ...]
    feature_layers: tuple[str, ...]
    discriminator_conditioning_keys: tuple[str, ...]
    teacher_timestep_probabilities: tuple[float, ...] | None = None
    distillation_weight: float = 2.5
    distillation_weighting: ADDDistillationWeighting = "exponential"
    r1_weight: float = 1.0e-5
    discriminator_updates_per_generator: int = 1

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, int) for value in self.student_timesteps):
            raise TypeError("student_timesteps must contain integers")
        timesteps = tuple(self.student_timesteps)
        if len(timesteps) != 4:
            raise ValueError("paper-faithful ADD training requires exactly four student timesteps")
        if any(value < 0 for value in timesteps):
            raise ValueError("student_timesteps must be non-negative")
        if any(left >= right for left, right in zip(timesteps, timesteps[1:], strict=False)):
            raise ValueError("student_timesteps must be strictly increasing")
        if (
            isinstance(self.teacher_timestep_min, bool)
            or not isinstance(self.teacher_timestep_min, int)
            or isinstance(self.teacher_timestep_max, bool)
            or not isinstance(self.teacher_timestep_max, int)
        ):
            raise TypeError("teacher timestep bounds must be integers")
        teacher_min = self.teacher_timestep_min
        teacher_max = self.teacher_timestep_max
        if teacher_min < 0 or teacher_max < teacher_min:
            raise ValueError("teacher timestep bounds must satisfy 0 <= min <= max")
        probabilities = self.teacher_timestep_probabilities
        if probabilities is not None:
            resolved_probabilities = tuple(
                _finite_float(value, field_name="teacher_timestep_probabilities value") for value in probabilities
            )
            if len(resolved_probabilities) != teacher_max - teacher_min + 1:
                raise ValueError("teacher_timestep_probabilities must align with the inclusive teacher range")
            if any(value < 0.0 for value in resolved_probabilities):
                raise ValueError("teacher_timestep_probabilities must be finite and non-negative")
            total_probability = sum(resolved_probabilities)
            if total_probability <= 0.0:
                raise ValueError("teacher_timestep_probabilities must contain positive mass")
            probabilities = tuple(value / total_probability for value in resolved_probabilities)
        resolutions = tuple(_positive_int(value, field_name="feature resolution") for value in self.feature_resolutions)
        if not resolutions or len(set(resolutions)) != len(resolutions):
            raise ValueError("feature_resolutions must be non-empty and unique")
        if any(not isinstance(value, str) for value in self.feature_layers):
            raise TypeError("feature_layers must contain strings")
        layers = tuple(value.strip() for value in self.feature_layers)
        if not layers or any(not value for value in layers) or len(set(layers)) != len(layers):
            raise ValueError("feature_layers must contain unique non-empty module paths")
        if any(not isinstance(value, str) for value in self.discriminator_conditioning_keys):
            raise TypeError("discriminator_conditioning_keys must contain strings")
        conditioning_keys = tuple(value.strip() for value in self.discriminator_conditioning_keys)
        if any(not value for value in conditioning_keys) or len(set(conditioning_keys)) != len(conditioning_keys):
            raise ValueError("discriminator_conditioning_keys must be unique non-empty strings")
        distillation_weight = _finite_float(
            self.distillation_weight,
            field_name="distillation_weight",
        )
        r1_weight = _finite_float(self.r1_weight, field_name="r1_weight")
        if distillation_weight <= 0.0:
            raise ValueError("distillation_weight must be finite and positive")
        if self.distillation_weighting not in {"exponential", "sds"}:
            raise ValueError("distillation_weighting must be 'exponential' or 'sds'")
        if r1_weight < 0.0:
            raise ValueError("r1_weight must be finite and non-negative")
        discriminator_updates = _positive_int(
            self.discriminator_updates_per_generator,
            field_name="discriminator_updates_per_generator",
        )
        object.__setattr__(self, "student_timesteps", timesteps)
        object.__setattr__(self, "teacher_timestep_min", teacher_min)
        object.__setattr__(self, "teacher_timestep_max", teacher_max)
        object.__setattr__(self, "teacher_timestep_probabilities", probabilities)
        object.__setattr__(self, "feature_resolutions", resolutions)
        object.__setattr__(self, "feature_layers", layers)
        object.__setattr__(self, "discriminator_conditioning_keys", conditioning_keys)
        object.__setattr__(self, "distillation_weight", distillation_weight)
        object.__setattr__(self, "r1_weight", r1_weight)
        object.__setattr__(self, "discriminator_updates_per_generator", discriminator_updates)

    @classmethod
    def from_recipe(cls, spec: AdversarialDiffusionAlgorithmSpec) -> ADDConfig:
        """Materialize the executable objective from its strict recipe section."""

        if not isinstance(spec, AdversarialDiffusionAlgorithmSpec):
            raise TypeError("spec must be AdversarialDiffusionAlgorithmSpec")
        return cls(
            student_timesteps=spec.student_timesteps,
            teacher_timestep_min=spec.teacher_timestep_min,
            teacher_timestep_max=spec.teacher_timestep_max,
            teacher_timestep_probabilities=spec.teacher_timestep_probabilities,
            feature_resolutions=spec.feature_resolutions,
            feature_layers=spec.feature_layers,
            discriminator_conditioning_keys=(spec.discriminator_conditioning_keys),
            distillation_weight=spec.distillation_weight,
            distillation_weighting=spec.distillation_weighting,
            r1_weight=spec.r1_weight,
            discriminator_updates_per_generator=(spec.discriminator_updates_per_generator),
        )

    @property
    def feature_keys(self) -> tuple[tuple[int, str], ...]:
        return tuple((resolution, layer) for resolution in self.feature_resolutions for layer in self.feature_layers)

    @property
    def digest(self) -> str:
        return canonical_sha256({"kind": "adversarial-diffusion-distillation", **asdict(self)})


def add_execution_digest(
    config: ADDConfig,
    student_schedule: ADDNoiseSchedule,
    teacher_schedule: ADDNoiseSchedule,
) -> str:
    """Bind every behavior-bearing objective schedule into resume identity."""

    if not isinstance(config, ADDConfig):
        raise TypeError("config must be ADDConfig")
    if not isinstance(student_schedule, ADDNoiseSchedule):
        raise TypeError("student_schedule must be ADDNoiseSchedule")
    if not isinstance(teacher_schedule, ADDNoiseSchedule):
        raise TypeError("teacher_schedule must be ADDNoiseSchedule")
    return canonical_sha256(
        {
            "config": config.digest,
            "student_schedule": student_schedule.digest,
            "teacher_schedule": teacher_schedule.digest,
        }
    )


__all__ = [
    "ADDConfig",
    "ADDDistillationWeighting",
    "ADDNoiseSchedule",
    "add_execution_digest",
]
