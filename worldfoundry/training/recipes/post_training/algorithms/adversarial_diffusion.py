"""Strict behavior contract for Adversarial Diffusion Distillation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real

from ..common import strict_mapping


def _checkpoint(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty checkpoint reference")
    return value.strip()


def _finite_real(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number")
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    return resolved


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _alpha_cumprods(value: object, *, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{field_name} must be a sequence")
    resolved = tuple(_finite_real(item, field_name=f"{field_name} value") for item in value)
    if len(resolved) < 2:
        raise ValueError(f"{field_name} must contain at least two timesteps")
    if any(not 0.0 <= item <= 1.0 for item in resolved):
        raise ValueError(f"{field_name} values must lie in [0,1]")
    if any(left < right for left, right in zip(resolved, resolved[1:])):
        raise ValueError(f"{field_name} must be non-increasing")
    if resolved[0] == resolved[-1]:
        raise ValueError(f"{field_name} must describe a non-degenerate process")
    return resolved


@dataclass(frozen=True, slots=True)
class AdversarialDiffusionAlgorithmSpec:
    """Every model, schedule, objective, and update choice executed by ADD."""

    teacher_checkpoint: str
    decoder_checkpoint: str
    feature_checkpoint: str
    student_alpha_cumprods: tuple[float, ...]
    teacher_alpha_cumprods: tuple[float, ...]
    student_timesteps: tuple[int, ...]
    teacher_timestep_min: int
    teacher_timestep_max: int
    feature_resolutions: tuple[int, ...]
    feature_layers: tuple[str, ...]
    discriminator_conditioning_keys: tuple[str, ...] = ()
    teacher_training_loss_weights: tuple[float, ...] | None = None
    teacher_timestep_probabilities: tuple[float, ...] | None = None
    distillation_weight: float = 2.5
    distillation_weighting: str = "exponential"
    r1_weight: float = 1.0e-5
    discriminator_updates_per_generator: int = 1
    type: str = "adversarial-diffusion-distillation"

    def __post_init__(self) -> None:
        algorithm_type = str(self.type).strip().lower().replace("_", "-")
        if algorithm_type != "adversarial-diffusion-distillation":
            raise ValueError("ADD algorithm type must be 'adversarial-diffusion-distillation'")
        checkpoints = {
            name: _checkpoint(getattr(self, name), field_name=name)
            for name in (
                "teacher_checkpoint",
                "decoder_checkpoint",
                "feature_checkpoint",
            )
        }
        student_alphas = _alpha_cumprods(
            self.student_alpha_cumprods,
            field_name="student_alpha_cumprods",
        )
        teacher_alphas = _alpha_cumprods(
            self.teacher_alpha_cumprods,
            field_name="teacher_alpha_cumprods",
        )
        if not isinstance(self.student_timesteps, (tuple, list)):
            raise TypeError("student_timesteps must be a sequence")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in self.student_timesteps):
            raise TypeError("student_timesteps must contain integers")
        student_timesteps = tuple(self.student_timesteps)
        if len(student_timesteps) != 4:
            raise ValueError("paper-faithful ADD training requires exactly four student timesteps")
        if any(value < 0 for value in student_timesteps) or any(
            left >= right for left, right in zip(student_timesteps, student_timesteps[1:])
        ):
            raise ValueError("student_timesteps must be non-negative and strictly increasing")
        if student_timesteps[-1] != len(student_alphas) - 1:
            raise ValueError("the final student timestep must address the student schedule terminal")
        if student_alphas[-1] != 0.0:
            raise ValueError("the final ADD student timestep must have exactly zero terminal SNR")
        if (
            isinstance(self.teacher_timestep_min, bool)
            or not isinstance(self.teacher_timestep_min, int)
            or isinstance(self.teacher_timestep_max, bool)
            or not isinstance(self.teacher_timestep_max, int)
        ):
            raise TypeError("teacher timestep bounds must be integers")
        teacher_min = self.teacher_timestep_min
        teacher_max = self.teacher_timestep_max
        if not 0 <= teacher_min <= teacher_max < len(teacher_alphas):
            raise ValueError("teacher timestep bounds must lie inside teacher_alpha_cumprods")
        if any(not 0.0 < value < 1.0 for value in teacher_alphas[teacher_min : teacher_max + 1]):
            raise ValueError("the selected teacher range requires non-zero signal and noise power")
        teacher_weights = self.teacher_training_loss_weights
        if teacher_weights is not None:
            if not isinstance(teacher_weights, (tuple, list)):
                raise TypeError("teacher_training_loss_weights must be a sequence")
            teacher_weights = tuple(
                _finite_real(
                    value,
                    field_name="teacher_training_loss_weights value",
                )
                for value in teacher_weights
            )
            if len(teacher_weights) != len(teacher_alphas):
                raise ValueError("teacher_training_loss_weights must align with teacher_alpha_cumprods")
            if any(value <= 0.0 for value in teacher_weights):
                raise ValueError("teacher_training_loss_weights values must be positive")
        teacher_probabilities = self.teacher_timestep_probabilities
        if teacher_probabilities is not None:
            if not isinstance(teacher_probabilities, (tuple, list)):
                raise TypeError("teacher_timestep_probabilities must be a sequence")
            teacher_probabilities = tuple(
                _finite_real(
                    value,
                    field_name="teacher_timestep_probabilities value",
                )
                for value in teacher_probabilities
            )
            if len(teacher_probabilities) != teacher_max - teacher_min + 1:
                raise ValueError("teacher_timestep_probabilities must align with the inclusive teacher range")
            if any(value < 0.0 for value in teacher_probabilities):
                raise ValueError("teacher_timestep_probabilities values must be non-negative")
            probability_mass = sum(teacher_probabilities)
            if probability_mass <= 0.0:
                raise ValueError("teacher_timestep_probabilities must contain positive mass")
            teacher_probabilities = tuple(value / probability_mass for value in teacher_probabilities)
        if not isinstance(self.feature_resolutions, (tuple, list)):
            raise TypeError("feature_resolutions must be a sequence")
        feature_resolutions = tuple(
            _positive_int(value, field_name="feature_resolutions value") for value in self.feature_resolutions
        )
        if not feature_resolutions or len(set(feature_resolutions)) != len(feature_resolutions):
            raise ValueError("feature_resolutions must be non-empty and unique")
        if not isinstance(self.feature_layers, (tuple, list)) or any(
            not isinstance(value, str) for value in self.feature_layers
        ):
            raise TypeError("feature_layers must be a string sequence")
        feature_layers = tuple(value.strip() for value in self.feature_layers)
        if (
            not feature_layers
            or any(not value for value in feature_layers)
            or len(set(feature_layers)) != len(feature_layers)
        ):
            raise ValueError("feature_layers must contain unique non-empty module paths")
        if not isinstance(self.discriminator_conditioning_keys, (tuple, list)) or any(
            not isinstance(value, str) for value in self.discriminator_conditioning_keys
        ):
            raise TypeError("discriminator_conditioning_keys must be a string sequence")
        conditioning_keys = tuple(value.strip() for value in self.discriminator_conditioning_keys)
        if any(not value for value in conditioning_keys) or len(set(conditioning_keys)) != len(conditioning_keys):
            raise ValueError("discriminator_conditioning_keys must be unique non-empty strings")
        distillation_weight = _finite_real(
            self.distillation_weight,
            field_name="distillation_weight",
        )
        if distillation_weight <= 0.0:
            raise ValueError("distillation_weight must be positive")
        weighting = str(self.distillation_weighting).strip().lower().replace("-", "_")
        if weighting not in {"exponential", "sds"}:
            raise ValueError("distillation_weighting must be 'exponential' or 'sds'")
        if weighting == "exponential" and teacher_weights is not None:
            raise ValueError("teacher_training_loss_weights are unused by exponential ADD")
        if weighting == "sds" and teacher_weights is None:
            raise ValueError("SDS ADD requires teacher_training_loss_weights")
        r1_weight = _finite_real(self.r1_weight, field_name="r1_weight")
        if r1_weight < 0.0:
            raise ValueError("r1_weight must be non-negative")
        discriminator_updates = _positive_int(
            self.discriminator_updates_per_generator,
            field_name="discriminator_updates_per_generator",
        )

        object.__setattr__(self, "type", algorithm_type)
        for name, value in checkpoints.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "student_alpha_cumprods", student_alphas)
        object.__setattr__(self, "teacher_alpha_cumprods", teacher_alphas)
        object.__setattr__(self, "student_timesteps", student_timesteps)
        object.__setattr__(self, "teacher_timestep_min", teacher_min)
        object.__setattr__(self, "teacher_timestep_max", teacher_max)
        object.__setattr__(
            self,
            "teacher_training_loss_weights",
            teacher_weights,
        )
        object.__setattr__(
            self,
            "teacher_timestep_probabilities",
            teacher_probabilities,
        )
        object.__setattr__(self, "feature_resolutions", feature_resolutions)
        object.__setattr__(self, "feature_layers", feature_layers)
        object.__setattr__(
            self,
            "discriminator_conditioning_keys",
            conditioning_keys,
        )
        object.__setattr__(self, "distillation_weight", distillation_weight)
        object.__setattr__(self, "distillation_weighting", weighting)
        object.__setattr__(self, "r1_weight", r1_weight)
        object.__setattr__(
            self,
            "discriminator_updates_per_generator",
            discriminator_updates,
        )


ADVERSARIAL_DIFFUSION_ALGORITHM_FIELDS = frozenset(AdversarialDiffusionAlgorithmSpec.__dataclass_fields__)


def parse_adversarial_diffusion_algorithm(
    value: object,
) -> AdversarialDiffusionAlgorithmSpec:
    """Parse an ADD section while rejecting undeclared fields."""

    return AdversarialDiffusionAlgorithmSpec(
        **strict_mapping(
            value,
            field_name="algorithm",
            allowed=set(ADVERSARIAL_DIFFUSION_ALGORITHM_FIELDS),
        )
    )


__all__ = [
    "AdversarialDiffusionAlgorithmSpec",
    "parse_adversarial_diffusion_algorithm",
]
