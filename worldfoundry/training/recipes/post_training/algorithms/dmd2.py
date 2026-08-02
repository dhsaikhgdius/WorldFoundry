"""Strict recipe contract for DMD2 distribution matching plus GAN training."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..common import positive_int, strict_mapping
from .dmd import _normalize_few_step_schedule

DMD2_ALGORITHM_FIELDS = {
    "type",
    "student_timesteps",
    "student_sigmas",
    "real_score_checkpoint",
    "guidance_checkpoint",
    "normalization_axes",
    "num_train_timesteps",
    "score_min_sigma",
    "score_max_sigma",
    "score_flow_shift",
    "teacher_guidance_scale",
    "generator_update_interval",
    "student_scheduler_cadence",
    "normalization_epsilon",
    "score_timestep_mode",
    "distribution_matching_weight",
    "generator_adversarial_weight",
    "guidance_denoising_weight",
    "guidance_adversarial_weight",
    "diffusion_gan_max_sigma",
}


def _finite_float(value: object, *, field_name: str) -> float:
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    return resolved


@dataclass(frozen=True, slots=True)
class DMD2AlgorithmSpec:
    """Every choice consumed by the flow-matching-only native DMD2 runtime."""

    student_timesteps: tuple[float, ...]
    student_sigmas: tuple[float, ...]
    real_score_checkpoint: str
    guidance_checkpoint: str
    normalization_axes: tuple[int, ...]
    num_train_timesteps: int = 1000
    score_min_sigma: float = 0.02
    score_max_sigma: float = 0.98
    score_flow_shift: float = 1.0
    teacher_guidance_scale: float = 6.0
    generator_update_interval: int = 5
    student_scheduler_cadence: str = "iteration"
    normalization_epsilon: float = 0.0
    score_timestep_mode: str = "per-sample"
    distribution_matching_weight: float = 1.0
    generator_adversarial_weight: float = 1.0
    guidance_denoising_weight: float = 1.0
    guidance_adversarial_weight: float = 1.0
    diffusion_gan_max_sigma: float | None = None
    type: str = "dmd2"

    def __post_init__(self) -> None:
        algorithm_type = str(self.type).strip().lower().replace("_", "-")
        if algorithm_type != "dmd2":
            raise ValueError("DMD2 algorithm type must be 'dmd2'")
        timesteps, sigmas = _normalize_few_step_schedule(
            self.student_timesteps,
            self.student_sigmas,
        )
        for name in ("real_score_checkpoint", "guidance_checkpoint"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty checkpoint reference")
            object.__setattr__(self, name, value.strip())
        axes = tuple(int(axis) for axis in self.normalization_axes)
        if not axes or any(axis <= 0 for axis in axes):
            raise ValueError("normalization_axes must explicitly list positive non-batch axes")
        if len(set(axes)) != len(axes) or any(left >= right for left, right in zip(axes, axes[1:])):
            raise ValueError("normalization_axes must be unique and strictly increasing")
        cadence = str(self.student_scheduler_cadence).strip().lower().replace("_", "-")
        if cadence not in {"iteration", "generator-update"}:
            raise ValueError("student_scheduler_cadence must be 'iteration' or 'generator-update'")
        timestep_mode = str(self.score_timestep_mode).strip().lower().replace("_", "-")
        if timestep_mode not in {"per-sample", "batch-shared"}:
            raise ValueError("score_timestep_mode must be 'per-sample' or 'batch-shared'")
        minimum = _finite_float(self.score_min_sigma, field_name="score_min_sigma")
        maximum = _finite_float(self.score_max_sigma, field_name="score_max_sigma")
        if not 0.0 <= minimum < maximum <= 1.0:
            raise ValueError("DMD2 score sigma bounds must satisfy 0 <= min < max <= 1")
        flow_shift = _finite_float(self.score_flow_shift, field_name="score_flow_shift")
        normalization_epsilon = _finite_float(
            self.normalization_epsilon,
            field_name="normalization_epsilon",
        )
        if flow_shift <= 0 or normalization_epsilon < 0:
            raise ValueError("score_flow_shift must be positive and normalization_epsilon non-negative")
        teacher_guidance_scale = _finite_float(
            self.teacher_guidance_scale,
            field_name="teacher_guidance_scale",
        )
        weights: dict[str, float] = {}
        for name in (
            "distribution_matching_weight",
            "generator_adversarial_weight",
            "guidance_denoising_weight",
            "guidance_adversarial_weight",
        ):
            value = _finite_float(getattr(self, name), field_name=name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            weights[name] = value
        if weights["distribution_matching_weight"] + weights["generator_adversarial_weight"] <= 0:
            raise ValueError("DMD2 generator must enable distribution matching or adversarial loss")
        if weights["guidance_denoising_weight"] + weights["guidance_adversarial_weight"] <= 0:
            raise ValueError("DMD2 guidance must enable denoising or adversarial loss")
        diffusion_gan_max_sigma = self.diffusion_gan_max_sigma
        if diffusion_gan_max_sigma is not None:
            diffusion_gan_max_sigma = _finite_float(
                diffusion_gan_max_sigma,
                field_name="diffusion_gan_max_sigma",
            )
            if not 0.0 < diffusion_gan_max_sigma <= 1.0:
                raise ValueError("diffusion_gan_max_sigma must be in (0,1] when configured")

        object.__setattr__(self, "type", algorithm_type)
        object.__setattr__(self, "student_timesteps", timesteps)
        object.__setattr__(self, "student_sigmas", sigmas)
        object.__setattr__(self, "normalization_axes", axes)
        num_train_timesteps = positive_int(
            self.num_train_timesteps,
            field_name="algorithm.num_train_timesteps",
        )
        if num_train_timesteps < 2:
            raise ValueError("algorithm.num_train_timesteps must be at least two")
        object.__setattr__(self, "num_train_timesteps", num_train_timesteps)
        object.__setattr__(
            self,
            "generator_update_interval",
            positive_int(
                self.generator_update_interval,
                field_name="algorithm.generator_update_interval",
            ),
        )
        object.__setattr__(self, "student_scheduler_cadence", cadence)
        object.__setattr__(self, "score_timestep_mode", timestep_mode)
        object.__setattr__(self, "score_min_sigma", minimum)
        object.__setattr__(self, "score_max_sigma", maximum)
        object.__setattr__(self, "score_flow_shift", flow_shift)
        object.__setattr__(self, "teacher_guidance_scale", teacher_guidance_scale)
        object.__setattr__(self, "normalization_epsilon", normalization_epsilon)
        for name, value in weights.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "diffusion_gan_max_sigma", diffusion_gan_max_sigma)


def parse_dmd2_algorithm(value: object) -> DMD2AlgorithmSpec:
    """Parse DMD2 without accepting inactive or misspelled fields."""

    return DMD2AlgorithmSpec(
        **strict_mapping(
            value,
            field_name="algorithm",
            allowed=DMD2_ALGORITHM_FIELDS,
        )
    )


__all__ = ["DMD2AlgorithmSpec"]
