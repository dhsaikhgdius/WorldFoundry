"""Strict behavior contract for adaptive video distillation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..common import positive_int, strict_mapping
from .auxiliary_optimizers import (
    AuxiliaryOptimizerRule,
    forbids_auxiliary,
    requires_auxiliary,
)
from .dmd import _normalize_few_step_schedule


def _finite_float(value: object, *, field_name: str) -> float:
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    return resolved


@dataclass(frozen=True, slots=True)
class AdaptiveVideoAlgorithmSpec:
    """DMD plus adaptive real-data regression and temporal regularization."""

    student_timesteps: tuple[float, ...]
    student_sigmas: tuple[float, ...]
    real_score_checkpoint: str
    fake_score_checkpoint: str
    num_train_timesteps: int = 1000
    score_min_sigma: float = 0.02
    score_max_sigma: float = 0.98
    score_flow_shift: float = 5.0
    teacher_guidance_scale: float = 5.0
    generator_update_interval: int = 5
    student_scheduler_cadence: str = "iteration"
    normalization_epsilon: float = 0.0
    regression_ema_decay: float = 0.95
    regression_sensitivity: float = 3.0
    regression_loss_weight: float = 1.0
    temporal_regularization_weight: float = 0.05
    temporal_loss_cutoff: float = 0.8
    temporal_epsilon: float = 1.0e-6
    type: str = "adaptive-video-distillation"

    def __post_init__(self) -> None:
        algorithm_type = str(self.type).strip().lower().replace("_", "-")
        if algorithm_type != "adaptive-video-distillation":
            raise ValueError(
                "adaptive video algorithm type must be 'adaptive-video-distillation'"
            )
        timesteps, sigmas = _normalize_few_step_schedule(
            self.student_timesteps,
            self.student_sigmas,
        )
        for name in ("real_score_checkpoint", "fake_score_checkpoint"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty checkpoint reference")
            object.__setattr__(self, name, value.strip())
        cadence = str(self.student_scheduler_cadence).strip().lower().replace("_", "-")
        if cadence not in {"iteration", "generator-update"}:
            raise ValueError(
                "student_scheduler_cadence must be 'iteration' or 'generator-update'"
            )
        minimum = _finite_float(self.score_min_sigma, field_name="score_min_sigma")
        maximum = _finite_float(self.score_max_sigma, field_name="score_max_sigma")
        if not 0.0 <= minimum < maximum <= 1.0:
            raise ValueError("score sigma bounds must satisfy 0 <= min < max <= 1")
        normalized: dict[str, float] = {}
        for name in (
            "score_flow_shift",
            "teacher_guidance_scale",
            "normalization_epsilon",
            "regression_ema_decay",
            "regression_sensitivity",
            "regression_loss_weight",
            "temporal_regularization_weight",
            "temporal_loss_cutoff",
            "temporal_epsilon",
        ):
            normalized[name] = _finite_float(getattr(self, name), field_name=name)
        if normalized["score_flow_shift"] <= 0:
            raise ValueError("score_flow_shift must be positive")
        if normalized["normalization_epsilon"] < 0:
            raise ValueError("normalization_epsilon must be non-negative")
        if not 0.0 <= normalized["regression_ema_decay"] < 1.0:
            raise ValueError("regression_ema_decay must be in [0,1)")
        if normalized["regression_sensitivity"] <= 0:
            raise ValueError("regression_sensitivity must be positive")
        if normalized["regression_loss_weight"] <= 0:
            raise ValueError("regression_loss_weight must be positive")
        if normalized["temporal_regularization_weight"] <= 0:
            raise ValueError("temporal_regularization_weight must be positive")
        if normalized["temporal_epsilon"] <= 0:
            raise ValueError("temporal_epsilon must be positive")
        object.__setattr__(self, "type", algorithm_type)
        object.__setattr__(self, "student_timesteps", timesteps)
        object.__setattr__(self, "student_sigmas", sigmas)
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
        object.__setattr__(self, "score_min_sigma", minimum)
        object.__setattr__(self, "score_max_sigma", maximum)
        for name, value in normalized.items():
            object.__setattr__(self, name, value)

    def auxiliary_optimizer_rules(self) -> tuple[AuxiliaryOptimizerRule, ...]:
        return (
            requires_auxiliary(
                "fake_score_optimizer",
                "adaptive video distillation requires fake_score_optimizer",
            ),
            forbids_auxiliary(
                "guidance_optimizer",
                "discriminator_optimizer",
                message=f"{self.type} only accepts fake_score_optimizer",
            ),
        )


ADAPTIVE_VIDEO_ALGORITHM_FIELDS = frozenset(
    AdaptiveVideoAlgorithmSpec.__dataclass_fields__
)


def parse_adaptive_video_algorithm(value: object) -> AdaptiveVideoAlgorithmSpec:
    return AdaptiveVideoAlgorithmSpec(
        **strict_mapping(
            value,
            field_name="algorithm",
            allowed=set(ADAPTIVE_VIDEO_ALGORITHM_FIELDS),
        )
    )


__all__ = ["AdaptiveVideoAlgorithmSpec", "parse_adaptive_video_algorithm"]
