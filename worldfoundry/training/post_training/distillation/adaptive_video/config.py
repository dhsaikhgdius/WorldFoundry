"""Execution configuration for adaptive video distillation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from worldfoundry.core.io.integrity import canonical_sha256
from worldfoundry.training.recipes.post_training.algorithms.adaptive_video import (
    AdaptiveVideoAlgorithmSpec,
)

from ..dmd.objective import DMDConfig, FewStepSchedule


@dataclass(frozen=True, slots=True)
class AdaptiveVideoConfig:
    dmd: DMDConfig
    regression_ema_decay: float = 0.95
    regression_sensitivity: float = 3.0
    regression_loss_weight: float = 1.0
    temporal_regularization_weight: float = 0.05
    temporal_loss_cutoff: float = 0.8
    temporal_epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if not isinstance(self.dmd, DMDConfig):
            raise TypeError("dmd must be DMDConfig")
        values: dict[str, float] = {}
        for name in (
            "regression_ema_decay",
            "regression_sensitivity",
            "regression_loss_weight",
            "temporal_regularization_weight",
            "temporal_loss_cutoff",
            "temporal_epsilon",
        ):
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            values[name] = value
        if not 0.0 <= values["regression_ema_decay"] < 1.0:
            raise ValueError("regression_ema_decay must be in [0,1)")
        if values["regression_sensitivity"] <= 0:
            raise ValueError("regression_sensitivity must be positive")
        if values["regression_loss_weight"] <= 0:
            raise ValueError("regression_loss_weight must be positive")
        if values["temporal_regularization_weight"] <= 0:
            raise ValueError("temporal_regularization_weight must be positive")
        if values["temporal_epsilon"] <= 0:
            raise ValueError("temporal_epsilon must be positive")
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @classmethod
    def from_recipe(cls, spec: AdaptiveVideoAlgorithmSpec) -> AdaptiveVideoConfig:
        if not isinstance(spec, AdaptiveVideoAlgorithmSpec):
            raise TypeError("spec must be AdaptiveVideoAlgorithmSpec")
        schedule = FewStepSchedule(spec.student_timesteps, spec.student_sigmas)
        return cls(
            dmd=DMDConfig(
                schedule=schedule,
                num_train_timesteps=spec.num_train_timesteps,
                score_min_sigma=spec.score_min_sigma,
                score_max_sigma=spec.score_max_sigma,
                score_flow_shift=spec.score_flow_shift,
                teacher_guidance_scale=spec.teacher_guidance_scale,
                normalization_epsilon=spec.normalization_epsilon,
                shared_score_timestep=False,
                per_sample_normalization=True,
            ),
            regression_ema_decay=spec.regression_ema_decay,
            regression_sensitivity=spec.regression_sensitivity,
            regression_loss_weight=spec.regression_loss_weight,
            temporal_regularization_weight=spec.temporal_regularization_weight,
            temporal_loss_cutoff=spec.temporal_loss_cutoff,
            temporal_epsilon=spec.temporal_epsilon,
        )

    @property
    def digest(self) -> str:
        return canonical_sha256(
            {
                "schema": "worldfoundry-adaptive-video-config",
                "dmd_digest": self.dmd.digest,
                "regression_ema_decay": self.regression_ema_decay,
                "regression_sensitivity": self.regression_sensitivity,
                "regression_loss_weight": self.regression_loss_weight,
                "temporal_regularization_weight": self.temporal_regularization_weight,
                "temporal_loss_cutoff": self.temporal_loss_cutoff,
                "temporal_epsilon": self.temporal_epsilon,
            }
        )


__all__ = ["AdaptiveVideoConfig"]
