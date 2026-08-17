"""Behavior contract for two-stage diagonal video distillation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..common import positive_int, strict_mapping
from .auxiliary_optimizers import (
    AuxiliaryOptimizerRule,
    forbids_auxiliary,
    requires_auxiliary,
)

DIAGONAL_ALGORITHM_FIELDS = {
    "type",
    "stage",
    "real_score_checkpoint",
    "fake_score_checkpoint",
    "fixed_teacher_checkpoint",
    "frames_per_block",
    "frame_dim",
    "latent_channels",
    "generator_update_interval",
    "student_scheduler_cadence",
    "ema_decay",
    "ema_start_step",
}


@dataclass(frozen=True, slots=True)
class DiagonalAlgorithmSpec:
    """Released stage schedule, model roles, cadence, and EMA behavior."""

    real_score_checkpoint: str
    fake_score_checkpoint: str
    fixed_teacher_checkpoint: str
    stage: str = "stage-two"
    frames_per_block: int = 3
    frame_dim: int = 2
    latent_channels: int = 16
    generator_update_interval: int = 5
    student_scheduler_cadence: str = "iteration"
    ema_decay: float = 0.99
    ema_start_step: int = 200
    type: str = "diagonal-distillation"

    def __post_init__(self) -> None:
        algorithm_type = str(self.type).strip().lower().replace("_", "-")
        if algorithm_type != "diagonal-distillation":
            raise ValueError(
                "Diagonal algorithm type must be 'diagonal-distillation'"
            )
        stage = str(self.stage).strip().lower().replace("_", "-")
        if stage not in {"stage-one", "stage-two"}:
            raise ValueError("Diagonal stage must be 'stage-one' or 'stage-two'")
        for name in (
            "real_score_checkpoint",
            "fake_score_checkpoint",
            "fixed_teacher_checkpoint",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty checkpoint reference")
            object.__setattr__(self, name, value.strip())
        if isinstance(self.frame_dim, bool) or not isinstance(self.frame_dim, int):
            raise TypeError("frame_dim must be an integer")
        if self.frame_dim == 0:
            raise ValueError("frame_dim cannot be the batch dimension")
        cadence = str(self.student_scheduler_cadence).strip().lower().replace("_", "-")
        if cadence not in {"iteration", "generator-update"}:
            raise ValueError(
                "student_scheduler_cadence must be 'iteration' or 'generator-update'"
            )
        decay = float(self.ema_decay)
        if not isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("ema_decay must be finite and in [0,1)")
        if isinstance(self.ema_start_step, bool) or not isinstance(
            self.ema_start_step,
            int,
        ):
            raise TypeError("ema_start_step must be an integer")
        if self.ema_start_step < 0:
            raise ValueError("ema_start_step must be non-negative")
        object.__setattr__(self, "type", algorithm_type)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(
            self,
            "frames_per_block",
            positive_int(
                self.frames_per_block,
                field_name="algorithm.frames_per_block",
            ),
        )
        object.__setattr__(
            self,
            "latent_channels",
            positive_int(
                self.latent_channels,
                field_name="algorithm.latent_channels",
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
        object.__setattr__(self, "student_scheduler_cadence", cadence)
        object.__setattr__(self, "ema_decay", decay)

    def auxiliary_optimizer_rules(self) -> tuple[AuxiliaryOptimizerRule, ...]:
        return (
            requires_auxiliary(
                "fake_score_optimizer",
                "diagonal distillation requires fake_score_optimizer",
            ),
            forbids_auxiliary(
                "guidance_optimizer",
                "discriminator_optimizer",
                message="diagonal distillation only accepts fake_score_optimizer",
            ),
        )


def parse_diagonal_algorithm(value: object) -> DiagonalAlgorithmSpec:
    """Parse a strict diagonal-distillation algorithm section."""

    return DiagonalAlgorithmSpec(
        **strict_mapping(
            value,
            field_name="algorithm",
            allowed=DIAGONAL_ALGORITHM_FIELDS,
        )
    )


__all__ = ["DiagonalAlgorithmSpec"]
