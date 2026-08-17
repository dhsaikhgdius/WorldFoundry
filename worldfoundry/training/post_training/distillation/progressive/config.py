"""Execution configuration for progressive DDIM distillation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

ProgressivePredictionType = Literal["sample", "epsilon", "v_prediction"]
ProgressiveLossWeight = Literal["constant", "snr", "snr_trunc", "v_mse"]
ProgressiveLearningRateAnneal = Literal["constant", "linear"]


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class ProgressiveDistillationConfig:
    """All values that alter the official halving procedure."""

    start_num_steps: int = 8192
    end_num_steps: int = 4
    optimizer_steps_per_stage: int = 50000
    prediction_type: ProgressivePredictionType = "sample"
    loss_weight: ProgressiveLossWeight = "snr_trunc"
    logsnr_min: float = -20.0
    logsnr_max: float = 20.0
    ema_decay: float = 0.0
    learning_rate_anneal: ProgressiveLearningRateAnneal = "linear"

    def __post_init__(self) -> None:
        start = _positive_int(self.start_num_steps, field_name="start_num_steps")
        end = _positive_int(self.end_num_steps, field_name="end_num_steps")
        stage_steps = _positive_int(
            self.optimizer_steps_per_stage,
            field_name="optimizer_steps_per_stage",
        )
        if start <= end:
            raise ValueError("start_num_steps must exceed end_num_steps")
        current = start
        while current > end:
            if current % 2:
                raise ValueError("progressive teacher steps must halve exactly")
            current //= 2
        if current != end:
            raise ValueError("end_num_steps must be reachable by repeated halving")
        if self.prediction_type not in {"sample", "epsilon", "v_prediction"}:
            raise ValueError("unsupported progressive prediction_type")
        if self.loss_weight not in {"constant", "snr", "snr_trunc", "v_mse"}:
            raise ValueError("unsupported progressive loss_weight")
        minimum = float(self.logsnr_min)
        maximum = float(self.logsnr_max)
        if not isfinite(minimum) or not isfinite(maximum) or minimum >= maximum:
            raise ValueError("logsnr bounds must be finite and strictly ordered")
        decay = float(self.ema_decay)
        if not isfinite(decay) or not 0.0 <= decay < 1.0:
            raise ValueError("ema_decay must be finite and lie in [0,1)")
        anneal = str(self.learning_rate_anneal).strip().lower().replace("_", "-")
        if anneal not in {"constant", "linear"}:
            raise ValueError(
                "learning_rate_anneal must be 'constant' or 'linear'"
            )
        object.__setattr__(self, "start_num_steps", start)
        object.__setattr__(self, "end_num_steps", end)
        object.__setattr__(self, "optimizer_steps_per_stage", stage_steps)
        object.__setattr__(self, "logsnr_min", minimum)
        object.__setattr__(self, "logsnr_max", maximum)
        object.__setattr__(self, "ema_decay", decay)
        object.__setattr__(self, "learning_rate_anneal", anneal)

    @property
    def teacher_steps(self) -> tuple[int, ...]:
        values: list[int] = []
        current = self.start_num_steps
        while current > self.end_num_steps:
            values.append(current)
            current //= 2
        return tuple(values)

    @property
    def student_steps(self) -> tuple[int, ...]:
        return tuple(value // 2 for value in self.teacher_steps)

    @property
    def stage_count(self) -> int:
        return len(self.student_steps)


__all__ = [
    "ProgressiveDistillationConfig",
    "ProgressiveLearningRateAnneal",
    "ProgressiveLossWeight",
    "ProgressivePredictionType",
]
