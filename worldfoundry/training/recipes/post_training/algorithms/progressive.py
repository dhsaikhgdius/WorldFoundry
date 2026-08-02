"""Behavior contract for progressive DDIM distillation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..common import strict_mapping

PROGRESSIVE_DISTILLATION_ALGORITHM_FIELDS = {
    "type",
    "teacher_checkpoint",
    "start_num_steps",
    "end_num_steps",
    "optimizer_steps_per_stage",
    "prediction_type",
    "loss_weight",
    "logsnr_min",
    "logsnr_max",
    "ema_decay",
    "learning_rate_anneal",
}


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class ProgressiveDistillationAlgorithmSpec:
    """Every choice used by the official repeated-halving procedure."""

    teacher_checkpoint: str
    start_num_steps: int = 8192
    end_num_steps: int = 4
    optimizer_steps_per_stage: int = 50000
    prediction_type: str = "sample"
    loss_weight: str = "snr_trunc"
    logsnr_min: float = -20.0
    logsnr_max: float = 20.0
    ema_decay: float = 0.0
    learning_rate_anneal: str = "linear"
    type: str = "progressive-distillation"

    def __post_init__(self) -> None:
        algorithm_type = str(self.type).strip().lower().replace("_", "-")
        if algorithm_type != "progressive-distillation":
            raise ValueError(
                "progressive algorithm type must be 'progressive-distillation'"
            )
        checkpoint = str(self.teacher_checkpoint).strip()
        if not checkpoint:
            raise ValueError("teacher_checkpoint must be non-empty")
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
            raise ValueError(
                "end_num_steps must be reachable by repeated halving"
            )
        prediction = str(self.prediction_type).strip().lower().replace("-", "_")
        if prediction not in {"sample", "epsilon", "v_prediction"}:
            raise ValueError("unsupported progressive prediction_type")
        loss_weight = str(self.loss_weight).strip().lower().replace("-", "_")
        if loss_weight not in {"constant", "snr", "snr_trunc", "v_mse"}:
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

        object.__setattr__(self, "type", algorithm_type)
        object.__setattr__(self, "teacher_checkpoint", checkpoint)
        object.__setattr__(self, "start_num_steps", start)
        object.__setattr__(self, "end_num_steps", end)
        object.__setattr__(self, "optimizer_steps_per_stage", stage_steps)
        object.__setattr__(self, "prediction_type", prediction)
        object.__setattr__(self, "loss_weight", loss_weight)
        object.__setattr__(self, "logsnr_min", minimum)
        object.__setattr__(self, "logsnr_max", maximum)
        object.__setattr__(self, "ema_decay", decay)
        object.__setattr__(self, "learning_rate_anneal", anneal)


def parse_progressive_distillation_algorithm(
    value: object,
) -> ProgressiveDistillationAlgorithmSpec:
    return ProgressiveDistillationAlgorithmSpec(
        **strict_mapping(
            value,
            field_name="algorithm",
            allowed=PROGRESSIVE_DISTILLATION_ALGORITHM_FIELDS,
        )
    )


__all__ = ["ProgressiveDistillationAlgorithmSpec"]
