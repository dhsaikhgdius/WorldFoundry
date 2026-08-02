"""Behavior contract for online causal consistency distillation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..common import strict_mapping

CAUSAL_CONSISTENCY_ALGORITHM_FIELDS = {
    "type",
    "teacher_checkpoint",
    "num_levels",
    "num_train_timesteps",
    "flow_shift",
    "extra_terminal_step",
    "guidance_scale",
    "ema_decay",
    "frame_dim",
}


@dataclass(frozen=True, slots=True)
class CausalConsistencyAlgorithmSpec:
    """Adjacent teacher ODE step and EMA target choices."""

    teacher_checkpoint: str
    num_levels: int = 48
    num_train_timesteps: int = 1000
    flow_shift: float = 5.0
    extra_terminal_step: bool = True
    guidance_scale: float = 3.0
    ema_decay: float = 0.99
    frame_dim: int = 2
    type: str = "causal-consistency"

    def __post_init__(self) -> None:
        algorithm_type = str(self.type).strip().lower().replace("_", "-")
        if algorithm_type != "causal-consistency":
            raise ValueError(
                "causal consistency algorithm type must be 'causal-consistency'"
            )
        checkpoint = str(self.teacher_checkpoint).strip()
        if not checkpoint:
            raise ValueError("teacher_checkpoint must be non-empty")
        for name in ("num_levels", "num_train_timesteps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        if not isinstance(self.extra_terminal_step, bool):
            raise TypeError("extra_terminal_step must be bool")
        normalized: dict[str, float] = {}
        for name in ("flow_shift", "guidance_scale", "ema_decay"):
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            normalized[name] = value
        if normalized["flow_shift"] <= 0:
            raise ValueError("flow_shift must be positive")
        if not 0 <= normalized["ema_decay"] < 1:
            raise ValueError("ema_decay must lie in [0,1)")
        if isinstance(self.frame_dim, bool) or not isinstance(self.frame_dim, int):
            raise TypeError("frame_dim must be an integer")
        if self.frame_dim == 0:
            raise ValueError("frame_dim cannot be the batch dimension")
        object.__setattr__(self, "type", algorithm_type)
        object.__setattr__(self, "teacher_checkpoint", checkpoint)
        for name, value in normalized.items():
            object.__setattr__(self, name, value)


def parse_causal_consistency_algorithm(
    value: object,
) -> CausalConsistencyAlgorithmSpec:
    return CausalConsistencyAlgorithmSpec(
        **strict_mapping(
            value,
            field_name="algorithm",
            allowed=CAUSAL_CONSISTENCY_ALGORITHM_FIELDS,
        )
    )


__all__ = ["CausalConsistencyAlgorithmSpec"]
