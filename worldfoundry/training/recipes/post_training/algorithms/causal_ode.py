"""Behavior contract for causal PF-ODE trajectory distillation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..common import strict_mapping

CAUSAL_ODE_ALGORITHM_FIELDS = {
    "type",
    "raw_denoising_steps",
    "num_train_timesteps",
    "flow_shift",
    "extra_terminal_step",
    "frame_dim",
}


@dataclass(frozen=True, slots=True)
class CausalODEAlgorithmSpec:
    """Fields that determine trajectory indexing and causal regression."""

    raw_denoising_steps: tuple[int, ...] = (1000, 750, 500, 250)
    num_train_timesteps: int = 1000
    flow_shift: float = 5.0
    extra_terminal_step: bool = True
    frame_dim: int = 2
    type: str = "causal-ode"

    def __post_init__(self) -> None:
        algorithm_type = str(self.type).strip().lower().replace("_", "-")
        if algorithm_type != "causal-ode":
            raise ValueError("causal ODE algorithm type must be 'causal-ode'")
        if (
            isinstance(self.num_train_timesteps, bool)
            or not isinstance(self.num_train_timesteps, int)
            or self.num_train_timesteps < 2
        ):
            raise ValueError("num_train_timesteps must be an integer >= 2")
        steps = tuple(self.raw_denoising_steps)
        if not steps:
            raise ValueError("raw_denoising_steps cannot be empty")
        if any(isinstance(step, bool) or not isinstance(step, int) for step in steps):
            raise TypeError("raw_denoising_steps must contain integers")
        if any(step < 0 or step > self.num_train_timesteps for step in steps):
            raise ValueError("raw_denoising_steps must lie in the training timeline")
        if any(left <= right for left, right in zip(steps, steps[1:], strict=False)):
            raise ValueError("raw_denoising_steps must be strictly descending")
        shift = float(self.flow_shift)
        if not isfinite(shift) or shift <= 0:
            raise ValueError("flow_shift must be finite and positive")
        if not isinstance(self.extra_terminal_step, bool):
            raise TypeError("extra_terminal_step must be bool")
        if isinstance(self.frame_dim, bool) or not isinstance(self.frame_dim, int):
            raise TypeError("frame_dim must be an integer")
        if self.frame_dim == 0:
            raise ValueError("frame_dim cannot be the batch dimension")
        object.__setattr__(self, "type", algorithm_type)
        object.__setattr__(self, "raw_denoising_steps", steps)
        object.__setattr__(self, "flow_shift", shift)


def parse_causal_ode_algorithm(value: object) -> CausalODEAlgorithmSpec:
    return CausalODEAlgorithmSpec(
        **strict_mapping(
            value,
            field_name="algorithm",
            allowed=CAUSAL_ODE_ALGORITHM_FIELDS,
        )
    )


__all__ = ["CausalODEAlgorithmSpec"]
