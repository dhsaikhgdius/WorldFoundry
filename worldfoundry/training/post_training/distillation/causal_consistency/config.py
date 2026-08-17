"""Executed flow schedule and hyperparameters for causal consistency."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from worldfoundry.core.nn.diffusion_schedulers import FlowMatchScheduler


@dataclass(frozen=True, slots=True)
class CausalConsistencySchedule:
    """Adjacent flow levels produced by the shared core scheduler."""

    timesteps: tuple[float, ...]
    sigmas: tuple[float, ...]
    num_train_timesteps: int
    extra_terminal_step: bool

    def __post_init__(self) -> None:
        if len(self.timesteps) < 2 or len(self.timesteps) != len(self.sigmas):
            raise ValueError("causal consistency schedule needs aligned adjacent levels")
        if any(not isfinite(value) or value < 0 for value in self.timesteps):
            raise ValueError("causal consistency timesteps must be finite and non-negative")
        if any(left <= right for left, right in zip(self.timesteps, self.timesteps[1:], strict=False)):
            raise ValueError("causal consistency timesteps must be strictly descending")
        if any(not isfinite(sigma) or not 0 <= sigma <= 1 for sigma in self.sigmas):
            raise ValueError("causal consistency sigmas must be finite and lie in [0,1]")
        if not isinstance(self.extra_terminal_step, bool):
            raise TypeError("extra_terminal_step must be bool")

    @property
    def pair_count(self) -> int:
        return len(self.timesteps) - 1


@dataclass(frozen=True, slots=True)
class CausalConsistencyConfig:
    """Every field directly controls Causal Consistency execution."""

    num_levels: int = 48
    num_train_timesteps: int = 1000
    flow_shift: float = 5.0
    extra_terminal_step: bool = True
    guidance_scale: float = 3.0
    ema_decay: float = 0.99
    frame_dim: int = 2

    def __post_init__(self) -> None:
        for name in ("num_levels", "num_train_timesteps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        if not isinstance(self.extra_terminal_step, bool):
            raise TypeError("extra_terminal_step must be bool")
        for name in ("flow_shift", "guidance_scale", "ema_decay"):
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.flow_shift <= 0:
            raise ValueError("flow_shift must be positive")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must lie in [0,1)")
        if isinstance(self.frame_dim, bool) or not isinstance(self.frame_dim, int):
            raise TypeError("frame_dim must be an integer")
        if self.frame_dim == 0:
            raise ValueError("frame_dim cannot be the batch dimension")


def build_causal_consistency_schedule(config: CausalConsistencyConfig) -> CausalConsistencySchedule:
    """Build the exact adjacent schedule consumed by teacher and students."""

    if not isinstance(config, CausalConsistencyConfig):
        raise TypeError("config must be CausalConsistencyConfig")
    scheduler = FlowMatchScheduler(
        num_inference_steps=config.num_levels,
        num_train_timesteps=config.num_train_timesteps,
        shift=config.flow_shift,
        sigma_min=0.0,
        extra_one_step=config.extra_terminal_step,
    )
    return CausalConsistencySchedule(
        timesteps=tuple(float(value) for value in scheduler.timesteps.tolist()),
        sigmas=tuple(float(value) for value in scheduler.sigmas.tolist()),
        num_train_timesteps=config.num_train_timesteps,
        extra_terminal_step=config.extra_terminal_step,
    )


__all__ = [
    "CausalConsistencyConfig",
    "CausalConsistencySchedule",
    "build_causal_consistency_schedule",
]
