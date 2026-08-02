"""Executed schedule contract for Causal ODE trajectory regression."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

from worldfoundry.core.io.integrity import canonical_sha256
from worldfoundry.core.nn.diffusion_schedulers import FlowMatchScheduler


def warped_causal_ode_timesteps(
    raw_denoising_steps: tuple[int, ...],
    *,
    num_train_timesteps: int = 1000,
    flow_shift: float = 5.0,
    extra_terminal_step: bool = True,
) -> tuple[float, ...]:
    """Map released raw Wan indices through the executed flow schedule."""

    if isinstance(num_train_timesteps, bool) or int(num_train_timesteps) < 2:
        raise ValueError("num_train_timesteps must be an integer >= 2")
    if not isinstance(extra_terminal_step, bool):
        raise TypeError("extra_terminal_step must be bool")
    shift = float(flow_shift)
    if not isfinite(shift) or shift <= 0:
        raise ValueError("flow_shift must be finite and positive")
    steps = tuple(raw_denoising_steps)
    if not steps:
        raise ValueError("raw_denoising_steps cannot be empty")
    if any(isinstance(step, bool) or not isinstance(step, int) for step in steps):
        raise TypeError("raw_denoising_steps must contain integers")
    train_steps = int(num_train_timesteps)
    if any(step < 0 or step > train_steps for step in steps):
        raise ValueError("raw denoising steps must lie in [0,num_train_timesteps]")

    scheduler = FlowMatchScheduler(
        num_inference_steps=train_steps,
        num_train_timesteps=train_steps,
        shift=shift,
        sigma_min=0.0,
        extra_one_step=extra_terminal_step,
    )
    mapped: list[float] = []
    for step in steps:
        if step == 0:
            mapped.append(0.0)
        else:
            mapped.append(float(scheduler.timesteps[train_steps - step].item()))
    return tuple(mapped)


@dataclass(frozen=True, slots=True)
class CausalODEConfig:
    """Effective trajectory timesteps consumed by the causal student."""

    trajectory_timesteps: tuple[float, ...]
    frame_dim: int = 2

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.trajectory_timesteps)
        if not values:
            raise ValueError("trajectory_timesteps cannot be empty")
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError("trajectory_timesteps must be finite and non-negative")
        if any(left <= right for left, right in zip(values, values[1:], strict=False)):
            raise ValueError("trajectory_timesteps must be strictly descending")
        if isinstance(self.frame_dim, bool) or not isinstance(self.frame_dim, int):
            raise TypeError("frame_dim must be an integer")
        if self.frame_dim == 0:
            raise ValueError("frame_dim cannot be the batch dimension")
        object.__setattr__(self, "trajectory_timesteps", values)

    @property
    def digest(self) -> str:
        return canonical_sha256({"schema": "worldfoundry-causal-ode-config", **asdict(self)})


__all__ = ["CausalODEConfig", "warped_causal_ode_timesteps"]
