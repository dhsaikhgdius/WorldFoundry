"""Strict execution configuration for native self-forcing rollout."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from ..dmd.objective import FewStepSchedule

ExitStepMode = Literal["sequence", "block"]


def shifted_few_step_schedule(
    timesteps: tuple[float, ...],
    *,
    num_train_timesteps: int,
    flow_shift: float,
) -> FewStepSchedule:
    """Warp released diffusion indices onto their effective flow timeline."""

    if isinstance(num_train_timesteps, bool) or int(num_train_timesteps) < 2:
        raise ValueError("num_train_timesteps must be an integer >= 2")
    shift = float(flow_shift)
    if not isfinite(shift) or shift <= 0:
        raise ValueError("flow_shift must be finite and positive")
    scale = float(num_train_timesteps)
    raw = tuple(float(value) for value in timesteps)
    sigmas = tuple(shift * (value / scale) / (1.0 + (shift - 1.0) * (value / scale)) for value in raw)
    return FewStepSchedule(
        timesteps=tuple(sigma * scale for sigma in sigmas),
        sigmas=sigmas,
    )


@dataclass(frozen=True, slots=True)
class SelfForcingConfig:
    """Causal chunk geometry and exit-step distribution."""

    schedule: FewStepSchedule
    frames_per_block: int
    frame_dim: int = 2
    exit_step_mode: ExitStepMode = "sequence"

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, FewStepSchedule):
            raise TypeError("schedule must be a FewStepSchedule")
        if isinstance(self.frames_per_block, bool) or int(self.frames_per_block) <= 0:
            raise ValueError("frames_per_block must be a positive integer")
        if isinstance(self.frame_dim, bool) or not isinstance(self.frame_dim, int):
            raise TypeError("frame_dim must be an integer")
        if self.frame_dim == 0:
            raise ValueError("frame_dim cannot be the batch dimension")
        mode = str(self.exit_step_mode).strip().lower().replace("_", "-")
        if mode not in {"sequence", "block"}:
            raise ValueError("exit_step_mode must be 'sequence' or 'block'")
        object.__setattr__(self, "frames_per_block", int(self.frames_per_block))
        object.__setattr__(self, "exit_step_mode", mode)

__all__ = ["ExitStepMode", "SelfForcingConfig", "shifted_few_step_schedule"]
