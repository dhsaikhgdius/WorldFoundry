"""Execution configuration for native Self-Gradient-Forcing replay."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from worldfoundry.core.io.integrity import canonical_sha256

from ..dmd.objective import FewStepSchedule
from ..self_forcing.config import shifted_few_step_schedule

CacheTargetMode = Literal["exit", "final-clean"]
ExitStepRankMode = Literal["local", "synchronized"]


def shifted_flow_timestep(
    timestep: float,
    *,
    num_train_timesteps: int,
    flow_shift: float,
) -> tuple[float, float]:
    """Map a raw flow index to the model timestep and effective sigma."""

    if isinstance(num_train_timesteps, bool) or int(num_train_timesteps) < 2:
        raise ValueError("num_train_timesteps must be an integer >= 2")
    raw = float(timestep)
    shift = float(flow_shift)
    if not isfinite(raw) or not 0 <= raw <= int(num_train_timesteps):
        raise ValueError("timestep must be finite and within the training timeline")
    if not isfinite(shift) or shift <= 0:
        raise ValueError("flow_shift must be finite and positive")
    base_sigma = raw / float(num_train_timesteps)
    sigma = shift * base_sigma / (1.0 + (shift - 1.0) * base_sigma)
    return sigma * int(num_train_timesteps), sigma


@dataclass(frozen=True, slots=True)
class SelfGradientForcingConfig:
    """The fields consumed by the bounded two-pass training rollout."""

    schedule: FewStepSchedule
    frames_per_block: int
    frame_dim: int = 2
    context_timestep: float = 0.0
    context_sigma: float = 0.0
    cache_target_mode: CacheTargetMode = "exit"
    exit_step_rank_mode: ExitStepRankMode = "local"
    match_context: bool = True
    last_step_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, FewStepSchedule):
            raise TypeError("schedule must be a FewStepSchedule")
        if isinstance(self.frames_per_block, bool) or int(self.frames_per_block) <= 0:
            raise ValueError("frames_per_block must be a positive integer")
        if isinstance(self.frame_dim, bool) or not isinstance(self.frame_dim, int):
            raise TypeError("frame_dim must be an integer")
        if self.frame_dim == 0:
            raise ValueError("frame_dim cannot be the batch dimension")
        context_timestep = float(self.context_timestep)
        context_sigma = float(self.context_sigma)
        if not isfinite(context_timestep) or context_timestep < 0:
            raise ValueError("context_timestep must be finite and non-negative")
        if not isfinite(context_sigma) or not 0 <= context_sigma <= 1:
            raise ValueError("context_sigma must be finite and in [0,1]")
        cache_mode = str(self.cache_target_mode).strip().lower().replace("_", "-")
        if cache_mode not in {"exit", "final-clean"}:
            raise ValueError("cache_target_mode must be 'exit' or 'final-clean'")
        rank_mode = str(self.exit_step_rank_mode).strip().lower().replace("_", "-")
        if rank_mode not in {"local", "synchronized"}:
            raise ValueError("exit_step_rank_mode must be 'local' or 'synchronized'")
        if not isinstance(self.match_context, bool) or not isinstance(self.last_step_only, bool):
            raise TypeError("match_context and last_step_only must be bool values")
        object.__setattr__(self, "frames_per_block", int(self.frames_per_block))
        object.__setattr__(self, "context_timestep", context_timestep)
        object.__setattr__(self, "context_sigma", context_sigma)
        object.__setattr__(self, "cache_target_mode", cache_mode)
        object.__setattr__(self, "exit_step_rank_mode", rank_mode)

    @property
    def digest(self) -> str:
        return canonical_sha256(
            {
                "schema": "worldfoundry-self-gradient-forcing-config",
                "schedule_digest": self.schedule.digest,
                "frames_per_block": self.frames_per_block,
                "frame_dim": self.frame_dim,
                "context_timestep": self.context_timestep,
                "context_sigma": self.context_sigma,
                "cache_target_mode": self.cache_target_mode,
                "exit_step_rank_mode": self.exit_step_rank_mode,
                "match_context": self.match_context,
                "last_step_only": self.last_step_only,
            }
        )

    @classmethod
    def from_raw_timesteps(
        cls,
        timesteps: tuple[float, ...] = (1000.0, 750.0, 500.0, 250.0),
        *,
        num_train_timesteps: int = 1000,
        flow_shift: float = 5.0,
        frames_per_block: int,
        frame_dim: int = 2,
        context_timestep: float = 0.0,
        cache_target_mode: CacheTargetMode = "exit",
        exit_step_rank_mode: ExitStepRankMode = "local",
        match_context: bool = True,
        last_step_only: bool = False,
    ) -> SelfGradientForcingConfig:
        schedule = shifted_few_step_schedule(
            timesteps,
            num_train_timesteps=num_train_timesteps,
            flow_shift=flow_shift,
        )
        effective_context, context_sigma = shifted_flow_timestep(
            context_timestep,
            num_train_timesteps=num_train_timesteps,
            flow_shift=flow_shift,
        )
        return cls(
            schedule=schedule,
            frames_per_block=frames_per_block,
            frame_dim=frame_dim,
            context_timestep=effective_context,
            context_sigma=context_sigma,
            cache_target_mode=cache_target_mode,
            exit_step_rank_mode=exit_step_rank_mode,
            match_context=match_context,
            last_step_only=last_step_only,
        )


__all__ = [
    "CacheTargetMode",
    "ExitStepRankMode",
    "SelfGradientForcingConfig",
    "shifted_flow_timestep",
]
