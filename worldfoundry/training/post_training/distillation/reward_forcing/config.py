"""Behavior-bearing configuration for Reward-Forcing Re-DMD."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from worldfoundry.training.recipes.post_training.algorithms.reward_forcing import (
    RewardForcingAlgorithmSpec,
)

from ..dmd.objective import DMDConfig, FewStepSchedule
from ..self_forcing.config import SelfForcingConfig, shifted_few_step_schedule


def _positive_float(value: object, *, field_name: str) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return resolved


def _non_negative_float(value: object, *, field_name: str) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return resolved


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class RewardForcingConfig:
    """Released 21-frame Re-DMD behavior over a native causal rollout."""

    denoising_timesteps: tuple[float, ...] = (1000.0, 750.0, 500.0, 250.0)
    num_train_timesteps: int = 1000
    denoising_flow_shift: float = 5.0
    frames_per_block: int = 3
    training_frames: int = 21
    frame_dim: int = 2
    same_step_across_blocks: bool = True
    local_attention_frames: int = 9
    ema_sink_frames: int = 3
    ema_sink_decay: float = 0.999
    score_min_sigma: float = 0.02
    score_max_sigma: float = 0.98
    score_flow_shift: float = 5.0
    teacher_guidance_scale: float = 3.0
    normalization_epsilon: float = 0.0
    reward_beta: float = 2.0
    generator_update_interval: int = 5
    student_scheduler_cadence: str = "generator-update"
    ema_decay: float = 0.99
    ema_start_step: int = 200

    @classmethod
    def from_recipe(cls, algorithm: RewardForcingAlgorithmSpec) -> RewardForcingConfig:
        """Project the strict recipe spec onto execution-only Re-DMD behavior."""

        if not isinstance(algorithm, RewardForcingAlgorithmSpec):
            raise TypeError("algorithm must be RewardForcingAlgorithmSpec")
        return cls(
            denoising_timesteps=algorithm.denoising_timesteps,
            num_train_timesteps=algorithm.num_train_timesteps,
            denoising_flow_shift=algorithm.denoising_flow_shift,
            frames_per_block=algorithm.frames_per_block,
            training_frames=algorithm.training_frames,
            frame_dim=algorithm.frame_dim,
            same_step_across_blocks=algorithm.same_step_across_blocks,
            local_attention_frames=algorithm.local_attention_frames,
            ema_sink_frames=algorithm.ema_sink_frames,
            ema_sink_decay=algorithm.ema_sink_decay,
            score_min_sigma=algorithm.score_min_sigma,
            score_max_sigma=algorithm.score_max_sigma,
            score_flow_shift=algorithm.score_flow_shift,
            teacher_guidance_scale=algorithm.teacher_guidance_scale,
            normalization_epsilon=algorithm.normalization_epsilon,
            reward_beta=algorithm.reward_beta,
            generator_update_interval=algorithm.generator_update_interval,
            student_scheduler_cadence=algorithm.student_scheduler_cadence,
            ema_decay=algorithm.ema_decay,
            ema_start_step=algorithm.ema_start_step,
        )

    def __post_init__(self) -> None:
        train_timesteps = _positive_int(
            self.num_train_timesteps,
            field_name="num_train_timesteps",
        )
        timesteps = tuple(float(value) for value in self.denoising_timesteps)
        if not timesteps or any(not isfinite(value) or value <= 0.0 or value > train_timesteps for value in timesteps):
            raise ValueError("denoising_timesteps must be finite values in (0,num_train_timesteps]")
        if any(left <= right for left, right in zip(timesteps, timesteps[1:])):
            raise ValueError("denoising_timesteps must be strictly descending")
        denoising_shift = _positive_float(
            self.denoising_flow_shift,
            field_name="denoising_flow_shift",
        )
        blocks = _positive_int(self.frames_per_block, field_name="frames_per_block")
        frames = _positive_int(self.training_frames, field_name="training_frames")
        if frames % blocks:
            raise ValueError("training_frames must be divisible by frames_per_block")
        if isinstance(self.frame_dim, bool) or not isinstance(self.frame_dim, int):
            raise TypeError("frame_dim must be an integer")
        if self.frame_dim == 0:
            raise ValueError("frame_dim cannot be the batch dimension")
        if not isinstance(self.same_step_across_blocks, bool):
            raise TypeError("same_step_across_blocks must be a bool")
        local_frames = _positive_int(
            self.local_attention_frames,
            field_name="local_attention_frames",
        )
        sink_frames = _positive_int(
            self.ema_sink_frames,
            field_name="ema_sink_frames",
        )
        if sink_frames != blocks:
            raise ValueError("released EMA-Sink requires ema_sink_frames to equal frames_per_block")
        if sink_frames >= local_frames or local_frames % blocks:
            raise ValueError("local_attention_frames must be a block-aligned window larger than EMA-Sink")
        sink_decay = float(self.ema_sink_decay)
        if not isfinite(sink_decay) or not 0.0 < sink_decay < 1.0:
            raise ValueError("ema_sink_decay must be finite and in (0,1)")
        minimum = float(self.score_min_sigma)
        maximum = float(self.score_max_sigma)
        if not all(isfinite(value) for value in (minimum, maximum)) or not (0.0 <= minimum < maximum <= 1.0):
            raise ValueError("score sigma bounds must satisfy 0 <= min < max <= 1")
        score_shift = _positive_float(
            self.score_flow_shift,
            field_name="score_flow_shift",
        )
        guidance = float(self.teacher_guidance_scale)
        if not isfinite(guidance):
            raise ValueError("teacher_guidance_scale must be finite")
        epsilon = _non_negative_float(
            self.normalization_epsilon,
            field_name="normalization_epsilon",
        )
        beta = _non_negative_float(self.reward_beta, field_name="reward_beta")
        interval = _positive_int(
            self.generator_update_interval,
            field_name="generator_update_interval",
        )
        cadence = str(self.student_scheduler_cadence).strip().lower().replace("_", "-")
        if cadence not in {"iteration", "generator-update"}:
            raise ValueError("student_scheduler_cadence must be 'iteration' or 'generator-update'")
        decay = float(self.ema_decay)
        if not isfinite(decay) or not 0.0 < decay < 1.0:
            raise ValueError("ema_decay must be finite and in (0,1)")
        ema_start = _non_negative_int(self.ema_start_step, field_name="ema_start_step")
        object.__setattr__(self, "denoising_timesteps", timesteps)
        object.__setattr__(self, "num_train_timesteps", train_timesteps)
        object.__setattr__(self, "denoising_flow_shift", denoising_shift)
        object.__setattr__(self, "frames_per_block", blocks)
        object.__setattr__(self, "training_frames", frames)
        object.__setattr__(self, "local_attention_frames", local_frames)
        object.__setattr__(self, "ema_sink_frames", sink_frames)
        object.__setattr__(self, "ema_sink_decay", sink_decay)
        object.__setattr__(self, "score_min_sigma", minimum)
        object.__setattr__(self, "score_max_sigma", maximum)
        object.__setattr__(self, "score_flow_shift", score_shift)
        object.__setattr__(self, "teacher_guidance_scale", guidance)
        object.__setattr__(self, "normalization_epsilon", epsilon)
        object.__setattr__(self, "reward_beta", beta)
        object.__setattr__(self, "generator_update_interval", interval)
        object.__setattr__(self, "student_scheduler_cadence", cadence)
        object.__setattr__(self, "ema_decay", decay)
        object.__setattr__(self, "ema_start_step", ema_start)

    @property
    def schedule(self) -> FewStepSchedule:
        return shifted_few_step_schedule(
            self.denoising_timesteps,
            num_train_timesteps=self.num_train_timesteps,
            flow_shift=self.denoising_flow_shift,
        )

    @property
    def rollout_config(self) -> SelfForcingConfig:
        return SelfForcingConfig(
            schedule=self.schedule,
            frames_per_block=self.frames_per_block,
            frame_dim=self.frame_dim,
            exit_step_mode=("sequence" if self.same_step_across_blocks else "block"),
        )

    @property
    def dmd_config(self) -> DMDConfig:
        return DMDConfig(
            schedule=self.schedule,
            num_train_timesteps=self.num_train_timesteps,
            score_min_sigma=self.score_min_sigma,
            score_max_sigma=self.score_max_sigma,
            score_flow_shift=self.score_flow_shift,
            teacher_guidance_scale=self.teacher_guidance_scale,
            normalization_epsilon=self.normalization_epsilon,
            shared_score_timestep=False,
            per_sample_normalization=True,
        )

__all__ = ["RewardForcingConfig"]
