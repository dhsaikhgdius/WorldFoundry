"""Pure recipe contract for Reward-Forcing Re-DMD training."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..common import strict_mapping
from .auxiliary_optimizers import (
    AuxiliaryOptimizerRule,
    forbids_auxiliary,
    requires_auxiliary,
)

REWARD_FORCING_ALGORITHM_FIELDS = {
    "type",
    "real_score_checkpoint",
    "fake_score_checkpoint",
    "reward_decoder_checkpoint",
    "motion_reward_checkpoint",
    "motion_reward_calibration_mean",
    "motion_reward_calibration_std",
    "motion_reward_normalization_epsilon",
    "denoising_timesteps",
    "num_train_timesteps",
    "denoising_flow_shift",
    "frames_per_block",
    "training_frames",
    "frame_dim",
    "same_step_across_blocks",
    "local_attention_frames",
    "ema_sink_frames",
    "ema_sink_decay",
    "score_min_sigma",
    "score_max_sigma",
    "score_flow_shift",
    "teacher_guidance_scale",
    "normalization_epsilon",
    "reward_beta",
    "generator_update_interval",
    "student_scheduler_cadence",
    "ema_decay",
    "ema_start_step",
}


def _finite_float(value: object, *, field_name: str) -> float:
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    return resolved


def _positive_float(value: object, *, field_name: str) -> float:
    resolved = _finite_float(value, field_name=field_name)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive")
    return resolved


def _non_negative_float(value: object, *, field_name: str) -> float:
    resolved = _finite_float(value, field_name=field_name)
    if resolved < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return resolved


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"{field_name} must be a positive integer")
    return value


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{field_name} must be a non-negative integer")
    return value


def _role_reference(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class RewardForcingAlgorithmSpec:
    """Strict behavior and independently loaded roles for rewarded DMD."""

    real_score_checkpoint: str = "default"
    fake_score_checkpoint: str = "default"
    reward_decoder_checkpoint: str = "default"
    motion_reward_checkpoint: str = "default"
    motion_reward_calibration_mean: float = 1.1646
    motion_reward_calibration_std: float = 1.3811
    motion_reward_normalization_epsilon: float = 0.0
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
    type: str = "reward-forcing"

    def __post_init__(self) -> None:
        algorithm_type = str(self.type).strip().lower().replace("_", "-")
        if algorithm_type != "reward-forcing":
            raise ValueError("Reward-Forcing algorithm type must be 'reward-forcing'")
        object.__setattr__(self, "type", algorithm_type)

        for name in (
            "real_score_checkpoint",
            "fake_score_checkpoint",
            "reward_decoder_checkpoint",
            "motion_reward_checkpoint",
        ):
            object.__setattr__(
                self,
                name,
                _role_reference(
                    getattr(self, name),
                    field_name=f"algorithm.{name}",
                ),
            )

        train_timesteps = _positive_int(
            self.num_train_timesteps,
            field_name="algorithm.num_train_timesteps",
        )
        timesteps = tuple(float(value) for value in self.denoising_timesteps)
        if not timesteps or any(not isfinite(value) or value <= 0 or value > train_timesteps for value in timesteps):
            raise ValueError("algorithm.denoising_timesteps must be finite values in (0,num_train_timesteps]")
        if any(left <= right for left, right in zip(timesteps, timesteps[1:])):
            raise ValueError("algorithm.denoising_timesteps must be strictly descending")
        object.__setattr__(self, "num_train_timesteps", train_timesteps)
        object.__setattr__(self, "denoising_timesteps", timesteps)
        object.__setattr__(
            self,
            "denoising_flow_shift",
            _positive_float(
                self.denoising_flow_shift,
                field_name="algorithm.denoising_flow_shift",
            ),
        )

        blocks = _positive_int(
            self.frames_per_block,
            field_name="algorithm.frames_per_block",
        )
        frames = _positive_int(
            self.training_frames,
            field_name="algorithm.training_frames",
        )
        if frames % blocks:
            raise ValueError("algorithm.training_frames must be divisible by frames_per_block")
        if isinstance(self.frame_dim, bool) or not isinstance(self.frame_dim, int):
            raise TypeError("algorithm.frame_dim must be an integer")
        if self.frame_dim == 0:
            raise ValueError("algorithm.frame_dim cannot be the batch dimension")
        if not isinstance(self.same_step_across_blocks, bool):
            raise TypeError("algorithm.same_step_across_blocks must be a bool")
        local_frames = _positive_int(
            self.local_attention_frames,
            field_name="algorithm.local_attention_frames",
        )
        sink_frames = _positive_int(
            self.ema_sink_frames,
            field_name="algorithm.ema_sink_frames",
        )
        if sink_frames != blocks:
            raise ValueError("Reward-Forcing EMA-Sink requires ema_sink_frames to equal frames_per_block")
        if sink_frames >= local_frames or local_frames % blocks:
            raise ValueError("algorithm.local_attention_frames must be a block-aligned window larger than EMA-Sink")
        object.__setattr__(self, "frames_per_block", blocks)
        object.__setattr__(self, "training_frames", frames)
        object.__setattr__(self, "local_attention_frames", local_frames)
        object.__setattr__(self, "ema_sink_frames", sink_frames)

        for name in ("ema_sink_decay", "ema_decay"):
            decay = _finite_float(
                getattr(self, name),
                field_name=f"algorithm.{name}",
            )
            if not 0 < decay < 1:
                raise ValueError(f"algorithm.{name} must be in (0,1)")
            object.__setattr__(self, name, decay)

        minimum = _finite_float(
            self.score_min_sigma,
            field_name="algorithm.score_min_sigma",
        )
        maximum = _finite_float(
            self.score_max_sigma,
            field_name="algorithm.score_max_sigma",
        )
        if not 0 <= minimum < maximum <= 1:
            raise ValueError("algorithm score sigma bounds must satisfy 0 <= min < max <= 1")
        object.__setattr__(self, "score_min_sigma", minimum)
        object.__setattr__(self, "score_max_sigma", maximum)
        object.__setattr__(
            self,
            "score_flow_shift",
            _positive_float(
                self.score_flow_shift,
                field_name="algorithm.score_flow_shift",
            ),
        )
        object.__setattr__(
            self,
            "teacher_guidance_scale",
            _finite_float(
                self.teacher_guidance_scale,
                field_name="algorithm.teacher_guidance_scale",
            ),
        )
        object.__setattr__(
            self,
            "normalization_epsilon",
            _non_negative_float(
                self.normalization_epsilon,
                field_name="algorithm.normalization_epsilon",
            ),
        )
        object.__setattr__(
            self,
            "reward_beta",
            _non_negative_float(
                self.reward_beta,
                field_name="algorithm.reward_beta",
            ),
        )
        object.__setattr__(
            self,
            "generator_update_interval",
            _positive_int(
                self.generator_update_interval,
                field_name="algorithm.generator_update_interval",
            ),
        )
        cadence = str(self.student_scheduler_cadence).strip().lower().replace("_", "-")
        if cadence not in {"iteration", "generator-update"}:
            raise ValueError("algorithm.student_scheduler_cadence must be 'iteration' or 'generator-update'")
        object.__setattr__(self, "student_scheduler_cadence", cadence)
        ema_start = _non_negative_int(
            self.ema_start_step,
            field_name="algorithm.ema_start_step",
        )
        object.__setattr__(self, "ema_start_step", ema_start)

        object.__setattr__(
            self,
            "motion_reward_calibration_mean",
            _finite_float(
                self.motion_reward_calibration_mean,
                field_name="algorithm.motion_reward_calibration_mean",
            ),
        )
        object.__setattr__(
            self,
            "motion_reward_calibration_std",
            _positive_float(
                self.motion_reward_calibration_std,
                field_name="algorithm.motion_reward_calibration_std",
            ),
        )
        object.__setattr__(
            self,
            "motion_reward_normalization_epsilon",
            _non_negative_float(
                self.motion_reward_normalization_epsilon,
                field_name="algorithm.motion_reward_normalization_epsilon",
            ),
        )

    def auxiliary_optimizer_rules(self) -> tuple[AuxiliaryOptimizerRule, ...]:
        return (
            requires_auxiliary("fake_score_optimizer", "Reward-Forcing requires fake_score_optimizer"),
            forbids_auxiliary(
                "guidance_optimizer",
                "discriminator_optimizer",
                message="Reward-Forcing only accepts fake_score_optimizer",
            ),
        )


def parse_reward_forcing_algorithm(value: object) -> RewardForcingAlgorithmSpec:
    """Parse a strict Reward-Forcing algorithm section."""

    return RewardForcingAlgorithmSpec(
        **strict_mapping(
            value,
            field_name="algorithm",
            allowed=REWARD_FORCING_ALGORITHM_FIELDS,
        )
    )


__all__ = ["RewardForcingAlgorithmSpec", "parse_reward_forcing_algorithm"]
