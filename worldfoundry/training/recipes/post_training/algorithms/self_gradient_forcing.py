"""Behavior contract for causal Self-Gradient-Forcing distillation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..common import positive_int, strict_mapping

SELF_GRADIENT_FORCING_ALGORITHM_FIELDS = {
    "type",
    "real_score_checkpoint",
    "fake_score_checkpoint",
    "denoising_timesteps",
    "denoising_flow_shift",
    "num_train_timesteps",
    "frames_per_block",
    "frame_dim",
    "context_timestep",
    "cache_target_mode",
    "exit_step_rank_mode",
    "match_context",
    "last_step_only",
    "score_min_sigma",
    "score_max_sigma",
    "score_flow_shift",
    "teacher_guidance_scale",
    "normalization_epsilon",
    "generator_update_interval",
    "student_scheduler_cadence",
    "ema_decay",
    "ema_start_step",
}


@dataclass(frozen=True, slots=True)
class SelfGradientForcingAlgorithmSpec:
    """Two-pass causal replay, DMD scoring, cadence, and EMA choices."""

    real_score_checkpoint: str
    fake_score_checkpoint: str
    denoising_timesteps: tuple[float, ...] = (1000.0, 750.0, 500.0, 250.0)
    denoising_flow_shift: float = 5.0
    num_train_timesteps: int = 1000
    frames_per_block: int = 3
    frame_dim: int = 2
    context_timestep: float = 0.0
    cache_target_mode: str = "exit"
    exit_step_rank_mode: str = "local"
    match_context: bool = True
    last_step_only: bool = False
    score_min_sigma: float = 0.02
    score_max_sigma: float = 0.98
    score_flow_shift: float = 5.0
    teacher_guidance_scale: float = 3.0
    normalization_epsilon: float = 0.0
    generator_update_interval: int = 5
    student_scheduler_cadence: str = "iteration"
    ema_decay: float = 0.99
    ema_start_step: int = 200
    type: str = "self-gradient-forcing"

    def __post_init__(self) -> None:
        algorithm_type = str(self.type).strip().lower().replace("_", "-")
        if algorithm_type != "self-gradient-forcing":
            raise ValueError(
                "Self-Gradient-Forcing type must be 'self-gradient-forcing'"
            )
        for name in ("real_score_checkpoint", "fake_score_checkpoint"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty checkpoint reference")
            object.__setattr__(self, name, value.strip())
        timesteps = tuple(float(value) for value in self.denoising_timesteps)
        if not timesteps or any(not isfinite(value) or value <= 0 for value in timesteps):
            raise ValueError(
                "denoising_timesteps must be non-empty, finite, and positive"
            )
        if any(left <= right for left, right in zip(timesteps, timesteps[1:])):
            raise ValueError("denoising_timesteps must be strictly descending")
        num_train_timesteps = positive_int(
            self.num_train_timesteps,
            field_name="algorithm.num_train_timesteps",
        )
        if any(value > num_train_timesteps for value in timesteps):
            raise ValueError("denoising_timesteps exceed the training timeline")
        denoising_shift = float(self.denoising_flow_shift)
        score_shift = float(self.score_flow_shift)
        if not isfinite(denoising_shift) or denoising_shift <= 0:
            raise ValueError("denoising_flow_shift must be finite and positive")
        if not isfinite(score_shift) or score_shift <= 0:
            raise ValueError("score_flow_shift must be finite and positive")
        context_timestep = float(self.context_timestep)
        if (
            not isfinite(context_timestep)
            or not 0 <= context_timestep <= num_train_timesteps
        ):
            raise ValueError("context_timestep must lie on the training timeline")
        cache_mode = str(self.cache_target_mode).strip().lower().replace("_", "-")
        if cache_mode not in {"exit", "final-clean"}:
            raise ValueError("cache_target_mode must be 'exit' or 'final-clean'")
        rank_mode = str(self.exit_step_rank_mode).strip().lower().replace("_", "-")
        if rank_mode not in {"local", "synchronized"}:
            raise ValueError("exit_step_rank_mode must be 'local' or 'synchronized'")
        if not isinstance(self.match_context, bool) or not isinstance(
            self.last_step_only,
            bool,
        ):
            raise TypeError("match_context and last_step_only must be bool values")
        if isinstance(self.frame_dim, bool) or not isinstance(self.frame_dim, int):
            raise TypeError("frame_dim must be an integer")
        if self.frame_dim == 0:
            raise ValueError("frame_dim cannot be the batch dimension")
        minimum = float(self.score_min_sigma)
        maximum = float(self.score_max_sigma)
        if (
            not isfinite(minimum)
            or not isfinite(maximum)
            or not 0 <= minimum < maximum <= 1
        ):
            raise ValueError("score sigma bounds must satisfy 0 <= min < max <= 1")
        guidance = float(self.teacher_guidance_scale)
        epsilon = float(self.normalization_epsilon)
        if not isfinite(guidance):
            raise ValueError("teacher_guidance_scale must be finite")
        if not isfinite(epsilon) or epsilon < 0:
            raise ValueError("normalization_epsilon must be finite and non-negative")
        cadence = str(self.student_scheduler_cadence).strip().lower().replace("_", "-")
        if cadence not in {"iteration", "generator-update"}:
            raise ValueError(
                "student_scheduler_cadence must be 'iteration' or 'generator-update'"
            )
        decay = float(self.ema_decay)
        if not isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("ema_decay must be finite and in [0,1)")
        if isinstance(self.ema_start_step, bool) or not isinstance(
            self.ema_start_step,
            int,
        ):
            raise TypeError("ema_start_step must be an integer")
        if self.ema_start_step < 0:
            raise ValueError("ema_start_step must be non-negative")
        object.__setattr__(self, "type", algorithm_type)
        object.__setattr__(self, "denoising_timesteps", timesteps)
        object.__setattr__(self, "num_train_timesteps", num_train_timesteps)
        object.__setattr__(self, "denoising_flow_shift", denoising_shift)
        object.__setattr__(self, "score_flow_shift", score_shift)
        object.__setattr__(self, "context_timestep", context_timestep)
        object.__setattr__(self, "cache_target_mode", cache_mode)
        object.__setattr__(self, "exit_step_rank_mode", rank_mode)
        object.__setattr__(self, "score_min_sigma", minimum)
        object.__setattr__(self, "score_max_sigma", maximum)
        object.__setattr__(self, "teacher_guidance_scale", guidance)
        object.__setattr__(self, "normalization_epsilon", epsilon)
        object.__setattr__(
            self,
            "frames_per_block",
            positive_int(
                self.frames_per_block,
                field_name="algorithm.frames_per_block",
            ),
        )
        object.__setattr__(
            self,
            "generator_update_interval",
            positive_int(
                self.generator_update_interval,
                field_name="algorithm.generator_update_interval",
            ),
        )
        object.__setattr__(self, "student_scheduler_cadence", cadence)
        object.__setattr__(self, "ema_decay", decay)


def parse_self_gradient_forcing_algorithm(
    value: object,
) -> SelfGradientForcingAlgorithmSpec:
    return SelfGradientForcingAlgorithmSpec(
        **strict_mapping(
            value,
            field_name="algorithm",
            allowed=SELF_GRADIENT_FORCING_ALGORITHM_FIELDS,
        )
    )


__all__ = ["SelfGradientForcingAlgorithmSpec"]
