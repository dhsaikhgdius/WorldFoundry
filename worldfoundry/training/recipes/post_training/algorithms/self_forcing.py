"""Pure recipe contract for causal Self-Forcing distribution training."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..common import positive_int, strict_mapping

SELF_FORCING_ALGORITHM_FIELDS = {
    "type",
    "distribution_objective",
    "real_score_model_recipe",
    "real_score_checkpoint",
    "fake_score_model_recipe",
    "fake_score_checkpoint",
    "denoising_timesteps",
    "denoising_flow_shift",
    "num_train_timesteps",
    "frames_per_block",
    "frame_dim",
    "exit_step_mode",
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
class SelfForcingAlgorithmSpec:
    """Official causal rollout with a holistic DMD distribution objective."""

    denoising_timesteps: tuple[float, ...]
    distribution_objective: str = "dmd"
    real_score_model_recipe: str = "wan2.1-t2v-14b"
    real_score_checkpoint: str = "default"
    fake_score_model_recipe: str = "wan2.1-t2v-1.3b"
    fake_score_checkpoint: str = "default"
    denoising_flow_shift: float = 5.0
    num_train_timesteps: int = 1000
    frames_per_block: int = 3
    frame_dim: int = 2
    exit_step_mode: str = "sequence"
    score_min_sigma: float = 0.02
    score_max_sigma: float = 0.98
    score_flow_shift: float = 5.0
    teacher_guidance_scale: float = 3.0
    normalization_epsilon: float = 0.0
    generator_update_interval: int = 5
    student_scheduler_cadence: str = "iteration"
    ema_decay: float = 0.99
    ema_start_step: int = 200
    type: str = "self-forcing"

    def __post_init__(self) -> None:
        algorithm_type = str(self.type).strip().lower().replace("_", "-")
        if algorithm_type != "self-forcing":
            raise ValueError("Self-Forcing algorithm type must be 'self-forcing'")
        objective = str(self.distribution_objective).strip().lower().replace("_", "-")
        if objective != "dmd":
            raise ValueError("Self-Forcing currently supports distribution_objective='dmd' only")
        role_strings: dict[str, str] = {}
        for name in (
            "real_score_model_recipe",
            "real_score_checkpoint",
            "fake_score_model_recipe",
            "fake_score_checkpoint",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            normalized = value.strip()
            if name.endswith("model_recipe"):
                normalized = normalized.lower().replace("_", "-")
            role_strings[name] = normalized
        timesteps = tuple(float(value) for value in self.denoising_timesteps)
        if not timesteps or any(not isfinite(value) or value <= 0 for value in timesteps):
            raise ValueError("denoising_timesteps must be non-empty, finite, and positive")
        if any(left <= right for left, right in zip(timesteps, timesteps[1:])):
            raise ValueError("denoising_timesteps must be strictly descending")
        flow_shift = float(self.denoising_flow_shift)
        score_shift = float(self.score_flow_shift)
        if not isfinite(flow_shift) or flow_shift <= 0:
            raise ValueError("denoising_flow_shift must be finite and positive")
        if not isfinite(score_shift) or score_shift <= 0:
            raise ValueError("score_flow_shift must be finite and positive")
        minimum = float(self.score_min_sigma)
        maximum = float(self.score_max_sigma)
        if not isfinite(minimum) or not isfinite(maximum) or not 0 <= minimum < maximum <= 1:
            raise ValueError("score sigma bounds must satisfy 0 <= min < max <= 1")
        guidance_scale = float(self.teacher_guidance_scale)
        if not isfinite(guidance_scale):
            raise ValueError("teacher_guidance_scale must be finite")
        epsilon = float(self.normalization_epsilon)
        if not isfinite(epsilon) or epsilon < 0:
            raise ValueError("normalization_epsilon must be finite and non-negative")
        mode = str(self.exit_step_mode).strip().lower().replace("_", "-")
        if mode not in {"sequence", "block"}:
            raise ValueError("exit_step_mode must be 'sequence' or 'block'")
        if isinstance(self.frame_dim, bool):
            raise TypeError("frame_dim must be an integer")
        frame_dim = int(self.frame_dim)
        if frame_dim == 0:
            raise ValueError("frame_dim cannot be the batch dimension")
        cadence = str(self.student_scheduler_cadence).strip().lower().replace("_", "-")
        if cadence not in {"iteration", "generator-update"}:
            raise ValueError("student_scheduler_cadence must be 'iteration' or 'generator-update'")
        decay = float(self.ema_decay)
        if not isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("ema_decay must be finite and in [0,1)")
        if isinstance(self.ema_start_step, bool):
            raise TypeError("ema_start_step must be a non-negative integer")
        ema_start_step = int(self.ema_start_step)
        if ema_start_step < 0:
            raise ValueError("ema_start_step must be a non-negative integer")
        train_timesteps = positive_int(
            self.num_train_timesteps,
            field_name="algorithm.num_train_timesteps",
        )
        if any(value > train_timesteps for value in timesteps):
            raise ValueError("denoising_timesteps cannot exceed num_train_timesteps")
        object.__setattr__(self, "type", algorithm_type)
        object.__setattr__(self, "distribution_objective", objective)
        for name, value in role_strings.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "denoising_timesteps", timesteps)
        object.__setattr__(self, "denoising_flow_shift", flow_shift)
        object.__setattr__(self, "score_min_sigma", minimum)
        object.__setattr__(self, "score_max_sigma", maximum)
        object.__setattr__(self, "score_flow_shift", score_shift)
        object.__setattr__(self, "teacher_guidance_scale", guidance_scale)
        object.__setattr__(self, "normalization_epsilon", epsilon)
        object.__setattr__(self, "num_train_timesteps", train_timesteps)
        object.__setattr__(
            self,
            "frames_per_block",
            positive_int(self.frames_per_block, field_name="algorithm.frames_per_block"),
        )
        object.__setattr__(self, "frame_dim", frame_dim)
        object.__setattr__(self, "exit_step_mode", mode)
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
        object.__setattr__(self, "ema_start_step", ema_start_step)


def parse_self_forcing_algorithm(value: object) -> SelfForcingAlgorithmSpec:
    """Parse a strict Self-Forcing algorithm section."""

    return SelfForcingAlgorithmSpec(
        **strict_mapping(
            value,
            field_name="algorithm",
            allowed=SELF_FORCING_ALGORITHM_FIELDS,
        )
    )


__all__ = ["SelfForcingAlgorithmSpec"]
