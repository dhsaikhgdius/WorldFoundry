"""Strict recipe contract for T2V-Turbo consistency distillation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..common import strict_mapping

T2V_TURBO_ALGORITHM_FIELDS = {
    "type",
    "teacher_checkpoint",
    "num_train_timesteps",
    "num_ddim_timesteps",
    "topk",
    "guidance_min",
    "guidance_max",
    "guidance_embedding_dim",
    "sigma_data",
    "timestep_scaling",
    "loss_type",
    "pseudo_huber_c",
    "distillation_weight",
    "image_reward_weight",
    "video_reward_weight",
    "image_reward_frames",
    "image_reward_batch_size",
    "video_reward_frames",
    "default_fps",
}


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{field_name} must be positive")
    return int(value)


@dataclass(frozen=True, slots=True)
class T2VTurboAlgorithmSpec:
    """All update-defining values in the released distillation loop."""

    teacher_checkpoint: str = "default"
    num_train_timesteps: int = 1000
    num_ddim_timesteps: int = 50
    topk: int = 20
    guidance_min: float = 5.0
    guidance_max: float = 15.0
    guidance_embedding_dim: int = 256
    sigma_data: float = 0.5
    timestep_scaling: float = 10.0
    loss_type: str = "pseudo_huber"
    pseudo_huber_c: float = 0.001
    distillation_weight: float = 1.0
    image_reward_weight: float = 0.0
    video_reward_weight: float = 0.0
    image_reward_frames: int = 5
    image_reward_batch_size: int = 1
    video_reward_frames: int = 8
    default_fps: int = 16
    type: str = "t2v-turbo-distillation"

    def __post_init__(self) -> None:
        algorithm_type = str(self.type).strip().lower().replace("_", "-")
        if algorithm_type != "t2v-turbo-distillation":
            raise ValueError("T2V-Turbo algorithm type must be 't2v-turbo-distillation'")
        checkpoint = str(self.teacher_checkpoint).strip()
        if not checkpoint:
            raise ValueError("teacher_checkpoint cannot be empty")
        for name in (
            "num_train_timesteps",
            "num_ddim_timesteps",
            "topk",
            "guidance_embedding_dim",
            "image_reward_frames",
            "image_reward_batch_size",
            "video_reward_frames",
            "default_fps",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), field_name=name))
        if self.num_train_timesteps % self.num_ddim_timesteps:
            raise ValueError("num_train_timesteps must be divisible by num_ddim_timesteps")
        minimum = float(self.guidance_min)
        maximum = float(self.guidance_max)
        if not isfinite(minimum) or not isfinite(maximum) or minimum < 0.0 or maximum < minimum:
            raise ValueError("guidance bounds must satisfy 0 <= min <= max")
        object.__setattr__(self, "guidance_min", minimum)
        object.__setattr__(self, "guidance_max", maximum)
        for name in ("sigma_data", "timestep_scaling", "pseudo_huber_c"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        for name in ("distillation_weight", "image_reward_weight", "video_reward_weight"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.distillation_weight + self.image_reward_weight + self.video_reward_weight == 0.0:
            raise ValueError("at least one T2V-Turbo objective weight must be positive")
        loss_type = str(self.loss_type).lower().replace("-", "_")
        if loss_type not in {"l2", "pseudo_huber"}:
            raise ValueError("loss_type must be l2 or pseudo_huber")
        object.__setattr__(self, "loss_type", loss_type)
        object.__setattr__(self, "type", algorithm_type)
        object.__setattr__(self, "teacher_checkpoint", checkpoint)


def parse_t2v_turbo_algorithm(value: object) -> T2VTurboAlgorithmSpec:
    return T2VTurboAlgorithmSpec(
        **strict_mapping(
            value,
            field_name="algorithm",
            allowed=T2V_TURBO_ALGORITHM_FIELDS,
        )
    )


__all__ = ["T2VTurboAlgorithmSpec", "parse_t2v_turbo_algorithm"]
