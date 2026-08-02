"""Behavior contract for latent consistency distillation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..common import strict_mapping

LATENT_CONSISTENCY_ALGORITHM_FIELDS = {
    "type",
    "teacher_checkpoint",
    "num_train_timesteps",
    "num_ddim_timesteps",
    "prediction_type",
    "guidance_coefficient_min",
    "guidance_coefficient_max",
    "guidance_embedding_dim",
    "guidance_embedding_scale",
    "guidance_embedding_max_period",
    "sigma_data",
    "timestep_scaling",
    "loss_type",
    "pseudo_huber_c",
    "ema_decay",
}


def _positive_float(value: object, *, field_name: str) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return resolved


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class LatentConsistencyAlgorithmSpec:
    """Every choice executed by the LCM teacher-target objective."""

    teacher_checkpoint: str
    num_train_timesteps: int = 1000
    num_ddim_timesteps: int = 50
    prediction_type: str = "epsilon"
    guidance_coefficient_min: float = 5.0
    guidance_coefficient_max: float = 15.0
    guidance_embedding_dim: int = 256
    guidance_embedding_scale: float = 1000.0
    guidance_embedding_max_period: float = 10000.0
    sigma_data: float = 0.5
    timestep_scaling: float = 10.0
    loss_type: str = "l2"
    pseudo_huber_c: float | None = None
    ema_decay: float = 0.95
    type: str = "latent-consistency"

    def __post_init__(self) -> None:
        algorithm_type = str(self.type).strip().lower().replace("_", "-")
        if algorithm_type != "latent-consistency":
            raise ValueError(
                "latent consistency algorithm type must be 'latent-consistency'"
            )
        checkpoint = str(self.teacher_checkpoint).strip()
        if not checkpoint:
            raise ValueError("teacher_checkpoint must be non-empty")
        train_steps = _positive_int(
            self.num_train_timesteps,
            field_name="num_train_timesteps",
        )
        ddim_steps = _positive_int(
            self.num_ddim_timesteps,
            field_name="num_ddim_timesteps",
        )
        if train_steps < 2:
            raise ValueError("num_train_timesteps must be at least two")
        if ddim_steps > train_steps:
            raise ValueError(
                "num_ddim_timesteps cannot exceed num_train_timesteps"
            )
        if train_steps % ddim_steps:
            raise ValueError(
                "num_train_timesteps must be divisible by num_ddim_timesteps"
            )
        prediction_type = str(self.prediction_type).strip().lower().replace("-", "_")
        if prediction_type not in {"epsilon", "v_prediction"}:
            raise ValueError(
                "prediction_type must be 'epsilon' or 'v_prediction'"
            )
        minimum = float(self.guidance_coefficient_min)
        maximum = float(self.guidance_coefficient_max)
        if (
            not isfinite(minimum)
            or not isfinite(maximum)
            or minimum < 0
            or maximum < minimum
        ):
            raise ValueError(
                "guidance coefficient bounds must satisfy 0 <= min <= max"
            )
        embedding_dim = _positive_int(
            self.guidance_embedding_dim,
            field_name="guidance_embedding_dim",
        )
        if embedding_dim < 4:
            raise ValueError("guidance_embedding_dim must be at least four")
        embedding_scale = _positive_float(
            self.guidance_embedding_scale,
            field_name="guidance_embedding_scale",
        )
        max_period = _positive_float(
            self.guidance_embedding_max_period,
            field_name="guidance_embedding_max_period",
        )
        if max_period <= 1:
            raise ValueError("guidance_embedding_max_period must be greater than one")
        sigma_data = _positive_float(self.sigma_data, field_name="sigma_data")
        timestep_scaling = _positive_float(
            self.timestep_scaling,
            field_name="timestep_scaling",
        )
        loss_type = str(self.loss_type).strip().lower().replace("-", "_")
        if loss_type not in {"l2", "pseudo_huber"}:
            raise ValueError("loss_type must be 'l2' or 'pseudo_huber'")
        if loss_type == "l2":
            if self.pseudo_huber_c is not None:
                raise ValueError("pseudo_huber_c is unused by l2 loss")
            pseudo_huber_c = None
        else:
            pseudo_huber_c = _positive_float(
                0.001 if self.pseudo_huber_c is None else self.pseudo_huber_c,
                field_name="pseudo_huber_c",
            )
        ema_decay = float(self.ema_decay)
        if not isfinite(ema_decay) or not 0 <= ema_decay < 1:
            raise ValueError("ema_decay must be finite and lie in [0,1)")

        object.__setattr__(self, "type", algorithm_type)
        object.__setattr__(self, "teacher_checkpoint", checkpoint)
        object.__setattr__(self, "num_train_timesteps", train_steps)
        object.__setattr__(self, "num_ddim_timesteps", ddim_steps)
        object.__setattr__(self, "prediction_type", prediction_type)
        object.__setattr__(self, "guidance_coefficient_min", minimum)
        object.__setattr__(self, "guidance_coefficient_max", maximum)
        object.__setattr__(self, "guidance_embedding_dim", embedding_dim)
        object.__setattr__(self, "guidance_embedding_scale", embedding_scale)
        object.__setattr__(self, "guidance_embedding_max_period", max_period)
        object.__setattr__(self, "sigma_data", sigma_data)
        object.__setattr__(self, "timestep_scaling", timestep_scaling)
        object.__setattr__(self, "loss_type", loss_type)
        object.__setattr__(self, "pseudo_huber_c", pseudo_huber_c)
        object.__setattr__(self, "ema_decay", ema_decay)


def parse_latent_consistency_algorithm(
    value: object,
) -> LatentConsistencyAlgorithmSpec:
    return LatentConsistencyAlgorithmSpec(
        **strict_mapping(
            value,
            field_name="algorithm",
            allowed=LATENT_CONSISTENCY_ALGORITHM_FIELDS,
        )
    )


__all__ = ["LatentConsistencyAlgorithmSpec"]
