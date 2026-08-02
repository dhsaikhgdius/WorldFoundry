"""Behavior-bearing schedules and hyperparameters for latent consistency."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Literal

from worldfoundry.core.io.integrity import canonical_sha256

LatentConsistencyPredictionType = Literal["epsilon", "v_prediction"]
LatentConsistencyLossType = Literal["l2", "pseudo_huber"]


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _positive_float(value: object, *, field_name: str) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return resolved


@dataclass(frozen=True, slots=True)
class LatentConsistencyNoiseSchedule:
    """The teacher DDPM cumulative-alpha schedule consumed by distillation."""

    alpha_cumprods: tuple[float, ...]

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.alpha_cumprods)
        if len(values) < 2:
            raise ValueError("alpha_cumprods must contain at least two timesteps")
        if any(not isfinite(value) or not 0.0 < value <= 1.0 for value in values):
            raise ValueError("alpha_cumprods must be finite and lie in (0,1]")
        if any(left < right for left, right in zip(values, values[1:], strict=False)):
            raise ValueError("alpha_cumprods must be non-increasing")
        if values[0] == values[-1]:
            raise ValueError("alpha_cumprods must describe a non-degenerate noise process")
        object.__setattr__(self, "alpha_cumprods", values)

    @property
    def num_train_timesteps(self) -> int:
        return len(self.alpha_cumprods)

    @property
    def digest(self) -> str:
        return canonical_sha256(
            {
                "kind": "latent-consistency-noise-schedule",
                "alpha_cumprods": self.alpha_cumprods,
            }
        )


@dataclass(frozen=True, slots=True)
class LatentConsistencyDDIMSchedule:
    """Discrete start/end pairs and previous alphas used by the DDIM solver."""

    start_timesteps: tuple[int, ...]
    end_timesteps: tuple[int, ...]
    previous_alpha_cumprods: tuple[float, ...]
    step_size: int

    def __post_init__(self) -> None:
        count = len(self.start_timesteps)
        if count == 0 or len(self.end_timesteps) != count:
            raise ValueError("DDIM start and end schedules must be aligned and non-empty")
        if len(self.previous_alpha_cumprods) != count:
            raise ValueError("DDIM previous alphas must align with timestep pairs")
        if isinstance(self.step_size, bool) or self.step_size <= 0:
            raise ValueError("DDIM step_size must be positive")
        if any(value < 0 for value in (*self.start_timesteps, *self.end_timesteps)):
            raise ValueError("DDIM timesteps must be non-negative")
        if any(
            left >= right
            for left, right in zip(
                self.start_timesteps,
                self.start_timesteps[1:],
                strict=False,
            )
        ):
            raise ValueError("DDIM start timesteps must be strictly ascending")
        if any(end > start for start, end in zip(self.start_timesteps, self.end_timesteps, strict=True)):
            raise ValueError("each DDIM end timestep must not exceed its start")

    @property
    def pair_count(self) -> int:
        return len(self.start_timesteps)


@dataclass(frozen=True, slots=True)
class LatentConsistencyConfig:
    """The complete model-neutral latent consistency objective configuration."""

    num_ddim_timesteps: int = 50
    prediction_type: LatentConsistencyPredictionType = "epsilon"
    guidance_coefficient_min: float = 5.0
    guidance_coefficient_max: float = 15.0
    guidance_embedding_dim: int = 256
    guidance_embedding_scale: float = 1000.0
    guidance_embedding_max_period: float = 10000.0
    sigma_data: float = 0.5
    timestep_scaling: float = 10.0
    loss_type: LatentConsistencyLossType = "l2"
    pseudo_huber_c: float | None = None
    ema_decay: float = 0.95

    def __post_init__(self) -> None:
        ddim_steps = _positive_int(
            self.num_ddim_timesteps,
            field_name="num_ddim_timesteps",
        )
        if self.prediction_type not in {"epsilon", "v_prediction"}:
            raise ValueError("prediction_type must be 'epsilon' or 'v_prediction'")
        minimum = float(self.guidance_coefficient_min)
        maximum = float(self.guidance_coefficient_max)
        if not isfinite(minimum) or not isfinite(maximum) or minimum < 0 or maximum < minimum:
            raise ValueError("guidance coefficient bounds must be finite and satisfy 0 <= min <= max")
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
        if max_period <= 1.0:
            raise ValueError("guidance_embedding_max_period must be greater than one")
        sigma_data = _positive_float(self.sigma_data, field_name="sigma_data")
        timestep_scaling = _positive_float(
            self.timestep_scaling,
            field_name="timestep_scaling",
        )
        if self.loss_type not in {"l2", "pseudo_huber"}:
            raise ValueError("loss_type must be 'l2' or 'pseudo_huber'")
        if self.loss_type == "l2":
            if self.pseudo_huber_c is not None:
                raise ValueError("pseudo_huber_c is unused by l2 loss")
            huber_c = None
        else:
            huber_c = _positive_float(
                0.001 if self.pseudo_huber_c is None else self.pseudo_huber_c,
                field_name="pseudo_huber_c",
            )
        ema_decay = float(self.ema_decay)
        if not isfinite(ema_decay) or not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be finite and lie in [0,1)")

        object.__setattr__(self, "num_ddim_timesteps", ddim_steps)
        object.__setattr__(self, "guidance_coefficient_min", minimum)
        object.__setattr__(self, "guidance_coefficient_max", maximum)
        object.__setattr__(self, "guidance_embedding_dim", embedding_dim)
        object.__setattr__(self, "guidance_embedding_scale", embedding_scale)
        object.__setattr__(self, "guidance_embedding_max_period", max_period)
        object.__setattr__(self, "sigma_data", sigma_data)
        object.__setattr__(self, "timestep_scaling", timestep_scaling)
        object.__setattr__(self, "pseudo_huber_c", huber_c)
        object.__setattr__(self, "ema_decay", ema_decay)

    @property
    def digest(self) -> str:
        return canonical_sha256({"kind": "latent-consistency-config", **asdict(self)})


def build_latent_consistency_ddim_schedule(
    noise_schedule: LatentConsistencyNoiseSchedule,
    config: LatentConsistencyConfig,
) -> LatentConsistencyDDIMSchedule:
    """Build the exact evenly-spaced DDIM pairs used for teacher targets."""

    if not isinstance(noise_schedule, LatentConsistencyNoiseSchedule):
        raise TypeError("noise_schedule must be LatentConsistencyNoiseSchedule")
    if not isinstance(config, LatentConsistencyConfig):
        raise TypeError("config must be LatentConsistencyConfig")
    train_steps = noise_schedule.num_train_timesteps
    ddim_steps = config.num_ddim_timesteps
    if ddim_steps > train_steps:
        raise ValueError("num_ddim_timesteps cannot exceed the teacher schedule length")
    if train_steps % ddim_steps:
        raise ValueError("the teacher schedule length must be divisible by num_ddim_timesteps")
    step_size = train_steps // ddim_steps
    starts = tuple((index + 1) * step_size - 1 for index in range(ddim_steps))
    ends = (0, *starts[:-1])
    alpha_cumprods = noise_schedule.alpha_cumprods
    previous_alphas = (
        alpha_cumprods[0],
        *(alpha_cumprods[timestep] for timestep in starts[:-1]),
    )
    return LatentConsistencyDDIMSchedule(
        start_timesteps=starts,
        end_timesteps=ends,
        previous_alpha_cumprods=previous_alphas,
        step_size=step_size,
    )


__all__ = [
    "LatentConsistencyConfig",
    "LatentConsistencyDDIMSchedule",
    "LatentConsistencyLossType",
    "LatentConsistencyNoiseSchedule",
    "LatentConsistencyPredictionType",
    "build_latent_consistency_ddim_schedule",
]
