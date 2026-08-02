"""Typed execution seams for WorldFoundry-native rCM."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from ...shared.contracts import TensorLike, freeze_mapping, non_empty_ids, tensor_shape


@dataclass(frozen=True, slots=True)
class RCMTrainingBatch:
    """Clean latent samples and both classifier-free guidance branches."""

    sample_ids: tuple[str, ...]
    clean_latents: TensorLike
    conditioning: Mapping[str, object]
    unconditional_conditioning: Mapping[str, object]

    def __post_init__(self) -> None:
        sample_ids = non_empty_ids(self.sample_ids, field_name="sample_ids", unique=True)
        shape = tensor_shape(self.clean_latents, field_name="clean_latents")
        if len(shape) < 2 or shape[0] != len(sample_ids) or any(size == 0 for size in shape[1:]):
            raise ValueError("clean_latents must be a non-empty [B,...] tensor")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(
            self,
            "conditioning",
            freeze_mapping(self.conditioning, field_name="conditioning"),
        )
        object.__setattr__(
            self,
            "unconditional_conditioning",
            freeze_mapping(
                self.unconditional_conditioning,
                field_name="unconditional_conditioning",
            ),
        )

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)


@dataclass(frozen=True, slots=True)
class RCMPrediction:
    """Clean-data and TrigFlow-velocity views of one model prediction."""

    clean_latents: TensorLike
    velocity: TensorLike


@runtime_checkable
class RCMPredictionAdapter(Protocol):
    """Bidirectional rCM model seam implemented by native model adapters."""

    module: object

    def predict(
        self,
        noisy_latents: TensorLike,
        trig_timesteps: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> RCMPrediction: ...


@runtime_checkable
class RCMExactJVPAdapter(RCMPredictionAdapter, Protocol):
    """Continuous-rCM seam; the capability bit must be explicitly true."""

    supports_exact_jvp: bool

    def predict_with_directional_derivative(
        self,
        noisy_latents: TensorLike,
        trig_timesteps: TensorLike,
        tangent_latents: TensorLike,
        tangent_timesteps: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
    ) -> tuple[RCMPrediction, TensorLike]: ...


@dataclass(frozen=True, slots=True)
class RCMLossResult:
    loss: TensorLike
    metrics: Mapping[str, object]


@runtime_checkable
class RCMLossAdapter(Protocol):
    """Objective seam consumed by the exact rCM phase state machine."""

    config_digest: str

    def loss_denominator(
        self,
        batch: RCMTrainingBatch,
        *,
        role: Literal["student", "fake-score"],
    ) -> object: ...

    def student_loss(
        self,
        batch: RCMTrainingBatch,
        *,
        iteration: int,
        effective_student_iteration: int,
        include_dmd: bool,
        generator: object | None = None,
    ) -> RCMLossResult: ...

    def fake_score_loss(
        self,
        batch: RCMTrainingBatch,
        *,
        effective_fake_iteration: int,
        generator: object | None = None,
    ) -> RCMLossResult: ...


__all__ = [
    "RCMLossAdapter",
    "RCMLossResult",
    "RCMExactJVPAdapter",
    "RCMPrediction",
    "RCMPredictionAdapter",
    "RCMTrainingBatch",
]
