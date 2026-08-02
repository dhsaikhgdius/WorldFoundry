"""Functional contracts for native sCM-LADD distillation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ...shared.contracts import TensorLike, freeze_mapping, non_empty_ids, tensor_shape


@dataclass(frozen=True, slots=True)
class SCMLADDTrainingBatch:
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
class SCMVelocityPrediction:
    """TrigFlow velocity and the optional learned sCM log-variance."""

    velocity: TensorLike
    log_variance: TensorLike | None = None


@runtime_checkable
class TrigFlowPredictionAdapter(Protocol):
    """Model-family seam used by continuous-time consistency distillation."""

    module: object
    checkpoint_identity: str

    def predict_velocity(
        self,
        scaled_noisy_latents: TensorLike,
        trig_timesteps: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        guidance_embedding_scale: float,
        return_log_variance: bool = False,
        branch: str = "positive",
    ) -> SCMVelocityPrediction: ...


@runtime_checkable
class SCMLADDDiscriminatorAdapter(Protocol):
    """Frozen teacher features plus trainable latent discriminator heads."""

    module: object
    feature_module: object
    head_block_ids: tuple[int, ...]

    def predict_logits(
        self,
        scaled_noisy_latents: TensorLike,
        trig_timesteps: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        head_block_ids: tuple[int, ...],
    ) -> TensorLike: ...


@dataclass(frozen=True, slots=True)
class SCMLADDLossResult:
    loss: TensorLike
    metrics: Mapping[str, object]


@runtime_checkable
class SCMLADDLossAdapter(Protocol):
    """Loss seam consumed by the alternating native optimizer engine."""

    config_digest: str

    def loss_denominator(
        self,
        batch: SCMLADDTrainingBatch,
        *,
        role: str,
    ) -> object: ...

    def generator_loss(
        self,
        batch: SCMLADDTrainingBatch,
        *,
        training_iteration: int,
        generator: object | None = None,
    ) -> SCMLADDLossResult: ...

    def discriminator_loss(
        self,
        batch: SCMLADDTrainingBatch,
        *,
        generator: object | None = None,
    ) -> SCMLADDLossResult: ...


__all__ = [
    "SCMLADDDiscriminatorAdapter",
    "SCMLADDLossAdapter",
    "SCMLADDLossResult",
    "SCMLADDTrainingBatch",
    "SCMVelocityPrediction",
    "TrigFlowPredictionAdapter",
]
