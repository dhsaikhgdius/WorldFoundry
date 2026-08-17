"""Typed model seams and condition-matched batches for native DFD."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from ...shared.contracts import TensorLike, freeze_mapping, non_empty_ids, tensor_shape


@dataclass(frozen=True, slots=True)
class DFDTrainingBatch:
    """Real latents and the exact conditions used to generate their paired samples."""

    sample_ids: tuple[str, ...]
    real_latents: TensorLike
    conditioning: Mapping[str, object]
    unconditional_conditioning: Mapping[str, object]
    sample_weights: TensorLike | None = None

    def __post_init__(self) -> None:
        sample_ids = non_empty_ids(self.sample_ids, field_name="sample_ids", unique=True)
        shape = tensor_shape(self.real_latents, field_name="real_latents")
        if len(shape) < 2 or shape[0] != len(sample_ids) or any(size == 0 for size in shape[1:]):
            raise ValueError("real_latents must be a non-empty [B,...] tensor")
        if self.sample_weights is not None and tensor_shape(
            self.sample_weights,
            field_name="sample_weights",
        ) != (len(sample_ids),):
            raise ValueError("sample_weights must have shape [B]")
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


@runtime_checkable
class DFDPredictionAdapter(Protocol):
    """Clean prediction and forward corruption implemented by each model family."""

    module: object
    checkpoint_identity: str
    noise_process_kind: str

    def add_noise(
        self,
        clean_latents: TensorLike,
        noise: TensorLike,
        timesteps: TensorLike,
    ) -> TensorLike: ...

    def predict_clean(
        self,
        noisy_latents: TensorLike,
        timesteps: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> TensorLike: ...


@runtime_checkable
class DFDFakeScoreAdapter(DFDPredictionAdapter, Protocol):
    """Fake-score denoising objective in the model's native prediction space."""

    def denoising_loss_per_sample(
        self,
        clean_latents: TensorLike,
        noisy_latents: TensorLike,
        noise: TensorLike,
        timesteps: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
    ) -> TensorLike: ...


@runtime_checkable
class DFDDiscriminatorAdapter(Protocol):
    """Trainable discriminator head over the teacher's model-family features."""

    module: object
    checkpoint_identity: str

    def discriminator_logits(
        self,
        noisy_latents: TensorLike,
        timesteps: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
    ) -> TensorLike: ...


@runtime_checkable
class DFDLossAdapter(Protocol):
    data_forcing_probability: float
    student_update_frequency: int

    def loss_denominator(
        self,
        batch: DFDTrainingBatch,
        *,
        role: Literal["student", "guidance"],
    ) -> object: ...

    def student_loss(
        self,
        batch: DFDTrainingBatch,
        *,
        data_forcing: bool,
        generator: object | None = None,
    ) -> object: ...

    def guidance_loss(
        self,
        batch: DFDTrainingBatch,
        *,
        generator: object | None = None,
    ) -> object: ...


__all__ = [
    "DFDDiscriminatorAdapter",
    "DFDFakeScoreAdapter",
    "DFDLossAdapter",
    "DFDPredictionAdapter",
    "DFDTrainingBatch",
]
