"""Functional model and batch seams for native DMD2 training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from ...shared.contracts import TensorLike, freeze_mapping, non_empty_ids, tensor_shape


@dataclass(frozen=True, slots=True)
class DMD2TrainingBatch:
    """Generated prompts paired with real latent samples for one DMD2 update."""

    sample_ids: tuple[str, ...]
    real_sample_ids: tuple[str, ...]
    real_latents: TensorLike
    conditioning: Mapping[str, object]
    unconditional_conditioning: Mapping[str, object]
    real_conditioning: Mapping[str, object]
    sample_weights: TensorLike | None = None

    def __post_init__(self) -> None:
        sample_ids = non_empty_ids(self.sample_ids, field_name="sample_ids", unique=True)
        real_sample_ids = non_empty_ids(
            self.real_sample_ids,
            field_name="real_sample_ids",
            unique=True,
        )
        if len(sample_ids) != len(real_sample_ids):
            raise ValueError("DMD2 generated and real batches must have equal batch size")
        shape = tensor_shape(self.real_latents, field_name="real_latents")
        if len(shape) < 2 or shape[0] != len(sample_ids) or any(size == 0 for size in shape[1:]):
            raise ValueError("real_latents must be a non-empty [B,...] tensor")
        if self.sample_weights is not None and tensor_shape(
            self.sample_weights,
            field_name="sample_weights",
        ) != (len(sample_ids),):
            raise ValueError("sample_weights must have shape [B]")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "real_sample_ids", real_sample_ids)
        for name in ("conditioning", "unconditional_conditioning", "real_conditioning"):
            object.__setattr__(
                self,
                name,
                freeze_mapping(getattr(self, name), field_name=name),
            )

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)


@runtime_checkable
class DMD2PredictionAdapter(Protocol):
    """Clean-prediction and corruption seam implemented by each model family."""

    module: object
    noise_process_kind: str
    checkpoint_identity: str

    def add_noise(
        self,
        clean_latents: TensorLike,
        noise: TensorLike,
        noise_levels: TensorLike,
    ) -> TensorLike: ...

    def predict_clean(
        self,
        noisy_latents: TensorLike,
        noise_levels: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> TensorLike: ...


@runtime_checkable
class DMD2GuidanceAdapter(DMD2PredictionAdapter, Protocol):
    """Shared fake-score backbone and discriminator-head functional seam."""

    def denoising_loss_per_sample(
        self,
        clean_latents: TensorLike,
        noisy_latents: TensorLike,
        noise: TensorLike,
        noise_levels: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
    ) -> TensorLike: ...


@runtime_checkable
class DMD2FusedGuidanceAdapter(DMD2GuidanceAdapter, Protocol):
    """Optional one-backbone-forward seam for coupled score and GAN losses."""

    def predict_clean_and_logits(
        self,
        noisy_latents: TensorLike,
        noise_levels: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
    ) -> tuple[TensorLike, TensorLike]: ...

    def denoising_loss_from_clean_per_sample(
        self,
        clean_latents: TensorLike,
        predicted_clean: TensorLike,
        noise_levels: TensorLike,
        *,
        conditioning: Mapping[str, object],
    ) -> TensorLike: ...

    def discriminator_logits(
        self,
        latents: TensorLike,
        noise_levels: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
    ) -> TensorLike: ...


@runtime_checkable
class DMD2LossAdapter(Protocol):
    """Loss seam consumed by the two-optimizer DMD2 engine."""

    def loss_denominator(
        self,
        batch: DMD2TrainingBatch,
        *,
        role: Literal["generator", "guidance"],
    ) -> object: ...

    def generator_loss(
        self,
        batch: DMD2TrainingBatch,
        *,
        generator: object | None = None,
    ) -> object: ...

    def guidance_loss(
        self,
        batch: DMD2TrainingBatch,
        *,
        generator: object | None = None,
    ) -> object: ...


__all__ = [
    "DMD2GuidanceAdapter",
    "DMD2FusedGuidanceAdapter",
    "DMD2LossAdapter",
    "DMD2PredictionAdapter",
    "DMD2TrainingBatch",
]
