"""Functional seams for native Score Identity Distillation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from ...shared.contracts import TensorLike, freeze_mapping, non_empty_ids, tensor_shape


@dataclass(frozen=True, slots=True)
class SIDTrainingBatch:
    """Prompt-only student inputs with optional real latents for DiffusionGAN."""

    sample_ids: tuple[str, ...]
    latent_template: TensorLike
    conditioning: Mapping[str, object]
    unconditional_conditioning: Mapping[str, object]
    sample_weights: TensorLike | None = None
    real_sample_ids: tuple[str, ...] = ()
    real_latents: TensorLike | None = None
    real_conditioning: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        sample_ids = non_empty_ids(self.sample_ids, field_name="sample_ids", unique=True)
        shape = tensor_shape(self.latent_template, field_name="latent_template")
        if len(shape) < 2 or shape[0] != len(sample_ids) or any(size == 0 for size in shape[1:]):
            raise ValueError("latent_template must be a non-empty [B,...] tensor")
        if self.sample_weights is not None and tensor_shape(
            self.sample_weights,
            field_name="sample_weights",
        ) != (len(sample_ids),):
            raise ValueError("sample_weights must have shape [B]")

        has_real = self.real_latents is not None
        if has_real != bool(self.real_sample_ids) or has_real != (self.real_conditioning is not None):
            raise ValueError(
                "real_sample_ids, real_latents, and real_conditioning must be provided together"
            )
        if has_real:
            real_ids = non_empty_ids(
                self.real_sample_ids,
                field_name="real_sample_ids",
                unique=True,
            )
            if len(real_ids) != len(sample_ids):
                raise ValueError("SiD generated and real batches must have equal batch size")
            if tensor_shape(self.real_latents, field_name="real_latents") != shape:
                raise ValueError("SiD real latents must match latent_template shape")
            object.__setattr__(self, "real_sample_ids", real_ids)
            object.__setattr__(
                self,
                "real_conditioning",
                freeze_mapping(self.real_conditioning, field_name="real_conditioning"),
            )
        else:
            object.__setattr__(self, "real_sample_ids", ())

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

    @property
    def has_real_samples(self) -> bool:
        return self.real_latents is not None


@runtime_checkable
class SIDPredictionAdapter(Protocol):
    """Flow corruption and prediction implemented by a native model family."""

    module: object
    checkpoint_identity: str
    noise_process_kind: str
    noise_process_digest: str

    def add_noise(
        self,
        clean_latents: TensorLike,
        noise: TensorLike,
        sigmas: TensorLike,
    ) -> TensorLike: ...

    def predict_velocity(
        self,
        noisy_latents: TensorLike,
        sigmas: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> TensorLike: ...

    def predict_clean(
        self,
        noisy_latents: TensorLike,
        sigmas: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> TensorLike: ...


@runtime_checkable
class SIDDiscriminatorAdapter(Protocol):
    """Optional discriminator feature head owned by the fake-score role."""

    module: object

    def discriminator_logits(
        self,
        noisy_latents: TensorLike,
        sigmas: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
    ) -> TensorLike: ...


@runtime_checkable
class SIDLossAdapter(Protocol):
    """Two-role loss seam consumed by the SiD optimizer engine."""

    config_digest: str
    num_student_steps: int

    def loss_denominator(
        self,
        batch: SIDTrainingBatch,
        *,
        role: Literal["fake-score", "generator"],
    ) -> object: ...

    def fake_score_loss(
        self,
        batch: SIDTrainingBatch,
        *,
        target_index: int,
        generator: object | None = None,
    ) -> object: ...

    def generator_loss(
        self,
        batch: SIDTrainingBatch,
        *,
        target_index: int,
        generator: object | None = None,
    ) -> object: ...


__all__ = [
    "SIDDiscriminatorAdapter",
    "SIDLossAdapter",
    "SIDPredictionAdapter",
    "SIDTrainingBatch",
]
