"""Typed model and batch seams for native adversarial diffusion distillation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ...shared.contracts import TensorLike, freeze_mapping, non_empty_ids, tensor_shape


@dataclass(frozen=True, slots=True)
class ADDTrainingBatch:
    """Clean diffusion states and aligned real images for one ADD update."""

    sample_ids: tuple[str, ...]
    clean_latents: TensorLike
    real_images: TensorLike
    conditioning: Mapping[str, object]
    discriminator_conditioning: Mapping[str, object]

    def __post_init__(self) -> None:
        sample_ids = non_empty_ids(self.sample_ids, field_name="sample_ids", unique=True)
        latent_shape = tensor_shape(self.clean_latents, field_name="clean_latents")
        image_shape = tensor_shape(self.real_images, field_name="real_images")
        if len(latent_shape) < 2 or latent_shape[0] != len(sample_ids):
            raise ValueError("clean_latents must be a non-empty [B,...] tensor")
        if any(size == 0 for size in latent_shape[1:]):
            raise ValueError("clean_latents cannot contain empty non-batch dimensions")
        if len(image_shape) != 4 or image_shape[0] != len(sample_ids):
            raise ValueError("real_images must have shape [B,C,H,W]")
        if any(size == 0 for size in image_shape[1:]):
            raise ValueError("real_images cannot contain empty non-batch dimensions")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(
            self,
            "conditioning",
            freeze_mapping(self.conditioning, field_name="conditioning"),
        )
        object.__setattr__(
            self,
            "discriminator_conditioning",
            freeze_mapping(
                self.discriminator_conditioning,
                field_name="discriminator_conditioning",
            ),
        )

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)


@runtime_checkable
class ADDPredictionAdapter(Protocol):
    """A student or teacher denoiser that returns clean-state predictions."""

    module: object
    checkpoint_identity: str

    def predict_clean(
        self,
        noisy_latents: TensorLike,
        timesteps: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
    ) -> TensorLike: ...


@runtime_checkable
class ADDDecoderAdapter(Protocol):
    """A frozen differentiable decoder from diffusion state to RGB image."""

    module: object
    checkpoint_identity: str

    def decode(self, clean_latents: TensorLike) -> TensorLike: ...


@dataclass(frozen=True, slots=True)
class ADDDiscriminatorHeadOutput:
    """One feature tap, its head input, and the resulting logits."""

    resolution: int
    layer: str
    features: TensorLike
    logits: TensorLike


@dataclass(frozen=True, slots=True)
class ADDDiscriminatorOutput:
    """Ordered Cartesian product of configured image scales and ViT layers."""

    heads: tuple[ADDDiscriminatorHeadOutput, ...]

    @property
    def keys(self) -> tuple[tuple[int, str], ...]:
        return tuple((head.resolution, head.layer) for head in self.heads)


@runtime_checkable
class ADDDiscriminatorAdapter(Protocol):
    """Frozen feature network plus synchronized trainable discriminator heads."""

    module: object
    feature_module: object
    checkpoint_identity: str
    feature_resolutions: tuple[int, ...]
    feature_layers: tuple[str, ...]
    conditioning_keys: tuple[str, ...]

    def predict(
        self,
        images: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        track_image_grad: bool,
        require_r1_inputs: bool,
    ) -> ADDDiscriminatorOutput: ...


@dataclass(frozen=True, slots=True)
class ADDLossResult:
    loss: TensorLike
    metrics: Mapping[str, object]


@runtime_checkable
class ADDLossAdapter(Protocol):
    """Loss seam consumed by the atomic generator/discriminator engine."""

    def loss_denominator(self, batch: ADDTrainingBatch, *, role: str) -> object: ...

    def generator_loss(
        self,
        batch: ADDTrainingBatch,
        *,
        generator: object | None = None,
    ) -> ADDLossResult: ...

    def discriminator_loss(
        self,
        batch: ADDTrainingBatch,
        *,
        generator: object | None = None,
    ) -> ADDLossResult: ...


__all__ = [
    "ADDDecoderAdapter",
    "ADDDiscriminatorAdapter",
    "ADDDiscriminatorHeadOutput",
    "ADDDiscriminatorOutput",
    "ADDLossAdapter",
    "ADDLossResult",
    "ADDPredictionAdapter",
    "ADDTrainingBatch",
]
