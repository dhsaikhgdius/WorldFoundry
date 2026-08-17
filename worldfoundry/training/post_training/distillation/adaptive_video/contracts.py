"""Functional batches for adaptive video distillation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ...shared.contracts import (
    TensorLike,
    freeze_mapping,
    is_broadcastable,
    non_empty_ids,
    tensor_shape,
)
from ..dmd.contracts import DMDLossAdapter, DMDTrainingBatch


@dataclass(frozen=True, slots=True)
class AdaptiveVideoRealBatch:
    """Fresh real-video latents consumed only on generator iterations."""

    sample_ids: tuple[str, ...]
    latents: TensorLike
    conditioning: Mapping[str, object]
    loss_mask: TensorLike | None = None
    sample_weights: TensorLike | None = None

    def __post_init__(self) -> None:
        sample_ids = non_empty_ids(
            self.sample_ids,
            field_name="sample_ids",
            unique=True,
        )
        shape = tensor_shape(self.latents, field_name="latents")
        if len(shape) < 3 or shape[0] != len(sample_ids) or shape[1] < 2:
            raise ValueError("real video latents must have shape [B,F,...] with F >= 2")
        if self.loss_mask is not None:
            mask_shape = tensor_shape(self.loss_mask, field_name="loss_mask")
            if not is_broadcastable(mask_shape, shape):
                if not (
                    len(mask_shape) + 1 == len(shape)
                    and mask_shape[:1] == shape[:1]
                    and is_broadcastable(
                        (mask_shape[0], 1, *mask_shape[1:]),
                        shape,
                    )
                ):
                    raise ValueError(
                        f"real loss_mask shape {mask_shape} cannot broadcast to {shape}"
                    )
        if self.sample_weights is not None and tensor_shape(
            self.sample_weights,
            field_name="sample_weights",
        ) != (len(sample_ids),):
            raise ValueError("real sample_weights must have shape [B]")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(
            self,
            "conditioning",
            freeze_mapping(self.conditioning, field_name="conditioning"),
        )

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)


@dataclass(frozen=True, slots=True, kw_only=True)
class AdaptiveVideoTrainingBatch(DMDTrainingBatch):
    """Prompt-only student samples paired with a fresh real-video batch."""

    real_sample_ids: tuple[str, ...]
    real_latents: TensorLike
    real_conditioning: Mapping[str, object]
    real_loss_mask: TensorLike | None = None
    real_sample_weights: TensorLike | None = None

    def __post_init__(self) -> None:
        DMDTrainingBatch.__post_init__(self)
        generated_shape = tensor_shape(self.clean_latents, field_name="clean_latents")
        if len(generated_shape) < 3 or generated_shape[1] < 2:
            raise ValueError(
                "adaptive video clean_latents must have shape [B,F,...] with F >= 2"
            )
        real_ids = non_empty_ids(
            self.real_sample_ids,
            field_name="real_sample_ids",
            unique=True,
        )
        if len(real_ids) != self.batch_size:
            raise ValueError("generated and real adaptive-video batches must have equal size")
        real_shape = tensor_shape(self.real_latents, field_name="real_latents")
        if real_shape != generated_shape:
            raise ValueError("real_latents must match the generated latent shape")
        if self.real_loss_mask is not None:
            mask_shape = tensor_shape(self.real_loss_mask, field_name="real_loss_mask")
            if not is_broadcastable(mask_shape, real_shape):
                if not (
                    len(mask_shape) + 1 == len(real_shape)
                    and mask_shape[:1] == real_shape[:1]
                    and is_broadcastable(
                        (mask_shape[0], 1, *mask_shape[1:]),
                        real_shape,
                    )
                ):
                    raise ValueError(
                        f"real_loss_mask shape {mask_shape} cannot broadcast to {real_shape}"
                    )
        if self.real_sample_weights is not None and tensor_shape(
            self.real_sample_weights,
            field_name="real_sample_weights",
        ) != (self.batch_size,):
            raise ValueError("real_sample_weights must have shape [B]")
        object.__setattr__(self, "real_sample_ids", real_ids)
        object.__setattr__(
            self,
            "real_conditioning",
            freeze_mapping(self.real_conditioning, field_name="real_conditioning"),
        )

    @classmethod
    def combine(
        cls,
        generated: DMDTrainingBatch,
        real: AdaptiveVideoRealBatch,
    ) -> AdaptiveVideoTrainingBatch:
        if not isinstance(generated, DMDTrainingBatch):
            raise TypeError("generated must be DMDTrainingBatch")
        if not isinstance(real, AdaptiveVideoRealBatch):
            raise TypeError("real must be AdaptiveVideoRealBatch")
        return cls(
            sample_ids=generated.sample_ids,
            clean_latents=generated.clean_latents,
            conditioning=generated.conditioning,
            unconditional_conditioning=generated.unconditional_conditioning,
            loss_mask=generated.loss_mask,
            sample_weights=generated.sample_weights,
            real_sample_ids=real.sample_ids,
            real_latents=real.latents,
            real_conditioning=real.conditioning,
            real_loss_mask=real.loss_mask,
            real_sample_weights=real.sample_weights,
        )


@runtime_checkable
class AdaptiveVideoLossAdapter(DMDLossAdapter, Protocol):
    """DMD loss seam plus checkpointable adaptive-regression state."""

    def commit_generator_step(self, results: tuple[object, ...]) -> None: ...

    def adaptive_state_dict(self) -> Mapping[str, object]: ...

    def load_adaptive_state_dict(self, state_dict: Mapping[str, object]) -> None: ...


__all__ = [
    "AdaptiveVideoLossAdapter",
    "AdaptiveVideoRealBatch",
    "AdaptiveVideoTrainingBatch",
]
