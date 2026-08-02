"""Typed batches and loss adapters owned by distribution matching."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from ...shared.contracts import (
    TensorLike,
    freeze_mapping,
    is_broadcastable,
    non_empty_ids,
    tensor_shape,
)


@dataclass(frozen=True, slots=True)
class DMDTrainingBatch:
    """Prepared clean latents and both teacher conditioning branches."""

    sample_ids: tuple[str, ...]
    clean_latents: TensorLike
    conditioning: Mapping[str, object]
    unconditional_conditioning: Mapping[str, object]
    loss_mask: TensorLike | None = None
    sample_weights: TensorLike | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sample_ids = non_empty_ids(self.sample_ids, field_name="sample_ids", unique=True)
        shape = tensor_shape(self.clean_latents, field_name="clean_latents")
        if len(shape) < 2 or shape[0] != len(sample_ids) or any(size == 0 for size in shape[1:]):
            raise ValueError("clean_latents must be a non-empty [B,...] tensor")
        if self.loss_mask is not None:
            mask_shape = tensor_shape(self.loss_mask, field_name="loss_mask")
            if not is_broadcastable(mask_shape, shape):
                if not (
                    len(mask_shape) + 1 == len(shape)
                    and mask_shape[:1] == shape[:1]
                    and is_broadcastable((mask_shape[0], 1, *mask_shape[1:]), shape)
                ):
                    raise ValueError(f"loss_mask shape {mask_shape} cannot broadcast to {shape}")
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
        object.__setattr__(
            self,
            "metadata",
            freeze_mapping(self.metadata, field_name="metadata"),
        )

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)


@runtime_checkable
class DMDLossAdapter(Protocol):
    """Loss seam consumed by the native two-optimizer DMD engine."""

    schedule_digest: str

    def loss_denominator(
        self,
        batch: DMDTrainingBatch,
        *,
        role: Literal["generator", "fake-score"],
    ) -> object: ...

    def generator_loss(
        self,
        batch: DMDTrainingBatch,
        *,
        generator: object | None = None,
    ) -> object: ...

    def fake_score_loss(
        self,
        batch: DMDTrainingBatch,
        *,
        generator: object | None = None,
    ) -> object: ...


__all__ = ["DMDLossAdapter", "DMDTrainingBatch"]
