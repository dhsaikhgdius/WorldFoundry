"""Typed execution seams for progressive distillation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ...shared.contracts import TensorLike, freeze_mapping, non_empty_ids, tensor_shape


@dataclass(frozen=True, slots=True)
class ProgressiveDistillationBatch:
    sample_ids: tuple[str, ...]
    clean_latents: TensorLike
    conditioning: Mapping[str, object]

    def __post_init__(self) -> None:
        sample_ids = non_empty_ids(
            self.sample_ids,
            field_name="sample_ids",
            unique=True,
        )
        shape = tensor_shape(self.clean_latents, field_name="clean_latents")
        if len(shape) < 3 or shape[0] != len(sample_ids):
            raise ValueError("clean_latents must have shape [B,...]")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(
            self,
            "conditioning",
            freeze_mapping(self.conditioning, field_name="conditioning"),
        )

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)


@dataclass(frozen=True, slots=True)
class ProgressiveRandomInputs:
    noise: TensorLike
    timestep_indices: TensorLike


@runtime_checkable
class ProgressivePredictionAdapter(Protocol):
    module: object
    checkpoint_identity: str

    def predict_model_output(
        self,
        noisy_latents: TensorLike,
        logsnr: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
    ) -> TensorLike: ...


__all__ = [
    "ProgressiveDistillationBatch",
    "ProgressivePredictionAdapter",
    "ProgressiveRandomInputs",
]
