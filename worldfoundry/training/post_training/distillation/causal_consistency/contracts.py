"""Typed clean-latent batches for native causal consistency."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ...shared.contracts import TensorLike, freeze_mapping, non_empty_ids, tensor_shape


@dataclass(frozen=True, slots=True)
class CausalConsistencyTrainingBatch:
    sample_ids: tuple[str, ...]
    clean_latents: TensorLike
    conditioning: Mapping[str, object]
    unconditional_conditioning: Mapping[str, object]

    def __post_init__(self) -> None:
        sample_ids = non_empty_ids(self.sample_ids, field_name="sample_ids", unique=True)
        shape = tensor_shape(self.clean_latents, field_name="clean_latents")
        if len(shape) < 3 or shape[0] != len(sample_ids) or any(size <= 0 for size in shape[1:]):
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


__all__ = ["CausalConsistencyTrainingBatch"]
