"""Typed batches and model seams for latent consistency distillation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ...shared.contracts import TensorLike, freeze_mapping, non_empty_ids, tensor_shape


@dataclass(frozen=True, slots=True)
class LatentConsistencyTrainingBatch:
    """Clean latents paired with positive and unconditional model conditions."""

    sample_ids: tuple[str, ...]
    clean_latents: TensorLike
    conditioning: Mapping[str, object]
    unconditional_conditioning: Mapping[str, object]

    def __post_init__(self) -> None:
        sample_ids = non_empty_ids(self.sample_ids, field_name="sample_ids", unique=True)
        shape = tensor_shape(self.clean_latents, field_name="clean_latents")
        if len(shape) < 2 or shape[0] != len(sample_ids) or any(size <= 0 for size in shape[1:]):
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
class LatentConsistencyRandomInputs:
    """All stochastic choices for one objective call, exposed for exact testing."""

    noise: TensorLike
    timestep_indices: TensorLike
    guidance_coefficients: TensorLike


@runtime_checkable
class LatentConsistencyPredictionAdapter(Protocol):
    """Native diffusion-model output for online, teacher, and EMA roles."""

    module: object
    checkpoint_identity: str

    def predict_model_output(
        self,
        noisy_latents: TensorLike,
        timesteps: TensorLike,
        *,
        guidance_embedding: TensorLike | None,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> TensorLike: ...


__all__ = [
    "LatentConsistencyPredictionAdapter",
    "LatentConsistencyRandomInputs",
    "LatentConsistencyTrainingBatch",
]
