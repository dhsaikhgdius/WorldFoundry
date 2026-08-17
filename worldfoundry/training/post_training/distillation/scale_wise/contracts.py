"""Typed batches and model seams for native scale-wise distillation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from ...shared.contracts import TensorLike, freeze_mapping, non_empty_ids, tensor_shape


@dataclass(frozen=True, slots=True)
class ScaleWiseTrainingBatch:
    """Current- and previous-scale VAE latents for one schedule interval."""

    sample_ids: tuple[str, ...]
    current_latents: TensorLike
    previous_latents: TensorLike
    conditioning: Mapping[str, object]
    unconditional_conditioning: Mapping[str, object]
    interval_index: int

    def __post_init__(self) -> None:
        sample_ids = non_empty_ids(
            self.sample_ids,
            field_name="sample_ids",
            unique=True,
        )
        current_shape = tensor_shape(
            self.current_latents,
            field_name="current_latents",
        )
        previous_shape = tensor_shape(
            self.previous_latents,
            field_name="previous_latents",
        )
        if len(current_shape) != 4 or current_shape[0] != len(sample_ids):
            raise ValueError("current_latents must have shape [B,C,H,W]")
        if len(previous_shape) != 4 or previous_shape[:2] != current_shape[:2]:
            raise ValueError(
                "previous_latents must match current_latents in batch and channels"
            )
        if current_shape[-2] != current_shape[-1]:
            raise ValueError("current_latents must use a square latent scale")
        if previous_shape[-2] != previous_shape[-1]:
            raise ValueError("previous_latents must use a square latent scale")
        if isinstance(self.interval_index, bool):
            raise TypeError("interval_index must be an integer")
        interval_index = int(self.interval_index)
        if interval_index < 0:
            raise ValueError("interval_index must be non-negative")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "interval_index", interval_index)
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
class ScaleWisePredictionAdapter(Protocol):
    """Flow velocity prediction exposed by a native model implementation."""

    module: object
    checkpoint_identity: str

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


@runtime_checkable
class ScaleWiseCriticAdapter(ScaleWisePredictionAdapter, Protocol):
    """Fake score, intermediate features, and its trainable classifier head."""

    def audit_scale_wise_critic(
        self,
        *,
        classifier_blocks: tuple[int, ...],
        mmd_blocks: tuple[int, ...],
        discriminator_layers: int,
    ) -> None: ...

    def predict_velocity_and_features(
        self,
        noisy_latents: TensorLike,
        sigmas: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        block_indices: tuple[int, ...],
        training: bool,
    ) -> tuple[TensorLike, tuple[TensorLike, ...]]: ...

    def extract_features(
        self,
        noisy_latents: TensorLike,
        sigmas: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        block_indices: tuple[int, ...],
        training: bool,
    ) -> tuple[TensorLike, ...]: ...

    def classify_features(
        self,
        pooled_features: tuple[TensorLike, ...],
    ) -> tuple[TensorLike, ...]: ...


@runtime_checkable
class ScaleWiseLossAdapter(Protocol):
    """Two-role loss seam consumed by the scale-wise optimizer engine."""

    num_intervals: int
    fake_updates_per_iteration: int
    batch_mmd: bool

    def loss_denominator(
        self,
        batch: ScaleWiseTrainingBatch,
        *,
        role: Literal["student", "fake-score"],
    ) -> object: ...

    def student_loss(
        self,
        batch: ScaleWiseTrainingBatch,
        *,
        generator: object | None = None,
    ) -> object: ...

    def fake_score_loss(
        self,
        batch: ScaleWiseTrainingBatch,
        *,
        generator: object | None = None,
    ) -> object: ...


__all__ = [
    "ScaleWiseCriticAdapter",
    "ScaleWiseLossAdapter",
    "ScaleWisePredictionAdapter",
    "ScaleWiseTrainingBatch",
]
