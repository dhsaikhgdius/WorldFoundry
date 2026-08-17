"""Typed seams for native SGMD roles and prompt batches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from ...shared.contracts import TensorLike, freeze_mapping, non_empty_ids, tensor_shape


@dataclass(frozen=True, slots=True)
class SGMDTrainingBatch:
    """Prompt-conditioned latent shape used to start an SGMD rollout from noise."""

    sample_ids: tuple[str, ...]
    latent_template: TensorLike
    conditioning: Mapping[str, object]
    unconditional_conditioning: Mapping[str, object]
    sample_weights: TensorLike | None = None

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
class SGMDPredictionAdapter(Protocol):
    """Velocity prediction exposed by a WorldFoundry model implementation."""

    module: object
    checkpoint_identity: str
    noise_process_kind: str

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
class SGMDLossAdapter(Protocol):
    """Two-phase loss seam consumed by the SGMD optimizer engine."""

    num_student_steps: int
    minimum_student_target_index: int

    def loss_denominator(
        self,
        batch: SGMDTrainingBatch,
        *,
        role: Literal["student", "fake-score"],
    ) -> object: ...

    def student_loss(
        self,
        batch: SGMDTrainingBatch,
        *,
        target_index: int,
        generator: object | None = None,
    ) -> object: ...

    def fake_score_loss(
        self,
        batch: SGMDTrainingBatch,
        *,
        target_index: int,
        generator: object | None = None,
    ) -> object: ...


__all__ = ["SGMDLossAdapter", "SGMDPredictionAdapter", "SGMDTrainingBatch"]
