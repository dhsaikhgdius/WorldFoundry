"""Functional seams shared by the native SenseFlow objective and engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import torch
from torch import Tensor

from ...shared.contracts import TensorLike
from ..dmd2.contracts import DMD2TrainingBatch

SenseFlowTrainingBatch = DMD2TrainingBatch


@runtime_checkable
class SenseFlowPredictionAdapter(Protocol):
    """Clean prediction and forward corruption for one flow model role."""

    module: object
    checkpoint_identity: str
    noise_process_kind: str
    noise_process_digest: str

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
class SenseFlowTeacherAdapter(SenseFlowPredictionAdapter, Protocol):
    """Architecture-aware guided teacher clean prediction."""

    def predict_guided_clean(
        self,
        noisy_latents: TensorLike,
        noise_levels: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        unconditional_conditioning: Mapping[str, object],
        guidance_scale: float,
    ) -> TensorLike: ...


@runtime_checkable
class SenseFlowFakeScoreAdapter(SenseFlowPredictionAdapter, Protocol):
    """Fake flow model plus its model-native denoising objective."""

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
class SenseFlowDiscriminatorAdapter(Protocol):
    """Image-domain VFM discriminator seam; decoding may live inside the adapter."""

    module: object
    checkpoint_identity: str
    frozen_feature_modules: tuple[object, ...]
    trainable_head_modules: tuple[object, ...]

    def logits(
        self,
        latents: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        reference_latents: TensorLike,
        training: bool,
    ) -> TensorLike: ...


@dataclass(frozen=True, slots=True)
class SenseFlowPreparedBatch:
    """One pre-update generator sample reused by fake-score and discriminator phases."""

    batch: SenseFlowTrainingBatch
    generated_clean: Tensor
    anchor_sigmas: Tensor
    anchor_index: int
    anchor_timestep: int
    backward_simulation: bool

    def __post_init__(self) -> None:
        if not isinstance(self.batch, SenseFlowTrainingBatch):
            raise TypeError("batch must be SenseFlowTrainingBatch")
        if not isinstance(self.generated_clean, Tensor) or self.generated_clean.shape != self.batch.real_latents.shape:
            raise ValueError("generated_clean must match the real latent shape")
        if self.generated_clean.requires_grad:
            raise ValueError("prepared generated_clean must be detached")
        if not isinstance(self.anchor_sigmas, Tensor) or self.anchor_sigmas.shape != (self.batch.batch_size,):
            raise ValueError("anchor_sigmas must have shape [B]")
        if isinstance(self.anchor_index, bool) or not isinstance(self.anchor_index, int) or self.anchor_index < 0:
            raise ValueError("anchor_index must be a non-negative integer")
        if isinstance(self.anchor_timestep, bool) or not isinstance(self.anchor_timestep, int):
            raise TypeError("anchor_timestep must be an integer")
        if not isinstance(self.backward_simulation, bool):
            raise TypeError("backward_simulation must be bool")


@dataclass(frozen=True, slots=True)
class SenseFlowLossResult:
    loss: Tensor
    metrics: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class SenseFlowGeneratorPhase:
    prepared: SenseFlowPreparedBatch
    loss_result: SenseFlowLossResult | None


@runtime_checkable
class SenseFlowLossAdapter(Protocol):
    """Three optimizer losses and the single shared generator rollout."""

    config_digest: str
    generator_update_interval: int
    ida_decay: float
    ida_enabled: bool
    student: SenseFlowPredictionAdapter
    teacher: SenseFlowTeacherAdapter
    fake_score: SenseFlowFakeScoreAdapter
    discriminator: SenseFlowDiscriminatorAdapter

    def loss_denominator(
        self,
        batch: SenseFlowTrainingBatch,
        *,
        role: Literal["generator", "fake-score", "discriminator"],
    ) -> object: ...

    def generator_phase(
        self,
        batch: SenseFlowTrainingBatch,
        *,
        update: bool,
        generator: torch.Generator,
    ) -> SenseFlowGeneratorPhase: ...

    def fake_score_loss(
        self,
        prepared: SenseFlowPreparedBatch,
        *,
        generator: torch.Generator,
    ) -> SenseFlowLossResult: ...

    def discriminator_loss(
        self,
        prepared: SenseFlowPreparedBatch,
    ) -> SenseFlowLossResult: ...


__all__ = [
    "SenseFlowDiscriminatorAdapter",
    "SenseFlowFakeScoreAdapter",
    "SenseFlowGeneratorPhase",
    "SenseFlowLossAdapter",
    "SenseFlowLossResult",
    "SenseFlowPredictionAdapter",
    "SenseFlowPreparedBatch",
    "SenseFlowTeacherAdapter",
    "SenseFlowTrainingBatch",
]
