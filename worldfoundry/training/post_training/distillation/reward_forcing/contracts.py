"""Typed batches and runtime seams for native Reward-Forcing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ...shared.contracts import TensorLike, non_empty_ids
from ..dmd.contracts import DMDTrainingBatch
from ..self_forcing.contracts import CausalChunkAdapter


@dataclass(frozen=True, slots=True)
class RewardForcingTrainingBatch(DMDTrainingBatch):
    """Prompt-bearing latent template consumed by Re-DMD and VideoReward."""

    prompts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        DMDTrainingBatch.__post_init__(self)
        prompts = non_empty_ids(self.prompts, field_name="prompts", unique=False)
        if len(prompts) != self.batch_size:
            raise ValueError("prompts must contain one entry per sample")
        object.__setattr__(self, "prompts", prompts)


@runtime_checkable
class RewardForcingDecoderAdapter(Protocol):
    """Frozen VAE boundary from clean latents to evaluator-ready videos."""

    module: object
    checkpoint_identity: str

    def decode_reward_videos(
        self,
        clean_latents: TensorLike,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
    ) -> TensorLike: ...


@runtime_checkable
class MotionQualityRewardAdapter(Protocol):
    """Return normalized MQ scores with explicit weight and module identity.

    ``owned_module`` is ``None`` when the evaluator owns no directly
    checkpointed ``nn.Module`` at this boundary.  In that case the mandatory
    ``checkpoint_identity`` still binds its loaded reward weights to the
    recipe and exact-resume identity.
    """

    checkpoint_identity: str
    owned_module: object | None
    calibration_mean: float
    calibration_std: float
    normalization_epsilon: float

    def score_motion_quality(
        self,
        videos: TensorLike,
        batch: RewardForcingTrainingBatch,
    ) -> TensorLike: ...


@runtime_checkable
class RewardForcingCausalAdapter(CausalChunkAdapter, Protocol):
    """Causal student whose cache path executes EMA-Sink semantics."""

    checkpoint_identity: str | None

    def audit_reward_forcing_cache(
        self,
        *,
        frames_per_block: int,
        local_attention_frames: int,
        ema_sink_frames: int,
        ema_sink_decay: float,
    ) -> None: ...


__all__ = [
    "MotionQualityRewardAdapter",
    "RewardForcingCausalAdapter",
    "RewardForcingDecoderAdapter",
    "RewardForcingTrainingBatch",
]
