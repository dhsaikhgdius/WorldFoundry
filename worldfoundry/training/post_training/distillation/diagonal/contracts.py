"""Functional model and rollout contracts for diagonal distillation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from worldfoundry.core.attention.kv_cache_policy import CacheState

from ..dmd.objective import FewStepPrediction
from ..self_forcing.contracts import CachePayload, CausalChunkAdapter


@runtime_checkable
class DiagonalCausalAdapter(CausalChunkAdapter, Protocol):
    """Causal clean prediction plus an arbitrary-noise context commit."""

    def commit_context_chunk(
        self,
        context_chunk: Tensor,
        context_timesteps: Tensor,
        *,
        block_index: int,
        start_frame: int,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        cache: CachePayload,
    ) -> CachePayload: ...


@dataclass(frozen=True, slots=True)
class DiagonalRollout:
    """One causal diagonal trajectory and the mask retaining student gradients."""

    clean_latents: Tensor
    gradient_mask: Tensor
    base_exit_indices: tuple[int, ...]
    block_exit_indices: tuple[int, ...]
    block_timesteps: tuple[tuple[float, ...], ...]
    cache_state: CacheState

    def __post_init__(self) -> None:
        if not isinstance(self.clean_latents, Tensor) or self.clean_latents.ndim < 2:
            raise TypeError("clean_latents must be a batched torch.Tensor")
        if not isinstance(self.gradient_mask, Tensor) or self.gradient_mask.dtype != torch.bool:
            raise TypeError("gradient_mask must be a boolean torch.Tensor")
        if self.gradient_mask.shape != self.clean_latents.shape:
            raise ValueError("gradient_mask must match clean_latents")
        count = len(self.base_exit_indices)
        if count == 0 or len(self.block_exit_indices) != count or len(self.block_timesteps) != count:
            raise ValueError("diagonal rollout metadata must contain one entry per block")
        if len(self.cache_state.blocks) != count:
            raise ValueError("one cache block must be committed per diagonal block")
        for base, clipped, timesteps in zip(
            self.base_exit_indices,
            self.block_exit_indices,
            self.block_timesteps,
            strict=True,
        ):
            if isinstance(base, bool) or isinstance(clipped, bool) or base < 0 or clipped < 0:
                raise ValueError("diagonal exit indices must be non-negative integers")
            if not timesteps or clipped >= len(timesteps):
                raise ValueError("a diagonal block exit falls outside its schedule")


@dataclass(frozen=True, slots=True)
class DiagonalFewStepPrediction(FewStepPrediction):
    """DMD prediction retaining the full diagonal trajectory contract."""

    rollout: DiagonalRollout

    def __post_init__(self) -> None:
        if self.clean_latents is not self.rollout.clean_latents:
            raise ValueError("prediction and rollout must share clean_latents")


__all__ = [
    "DiagonalCausalAdapter",
    "DiagonalFewStepPrediction",
    "DiagonalRollout",
]
