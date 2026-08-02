"""Functional model and cache boundaries for causal self-forcing rollout."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from worldfoundry.core.attention.kv_cache_policy import CacheState

CachePayload = object


@runtime_checkable
class CausalChunkAdapter(Protocol):
    """One clean-prediction call over a temporal chunk and its live cache.

    The adapter owns architecture-specific KV allocation, overwrite, and
    eviction.  WorldFoundry owns the temporal loop and passes the exact cache
    returned by one call into the next call.  Cache payloads must be tensors or
    nested mappings/lists/tuples of tensors and scalar bookkeeping so the
    rollout can detach them after a clean context commit.
    """

    module: nn.Module

    def initialize_cache(
        self,
        reference: Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
    ) -> CachePayload: ...

    def predict_clean_chunk(
        self,
        noisy_chunk: Tensor,
        timesteps: Tensor,
        sigmas: Tensor,
        *,
        block_index: int,
        start_frame: int,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        cache: CachePayload,
        training: bool,
    ) -> Tensor: ...

    def commit_clean_chunk(
        self,
        clean_chunk: Tensor,
        *,
        block_index: int,
        start_frame: int,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        cache: CachePayload,
    ) -> CachePayload: ...


@dataclass(frozen=True, slots=True)
class SelfForcingRollout:
    """A generated full sequence and the cache state after every clean commit."""

    clean_latents: Tensor
    exit_indices: tuple[int, ...]
    cache_state: CacheState

    def __post_init__(self) -> None:
        if not isinstance(self.clean_latents, torch.Tensor) or self.clean_latents.ndim < 2:
            raise TypeError("clean_latents must be a batched torch.Tensor")
        if not self.exit_indices:
            raise ValueError("exit_indices cannot be empty")
        if len(self.cache_state.blocks) != len(self.exit_indices):
            raise ValueError("one cache block must be committed for every exit index")


__all__ = ["CachePayload", "CausalChunkAdapter", "SelfForcingRollout"]
