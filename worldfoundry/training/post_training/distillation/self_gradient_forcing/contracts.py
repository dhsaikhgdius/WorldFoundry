"""Model and replay boundaries for Self-Gradient-Forcing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from torch import Tensor

from worldfoundry.core.attention.kv_cache_policy import CacheState

from ..self_forcing.contracts import CachePayload, CausalChunkAdapter


@runtime_checkable
class SelfGradientForcingAdapter(CausalChunkAdapter, Protocol):
    """Causal chunk calls plus the parallel teacher-forced replay call."""

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

    def predict_clean_teacher_forced(
        self,
        noisy_latents: Tensor,
        timesteps: Tensor,
        sigmas: Tensor,
        *,
        clean_context: Tensor,
        context_timesteps: Tensor,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
    ) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class SelfGradientForcingReplay:
    """The bounded no-grad replay and its single live parallel prediction."""

    clean_latents: Tensor
    noisy_at_exit: Tensor
    cache_targets: Tensor
    context_latents: Tensor
    exit_index: int
    timestep: float
    sigma: float
    cache_state: CacheState

    def __post_init__(self) -> None:
        tensors = (
            self.clean_latents,
            self.noisy_at_exit,
            self.cache_targets,
            self.context_latents,
        )
        if not all(isinstance(value, Tensor) for value in tensors):
            raise TypeError("Self-Gradient-Forcing replay values must be tensors")
        shapes = {tuple(value.shape) for value in tensors}
        if len(shapes) != 1 or self.clean_latents.ndim < 2:
            raise ValueError("all Self-Gradient-Forcing replay tensors must share a batched shape")
        if isinstance(self.exit_index, bool) or int(self.exit_index) < 0:
            raise ValueError("exit_index must be a non-negative integer")
        if self.noisy_at_exit.requires_grad or self.cache_targets.requires_grad or self.context_latents.requires_grad:
            raise RuntimeError("the first Self-Gradient-Forcing pass must not retain autograd state")
        if not isinstance(self.cache_state, CacheState):
            raise TypeError("cache_state must use the core CacheState contract")


__all__ = ["SelfGradientForcingAdapter", "SelfGradientForcingReplay"]
