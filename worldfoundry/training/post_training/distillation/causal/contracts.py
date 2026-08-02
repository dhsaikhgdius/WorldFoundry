"""Model-family boundaries shared by causal ODE and consistency training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from torch import Tensor, nn


@runtime_checkable
class CausalCleanPredictionAdapter(Protocol):
    """Predict clean latents while teacher-forcing the clean causal context."""

    module: nn.Module

    def predict_clean(
        self,
        noisy_latents: Tensor,
        timesteps: Tensor,
        *,
        clean_context: Tensor,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
    ) -> Tensor: ...


@runtime_checkable
class CausalVelocityPredictionAdapter(Protocol):
    """Predict causal flow velocity with the same clean teacher-forced context."""

    module: nn.Module

    def predict_velocity(
        self,
        noisy_latents: Tensor,
        timesteps: Tensor,
        *,
        clean_context: Tensor,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
    ) -> Tensor: ...


__all__ = ["CausalCleanPredictionAdapter", "CausalVelocityPredictionAdapter"]
