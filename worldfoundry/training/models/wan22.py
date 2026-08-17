"""Native Wan2.2 A14B dual-expert training adapter."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from worldfoundry.training.api.contracts import (
    ObjectiveBatch,
    PreparedBatch,
    TrainingBatch,
)

from .wan import WanTrainAdapter

WAN22_A14B_BOUNDARY_RATIO = 0.875
WAN22_A14B_LATENT_CHANNELS = 16
WAN22_DUAL_ATTENTION = "wan22-dual-attention"


def _slice_batched_value(value: object, indices: torch.Tensor, batch_size: int) -> object:
    if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == batch_size:
        return value.index_select(0, indices)
    if isinstance(value, Mapping):
        return {
            str(key): _slice_batched_value(item, indices, batch_size)
            for key, item in value.items()
        }
    return value


def _slice_objective_batch(batch: ObjectiveBatch, indices: torch.Tensor) -> ObjectiveBatch:
    batch_size = batch.batch_size
    selected = tuple(batch.sample_ids[index] for index in indices.tolist())
    return ObjectiveBatch(
        sample_ids=selected,
        model_input=_slice_batched_value(batch.model_input, indices, batch_size),
        target=_slice_batched_value(batch.target, indices, batch_size),
        sigmas=_slice_batched_value(batch.sigmas, indices, batch_size),
        timesteps=_slice_batched_value(batch.timesteps, indices, batch_size),
        conditioning=_slice_batched_value(batch.conditioning, indices, batch_size),
        noise=(
            None
            if batch.noise is None
            else _slice_batched_value(batch.noise, indices, batch_size)
        ),
        loss_mask=(
            None
            if batch.loss_mask is None
            else _slice_batched_value(batch.loss_mask, indices, batch_size)
        ),
        sample_weights=(
            None
            if batch.sample_weights is None
            else _slice_batched_value(batch.sample_weights, indices, batch_size)
        ),
        metadata=batch.metadata,
    )


class Wan22DualExpertModule(nn.Module):
    """One optimizer/FSDP surface containing both A14B experts."""

    def __init__(
        self,
        high_noise: WanTrainAdapter,
        low_noise: WanTrainAdapter,
    ) -> None:
        super().__init__()
        self.high_noise = high_noise.trainable_module
        self.low_noise = low_noise.trainable_module
        self._high_adapter = high_noise
        self._low_adapter = low_noise

    def forward(
        self,
        objective_batch: ObjectiveBatch,
        *,
        expert: str,
        training: bool,
        conditioning_branch: str,
    ) -> torch.Tensor:
        adapter = self._high_adapter if expert == "high-noise" else self._low_adapter
        return adapter.forward_model(
            objective_batch,
            training=training,
            branch=conditioning_branch,
        )


class Wan22TrainAdapter:
    """Route Wan2.2 A14B samples to high/low experts by effective sigma.

    Wan2.2 TI2V-5B is deliberately excluded: it is a single 48-channel
    per-token-timestep model, while A14B is a pair of independent 16-channel
    Wan transformers.
    """

    prediction_type = "flow_velocity"
    lora_target_preset = WAN22_DUAL_ATTENTION
    conditioning_dropout_owner = "none"

    def __init__(
        self,
        high_noise: WanTrainAdapter,
        low_noise: WanTrainAdapter,
        *,
        boundary_ratio: float = WAN22_A14B_BOUNDARY_RATIO,
    ) -> None:
        if not isinstance(high_noise, WanTrainAdapter) or not isinstance(
            low_noise,
            WanTrainAdapter,
        ):
            raise TypeError("Wan2.2 A14B experts must be WanTrainAdapter instances")
        if high_noise.trainable_module is low_noise.trainable_module:
            raise ValueError("Wan2.2 high- and low-noise experts must be independent")
        boundary = float(boundary_ratio)
        if not 0.0 < boundary < 1.0:
            raise ValueError("Wan2.2 boundary_ratio must be in (0, 1)")
        if (
            high_noise.expected_latent_channels != WAN22_A14B_LATENT_CHANNELS
            or low_noise.expected_latent_channels != WAN22_A14B_LATENT_CHANNELS
        ):
            raise ValueError("Wan2.2 A14B requires two 16-channel experts, not TI2V-5B")
        compatibility = (
            "model_timestep_scale",
            "num_train_timesteps",
            "expected_text_length",
            "expected_context_features",
            "temporal_compression",
            "spatial_compression",
            "patch_size",
        )
        if any(getattr(high_noise, name) != getattr(low_noise, name) for name in compatibility):
            raise ValueError("Wan2.2 A14B experts must share timestep, text, and latent geometry")

        self.high_noise = high_noise
        self.low_noise = low_noise
        self.boundary_ratio = boundary
        self.trainable_module: nn.Module = Wan22DualExpertModule(high_noise, low_noise)
        self.denoiser = None
        self.codec = high_noise.codec
        self.conditioner = high_noise.conditioner
        self.model_timestep_scale = high_noise.model_timestep_scale
        self.num_train_timesteps = high_noise.num_train_timesteps
        self.expected_latent_channels = WAN22_A14B_LATENT_CHANNELS
        self.expected_text_length = high_noise.expected_text_length
        self.expected_context_features = high_noise.expected_context_features
        self.temporal_compression = high_noise.temporal_compression
        self.spatial_compression = high_noise.spatial_compression
        self.patch_size = high_noise.patch_size
        self.gradient_checkpointing = high_noise.gradient_checkpointing
        self.attention_compatibility_mode = high_noise.attention_compatibility_mode
        self.conditioning_dropout_probability = 0.0
        self.fsdp_block_classes = tuple(
            dict.fromkeys((*high_noise.fsdp_block_classes, *low_noise.fsdp_block_classes))
        )
        self.frozen_modules = tuple(
            dict.fromkeys((*high_noise.frozen_modules, *low_noise.frozen_modules))
        )

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        prepared = self.high_noise.prepare_batch(batch)
        return PreparedBatch(
            sample_ids=prepared.sample_ids,
            clean_latents=prepared.clean_latents,
            conditioning=prepared.conditioning,
            loss_mask=prepared.loss_mask,
            sample_weights=prepared.sample_weights,
            metadata={
                **dict(prepared.metadata),
                "model_family": "wan2.2-a14b-dual-expert",
                "expert_boundary_ratio": self.boundary_ratio,
            },
        )

    def forward_train(self, batch: ObjectiveBatch) -> torch.Tensor:
        return self.forward_model(batch, training=True)

    def forward_model(
        self,
        batch: ObjectiveBatch,
        *,
        training: bool,
        branch: str = "positive",
    ) -> torch.Tensor:
        if not isinstance(batch, ObjectiveBatch):
            raise TypeError("batch must be an ObjectiveBatch")
        if not isinstance(batch.model_input, torch.Tensor):
            raise TypeError("Wan2.2 A14B requires tensor video latents")
        if not isinstance(batch.sigmas, torch.Tensor):
            raise TypeError("Wan2.2 A14B objective sigmas must be a torch.Tensor")
        sigmas = batch.sigmas.to(device=batch.model_input.device, dtype=torch.float32)
        if sigmas.ndim == 0:
            sigmas = sigmas.expand(batch.batch_size)
        else:
            sigmas = sigmas.reshape(batch.batch_size)
        # Validate before expert routing: a NaN compares False against the
        # boundary and would otherwise be silently routed to the low-noise
        # expert instead of failing closed.
        if not bool(torch.isfinite(sigmas).all()) or not bool(((0 <= sigmas) & (sigmas <= 1)).all()):
            raise ValueError("Wan2.2 A14B effective sigmas must be finite values in [0, 1]")
        high_indices = torch.nonzero(sigmas >= self.boundary_ratio, as_tuple=False).flatten()
        low_indices = torch.nonzero(sigmas < self.boundary_ratio, as_tuple=False).flatten()

        predictions: list[torch.Tensor] = []
        order: list[torch.Tensor] = []
        for expert, indices in (
            ("high-noise", high_indices),
            ("low-noise", low_indices),
        ):
            if indices.numel() == 0:
                continue
            predictions.append(
                self.trainable_module(
                    _slice_objective_batch(batch, indices),
                    expert=expert,
                    training=training,
                    conditioning_branch=branch,
                )
            )
            order.append(indices)
        routed_order = torch.cat(order)
        return torch.cat(predictions, dim=0).index_select(0, torch.argsort(routed_order))


def build_wan22_train_adapter(
    high_noise: WanTrainAdapter,
    low_noise: WanTrainAdapter,
    *,
    boundary_ratio: float = WAN22_A14B_BOUNDARY_RATIO,
) -> Wan22TrainAdapter:
    """Combine independently loaded native A14B expert adapters."""

    return Wan22TrainAdapter(
        high_noise,
        low_noise,
        boundary_ratio=boundary_ratio,
    )


__all__ = [
    "WAN22_A14B_BOUNDARY_RATIO",
    "WAN22_A14B_LATENT_CHANNELS",
    "WAN22_DUAL_ATTENTION",
    "Wan22DualExpertModule",
    "Wan22TrainAdapter",
    "build_wan22_train_adapter",
]
