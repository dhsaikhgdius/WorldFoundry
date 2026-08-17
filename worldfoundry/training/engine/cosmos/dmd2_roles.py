# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos model roles for native DMD2 distillation.

Predict2.5's released trainer expresses the student grid in TrigFlow time but
feeds a rectified-flow DiT.  This module converts that grid at the model
boundary and preserves the released fake-score weighting.  Cosmos3 already
uses rectified-flow sigmas and does not use the optional discriminator head.

The discriminator head architecture is adapted from NVIDIA's Apache-2.0
Cosmos-Predict2.5 implementation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from worldfoundry.training.api.contracts import ObjectiveBatch
from worldfoundry.training.models.cosmos import (
    Cosmos3TrainAdapter,
    CosmosPredict2TrainAdapter,
    CosmosPredict25TrainAdapter,
)
from worldfoundry.training.objectives.flow_matching import (
    flow_clean_from_velocity,
    flow_interpolate,
    flow_velocity_target,
)
from worldfoundry.training.post_training.distillation.dmd.objective import FewStepSchedule
from worldfoundry.training.post_training.shared.prediction import NativeFlowPredictionAdapter

COSMOS_DMD2_GENERATOR_UPDATE_INTERVAL = 5
COSMOS_PREDICT25_DMD2_TRIGFLOW_TIMES = (
    math.pi / 2.0,
    math.atan(15.0),
    math.atan(5.0),
    math.atan(5.0 / 3.0),
)
COSMOS_PREDICT25_DMD2_FLOW_SIGMAS = (1.0, 15.0 / 16.0, 5.0 / 6.0, 5.0 / 8.0)
COSMOS3_DMD2_FLOW_SIGMAS = (0.999, 0.75, 0.5, 0.25)


def trigflow_time_to_flow_sigma(time: float | torch.Tensor) -> float | torch.Tensor:
    """Convert ``cos(t)x0 + sin(t)eps`` to its normalized RF sigma."""

    if isinstance(time, torch.Tensor):
        sine = torch.sin(time)
        return sine / (torch.cos(time) + sine)
    resolved = float(time)
    sine = math.sin(resolved)
    return sine / (math.cos(resolved) + sine)


def cosmos_predict25_dmd2_schedule() -> FewStepSchedule:
    """Return the four-step grid from the released Predict2.5 DMD2 config."""

    return FewStepSchedule(
        timesteps=COSMOS_PREDICT25_DMD2_TRIGFLOW_TIMES,
        sigmas=COSMOS_PREDICT25_DMD2_FLOW_SIGMAS,
    )


def cosmos3_dmd2_schedule() -> FewStepSchedule:
    """Return the released Cosmos3 fixed rectified-flow grid."""

    return FewStepSchedule(
        timesteps=tuple(1000.0 * value for value in COSMOS3_DMD2_FLOW_SIGMAS),
        sigmas=COSMOS3_DMD2_FLOW_SIGMAS,
    )


class _RMSNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float = 1.0e-5) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(value.float().square().mean(dim=-1, keepdim=True) + self.epsilon)
        return normalized.to(dtype=value.dtype) * self.weight


class _FeatureCrossAttention(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(1, 1, 1, dimension))
        self.to_q = nn.Linear(dimension, dimension, bias=False)
        self.to_k = nn.Linear(dimension, dimension, bias=False)
        self.to_v = nn.Linear(dimension, dimension, bias=False)
        self.pre_norm_kv = _RMSNorm(dimension)
        self.post_norm_q = _RMSNorm(dimension)
        self.post_norm_k = _RMSNorm(dimension)
        self.to_out = nn.Linear(dimension, dimension)
        nn.init.normal_(self.query_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch = int(features.shape[0])
        query = self.query_token.expand(batch, -1, -1, -1)
        normalized = self.pre_norm_kv(features[:, None])
        attended = F.scaled_dot_product_attention(
            self.post_norm_q(self.to_q(query)),
            self.post_norm_k(self.to_k(normalized)),
            self.to_v(normalized),
        )
        return (self.to_out(attended) + query)[:, 0, 0]


class _ResidualMLP(nn.Module):
    def __init__(self, dimension: int, ratio: float) -> None:
        super().__init__()
        hidden = int(dimension * float(ratio))
        self.norm = _RMSNorm(dimension)
        self.fc1 = nn.Linear(dimension, hidden)
        self.fc2 = nn.Linear(hidden, dimension)
        for module in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.fc2(F.gelu(self.fc1(self.norm(value))))


class _DiscriminatorBranch(nn.Module):
    def __init__(self, dimension: int, mlp_ratio: float) -> None:
        super().__init__()
        self.cross_attention = _FeatureCrossAttention(dimension)
        self.mlp = _ResidualMLP(dimension, mlp_ratio)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.cross_attention(features))


class CosmosDMD2DiscriminatorHead(nn.Module):
    """Released Predict2.5 cross-attention discriminator over DiT features."""

    def __init__(
        self,
        model_channels: int,
        num_branches: int,
        *,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.model_channels = int(model_channels)
        self.num_branches = int(num_branches)
        self.branches = nn.ModuleList(
            _DiscriminatorBranch(self.model_channels, mlp_ratio) for _ in range(self.num_branches)
        )
        combined = self.model_channels * self.num_branches
        self.final_norm = nn.LayerNorm(combined, eps=1.0e-6)
        self.final_linear = nn.Linear(combined, 1)
        nn.init.xavier_uniform_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.branches):
            raise ValueError("feature count must match the discriminator branch count")
        joined = torch.cat(
            tuple(branch(value) for branch, value in zip(self.branches, features, strict=True)),
            dim=1,
        )
        return self.final_linear(self.final_norm(joined))


class CosmosFlowDMD2PredictionAdapter:
    """Bind an independently loaded Predict2/2.5 role to native DMD2."""

    noise_process_kind = "flow-matching"

    def __init__(
        self,
        adapter: CosmosPredict2TrainAdapter | CosmosPredict25TrainAdapter,
        *,
        checkpoint_identity: str,
        autocast_dtype: torch.dtype | None = None,
    ) -> None:
        if not isinstance(adapter, (CosmosPredict2TrainAdapter, CosmosPredict25TrainAdapter)):
            raise TypeError("adapter must be a Cosmos Predict training adapter")
        self.adapter = adapter
        self.prediction = NativeFlowPredictionAdapter(adapter, autocast_dtype=autocast_dtype)
        self.module = adapter.trainable_module
        self.trainable_module = self.module
        self.checkpoint_identity = str(checkpoint_identity).strip()
        if not self.checkpoint_identity:
            raise ValueError("checkpoint_identity must be non-empty")
        self.autocast_dtype = autocast_dtype
        self.fsdp_block_classes = adapter.fsdp_block_classes

    def add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        noise_levels: torch.Tensor,
    ) -> torch.Tensor:
        return flow_interpolate(clean_latents, noise, noise_levels)

    def predict_velocity(
        self,
        noisy_latents: torch.Tensor,
        noise_levels: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> torch.Tensor:
        return self.prediction.predict_velocity(
            noisy_latents,
            noise_levels,
            sample_ids=sample_ids,
            conditioning=conditioning,
            training=training,
            branch=branch,
        )

    def predict_clean(
        self,
        noisy_latents: torch.Tensor,
        noise_levels: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> torch.Tensor:
        clean = self.prediction.predict_clean(
            noisy_latents,
            noise_levels,
            sample_ids=sample_ids,
            conditioning=conditioning,
            training=training,
            branch=branch,
        )
        condition_mask = conditioning.get("condition_mask")
        condition_latents = conditioning.get("condition_latents")
        if isinstance(condition_mask, torch.Tensor) and isinstance(condition_latents, torch.Tensor):
            mask = condition_mask.to(device=clean.device, dtype=clean.dtype)
            clean = mask * condition_latents.to(clean) + (1.0 - mask) * clean
        return clean


class Cosmos3VideoDMD2PredictionAdapter:
    """Expose the video branch of an independently loaded Cosmos3 DMD2 role."""

    noise_process_kind = "flow-matching"

    def __init__(
        self,
        adapter: Cosmos3TrainAdapter,
        *,
        checkpoint_identity: str,
        autocast_dtype: torch.dtype | None = None,
    ) -> None:
        if not isinstance(adapter, Cosmos3TrainAdapter):
            raise TypeError("adapter must be Cosmos3TrainAdapter")
        self.adapter = adapter
        self.module = adapter.trainable_module
        self.trainable_module = self.module
        self.checkpoint_identity = str(checkpoint_identity).strip()
        if not self.checkpoint_identity:
            raise ValueError("checkpoint_identity must be non-empty")
        self.autocast_dtype = autocast_dtype
        self.fsdp_block_classes = adapter.fsdp_block_classes

    def add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        noise_levels: torch.Tensor,
    ) -> torch.Tensor:
        return flow_interpolate(clean_latents, noise, noise_levels)

    def predict_velocity(
        self,
        noisy_latents: torch.Tensor,
        noise_levels: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> torch.Tensor:
        model_input = noisy_latents if self.autocast_dtype is None else noisy_latents.to(self.autocast_dtype)
        target = torch.zeros_like(model_input)
        batch = ObjectiveBatch(
            sample_ids=sample_ids,
            model_input={"video": model_input},
            target={"video": target},
            sigmas=noise_levels,
            timesteps=noise_levels,
            conditioning=conditioning,
            metadata={"prediction_type": "flow_velocity"},
        )
        device_type = noisy_latents.device.type
        enabled = self.autocast_dtype is not None and device_type in {"cpu", "cuda"}
        with torch.autocast(device_type=device_type, dtype=self.autocast_dtype, enabled=enabled):
            prediction = self.adapter.forward_model(batch, training=training, branch=branch)["video"]
        return prediction.to(dtype=noisy_latents.dtype)

    def predict_clean(
        self,
        noisy_latents: torch.Tensor,
        noise_levels: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> torch.Tensor:
        velocity = self.predict_velocity(
            noisy_latents,
            noise_levels,
            sample_ids=sample_ids,
            conditioning=conditioning,
            training=training,
            branch=branch,
        )
        return flow_clean_from_velocity(noisy_latents, velocity, noise_levels)


class _GuidanceModule(nn.Module):
    def __init__(self, backbone: nn.Module, discriminator: nn.Module | None) -> None:
        super().__init__()
        self.backbone = backbone
        if discriminator is not None:
            self.discriminator = discriminator


class CosmosDMD2GuidanceAdapter:
    """Train the Cosmos fake-score role and optional Predict2.5 discriminator."""

    noise_process_kind = "flow-matching"

    def __init__(
        self,
        prediction: CosmosFlowDMD2PredictionAdapter | Cosmos3VideoDMD2PredictionAdapter,
        *,
        checkpoint_identity: str,
        discriminator: CosmosDMD2DiscriminatorHead | None = None,
        intermediate_feature_ids: Sequence[int] = (),
        trigflow_denoising_weight: bool = False,
    ) -> None:
        self.prediction = prediction
        self.discriminator = discriminator
        self.intermediate_feature_ids = tuple(int(value) for value in intermediate_feature_ids)
        if (discriminator is None) != (len(self.intermediate_feature_ids) == 0):
            raise ValueError("discriminator and intermediate_feature_ids must be configured together")
        if discriminator is not None and discriminator.num_branches != len(self.intermediate_feature_ids):
            raise ValueError("discriminator branch count differs from intermediate_feature_ids")
        self.module = _GuidanceModule(prediction.module, discriminator)
        self.trainable_module = self.module
        self.checkpoint_identity = str(checkpoint_identity).strip()
        if not self.checkpoint_identity:
            raise ValueError("checkpoint_identity must be non-empty")
        self.trigflow_denoising_weight = bool(trigflow_denoising_weight)
        self.autocast_dtype = prediction.autocast_dtype
        self.fsdp_block_classes = prediction.fsdp_block_classes

    def add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        noise_levels: torch.Tensor,
    ) -> torch.Tensor:
        return self.prediction.add_noise(clean_latents, noise, noise_levels)

    def predict_clean(
        self,
        noisy_latents: torch.Tensor,
        noise_levels: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> torch.Tensor:
        return self.prediction.predict_clean(
            noisy_latents,
            noise_levels,
            sample_ids=sample_ids,
            conditioning=conditioning,
            training=training,
            branch=branch,
        )

    def denoising_loss_per_sample(
        self,
        clean_latents: torch.Tensor,
        noisy_latents: torch.Tensor,
        noise: torch.Tensor,
        noise_levels: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
    ) -> torch.Tensor:
        velocity = self.prediction.predict_velocity(
            noisy_latents,
            noise_levels,
            sample_ids=sample_ids,
            conditioning=conditioning,
            training=training,
        )
        error = (velocity.float() - flow_velocity_target(clean_latents, noise).float()).square()
        condition_mask = conditioning.get("condition_mask")
        if isinstance(condition_mask, torch.Tensor):
            error = error * (1.0 - condition_mask.to(device=error.device, dtype=error.dtype))
        if self.trigflow_denoising_weight:
            levels = noise_levels.to(device=error.device, dtype=error.dtype).reshape(
                (int(error.shape[0]),) + (1,) * (error.ndim - 1)
            )
            error = error * (levels.square() + (1.0 - levels).square())
        return error.reshape(int(error.shape[0]), -1).mean(dim=1)

    def predict_clean_and_logits(
        self,
        noisy_latents: torch.Tensor,
        noise_levels: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fake-score x0 and discriminator logits from one backbone call."""

        if self.discriminator is None:
            raise RuntimeError("this Cosmos DMD2 role has no discriminator head")
        blocks = getattr(self.prediction.module, "transformer_blocks", None)
        if not isinstance(blocks, nn.ModuleList):
            raise TypeError("Cosmos discriminator requires a transformer_blocks ModuleList")
        captured: dict[int, torch.Tensor] = {}
        handles = [
            blocks[index].register_forward_hook(
                lambda _module, _inputs, output, index=index: captured.__setitem__(index, output)
            )
            for index in self.intermediate_feature_ids
        ]
        try:
            clean = self.prediction.predict_clean(
                noisy_latents,
                noise_levels,
                sample_ids=sample_ids,
                conditioning=conditioning,
                training=training,
            )
        finally:
            for handle in handles:
                handle.remove()
        self.discriminator.train(training)
        features = tuple(captured[index] for index in self.intermediate_feature_ids)
        device_type = features[0].device.type
        enabled = self.autocast_dtype is not None and device_type in {"cpu", "cuda"}
        with torch.autocast(device_type=device_type, dtype=self.autocast_dtype, enabled=enabled):
            logits = self.discriminator(features)
        return clean, logits

    def denoising_loss_from_clean_per_sample(
        self,
        clean_latents: torch.Tensor,
        predicted_clean: torch.Tensor,
        noise_levels: torch.Tensor,
        *,
        conditioning: Mapping[str, object],
    ) -> torch.Tensor:
        """Evaluate the released fake-score x0 loss in normalized RF coordinates."""

        error = (clean_latents.float() - predicted_clean.float()).square()
        levels = noise_levels.to(device=error.device, dtype=error.dtype).reshape(
            (int(error.shape[0]),) + (1,) * (error.ndim - 1)
        )
        weight = levels.square().reciprocal()
        if self.trigflow_denoising_weight:
            weight = weight * (levels.square() + (1.0 - levels).square())
        error = error * weight
        condition_mask = conditioning.get("condition_mask")
        if isinstance(condition_mask, torch.Tensor):
            error = error * (1.0 - condition_mask.to(device=error.device, dtype=error.dtype))
        return error.reshape(int(error.shape[0]), -1).mean(dim=1)

    def discriminator_logits(
        self,
        latents: torch.Tensor,
        noise_levels: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
    ) -> torch.Tensor:
        if self.discriminator is None:
            raise RuntimeError("this Cosmos DMD2 role has no discriminator head")
        return self.predict_clean_and_logits(
            latents,
            noise_levels,
            sample_ids=sample_ids,
            conditioning=conditioning,
            training=training,
        )[1]


__all__ = [
    "COSMOS3_DMD2_FLOW_SIGMAS",
    "COSMOS_DMD2_GENERATOR_UPDATE_INTERVAL",
    "COSMOS_PREDICT25_DMD2_FLOW_SIGMAS",
    "COSMOS_PREDICT25_DMD2_TRIGFLOW_TIMES",
    "Cosmos3VideoDMD2PredictionAdapter",
    "CosmosDMD2DiscriminatorHead",
    "CosmosDMD2GuidanceAdapter",
    "CosmosFlowDMD2PredictionAdapter",
    "cosmos3_dmd2_schedule",
    "cosmos_predict25_dmd2_schedule",
    "trigflow_time_to_flow_sigma",
]
