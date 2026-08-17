"""Training objectives released with Cosmos Predict.

The author trainer samples an EDM sigma ``r`` from a log-normal distribution,
uses a shifted logit-normal rectified-flow time and minimizes velocity MSE.
Video2World conditional-frame sampling remains explicit here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from math import isfinite

import torch

from worldfoundry.training.api.contracts import ObjectiveBatch, PreparedBatch, TrainStepResult
from worldfoundry.training.objectives.flow_matching import FlowMatchingConfig, FlowMatchingObjective

COSMOS_PREDICT_LOSS_SCALE = 1.0
COSMOS_PREDICT_LORA_CONDITIONAL_FRAME_PROBABILITIES = (0.333, 0.333, 0.334)
COSMOS3_NANO_CONDITIONING_CONFIG = {0: 0.7, 1: 0.2, 2: 0.1}
COSMOS3_NANO_CFG_DROPOUT = 0.1


class CosmosPredictFlowMatchingObjective(FlowMatchingObjective):
    """Released velocity objective for Predict2/2.5 Video2World."""

    def __init__(
        self,
        config: FlowMatchingConfig | None = None,
        *,
        loss_scale: float = COSMOS_PREDICT_LOSS_SCALE,
        conditional_frame_probabilities: Mapping[int, float] | tuple[float, ...] | None = None,
        conditioning_dropout_probability: float = 0.0,
    ) -> None:
        super().__init__(config)
        self.loss_scale = float(loss_scale)
        if not isfinite(self.loss_scale) or self.loss_scale <= 0.0:
            raise ValueError("Cosmos Predict loss_scale must be finite and positive")
        if conditional_frame_probabilities is None:
            self.conditional_frame_probabilities = None
        else:
            if isinstance(conditional_frame_probabilities, Mapping):
                values = tuple(
                    float(conditional_frame_probabilities.get(index, 0.0))
                    for index in range(max(int(key) for key in conditional_frame_probabilities) + 1)
                )
            else:
                values = tuple(float(value) for value in conditional_frame_probabilities)
            if not values or any(not isfinite(value) or value < 0.0 for value in values) or sum(values) <= 0.0:
                raise ValueError("conditional-frame probabilities must be finite non-negative weights")
            self.conditional_frame_probabilities = tuple(value / sum(values) for value in values)
        self.conditioning_dropout_probability = float(conditioning_dropout_probability)
        if not 0.0 <= self.conditioning_dropout_probability <= 1.0:
            raise ValueError("Cosmos Predict conditioning dropout probability must be in [0, 1]")

    def _apply_text_dropout(
        self,
        batch: ObjectiveBatch,
        *,
        generator: torch.Generator | None,
    ) -> ObjectiveBatch:
        probability = self.conditioning_dropout_probability
        if probability == 0.0:
            return batch
        context = batch.conditioning.get("context")
        if not isinstance(context, torch.Tensor) or int(context.shape[0]) != batch.batch_size:
            raise TypeError("Cosmos Predict text dropout requires batched context")
        dropped = (
            torch.rand(
                batch.batch_size,
                device=context.device,
                generator=generator,
            )
            < probability
        )
        keep = (~dropped).reshape((batch.batch_size,) + (1,) * (context.ndim - 1))
        conditioning = dict(batch.conditioning)
        conditioning["context"] = context * keep.to(dtype=context.dtype)
        metadata = dict(batch.metadata)
        metadata["conditioning_dropped_samples"] = dropped.sum()
        return replace(batch, conditioning=conditioning, metadata=metadata)

    def _sample_conditioning(
        self,
        batch: ObjectiveBatch,
        *,
        generator: torch.Generator | None,
    ) -> ObjectiveBatch:
        probabilities = self.conditional_frame_probabilities
        if probabilities is None:
            return batch
        clean = batch.model_input
        if not isinstance(clean, torch.Tensor):
            raise TypeError("Cosmos Predict conditioning requires tensor video latents")
        batch_size, _, frames, height, width = clean.shape
        if frames == 1:
            counts = torch.zeros(batch_size, device=clean.device, dtype=torch.long)
        else:
            weights = torch.tensor(probabilities, device=clean.device, dtype=torch.float32)
            counts = torch.multinomial(weights, batch_size, replacement=True, generator=generator)
            counts = counts.clamp_max(frames)
        frame_ids = torch.arange(frames, device=clean.device).reshape(1, frames)
        indicator = (frame_ids < counts[:, None]).to(dtype=clean.dtype)[:, None, :, None, None]
        mask = indicator.expand(batch_size, 1, frames, height, width)
        conditioning = dict(batch.conditioning)
        condition_latents = conditioning.get("condition_latents")
        if not isinstance(condition_latents, torch.Tensor):
            raise TypeError("Cosmos Predict conditioning requires cached condition_latents")
        conditioning["condition_mask"] = mask
        conditioning["condition_indicator"] = indicator
        return replace(batch, conditioning=conditioning)

    def corrupt(
        self,
        batch: PreparedBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> ObjectiveBatch:
        objective_batch = super().corrupt(batch, generator=generator)
        objective_batch = self._sample_conditioning(objective_batch, generator=generator)
        objective_batch = self._apply_text_dropout(objective_batch, generator=generator)
        metadata = dict(objective_batch.metadata)
        metadata.update(
            {
                "objective": "cosmos_predict_flow_matching",
                "cosmos_replace_conditioned_target": True,
                "loss_scale": self.loss_scale,
            }
        )
        return replace(objective_batch, metadata=metadata)

    def compute_loss(self, prediction: object, batch: ObjectiveBatch) -> TrainStepResult:
        result = super().compute_loss(prediction, batch)
        scale = self.loss_scale
        metrics = dict(result.metrics)
        metrics["loss_numerator"] = metrics["loss_numerator"] * scale
        metrics["loss_scale"] = torch.tensor(scale, device=result.loss.device, dtype=torch.float32)
        dropped = batch.metadata.get("conditioning_dropped_samples")
        if isinstance(dropped, torch.Tensor):
            metrics["conditioning_dropped_samples"] = dropped
        diagnostics = dict(result.diagnostics)
        diagnostics["author_parameterization"] = "flow-velocity-mse"
        diagnostics["loss_scale"] = scale
        return replace(
            result,
            loss=result.loss * scale,
            losses={name: value * scale for name, value in result.losses.items()},
            metrics=metrics,
            diagnostics=diagnostics,
        )


class Cosmos3VisionFlowMatchingObjective(FlowMatchingObjective):
    """Cosmos3 vision SFT with per-step T2V/I2V/V2V and CFG sampling.

    Conditioned frames stay clean in the adapter's modality state.  Their
    prediction error is zeroed, while the ordinary validity mask remains the
    loss denominator, matching ``normalize_loss_by_active=False`` in the
    released Nano recipe.
    """

    def __init__(
        self,
        config: FlowMatchingConfig | None = None,
        *,
        conditioning_config: Mapping[int, float] = COSMOS3_NANO_CONDITIONING_CONFIG,
        conditioning_dropout: float = COSMOS3_NANO_CFG_DROPOUT,
    ) -> None:
        super().__init__(config)
        items = sorted((int(count), float(weight)) for count, weight in conditioning_config.items())
        if not items or any(count < 0 or not isfinite(weight) or weight < 0.0 for count, weight in items):
            raise ValueError("Cosmos3 conditioning_config must contain non-negative counts and weights")
        total = sum(weight for _, weight in items)
        if total <= 0.0:
            raise ValueError("Cosmos3 conditioning_config must have positive total weight")
        dropout = float(conditioning_dropout)
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("Cosmos3 conditioning dropout must be in [0, 1]")
        self.conditioning_config = tuple((count, weight / total) for count, weight in items)
        self.conditioning_dropout = dropout

    def _sample_step_conditioning(
        self,
        batch: PreparedBatch,
        *,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clean_tree = batch.clean_latents
        if not isinstance(clean_tree, Mapping) or set(clean_tree) != {"video"}:
            raise TypeError("Cosmos3 vision SFT requires a video latent mapping")
        video = clean_tree["video"]
        if not isinstance(video, torch.Tensor) or video.ndim != 5:
            raise ValueError("Cosmos3 video latents must have shape [B,C,T,H,W]")
        batch_size, _, frames, _, _ = video.shape
        task_draws = torch.rand(batch_size, device=video.device, generator=generator)
        dropout_draws = torch.rand(batch_size, device=video.device, generator=generator)
        counts_table = torch.tensor(
            [count for count, _ in self.conditioning_config],
            device=video.device,
            dtype=torch.long,
        )
        cumulative = torch.tensor(
            [
                sum(weight for _, weight in self.conditioning_config[: index + 1])
                for index in range(len(self.conditioning_config))
            ],
            device=video.device,
            dtype=torch.float32,
        )
        selected = torch.searchsorted(cumulative, task_draws).clamp_max(len(self.conditioning_config) - 1)
        counts = counts_table[selected].clamp_max(max(0, int(frames) - 1))
        frame_ids = torch.arange(frames, device=video.device).reshape(1, frames)
        condition_mask = (frame_ids < counts[:, None])[:, None, :, None, None]
        denoise_mask = (~condition_mask).to(dtype=torch.float32)
        caption_dropout = dropout_draws < self.conditioning_dropout
        return counts, denoise_mask, caption_dropout

    def corrupt(
        self,
        batch: PreparedBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> ObjectiveBatch:
        counts, denoise_mask, caption_dropout = self._sample_step_conditioning(
            batch,
            generator=generator,
        )
        objective_batch = super().corrupt(batch, generator=generator)
        if self.conditioning_dropout > 0.0 and "empty_input_ids" not in objective_batch.conditioning:
            raise ValueError("Cosmos3 CFG dropout requires normal and empty cached input IDs")
        conditioning = dict(objective_batch.conditioning)
        conditioning["denoise_masks"] = {"video": denoise_mask}
        conditioning["num_conditional_frames"] = counts
        conditioning["caption_dropout_mask"] = caption_dropout
        metadata = dict(objective_batch.metadata)
        metadata.update(
            {
                "objective": "cosmos3_vision_flow_matching",
                "conditioning_config": dict(self.conditioning_config),
                "conditioning_dropout": self.conditioning_dropout,
                "normalize_loss_by_active": False,
            }
        )
        return replace(objective_batch, conditioning=conditioning, metadata=metadata)

    def compute_loss(self, prediction: object, batch: ObjectiveBatch) -> TrainStepResult:
        if not isinstance(prediction, Mapping) or not isinstance(batch.target, Mapping):
            raise TypeError("Cosmos3 vision prediction and target must be modality mappings")
        masks = batch.conditioning.get("denoise_masks")
        if not isinstance(masks, Mapping):
            raise TypeError("Cosmos3 objective requires denoise_masks")
        zero_condition_error: dict[str, torch.Tensor] = {}
        for name, target in batch.target.items():
            value = prediction[name]
            mask = masks[name]
            if not all(isinstance(item, torch.Tensor) for item in (value, target, mask)):
                raise TypeError("Cosmos3 objective tensors are incomplete")
            denoise = torch.broadcast_to(mask.to(device=value.device, dtype=value.dtype), value.shape)
            zero_condition_error[name] = target.to(value) + denoise * (value - target.to(value))
        result = super().compute_loss(zero_condition_error, batch)
        counts = batch.conditioning["num_conditional_frames"]
        dropped = batch.conditioning["caption_dropout_mask"]
        assert isinstance(counts, torch.Tensor) and isinstance(dropped, torch.Tensor)
        metrics = dict(result.metrics)
        metrics["conditional_frames_mean"] = counts.float().mean().detach()
        metrics["caption_dropout_count"] = dropped.sum().detach()
        diagnostics = dict(result.diagnostics)
        diagnostics.update(
            {
                "conditioning_config": dict(self.conditioning_config),
                "conditioning_dropout": self.conditioning_dropout,
                "normalize_loss_by_active": False,
            }
        )
        return replace(result, metrics=metrics, diagnostics=diagnostics)


__all__ = [
    "COSMOS3_NANO_CFG_DROPOUT",
    "COSMOS3_NANO_CONDITIONING_CONFIG",
    "COSMOS_PREDICT_LOSS_SCALE",
    "COSMOS_PREDICT_LORA_CONDITIONAL_FRAME_PROBABILITIES",
    "Cosmos3VisionFlowMatchingObjective",
    "CosmosPredictFlowMatchingObjective",
]
