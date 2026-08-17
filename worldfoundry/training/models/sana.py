"""Training adapter for native SANA image components.

The adapter bypasses the inference scheduler.  It encodes frozen pixel/text
inputs, translates effective flow sigmas to SANA's model timestep scale, and
calls the ordinary denoiser graph with autograd enabled.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from worldfoundry.base_models.diffusion_model.components import ComponentKey, ComponentKind
from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DenoiserInput,
    DiffusionRequest,
    SamplingConfig,
)
from worldfoundry.training.api.contracts import ObjectiveBatch, PreparedBatch, TrainingBatch

from ._shared import (
    component_module as _component_module,
    freeze_module as _freeze,
    merge_without_overwrite,
    module_device_dtype as _module_device_dtype,
)

SANA_DEFAULT_TRAIN_TIMESTEPS = 1000
SANA_600M_512_TRAIN_FLOW_SHIFT = 3.0


class SanaTrainAdapter:
    """Own SANA image conditioning, latent packing, and training forward.

    The native SANA graph owns classifier-free conditioning dropout through
    its ``CaptionEmbedder``.  ``conditioning_dropout_probability`` exposes the
    loaded value so the future assembler can fail closed if a recipe asks for
    a different probability instead of applying dropout twice.
    """

    prediction_type = "flow_velocity"
    lora_target_preset = "sana-attention"
    conditioning_dropout_owner = "denoiser"

    def __init__(
        self,
        denoiser: object,
        codec: object | None,
        conditioner: object | None,
        *,
        model_timestep_scale: float = SANA_DEFAULT_TRAIN_TIMESTEPS,
        num_train_timesteps: int = SANA_DEFAULT_TRAIN_TIMESTEPS,
        expected_latent_channels: int = 32,
        spatial_compression: int = 32,
    ) -> None:
        trainable_module = _component_module(denoiser, "model")
        if trainable_module is None:
            raise TypeError("SANA denoiser must expose an nn.Module as 'model'")
        if not callable(denoiser):
            raise TypeError("SANA denoiser must be callable")
        if codec is not None and not callable(getattr(codec, "encode", None)):
            raise TypeError("SANA codec must expose encode(images)")
        if conditioner is not None and not callable(getattr(conditioner, "encode", None)):
            raise TypeError("SANA conditioner must expose encode(request, device=..., dtype=...)")

        scale = float(model_timestep_scale)
        if not isfinite(scale) or scale <= 0:
            raise ValueError("model_timestep_scale must be finite and positive")
        if isinstance(num_train_timesteps, bool) or int(num_train_timesteps) < 2:
            raise ValueError("num_train_timesteps must be an integer of at least two")
        if isinstance(expected_latent_channels, bool) or int(expected_latent_channels) <= 0:
            raise ValueError("expected_latent_channels must be a positive integer")
        if isinstance(spatial_compression, bool) or int(spatial_compression) <= 0:
            raise ValueError("spatial_compression must be a positive integer")

        self.denoiser = denoiser
        self.codec = codec
        self.conditioner = conditioner
        self.trainable_module = trainable_module
        self.model_timestep_scale = scale
        self.num_train_timesteps = int(num_train_timesteps)
        self.expected_latent_channels = int(expected_latent_channels)
        self.spatial_compression = int(spatial_compression)

        codec_module = _component_module(codec, "model")
        conditioner_module = _component_module(conditioner, "encoder", "model")
        self.frozen_modules = tuple(module for module in (codec_module, conditioner_module) if module is not None)
        for module in self.frozen_modules:
            _freeze(module)
        self.trainable_module.train()

        blocks = getattr(
            self.trainable_module,
            "blocks",
            getattr(self.trainable_module, "transformer_blocks", ()),
        )
        self.fsdp_block_classes = tuple(dict.fromkeys(type(block) for block in blocks if isinstance(block, nn.Module)))
        y_embedder = getattr(self.trainable_module, "y_embedder", None)
        self.conditioning_dropout_probability = float(getattr(y_embedder, "uncond_prob", 0.0))

    def _keep_frozen_modules_in_eval(self) -> None:
        for module in self.frozen_modules:
            module.eval()

    def _encoded_conditioning(
        self,
        batch: TrainingBatch,
        *,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, object]:
        raw = dict(batch.conditions)
        raw.pop("clean_latents", None)
        raw.pop("latent_loss_mask", None)
        has_context = "context" in raw
        has_mask = "context_mask" in raw
        if has_context != has_mask:
            raise ValueError("precomputed SANA conditioning requires both context and context_mask")

        if has_context:
            values = raw
        else:
            if self.conditioner is None:
                raise RuntimeError("SANA batch has no precomputed context and the adapter has no conditioner")
            raw_inputs = batch.metadata.get("conditioner_inputs", {})
            if not isinstance(raw_inputs, Mapping):
                raise TypeError("metadata.conditioner_inputs must be a mapping")
            request = DiffusionRequest(
                prompt=batch.prompts,
                height=height,
                width=width,
                num_frames=1,
                sampling=SamplingConfig(guidance_scale=1.0),
                inputs=dict(raw_inputs),
                metadata=dict(batch.metadata),
            )
            with torch.no_grad():
                encoded = self.conditioner.encode(request, device=device, dtype=dtype)
            if not isinstance(encoded, Conditioning):
                raise TypeError(f"SANA conditioner returned {type(encoded).__name__}, expected Conditioning")
            values = dict(encoded.positive)
            merge_without_overwrite(values, encoded.shared, source_name="conditioner.shared", family="SANA")
            merge_without_overwrite(values, raw, source_name="TrainingBatch.conditions", family="SANA")

        values.setdefault(
            "img_hw",
            torch.tensor([[height, width]], device=device, dtype=dtype).expand(batch.batch_size, -1),
        )
        values.setdefault(
            "aspect_ratio",
            torch.full(
                (batch.batch_size, 1),
                float(height) / float(width),
                device=device,
                dtype=dtype,
            ),
        )
        values.setdefault(
            "cfg_scale",
            torch.ones(batch.batch_size, device=device, dtype=dtype),
        )

        normalized: dict[str, object] = {}
        for key, value in values.items():
            if not isinstance(value, torch.Tensor):
                normalized[key] = value
                continue
            target_dtype = dtype if key in {"context", "img_hw", "aspect_ratio", "cfg_scale"} else value.dtype
            normalized[key] = value.to(device=device, dtype=target_dtype)

        context = normalized.get("context")
        context_mask = normalized.get("context_mask")
        if not isinstance(context, torch.Tensor) or not isinstance(context_mask, torch.Tensor):
            raise TypeError("SANA conditioning requires tensor context and context_mask")
        if context.ndim != 4 or int(context.shape[0]) != batch.batch_size:
            raise ValueError(f"SANA context must be [B,1,L,C] with B={batch.batch_size}; got {tuple(context.shape)}")
        if context_mask.ndim < 2 or int(context_mask.shape[0]) != batch.batch_size:
            raise ValueError(f"SANA context_mask must start with B={batch.batch_size}; got {tuple(context_mask.shape)}")
        return normalized

    def prepare_prompt_conditioning(
        self,
        batch: TrainingBatch,
        *,
        height: int,
        width: int,
    ) -> Mapping[str, object]:
        """Encode a prompt-only batch without requiring pixels or cached latents."""

        if not isinstance(batch, TrainingBatch):
            raise TypeError("batch must be a TrainingBatch")
        if isinstance(height, bool) or isinstance(width, bool) or int(height) <= 0 or int(width) <= 0:
            raise ValueError("SANA prompt height and width must be positive integers")
        self._keep_frozen_modules_in_eval()
        device, dtype = _module_device_dtype(self.trainable_module)
        return self._encoded_conditioning(
            batch,
            height=int(height),
            width=int(width),
            device=device,
            dtype=dtype,
        )

    def allocate_latent_template(
        self,
        *,
        batch_size: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Allocate only the latent geometry needed by prompt-only sampling."""

        if isinstance(batch_size, bool) or int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if isinstance(height, bool) or isinstance(width, bool) or int(height) <= 0 or int(width) <= 0:
            raise ValueError("SANA prompt height and width must be positive integers")
        if int(height) % self.spatial_compression or int(width) % self.spatial_compression:
            raise ValueError("SANA prompt dimensions must be divisible by spatial_compression")
        device, dtype = _module_device_dtype(self.trainable_module)
        return torch.empty(
            (
                int(batch_size),
                self.expected_latent_channels,
                int(height) // self.spatial_compression,
                int(width) // self.spatial_compression,
            ),
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _latent_loss_mask(
        valid_mask: object | None,
        *,
        pixels: torch.Tensor,
        latent_height: int,
        latent_width: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if valid_mask is None:
            return None
        if not isinstance(valid_mask, torch.Tensor):
            raise TypeError("SANA valid_mask must be a torch.Tensor")
        mask = valid_mask.to(device=device)
        if mask.ndim + 1 == pixels.ndim and int(mask.shape[0]) == int(pixels.shape[0]):
            mask = mask.unsqueeze(1)
        try:
            mask = torch.broadcast_to(mask, pixels.shape)
        except RuntimeError as error:
            raise ValueError(
                f"valid_mask shape {tuple(valid_mask.shape)} cannot broadcast to pixels {tuple(pixels.shape)}"
            ) from error
        mask = mask.float()
        if not bool(torch.isfinite(mask).all()) or not bool((mask >= 0).all()):
            raise ValueError("valid_mask must contain finite non-negative weights")
        # Conservative channel reduction followed by area resampling gives a
        # fractional weight for partially valid codec cells.
        spatial = mask.amin(dim=1)[:, 0:1]
        return F.interpolate(spatial, size=(latent_height, latent_width), mode="area")

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        if not isinstance(batch, TrainingBatch):
            raise TypeError("batch must be a TrainingBatch")
        pixels = batch.pixel_values
        cached_latents = batch.conditions.get("clean_latents")
        if cached_latents is not None and pixels is not None:
            raise ValueError("SANA batch cannot provide both pixel_values and precomputed clean_latents")

        self._keep_frozen_modules_in_eval()
        device, dtype = _module_device_dtype(self.trainable_module)
        if cached_latents is not None:
            if not isinstance(cached_latents, torch.Tensor):
                raise TypeError("precomputed SANA clean_latents must be a torch.Tensor")
            if cached_latents.ndim != 4:
                raise ValueError(f"precomputed SANA clean_latents must be BCHW; got {tuple(cached_latents.shape)}")
            clean_latents = cached_latents
            pixel_height = int(
                batch.metadata.get(
                    "image_height",
                    int(clean_latents.shape[-2]) * self.spatial_compression,
                )
            )
            pixel_width = int(
                batch.metadata.get(
                    "image_width",
                    int(clean_latents.shape[-1]) * self.spatial_compression,
                )
            )
            pixel_shape: tuple[int, ...] | None = None
        else:
            if not isinstance(pixels, torch.Tensor):
                raise TypeError("SANA image training requires pixel_values or precomputed clean_latents")
            if pixels.ndim != 5 or int(pixels.shape[1]) != 3 or int(pixels.shape[2]) != 1:
                raise ValueError(f"SANA image pixels must be [B,3,1,H,W]; got {tuple(pixels.shape)}")
            if self.codec is None:
                raise RuntimeError("SANA batch has no precomputed clean_latents and the adapter has no codec")
            images = pixels[:, :, 0]
            with torch.no_grad():
                clean_latents = self.codec.encode(images)
            pixel_height = int(pixels.shape[-2])
            pixel_width = int(pixels.shape[-1])
            pixel_shape = tuple(int(size) for size in pixels.shape)
        if not isinstance(clean_latents, torch.Tensor) or clean_latents.ndim != 4:
            shape = getattr(clean_latents, "shape", None)
            raise TypeError(f"SANA clean latents must be a BCHW tensor; got {shape!r}")
        if int(clean_latents.shape[0]) != batch.batch_size:
            raise ValueError("SANA clean latents do not match the batch dimension")
        if int(clean_latents.shape[1]) != self.expected_latent_channels:
            raise ValueError(
                f"SANA clean latents contain {clean_latents.shape[1]} channels; "
                f"expected {self.expected_latent_channels}"
            )
        if pixel_height <= 0 or pixel_width <= 0:
            raise ValueError("SANA image_height and image_width must be positive")
        clean_latents = clean_latents.detach().to(device=device, dtype=dtype)

        conditioning = self._encoded_conditioning(
            batch,
            height=pixel_height,
            width=pixel_width,
            device=device,
            dtype=dtype,
        )
        if cached_latents is None:
            assert isinstance(pixels, torch.Tensor)
            loss_mask = self._latent_loss_mask(
                batch.valid_mask,
                pixels=pixels,
                latent_height=int(clean_latents.shape[-2]),
                latent_width=int(clean_latents.shape[-1]),
                device=device,
            )
        else:
            loss_mask = batch.conditions.get("latent_loss_mask")
            if loss_mask is not None:
                if not isinstance(loss_mask, torch.Tensor):
                    raise TypeError("precomputed SANA latent_loss_mask must be a torch.Tensor")
                loss_mask = loss_mask.detach().to(device=device, dtype=torch.float32)
        sample_weights = batch.sample_weights
        if isinstance(sample_weights, torch.Tensor):
            sample_weights = sample_weights.to(device=device, dtype=torch.float32)

        metadata: dict[str, Any] = dict(batch.metadata)
        metadata.update(
            {
                "model_family": "sana-image",
                "prediction_type": self.prediction_type,
                "model_timestep_scale": self.model_timestep_scale,
                "num_train_timesteps": self.num_train_timesteps,
                "pixel_shape": pixel_shape,
                "latent_shape": tuple(int(size) for size in clean_latents.shape),
                "latent_scaling_factor": float(
                    getattr(self.codec, "scaling_factor", batch.metadata.get("latent_scaling_factor", 1.0))
                ),
                "precomputed_latents": cached_latents is not None,
                "conditioning_dropout_owner": self.conditioning_dropout_owner,
                "conditioning_dropout_probability": self.conditioning_dropout_probability,
            }
        )
        return PreparedBatch(
            sample_ids=batch.sample_ids,
            clean_latents=clean_latents,
            conditioning=conditioning,
            loss_mask=loss_mask,
            sample_weights=sample_weights,
            metadata=metadata,
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
        """Forward a trainable policy or frozen teacher/reference role."""

        if not isinstance(training, bool):
            raise TypeError("training must be a bool")
        if branch not in {"positive", "negative"}:
            raise ValueError("branch must be 'positive' or 'negative'")
        if not isinstance(batch, ObjectiveBatch):
            raise TypeError("batch must be an ObjectiveBatch")
        if isinstance(batch.model_input, Mapping) or not isinstance(batch.model_input, torch.Tensor):
            raise TypeError("SANA image training requires one tensor model_input")
        latents = batch.model_input
        if latents.ndim != 4 or int(latents.shape[1]) != self.expected_latent_channels:
            raise ValueError(
                f"SANA image model_input must be [B,{self.expected_latent_channels},H,W]; got {tuple(latents.shape)}"
            )
        prediction_type = batch.metadata.get("prediction_type")
        if prediction_type is not None and prediction_type != self.prediction_type:
            raise ValueError(f"SANA adapter predicts {self.prediction_type!r}, got objective {prediction_type!r}")
        if not isinstance(batch.sigmas, torch.Tensor):
            raise TypeError("SANA objective sigmas must be a torch.Tensor")
        sigmas = batch.sigmas.to(device=latents.device, dtype=torch.float32)
        if sigmas.ndim == 0:
            sigmas = sigmas.expand(latents.shape[0])
        elif sigmas.numel() == latents.shape[0]:
            sigmas = sigmas.reshape(latents.shape[0])
        else:
            raise ValueError("SANA requires one effective sigma per sample")
        if not bool(torch.isfinite(sigmas).all()) or not bool(((0 <= sigmas) & (sigmas <= 1)).all()):
            raise ValueError("SANA effective sigmas must be finite values in [0, 1]")

        self.trainable_module.train(training)
        self._keep_frozen_modules_in_eval()
        model_timesteps = sigmas * self.model_timestep_scale
        output = self.denoiser(
            DenoiserInput(
                latents=latents,
                timestep=model_timesteps,
                next_timestep=torch.zeros_like(model_timesteps),
                conditioning=batch.conditioning,
                step_index=0,
                total_steps=self.num_train_timesteps,
                branch=branch,
            )
        )
        sample = getattr(output, "sample", None)
        if not isinstance(sample, torch.Tensor):
            raise TypeError(f"SANA denoiser returned {type(output).__name__} without a tensor sample")
        if sample.shape != latents.shape:
            raise ValueError(f"SANA prediction shape {tuple(sample.shape)} does not match input {tuple(latents.shape)}")
        return sample


def build_sana_train_adapter(
    components: Mapping[ComponentKey, object],
    **options: object,
) -> SanaTrainAdapter:
    """Build the adapter from ``NativeDiffusionAssembler.build_components``.

    Only trainable/encoding components are consumed.  In particular, the
    inference scheduler and latent initializer are intentionally ignored.
    """

    if not isinstance(components, Mapping):
        raise TypeError("components must be a mapping")

    def require(kind: ComponentKind, name: str = "main") -> object:
        key = ComponentKey(kind, name)
        try:
            return components[key]
        except KeyError as error:
            raise KeyError(f"SANA training components are missing {key}") from error

    return SanaTrainAdapter(
        denoiser=require(ComponentKind.DENOISER),
        codec=require(ComponentKind.LATENT_ENCODER, "codec"),
        conditioner=require(ComponentKind.CONDITIONER),
        **options,
    )


def build_cached_sana_train_adapter(
    components: Mapping[ComponentKey, object],
    **options: object,
) -> SanaTrainAdapter:
    """Build a SANA adapter that consumes only audited cached features."""

    if not isinstance(components, Mapping):
        raise TypeError("components must be a mapping")
    denoiser_key = ComponentKey(ComponentKind.DENOISER)
    unexpected = sorted(str(key) for key in components if key != denoiser_key)
    if unexpected:
        raise ValueError(f"cached SANA training accepts only the denoiser component; got {unexpected}")
    try:
        denoiser = components[denoiser_key]
    except KeyError as error:
        raise KeyError(f"cached SANA training components are missing {denoiser_key}") from error
    return SanaTrainAdapter(denoiser=denoiser, codec=None, conditioner=None, **options)


__all__ = [
    "SANA_600M_512_TRAIN_FLOW_SHIFT",
    "SANA_DEFAULT_TRAIN_TIMESTEPS",
    "SanaTrainAdapter",
    "build_cached_sana_train_adapter",
    "build_sana_train_adapter",
]
