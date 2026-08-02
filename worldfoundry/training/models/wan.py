"""Training adapter for the native Wan2.1 text-to-video graph."""

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

WAN_DEFAULT_TRAIN_TIMESTEPS = 1000
WAN_DEFAULT_TEXT_LENGTH = 512
WAN_DEFAULT_CONTEXT_FEATURES = 4096


def _component_module(component: object, *names: str) -> nn.Module | None:
    if isinstance(component, nn.Module):
        return component
    for name in names:
        value = getattr(component, name, None)
        if isinstance(value, nn.Module):
            return value
    return None


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    reference = next(module.parameters(), None)
    if reference is None:
        reference = next(module.buffers(), None)
    if reference is None:
        return torch.device("cpu"), torch.float32
    dtype = reference.dtype if reference.is_floating_point() else torch.float32
    return reference.device, dtype


def _freeze(module: nn.Module | None) -> None:
    if module is None:
        return
    module.requires_grad_(False)
    module.eval()


def _merge_without_overwrite(
    destination: dict[str, object],
    source: Mapping[str, object],
    *,
    source_name: str,
) -> None:
    overlap = sorted(set(destination) & set(source))
    if overlap:
        raise ValueError(f"{source_name} collides with encoded Wan conditioning keys: {overlap}")
    destination.update(source)


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive")
    return resolved


def wan_pixel_mask_to_latent(
    valid_mask: torch.Tensor,
    *,
    pixel_shape: tuple[int, int, int, int, int],
    latent_shape: tuple[int, int, int],
    temporal_compression: int = 4,
) -> torch.Tensor:
    """Project a pixel validity mask through Wan's causal codec geometry.

    The first latent frame represents the first pixel frame.  Every later
    latent frame represents the next ``temporal_compression`` pixel frames;
    averaging therefore preserves partial padding weights instead of treating
    the codec as an ordinary symmetric 3D downsampler.
    """

    if not isinstance(valid_mask, torch.Tensor):
        raise TypeError("Wan valid_mask must be a torch.Tensor")
    if len(pixel_shape) != 5 or len(latent_shape) != 3:
        raise ValueError("Wan pixel_shape and latent_shape must be BCTHW and THW")
    compression = _positive_int(
        temporal_compression,
        field_name="temporal_compression",
    )
    mask = valid_mask
    if mask.ndim + 1 == len(pixel_shape) and int(mask.shape[0]) == pixel_shape[0]:
        mask = mask.unsqueeze(1)
    try:
        mask = torch.broadcast_to(mask, pixel_shape)
    except RuntimeError as error:
        raise ValueError(
            f"valid_mask shape {tuple(valid_mask.shape)} cannot broadcast to pixels {pixel_shape}"
        ) from error
    mask = mask.float().amin(dim=1, keepdim=True)
    if not bool(torch.isfinite(mask).all()) or not bool(((0 <= mask) & (mask <= 1)).all()):
        raise ValueError("valid_mask must contain finite weights in [0, 1]")

    latent_frames, latent_height, latent_width = latent_shape
    pixel_frames = int(mask.shape[2])
    expected_latent_frames = 1 + (pixel_frames - 1) // compression
    if (pixel_frames - 1) % compression or expected_latent_frames != latent_frames:
        raise ValueError("pixel valid_mask temporal geometry differs from encoded Wan latents")
    spatial = F.interpolate(
        mask,
        size=(pixel_frames, latent_height, latent_width),
        mode="area",
    )
    first = spatial[:, :, :1]
    if latent_frames == 1:
        return first
    remaining = spatial[:, :, 1:]
    grouped = remaining.reshape(
        pixel_shape[0],
        1,
        latent_frames - 1,
        compression,
        latent_height,
        latent_width,
    ).mean(dim=3)
    return torch.cat((first, grouped), dim=2)


class WanTrainAdapter:
    """Own Wan video encoding, conditioning, mask projection, and forward."""

    prediction_type = "flow_velocity"
    lora_target_preset = "wan-attention"
    conditioning_dropout_owner = "none"

    def __init__(
        self,
        denoiser: object,
        codec: object | None,
        conditioner: object | None,
        *,
        model_timestep_scale: float = WAN_DEFAULT_TRAIN_TIMESTEPS,
        num_train_timesteps: int = WAN_DEFAULT_TRAIN_TIMESTEPS,
        expected_latent_channels: int = 16,
        temporal_compression: int = 4,
        spatial_compression: int = 8,
        expected_text_length: int = WAN_DEFAULT_TEXT_LENGTH,
        expected_context_features: int = WAN_DEFAULT_CONTEXT_FEATURES,
        gradient_checkpointing: bool = False,
        attention_compatibility_mode: bool = True,
    ) -> None:
        trainable_module = _component_module(denoiser, "model")
        if trainable_module is None:
            raise TypeError("Wan denoiser must expose an nn.Module as 'model'")
        if not callable(denoiser):
            raise TypeError("Wan denoiser must be callable")
        if codec is not None and not callable(getattr(codec, "encode", None)):
            raise TypeError("Wan codec must expose encode(videos)")
        if conditioner is not None and not callable(getattr(conditioner, "encode", None)):
            raise TypeError("Wan conditioner must expose encode(request, device=..., dtype=...)")

        scale = float(model_timestep_scale)
        if not isfinite(scale) or scale <= 0:
            raise ValueError("model_timestep_scale must be finite and positive")
        if not isinstance(gradient_checkpointing, bool):
            raise TypeError("gradient_checkpointing must be a bool")
        if not isinstance(attention_compatibility_mode, bool):
            raise TypeError("attention_compatibility_mode must be a bool")

        self.denoiser = denoiser
        self.codec = codec
        self.conditioner = conditioner
        self.trainable_module = trainable_module
        self.model_timestep_scale = scale
        self.num_train_timesteps = _positive_int(
            num_train_timesteps,
            field_name="num_train_timesteps",
        )
        if self.num_train_timesteps < 2:
            raise ValueError("num_train_timesteps must be at least two")
        self.expected_latent_channels = _positive_int(
            expected_latent_channels,
            field_name="expected_latent_channels",
        )
        self.temporal_compression = _positive_int(
            temporal_compression,
            field_name="temporal_compression",
        )
        self.spatial_compression = _positive_int(
            spatial_compression,
            field_name="spatial_compression",
        )
        self.expected_text_length = _positive_int(
            expected_text_length,
            field_name="expected_text_length",
        )
        self.expected_context_features = _positive_int(
            expected_context_features,
            field_name="expected_context_features",
        )
        self.gradient_checkpointing = gradient_checkpointing
        self.attention_compatibility_mode = attention_compatibility_mode

        set_attention_mode = getattr(
            self.trainable_module,
            "set_attention_compatibility_mode",
            None,
        )
        if not callable(set_attention_mode):
            raise TypeError("Wan trainable module must expose set_attention_compatibility_mode(enabled)")
        set_attention_mode(attention_compatibility_mode)
        if hasattr(self.denoiser, "manage_autocast"):
            self.denoiser.manage_autocast = False

        codec_module = _component_module(codec, "vae", "model")
        conditioner_module = _component_module(
            conditioner,
            "text_encoder",
            "encoder",
            "model",
        )
        frozen = tuple(dict.fromkeys(module for module in (codec_module, conditioner_module) if module is not None))
        self.frozen_modules = frozen
        for module in frozen:
            _freeze(module)
        self.trainable_module.train()

        blocks = getattr(self.trainable_module, "blocks", ())
        self.fsdp_block_classes = tuple(dict.fromkeys(type(block) for block in blocks if isinstance(block, nn.Module)))
        patch_size = tuple(int(value) for value in getattr(self.trainable_module, "patch_size", (1, 2, 2)))
        if len(patch_size) != 3 or any(value <= 0 for value in patch_size):
            raise ValueError("Wan trainable module must expose a positive three-axis patch_size")
        self.patch_size = patch_size
        self.conditioning_dropout_probability = 0.0

    def _keep_frozen_modules_in_eval(self) -> None:
        for module in self.frozen_modules:
            module.eval()

    def _validate_pixel_geometry(self, *, frames: int, height: int, width: int) -> None:
        if (frames - 1) % self.temporal_compression != 0:
            raise ValueError(
                "Wan video frame count must follow temporal codec geometry "
                f"1 + k*{self.temporal_compression}; got {frames}"
            )
        required_height = self.spatial_compression * self.patch_size[1]
        required_width = self.spatial_compression * self.patch_size[2]
        if height % required_height or width % required_width:
            raise ValueError(
                "Wan pixel geometry must remain divisible after codec and DiT patching: "
                f"height multiple {required_height}, width multiple {required_width}; "
                f"got {height}x{width}"
            )

    def _encoded_conditioning(
        self,
        batch: TrainingBatch,
        *,
        frames: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, object]:
        raw = dict(batch.conditions)
        for reserved in ("clean_latents", "latent_loss_mask", "valid_latent_mask"):
            raw.pop(reserved, None)

        if "context" in raw:
            values = raw
        else:
            if self.conditioner is None:
                raise RuntimeError("Wan batch has no precomputed context and the adapter has no conditioner")
            raw_inputs = batch.metadata.get("conditioner_inputs", {})
            if not isinstance(raw_inputs, Mapping):
                raise TypeError("metadata.conditioner_inputs must be a mapping")
            request = DiffusionRequest(
                prompt=batch.prompts,
                height=height,
                width=width,
                num_frames=frames,
                sampling=SamplingConfig(guidance_scale=1.0),
                inputs=dict(raw_inputs),
                metadata=dict(batch.metadata),
            )
            with torch.no_grad():
                encoded = self.conditioner.encode(request, device=device, dtype=dtype)
            if not isinstance(encoded, Conditioning):
                raise TypeError(f"Wan conditioner returned {type(encoded).__name__}, expected Conditioning")
            values = dict(encoded.positive)
            _merge_without_overwrite(values, encoded.shared, source_name="conditioner.shared")
            _merge_without_overwrite(values, raw, source_name="TrainingBatch.conditions")

        normalized: dict[str, object] = {}
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                target_dtype = dtype if value.is_floating_point() else value.dtype
                normalized[key] = value.detach().to(device=device, dtype=target_dtype)
            else:
                normalized[key] = value
        context = normalized.get("context")
        expected_shape = (
            batch.batch_size,
            self.expected_text_length,
            self.expected_context_features,
        )
        if not isinstance(context, torch.Tensor) or tuple(context.shape) != expected_shape:
            shape = getattr(context, "shape", None)
            raise ValueError(f"Wan context must have shape {expected_shape}; got {shape!r}")
        if not bool(torch.isfinite(context).all()):
            raise ValueError("Wan context contains NaN or infinity")
        return normalized

    def _pixel_mask_to_latent(
        self,
        valid_mask: object | None,
        *,
        pixels: torch.Tensor,
        latent_shape: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor | None:
        if valid_mask is None:
            return None
        if not isinstance(valid_mask, torch.Tensor):
            raise TypeError("Wan valid_mask must be a torch.Tensor")
        return wan_pixel_mask_to_latent(
            valid_mask.to(device=device),
            pixel_shape=tuple(int(value) for value in pixels.shape),
            latent_shape=latent_shape,
            temporal_compression=self.temporal_compression,
        )

    @staticmethod
    def _cached_loss_mask(
        batch: TrainingBatch,
        *,
        latent_shape: tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor | None:
        combined: torch.Tensor | None = None
        for name in ("latent_loss_mask", "valid_latent_mask"):
            value = batch.conditions.get(name)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"precomputed Wan {name} must be a torch.Tensor")
            mask = value.detach().to(device=device, dtype=torch.float32)
            try:
                mask = torch.broadcast_to(mask, latent_shape)
            except RuntimeError as error:
                raise ValueError(
                    f"precomputed Wan {name} shape {tuple(value.shape)} cannot broadcast to latents {latent_shape}"
                ) from error
            if not bool(torch.isfinite(mask).all()) or not bool((mask >= 0).all()):
                raise ValueError(f"precomputed Wan {name} must contain finite non-negative weights")
            if name == "valid_latent_mask" and not bool((mask <= 1).all()):
                raise ValueError("precomputed Wan valid_latent_mask must contain weights in [0, 1]")
            combined = mask if combined is None else combined * mask
        return combined

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        if not isinstance(batch, TrainingBatch):
            raise TypeError("batch must be a TrainingBatch")
        pixels = batch.pixel_values
        cached_latents = batch.conditions.get("clean_latents")
        if cached_latents is not None and pixels is not None:
            raise ValueError("Wan batch cannot provide both pixel_values and precomputed clean_latents")

        self._keep_frozen_modules_in_eval()
        device, dtype = _module_device_dtype(self.trainable_module)
        pixel_shape: tuple[int, ...] | None
        if cached_latents is None:
            if not isinstance(pixels, torch.Tensor):
                raise TypeError("Wan training requires pixel_values or precomputed clean_latents")
            if pixels.ndim != 5 or int(pixels.shape[1]) != 3:
                raise ValueError(f"Wan pixels must be [B,3,T,H,W]; got {tuple(pixels.shape)}")
            frames, height, width = (int(value) for value in pixels.shape[-3:])
            self._validate_pixel_geometry(frames=frames, height=height, width=width)
            if self.codec is None:
                raise RuntimeError("Wan batch has no precomputed clean_latents and the adapter has no codec")
            with torch.no_grad():
                clean_latents = self.codec.encode(pixels)
            pixel_shape = tuple(int(value) for value in pixels.shape)
        else:
            if not isinstance(cached_latents, torch.Tensor):
                raise TypeError("precomputed Wan clean_latents must be a torch.Tensor")
            clean_latents = cached_latents
            if clean_latents.ndim != 5:
                raise ValueError(f"precomputed Wan clean_latents must be BCTHW; got {tuple(clean_latents.shape)}")
            inferred_frames = (int(clean_latents.shape[2]) - 1) * self.temporal_compression + 1
            inferred_height = int(clean_latents.shape[3]) * self.spatial_compression
            inferred_width = int(clean_latents.shape[4]) * self.spatial_compression
            frames = int(batch.metadata.get("target_num_frames", inferred_frames))
            height = int(batch.metadata.get("target_height", inferred_height))
            width = int(batch.metadata.get("target_width", inferred_width))
            if (frames, height, width) != (inferred_frames, inferred_height, inferred_width):
                raise ValueError(
                    "cached Wan target geometry differs from codec-implied geometry: "
                    f"metadata={(frames, height, width)}, "
                    f"implied={(inferred_frames, inferred_height, inferred_width)}"
                )
            self._validate_pixel_geometry(frames=frames, height=height, width=width)
            pixel_shape = None

        if not isinstance(clean_latents, torch.Tensor) or clean_latents.ndim != 5:
            shape = getattr(clean_latents, "shape", None)
            raise TypeError(f"Wan clean latents must be a BCTHW tensor; got {shape!r}")
        if int(clean_latents.shape[0]) != batch.batch_size:
            raise ValueError("Wan clean latents do not match the batch dimension")
        if int(clean_latents.shape[1]) != self.expected_latent_channels:
            raise ValueError(
                f"Wan clean latents contain {clean_latents.shape[1]} channels; expected {self.expected_latent_channels}"
            )
        for axis, patch in zip(clean_latents.shape[-3:], self.patch_size):
            if int(axis) % patch:
                raise ValueError(
                    f"Wan latent shape {tuple(clean_latents.shape[-3:])} is not divisible "
                    f"by DiT patch size {self.patch_size}"
                )
        clean_latents = clean_latents.detach().to(device=device, dtype=dtype)

        conditioning = self._encoded_conditioning(
            batch,
            frames=frames,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
        )
        if cached_latents is None:
            assert isinstance(pixels, torch.Tensor)
            loss_mask = self._pixel_mask_to_latent(
                batch.valid_mask,
                pixels=pixels,
                latent_shape=tuple(int(value) for value in clean_latents.shape[-3:]),
                device=device,
            )
        else:
            loss_mask = self._cached_loss_mask(
                batch,
                latent_shape=tuple(int(value) for value in clean_latents.shape),
                device=device,
            )
        sample_weights = batch.sample_weights
        if isinstance(sample_weights, torch.Tensor):
            sample_weights = sample_weights.to(device=device, dtype=torch.float32)

        metadata: dict[str, Any] = dict(batch.metadata)
        metadata.update(
            {
                "model_family": "wan-video",
                "prediction_type": self.prediction_type,
                "model_timestep_scale": self.model_timestep_scale,
                "num_train_timesteps": self.num_train_timesteps,
                "pixel_shape": pixel_shape,
                "latent_shape": tuple(int(value) for value in clean_latents.shape),
                "precomputed_latents": cached_latents is not None,
                "latent_normalization": "official-channel-mean-std-deterministic-mean",
                "conditioning_dropout_owner": self.conditioning_dropout_owner,
                "conditioning_dropout_probability": self.conditioning_dropout_probability,
                "gradient_checkpointing": self.gradient_checkpointing,
                "attention_compatibility_mode": self.attention_compatibility_mode,
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
            raise TypeError("Wan video training requires one tensor model_input")
        latents = batch.model_input
        if latents.ndim != 5 or int(latents.shape[1]) != self.expected_latent_channels:
            raise ValueError(
                f"Wan model_input must be [B,{self.expected_latent_channels},T,H,W]; got {tuple(latents.shape)}"
            )
        prediction_type = batch.metadata.get("prediction_type")
        if prediction_type is not None and prediction_type != self.prediction_type:
            raise ValueError(f"Wan adapter predicts {self.prediction_type!r}, got objective {prediction_type!r}")
        if not isinstance(batch.sigmas, torch.Tensor):
            raise TypeError("Wan objective sigmas must be a torch.Tensor")
        sigmas = batch.sigmas.to(device=latents.device, dtype=torch.float32)
        if sigmas.ndim == 0:
            sigmas = sigmas.expand(latents.shape[0])
        elif sigmas.numel() == latents.shape[0]:
            sigmas = sigmas.reshape(latents.shape[0])
        else:
            raise ValueError("Wan requires one effective sigma per sample")
        if not bool(torch.isfinite(sigmas).all()) or not bool(((0 <= sigmas) & (sigmas <= 1)).all()):
            raise ValueError("Wan effective sigmas must be finite values in [0, 1]")

        self.trainable_module.train(training)
        self._keep_frozen_modules_in_eval()
        model_timesteps = sigmas * self.model_timestep_scale
        conditioning = dict(batch.conditioning)
        conditioning["use_gradient_checkpointing"] = self.gradient_checkpointing and training
        output = self.denoiser(
            DenoiserInput(
                latents=latents,
                timestep=model_timesteps,
                next_timestep=torch.zeros_like(model_timesteps),
                conditioning=conditioning,
                step_index=0,
                total_steps=self.num_train_timesteps,
                branch=branch,
            )
        )
        sample = getattr(output, "sample", None)
        if not isinstance(sample, torch.Tensor):
            raise TypeError(f"Wan denoiser returned {type(output).__name__} without a tensor sample")
        if sample.shape != latents.shape:
            raise ValueError(f"Wan prediction shape {tuple(sample.shape)} does not match input {tuple(latents.shape)}")
        return sample


def build_wan_train_adapter(
    components: Mapping[ComponentKey, object],
    **options: object,
) -> WanTrainAdapter:
    """Build a raw-video Wan adapter from native training components."""

    if not isinstance(components, Mapping):
        raise TypeError("components must be a mapping")

    def require(kind: ComponentKind, name: str = "main") -> object:
        key = ComponentKey(kind, name)
        try:
            return components[key]
        except KeyError as error:
            raise KeyError(f"Wan training components are missing {key}") from error

    return WanTrainAdapter(
        denoiser=require(ComponentKind.DENOISER),
        codec=require(ComponentKind.LATENT_ENCODER, "codec"),
        conditioner=require(ComponentKind.CONDITIONER),
        **options,
    )


def build_cached_wan_train_adapter(
    components: Mapping[ComponentKey, object],
    **options: object,
) -> WanTrainAdapter:
    """Build a Wan adapter that consumes only immutable cached features."""

    if not isinstance(components, Mapping):
        raise TypeError("components must be a mapping")
    denoiser_key = ComponentKey(ComponentKind.DENOISER)
    unexpected = sorted(str(key) for key in components if key != denoiser_key)
    if unexpected:
        raise ValueError(f"cached Wan training accepts only the denoiser component; got {unexpected}")
    try:
        denoiser = components[denoiser_key]
    except KeyError as error:
        raise KeyError(f"cached Wan training components are missing {denoiser_key}") from error
    return WanTrainAdapter(denoiser=denoiser, codec=None, conditioner=None, **options)


__all__ = [
    "WAN_DEFAULT_CONTEXT_FEATURES",
    "WAN_DEFAULT_TEXT_LENGTH",
    "WAN_DEFAULT_TRAIN_TIMESTEPS",
    "WanTrainAdapter",
    "build_cached_wan_train_adapter",
    "build_wan_train_adapter",
    "wan_pixel_mask_to_latent",
]
