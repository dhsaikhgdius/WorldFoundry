"""Native flow-matching adapter for LTX video transformers.

The adapter follows the tensor contract used by the Lightricks trainers: cached
``BCTHW`` video latents are patchified to ``[B, tokens, channels]``; intrinsic
image conditioning keeps selected tokens clean, assigns them timestep zero,
and removes them from the loss; the transformer output is a flow velocity.
"""

from __future__ import annotations

import random
from collections.abc import Mapping

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.components import ComponentKey, ComponentKind
from worldfoundry.base_models.diffusion_model.contracts import Conditioning, DiffusionRequest, SamplingConfig
from worldfoundry.base_models.diffusion_model.models.networks.ltx.modality import Modality
from worldfoundry.base_models.diffusion_model.models.networks.ltx.perturbations import BatchedPerturbationConfig
from worldfoundry.base_models.diffusion_model.models.representations.ltx.patchifiers import (
    AudioPatchifier,
    VideoLatentPatchifier,
    get_pixel_coords,
)
from worldfoundry.base_models.diffusion_model.models.representations.ltx.types import (
    AudioLatentShape,
    SpatioTemporalScaleFactors,
    VideoLatentShape,
)
from worldfoundry.training.api.contracts import ObjectiveBatch, PreparedBatch, TrainingBatch

from ._shared import (
    component_module as _module_from_component,
    module_device_dtype as _module_device_dtype,
)

LTX_DEFAULT_FPS = 24.0
LTX_DEFAULT_LATENT_CHANNELS = 128
LTX_DEFAULT_SPATIAL_COMPRESSION = 32
LTX_DEFAULT_TEMPORAL_COMPRESSION = 8

_CLEAN_CONDITIONING = "ltx_clean_conditioning_latents"
_CONDITIONING_MASK = "ltx_latent_conditioning_mask"
_AUDIO_POSITIONS = "ltx_audio_positions"
_POSITIONS = "ltx_video_positions"


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


def _context_features(velocity_model: nn.Module, *, attention_name: str = "attn2") -> int:
    blocks = getattr(velocity_model, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise TypeError("LTX velocity model must expose transformer_blocks")
    projection = getattr(getattr(blocks[0], attention_name, None), "to_k", None)
    if not isinstance(projection, nn.Linear):
        raise TypeError(f"LTX cross-attention must expose {attention_name}.to_k")
    return int(projection.in_features)


class LTXTrainAdapter:
    """Prepare cached LTX latents and call the native velocity transformer."""

    prediction_type = "flow_velocity"
    lora_target_preset = "ltx-attention"
    conditioning_dropout_owner = "none"

    def __init__(
        self,
        denoiser: object,
        conditioner: object | None = None,
        *,
        expected_latent_channels: int = LTX_DEFAULT_LATENT_CHANNELS,
        temporal_compression: int = LTX_DEFAULT_TEMPORAL_COMPRESSION,
        spatial_compression: int = LTX_DEFAULT_SPATIAL_COMPRESSION,
        default_fps: float = LTX_DEFAULT_FPS,
        first_frame_conditioning_probability: float = 0.0,
        per_sample_first_frame_conditioning: bool = True,
        causal_positions: bool = True,
        discrete_timesteps: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        trainable_module = _module_from_component(denoiser, "model")
        if trainable_module is None:
            raise TypeError("LTX denoiser must expose its transformer as model")
        velocity_model = getattr(trainable_module, "velocity_model", None)
        if not isinstance(velocity_model, nn.Module):
            raise TypeError("LTX transformer module must expose velocity_model")
        if conditioner is not None and not callable(getattr(conditioner, "encode", None)):
            raise TypeError("LTX conditioner must expose encode(request, device=..., dtype=...)")

        probability = float(first_frame_conditioning_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("first_frame_conditioning_probability must be in [0, 1]")
        fps = float(default_fps)
        if fps <= 0.0:
            raise ValueError("default_fps must be positive")

        self.denoiser = denoiser
        self.conditioner = conditioner
        self.trainable_module = trainable_module
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
        self.default_fps = fps
        self.first_frame_conditioning_probability = probability
        self.per_sample_first_frame_conditioning = bool(per_sample_first_frame_conditioning)
        self.causal_positions = bool(causal_positions)
        self.discrete_timesteps = bool(discrete_timesteps)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.patchifier = VideoLatentPatchifier(1)
        self.expected_context_features = _context_features(velocity_model)
        audio_projection = getattr(velocity_model, "audio_patchify_proj", None)
        self.supports_joint_audio = isinstance(audio_projection, nn.Linear)
        self.expected_audio_latent_channels = (
            int(audio_projection.in_features) if isinstance(audio_projection, nn.Linear) else None
        )
        self.expected_audio_context_features = (
            _context_features(velocity_model, attention_name="audio_attn2") if self.supports_joint_audio else None
        )
        self.audio_patchifier = AudioPatchifier(1) if self.supports_joint_audio else None

        input_projection = getattr(velocity_model, "patchify_proj", None)
        output_projection = getattr(velocity_model, "proj_out", None)
        if not isinstance(input_projection, nn.Linear) or not isinstance(output_projection, nn.Linear):
            raise TypeError("LTX velocity model must expose video input/output projections")
        if (int(input_projection.in_features), int(output_projection.out_features)) != (
            self.expected_latent_channels,
            self.expected_latent_channels,
        ):
            raise ValueError("LTX adapter latent channels differ from the loaded transformer")

        set_checkpointing = getattr(velocity_model, "set_gradient_checkpointing", None)
        if not callable(set_checkpointing):
            raise TypeError("LTX velocity model must expose set_gradient_checkpointing")
        set_checkpointing(self.gradient_checkpointing)

        blocks = velocity_model.transformer_blocks
        self.fsdp_block_classes = tuple(dict.fromkeys(type(block) for block in blocks))
        self.conditioning_dropout_probability = 0.0

        conditioner_module = _module_from_component(conditioner, "gemma", "text_encoder", "model")
        self.frozen_modules = () if conditioner_module is None else (conditioner_module,)
        for module in self.frozen_modules:
            module.requires_grad_(False)
            module.eval()
        self.trainable_module.train()

    def _keep_frozen_modules_in_eval(self) -> None:
        for module in self.frozen_modules:
            module.eval()

    def _conditioning(
        self,
        batch: TrainingBatch,
        *,
        frames: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, object]:
        values = {
            key: value
            for key, value in batch.conditions.items()
            if key
            not in {
                "clean_latents",
                "latent_loss_mask",
                "valid_latent_mask",
                "latent_conditioning_mask",
                "audio_context",
            }
        }
        if "video_context" not in values:
            if self.conditioner is None:
                raise RuntimeError("LTX cached training requires video_context and context_mask")
            request = DiffusionRequest(
                prompt=batch.prompts,
                height=height,
                width=width,
                num_frames=frames,
                sampling=SamplingConfig(guidance_scale=1.0),
                metadata=dict(batch.metadata),
            )
            with torch.no_grad():
                encoded = self.conditioner.encode(request, device=device, dtype=dtype)
            if not isinstance(encoded, Conditioning):
                raise TypeError("LTX conditioner must return Conditioning")
            overlap = set(values) & set(encoded.positive)
            if overlap:
                raise ValueError(f"LTX encoded conditioning collides with batch keys: {sorted(overlap)}")
            values = {**encoded.positive, **values}

        normalized: dict[str, object] = {}
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                if key == "context_mask" and value.dtype is torch.bool:
                    target_dtype = torch.int64
                else:
                    target_dtype = dtype if value.is_floating_point() and key != "context_mask" else value.dtype
                normalized[key] = value.detach().to(device=device, dtype=target_dtype)
            else:
                normalized[key] = value

        context = normalized.get("video_context")
        context_mask = normalized.get("context_mask")
        if not isinstance(context, torch.Tensor) or context.ndim != 3:
            raise ValueError("LTX video_context must be [B,sequence,features]")
        if tuple(context.shape[:1]) != (batch.batch_size,) or int(context.shape[-1]) != self.expected_context_features:
            raise ValueError(
                "LTX video_context shape differs from the loaded transformer: "
                f"got {tuple(context.shape)}, expected [B,sequence,{self.expected_context_features}]"
            )
        if not isinstance(context_mask, torch.Tensor) or tuple(context_mask.shape) != tuple(context.shape[:2]):
            raise ValueError("LTX context_mask must match video_context's batch and sequence dimensions")
        return normalized

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
                raise TypeError(f"LTX {name} must be a torch.Tensor")
            mask = value.detach().to(device=device, dtype=torch.float32)
            if mask.ndim == 4 and int(mask.shape[0]) == latent_shape[0]:
                mask = mask.unsqueeze(1)
            try:
                mask = torch.broadcast_to(mask, latent_shape)
            except RuntimeError as error:
                raise ValueError(f"LTX {name} cannot broadcast to clean latents") from error
            combined = mask if combined is None else combined * mask
        return combined

    def _conditioning_mask(
        self,
        batch: TrainingBatch,
        *,
        latent_shape: tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor | None:
        explicit = batch.conditions.get("latent_conditioning_mask")
        if explicit is not None:
            if not isinstance(explicit, torch.Tensor):
                raise TypeError("LTX latent_conditioning_mask must be a torch.Tensor")
            mask = explicit.detach().to(device=device)
            if mask.ndim == 4 and int(mask.shape[0]) == latent_shape[0]:
                mask = mask.unsqueeze(1)
            try:
                return torch.broadcast_to(mask.bool(), (latent_shape[0], 1, *latent_shape[2:]))
            except RuntimeError as error:
                raise ValueError("LTX latent_conditioning_mask cannot broadcast to video latents") from error
        if self.first_frame_conditioning_probability == 0.0:
            return None
        if latent_shape[2] == 1:
            return None
        if self.per_sample_first_frame_conditioning:
            selected = torch.rand(latent_shape[0], device=device) < self.first_frame_conditioning_probability
        else:
            apply_to_batch = random.random() < self.first_frame_conditioning_probability
            selected = torch.full((latent_shape[0],), apply_to_batch, device=device, dtype=torch.bool)
        mask = torch.zeros((latent_shape[0], 1, *latent_shape[2:]), device=device, dtype=torch.bool)
        mask[:, :, 0] = selected[:, None, None, None]
        return mask

    def _positions(
        self,
        *,
        latent_shape: VideoLatentShape,
        fps: float,
        device: torch.device,
    ) -> torch.Tensor:
        latent_coords = self.patchifier.get_patch_grid_bounds(latent_shape, device=device)
        positions = get_pixel_coords(
            latent_coords,
            SpatioTemporalScaleFactors(
                time=self.temporal_compression,
                height=self.spatial_compression,
                width=self.spatial_compression,
            ),
            causal_fix=self.causal_positions,
        ).float()
        positions[:, 0] /= fps
        return positions

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        if not isinstance(batch, TrainingBatch):
            raise TypeError("batch must be a TrainingBatch")
        if batch.pixel_values is not None:
            raise ValueError("LTX native trainer consumes precomputed normalized latents")
        clean_latents = batch.conditions.get("clean_latents")
        if not isinstance(clean_latents, torch.Tensor) or clean_latents.ndim != 5:
            raise ValueError("LTX clean_latents must be a BCTHW tensor")
        if (
            int(clean_latents.shape[0]) != batch.batch_size
            or int(clean_latents.shape[1]) != self.expected_latent_channels
        ):
            raise ValueError("LTX clean_latents batch or channel count is incompatible")

        device, dtype = _module_device_dtype(self.trainable_module)
        clean_latents = clean_latents.detach().to(device=device, dtype=dtype)
        latent_shape_tuple = tuple(int(value) for value in clean_latents.shape)
        latent_shape = VideoLatentShape.from_torch_shape(clean_latents.shape)
        frames = (latent_shape.frames - 1) * self.temporal_compression + 1
        height = latent_shape.height * self.spatial_compression
        width = latent_shape.width * self.spatial_compression
        declared_geometry = (
            int(batch.metadata.get("target_num_frames", frames)),
            int(batch.metadata.get("target_height", height)),
            int(batch.metadata.get("target_width", width)),
        )
        if declared_geometry != (frames, height, width):
            raise ValueError("LTX cache target geometry differs from its latent geometry")
        fps = float(batch.metadata.get("target_fps", self.default_fps))
        if fps <= 0.0:
            raise ValueError("LTX target_fps must be positive")

        self._keep_frozen_modules_in_eval()
        conditioning = self._conditioning(
            batch,
            frames=frames,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
        )
        conditioning[_POSITIONS] = self._positions(latent_shape=latent_shape, fps=fps, device=device)

        loss_mask = self._cached_loss_mask(batch, latent_shape=latent_shape_tuple, device=device)
        conditioning_mask = self._conditioning_mask(batch, latent_shape=latent_shape_tuple, device=device)
        if conditioning_mask is not None:
            conditioning[_CLEAN_CONDITIONING] = clean_latents
            conditioning[_CONDITIONING_MASK] = conditioning_mask
            generated = (~conditioning_mask).to(torch.float32)
            loss_mask = generated if loss_mask is None else loss_mask * generated

        sample_weights = batch.sample_weights
        if isinstance(sample_weights, torch.Tensor):
            sample_weights = sample_weights.to(device=device, dtype=torch.float32)
        return PreparedBatch(
            sample_ids=batch.sample_ids,
            clean_latents=clean_latents,
            conditioning=conditioning,
            loss_mask=loss_mask,
            sample_weights=sample_weights,
        )

    @staticmethod
    def _batch_sigmas(batch: ObjectiveBatch, latents: torch.Tensor) -> torch.Tensor:
        if not isinstance(batch.sigmas, torch.Tensor):
            raise TypeError("LTX objective sigmas must be a torch.Tensor")
        sigmas = batch.sigmas.to(device=latents.device, dtype=torch.float32)
        if sigmas.ndim == 0:
            sigmas = sigmas.expand(latents.shape[0])
        elif sigmas.numel() == latents.shape[0]:
            sigmas = sigmas.reshape(latents.shape[0])
        else:
            raise ValueError("LTX requires one sigma per sample")
        if not bool(torch.isfinite(sigmas).all()) or not bool(((0 <= sigmas) & (sigmas <= 1)).all()):
            raise ValueError("LTX effective sigmas must be finite values in [0, 1]")
        return sigmas

    def forward_train(self, batch: ObjectiveBatch) -> torch.Tensor:
        return self.forward_model(batch, training=True)

    def rollout_conditioning(
        self,
        latents: torch.Tensor,
        conditioning: Mapping[str, object],
        *,
        audio_latents: torch.Tensor | None = None,
        fps: float | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, object]:
        """Prepare connector outputs and native AV coordinates for RL rollout."""

        if latents.ndim != 5 or int(latents.shape[1]) != self.expected_latent_channels:
            raise ValueError("LTX rollout latents must be a compatible BCTHW tensor")
        values = dict(conditioning)
        context = values.get("video_context")
        context_mask = values.get("context_mask")
        if not isinstance(context, torch.Tensor) or context.ndim != 3:
            raise ValueError("LTX rollout video_context must be [B,sequence,features]")
        if int(context.shape[0]) != int(latents.shape[0]) or int(context.shape[-1]) != self.expected_context_features:
            raise ValueError("LTX rollout video_context does not match the latent batch or model")
        if not isinstance(context_mask, torch.Tensor) or tuple(context_mask.shape) != tuple(context.shape[:2]):
            raise ValueError("LTX rollout context_mask must match video_context")

        compute_dtype = dtype or latents.dtype
        values["video_context"] = context.to(device=latents.device, dtype=compute_dtype)
        values["context_mask"] = context_mask.to(device=latents.device, dtype=torch.int64)
        values[_POSITIONS] = self._positions(
            latent_shape=VideoLatentShape.from_torch_shape(latents.shape),
            fps=self.default_fps if fps is None else float(fps),
            device=latents.device,
        )
        if self.supports_joint_audio:
            audio_context = values.get("audio_context")
            if (
                not isinstance(audio_context, torch.Tensor)
                or audio_context.ndim != 3
                or int(audio_context.shape[0]) != int(latents.shape[0])
                or int(audio_context.shape[-1]) != self.expected_audio_context_features
            ):
                raise ValueError("LTX rollout audio_context does not match the latent batch or model")
            if (
                not isinstance(audio_latents, torch.Tensor)
                or audio_latents.ndim != 3
                or int(audio_latents.shape[0]) != int(latents.shape[0])
                or int(audio_latents.shape[-1]) != self.expected_audio_latent_channels
            ):
                raise ValueError("LTX rollout audio latents must be [B,audio_frames,audio_channels]")
            values["audio_context"] = audio_context.to(device=latents.device, dtype=compute_dtype)
            assert self.audio_patchifier is not None
            values[_AUDIO_POSITIONS] = self.audio_patchifier.get_patch_grid_bounds(
                AudioLatentShape(
                    batch=int(audio_latents.shape[0]),
                    channels=1,
                    frames=int(audio_latents.shape[1]),
                    mel_bins=int(audio_latents.shape[2]),
                ),
                device=latents.device,
            )
        return values

    def forward_joint_model(
        self,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        sigmas: torch.Tensor,
        conditioning: Mapping[str, object],
        *,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the LTX-2.x AV DiT without discarding its audio state."""

        if not self.supports_joint_audio:
            raise ValueError("joint LTX rollout requires an audio-video transformer")
        prepared = self.rollout_conditioning(
            video_latents,
            conditioning,
            audio_latents=audio_latents,
            dtype=video_latents.dtype,
        )
        resolved_sigmas = sigmas.to(device=video_latents.device, dtype=torch.float32).reshape(-1)
        if resolved_sigmas.numel() == 1:
            resolved_sigmas = resolved_sigmas.expand(int(video_latents.shape[0]))

        video_tokens = self.patchifier.patchify(video_latents)
        video = Modality(
            latent=video_tokens,
            sigma=resolved_sigmas,
            timesteps=resolved_sigmas[:, None].expand(-1, int(video_tokens.shape[1])),
            positions=prepared[_POSITIONS],
            context=prepared["video_context"],
            context_mask=prepared["context_mask"],
            attention_mask=prepared.get("video_attention_mask"),
        )
        audio = Modality(
            latent=audio_latents,
            sigma=resolved_sigmas,
            timesteps=resolved_sigmas[:, None].expand(-1, int(audio_latents.shape[1])),
            positions=prepared[_AUDIO_POSITIONS],
            context=prepared["audio_context"],
            context_mask=prepared["context_mask"],
            attention_mask=prepared.get("audio_attention_mask"),
        )
        self.trainable_module.train(training)
        self._keep_frozen_modules_in_eval()
        video_velocity, audio_velocity = self.trainable_module(
            video=video,
            audio=audio,
            perturbations=BatchedPerturbationConfig.empty(video_latents.shape[0]),
        )
        if not isinstance(video_velocity, torch.Tensor) or not isinstance(audio_velocity, torch.Tensor):
            raise TypeError("LTX AV transformer must return video and audio velocities")
        return (
            self.patchifier.unpatchify(
                video_velocity,
                VideoLatentShape.from_torch_shape(video_latents.shape),
            ),
            audio_velocity,
        )

    def forward_model(
        self,
        batch: ObjectiveBatch,
        *,
        training: bool,
        branch: str = "positive",
    ) -> torch.Tensor:
        if branch != "positive":
            raise ValueError("LTX video training exposes only positive conditioning")
        if not isinstance(batch, ObjectiveBatch):
            raise TypeError("batch must be an ObjectiveBatch")
        if isinstance(batch.model_input, Mapping) or not isinstance(batch.model_input, torch.Tensor):
            raise TypeError("LTX video training requires one BCTHW model input")
        latents = batch.model_input
        if latents.ndim != 5 or int(latents.shape[1]) != self.expected_latent_channels:
            raise ValueError("LTX model_input must be a compatible BCTHW tensor")

        sigmas = self._batch_sigmas(batch, latents)
        velocity_model = getattr(self.trainable_module, "velocity_model", None)
        if self.discrete_timesteps:
            scale = float(getattr(velocity_model, "timestep_scale_multiplier", 1000.0))
            effective_sigmas = torch.round(sigmas * scale) / scale
        else:
            effective_sigmas = sigmas

        tokens = self.patchifier.patchify(latents)
        token_timesteps = effective_sigmas[:, None].expand(-1, tokens.shape[1])
        conditioning_mask = batch.conditioning.get(_CONDITIONING_MASK)
        clean_conditioning = batch.conditioning.get(_CLEAN_CONDITIONING)
        if conditioning_mask is not None or clean_conditioning is not None:
            if not isinstance(conditioning_mask, torch.Tensor) or not isinstance(clean_conditioning, torch.Tensor):
                raise TypeError("LTX intrinsic conditioning requires clean latents and a mask")
            clean_tokens = self.patchifier.patchify(clean_conditioning.to(device=latents.device, dtype=latents.dtype))
            token_mask = self.patchifier.patchify(conditioning_mask.to(device=latents.device)).bool()
            tokens = torch.where(token_mask, clean_tokens, tokens)
            token_timesteps = torch.where(token_mask.squeeze(-1), torch.zeros_like(token_timesteps), token_timesteps)

        context = batch.conditioning.get("video_context")
        context_mask = batch.conditioning.get("context_mask")
        positions = batch.conditioning.get(_POSITIONS)
        attention_mask = batch.conditioning.get("video_attention_mask")
        if not all(isinstance(value, torch.Tensor) for value in (context, context_mask, positions)):
            raise TypeError("LTX objective batch is missing video context, mask, or positions")
        if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
            raise TypeError("LTX video_attention_mask must be a tensor")

        self.trainable_module.train(training)
        self._keep_frozen_modules_in_eval()
        video = Modality(
            latent=tokens,
            sigma=effective_sigmas,
            timesteps=token_timesteps,
            positions=positions,
            context=context,
            context_mask=context_mask,
            attention_mask=attention_mask,
        )
        video_velocity, audio_velocity = self.trainable_module(
            video=video,
            audio=None,
            perturbations=BatchedPerturbationConfig.empty(latents.shape[0]),
        )
        if not isinstance(video_velocity, torch.Tensor) or audio_velocity is not None:
            raise TypeError("LTX video training returned an unexpected modality set")
        if tuple(video_velocity.shape) != tuple(tokens.shape):
            raise ValueError("LTX video velocity shape differs from patchified latents")
        return self.patchifier.unpatchify(
            video_velocity,
            VideoLatentShape.from_torch_shape(latents.shape),
        )


def build_ltx_train_adapter(
    components: Mapping[ComponentKey, object],
    **options: object,
) -> LTXTrainAdapter:
    """Build an LTX adapter with a native prompt conditioner."""

    try:
        denoiser = components[ComponentKey(ComponentKind.DENOISER)]
        conditioner = components[ComponentKey(ComponentKind.CONDITIONER)]
    except KeyError as error:
        raise KeyError(f"LTX training components are missing {error.args[0]}") from error
    return LTXTrainAdapter(denoiser, conditioner, **options)


def build_cached_ltx_train_adapter(
    components: Mapping[ComponentKey, object],
    **options: object,
) -> LTXTrainAdapter:
    """Build the author-trainer-compatible cached-feature adapter."""

    denoiser_key = ComponentKey(ComponentKind.DENOISER)
    unexpected = tuple(key for key in components if key != denoiser_key)
    if unexpected:
        raise ValueError("cached LTX training accepts only the denoiser component")
    try:
        denoiser = components[denoiser_key]
    except KeyError as error:
        raise KeyError(f"LTX training components are missing {denoiser_key}") from error
    return LTXTrainAdapter(denoiser, None, **options)


__all__ = [
    "LTX_DEFAULT_FPS",
    "LTX_DEFAULT_LATENT_CHANNELS",
    "LTX_DEFAULT_SPATIAL_COMPRESSION",
    "LTX_DEFAULT_TEMPORAL_COMPRESSION",
    "LTXTrainAdapter",
    "build_cached_ltx_train_adapter",
    "build_ltx_train_adapter",
]
