"""Native training adapters for Cosmos Predict and Cosmos3 video generation.

Predict2 and Predict2.5 share a DiT implementation in WorldFoundry, but they
do not share an inference parameterization.  Predict2's public denoiser wraps
the DiT with EDM-style x0 preconditioning; its underlying DiT still predicts
the rectified-flow velocity on the normalized state.  Predict2.5 exposes that
velocity directly.  The separate adapters below preserve that distinction.

GEN3C is intentionally absent: its released repository contains inference,
not a model-author training loop, and its EDM x0 objective is not equivalent
to the flow-matching contracts implemented here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.components import ComponentKey, ComponentKind
from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DenoiserInput,
    DiffusionRequest,
    ModalityState,
    MultiModalDenoiserInput,
    SamplingConfig,
)
from worldfoundry.training.api.contracts import ObjectiveBatch, PreparedBatch, TrainingBatch
from worldfoundry.training.data.video_masks import project_causal_video_mask_to_latent

from ._shared import (
    component_module as _component_module,
    freeze_module as _freeze,
    merge_without_overwrite,
    module_device_dtype as _module_device_dtype,
)

COSMOS_DEFAULT_TRAIN_TIMESTEPS = 1000


def _sample_sigmas(batch: ObjectiveBatch, latents: torch.Tensor) -> torch.Tensor:
    if not isinstance(batch.sigmas, torch.Tensor):
        raise TypeError("Cosmos objective sigmas must be a torch.Tensor")
    sigmas = batch.sigmas.to(device=latents.device, dtype=torch.float32)
    if sigmas.ndim == 0:
        sigmas = sigmas.expand(int(latents.shape[0]))
    elif sigmas.numel() == int(latents.shape[0]):
        sigmas = sigmas.reshape(int(latents.shape[0]))
    else:
        raise ValueError("Cosmos requires one effective flow sigma per sample")
    if not bool(torch.isfinite(sigmas).all()) or not bool(((0 <= sigmas) & (sigmas <= 1)).all()):
        raise ValueError("Cosmos effective flow sigmas must be finite values in [0, 1]")
    return sigmas


def _to_model_values(
    values: Mapping[str, object],
    *,
    device: torch.device,
    dtype: torch.dtype,
    branch: str,
) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for name, value in values.items():
        if name == "negative_context":
            continue
        if isinstance(value, torch.Tensor):
            normalized[name] = value.to(
                device=device,
                dtype=dtype if value.is_floating_point() else value.dtype,
            )
        else:
            normalized[name] = value
    if branch == "negative" and isinstance(values.get("negative_context"), torch.Tensor):
        normalized["context"] = values["negative_context"].to(device=device, dtype=dtype)
    return normalized


def _condition_indicator(
    raw: dict[str, object],
    *,
    batch_size: int,
    latent_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    supplied = raw.pop("condition_indicator", None)
    if supplied is not None:
        if not isinstance(supplied, torch.Tensor):
            raise TypeError("Cosmos condition_indicator must be a torch.Tensor")
        indicator = supplied.to(device=device, dtype=dtype)
        if indicator.ndim == 2 and tuple(indicator.shape) == (batch_size, latent_frames):
            indicator = indicator[:, None, :, None, None]
        try:
            return torch.broadcast_to(indicator, (batch_size, 1, latent_frames, 1, 1))
        except RuntimeError as error:
            raise ValueError("Cosmos condition_indicator cannot broadcast to [B,1,T,1,1]") from error

    counts = raw.pop("num_conditional_frames", 0)
    if isinstance(counts, torch.Tensor):
        counts = counts.to(device=device, dtype=torch.long).reshape(-1)
        if counts.numel() == 1:
            counts = counts.expand(batch_size)
        elif counts.numel() != batch_size:
            raise ValueError("num_conditional_frames must be scalar or contain one value per sample")
    else:
        counts = torch.full((batch_size,), int(counts), device=device, dtype=torch.long)
    if bool(((counts < 0) | (counts > latent_frames)).any()):
        raise ValueError("num_conditional_frames falls outside the latent video")
    frame_ids = torch.arange(latent_frames, device=device).reshape(1, latent_frames)
    return (frame_ids < counts[:, None]).to(dtype=dtype)[:, None, :, None, None]


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
            raise TypeError(f"Cosmos {name} must be a torch.Tensor")
        try:
            mask = torch.broadcast_to(value.to(device=device, dtype=torch.float32), latent_shape)
        except RuntimeError as error:
            raise ValueError(f"Cosmos {name} cannot broadcast to the latent shape") from error
        combined = mask if combined is None else combined * mask
    return combined


class _CosmosVideoFlowTrainAdapter:
    prediction_type = "flow_velocity"
    conditioning_dropout_owner = "none"
    conditioning_dropout_probability = 0.0

    def __init__(
        self,
        denoiser: object,
        codec: object | None,
        conditioner: object | None,
        *,
        model_family: str,
        lora_target_preset: str,
        expected_latent_channels: int = 16,
        temporal_compression: int = 4,
        spatial_compression: int = 8,
        num_train_timesteps: int = COSMOS_DEFAULT_TRAIN_TIMESTEPS,
        conditional_frame_timestep: float = -1.0,
    ) -> None:
        trainable_module = _component_module(denoiser, "model")
        if trainable_module is None or not callable(denoiser):
            raise TypeError("Cosmos denoiser must expose a callable nn.Module as model")
        if codec is not None and not callable(getattr(codec, "encode", None)):
            raise TypeError("Cosmos codec must expose encode(pixels)")
        if conditioner is not None and not callable(getattr(conditioner, "encode", None)):
            raise TypeError("Cosmos conditioner must expose encode(request, device=..., dtype=...)")

        self.denoiser = denoiser
        self.codec = codec
        self.conditioner = conditioner
        self.trainable_module = trainable_module
        self.model_family = model_family
        self.lora_target_preset = lora_target_preset
        self.expected_latent_channels = int(expected_latent_channels)
        self.temporal_compression = int(temporal_compression)
        self.spatial_compression = int(spatial_compression)
        self.num_train_timesteps = int(num_train_timesteps)
        self.model_timestep_scale = float(self.num_train_timesteps)
        self.conditional_frame_timestep = float(conditional_frame_timestep)

        frozen = tuple(
            dict.fromkeys(
                module
                for module in (
                    _component_module(codec, "vae", "model"),
                    _component_module(conditioner, "encoder", "model"),
                )
                if module is not None
            )
        )
        self.frozen_modules = frozen
        for module in frozen:
            _freeze(module)
        self.trainable_module.train()

        blocks = getattr(self.trainable_module, "transformer_blocks", ())
        self.fsdp_block_classes = tuple(dict.fromkeys(type(block) for block in blocks if isinstance(block, nn.Module)))
        patch = tuple(
            int(value) for value in getattr(getattr(self.trainable_module, "config", None), "patch_size", (1, 2, 2))
        )
        if len(patch) != 3:
            raise ValueError("Cosmos Predict DiT must expose a three-axis patch_size")
        self.patch_size = patch

    def _keep_frozen_modules_in_eval(self) -> None:
        for module in self.frozen_modules:
            module.eval()

    def _encoded_context(
        self,
        batch: TrainingBatch,
        raw: dict[str, object],
        *,
        frames: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, object]:
        if "context" in raw:
            values = raw
        else:
            if self.conditioner is None:
                raise RuntimeError("Cosmos batch has no cached context and the adapter has no conditioner")
            request = DiffusionRequest(
                prompt=batch.prompts,
                height=height,
                width=width,
                num_frames=frames,
                sampling=SamplingConfig(guidance_scale=1.0),
                inputs=dict(batch.metadata.get("conditioner_inputs", {})),
            )
            with torch.no_grad():
                encoded = self.conditioner.encode(request, device=device, dtype=dtype)
            if not isinstance(encoded, Conditioning):
                raise TypeError("Cosmos conditioner must return Conditioning")
            values = dict(encoded.positive)
            merge_without_overwrite(values, encoded.shared, source_name="conditioner.shared", family="Cosmos")
            merge_without_overwrite(values, raw, source_name="TrainingBatch.conditions", family="Cosmos")
        normalized = _to_model_values(values, device=device, dtype=dtype, branch="positive")
        context = normalized.get("context")
        if not isinstance(context, torch.Tensor) or context.ndim < 3 or int(context.shape[0]) != batch.batch_size:
            raise ValueError("Cosmos context must start with the training batch dimension")
        negative_context = values.get("negative_context")
        if isinstance(negative_context, torch.Tensor):
            normalized["negative_context"] = negative_context.to(device=device, dtype=dtype)
        return normalized

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        if not isinstance(batch, TrainingBatch):
            raise TypeError("batch must be a TrainingBatch")
        self._keep_frozen_modules_in_eval()
        device, dtype = _module_device_dtype(self.trainable_module)
        cached = batch.conditions.get("clean_latents")
        pixels = batch.pixel_values
        if cached is not None and pixels is not None:
            raise ValueError("Cosmos batch cannot contain pixels and cached clean_latents together")
        if cached is None:
            if not isinstance(pixels, torch.Tensor) or pixels.ndim != 5 or int(pixels.shape[1]) != 3:
                raise TypeError("Cosmos video training requires BCTHW pixels or cached clean_latents")
            if self.codec is None:
                raise RuntimeError("Cosmos raw-video training requires a codec")
            frames, height, width = (int(value) for value in pixels.shape[-3:])
            with torch.no_grad():
                clean = self.codec.encode(pixels)
        else:
            if not isinstance(cached, torch.Tensor):
                raise TypeError("Cosmos cached clean_latents must be a torch.Tensor")
            clean = cached
            frames = (int(clean.shape[2]) - 1) * self.temporal_compression + 1
            height = int(clean.shape[3]) * self.spatial_compression
            width = int(clean.shape[4]) * self.spatial_compression
        if not isinstance(clean, torch.Tensor) or clean.ndim != 5:
            raise ValueError("Cosmos clean latents must have shape [B,C,T,H,W]")
        if int(clean.shape[0]) != batch.batch_size or int(clean.shape[1]) != self.expected_latent_channels:
            raise ValueError("Cosmos clean latent batch or channel count differs from the model")
        if any(int(size) % patch for size, patch in zip(clean.shape[-3:], self.patch_size, strict=True)):
            raise ValueError("Cosmos clean latent geometry is not divisible by the DiT patch size")
        clean = clean.detach().to(device=device, dtype=dtype)

        reserved = {
            "clean_latents",
            "latent_loss_mask",
            "valid_latent_mask",
            "condition_latents",
            "condition_mask",
            "condition_indicator",
            "initial_noise",
            "num_conditional_frames",
        }
        raw = {name: value for name, value in batch.conditions.items() if name not in reserved}
        layout_values = {
            name: batch.conditions[name]
            for name in ("condition_latents", "condition_mask", "condition_indicator", "num_conditional_frames")
            if name in batch.conditions
        }
        indicator = _condition_indicator(
            layout_values,
            batch_size=batch.batch_size,
            latent_frames=int(clean.shape[2]),
            device=device,
            dtype=dtype,
        )
        condition_latents = layout_values.get("condition_latents", clean)
        if not isinstance(condition_latents, torch.Tensor) or tuple(condition_latents.shape) != tuple(clean.shape):
            raise ValueError("Cosmos condition_latents must match clean_latents")
        condition_latents = condition_latents.detach().to(device=device, dtype=dtype)
        condition_mask = layout_values.get("condition_mask")
        if condition_mask is None:
            condition_mask = indicator.expand(-1, -1, -1, int(clean.shape[3]), int(clean.shape[4]))
        elif isinstance(condition_mask, torch.Tensor):
            condition_mask = torch.broadcast_to(
                condition_mask.to(device=device, dtype=dtype),
                (batch.batch_size, 1, *tuple(int(value) for value in clean.shape[-3:])),
            )
        else:
            raise TypeError("Cosmos condition_mask must be a torch.Tensor")

        context = self._encoded_context(
            batch,
            raw,
            frames=frames,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
        )
        context.update(
            {
                "condition_latents": condition_latents,
                "condition_mask": condition_mask,
                "condition_indicator": indicator,
                "conditional_frame_timestep": self.conditional_frame_timestep,
                "fps": float(batch.metadata.get("target_fps", context.get("fps", 16.0))),
            }
        )
        context.setdefault(
            "padding_mask",
            torch.zeros((batch.batch_size, 1, int(clean.shape[3]), int(clean.shape[4])), device=device, dtype=dtype),
        )

        if cached is None and batch.valid_mask is not None:
            assert isinstance(pixels, torch.Tensor)
            loss_mask = project_causal_video_mask_to_latent(
                batch.valid_mask.to(device=device),
                pixel_shape=tuple(int(value) for value in pixels.shape),
                latent_shape=tuple(int(value) for value in clean.shape[-3:]),
                temporal_compression=self.temporal_compression,
            )
        else:
            loss_mask = _cached_loss_mask(batch, latent_shape=tuple(clean.shape), device=device)
        sample_weights = batch.sample_weights
        if isinstance(sample_weights, torch.Tensor):
            sample_weights = sample_weights.to(device=device, dtype=torch.float32)

        metadata: dict[str, Any] = dict(batch.metadata)
        metadata.update(
            {
                "model_family": self.model_family,
                "prediction_type": self.prediction_type,
                "model_timestep_scale": self.model_timestep_scale,
            }
        )
        return PreparedBatch(
            sample_ids=batch.sample_ids,
            clean_latents=clean,
            conditioning=context,
            loss_mask=loss_mask,
            sample_weights=sample_weights,
            metadata=metadata,
        )

    def _video_batch(
        self,
        batch: ObjectiveBatch,
        *,
        branch: str,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        if not isinstance(batch, ObjectiveBatch) or isinstance(batch.model_input, Mapping):
            raise TypeError("Cosmos Predict training requires one tensor model_input")
        if not isinstance(batch.model_input, torch.Tensor) or batch.model_input.ndim != 5:
            raise ValueError("Cosmos Predict model_input must have shape [B,C,T,H,W]")
        latents = batch.model_input
        if int(latents.shape[1]) != self.expected_latent_channels:
            raise ValueError("Cosmos Predict model_input channel count differs from the model")
        sigmas = _sample_sigmas(batch, latents)
        values = _to_model_values(batch.conditioning, device=latents.device, dtype=latents.dtype, branch=branch)
        return latents, sigmas, values

    def forward_train(self, batch: ObjectiveBatch) -> torch.Tensor:
        return self.forward_model(batch, training=True)


class CosmosPredict2TrainAdapter(_CosmosVideoFlowTrainAdapter):
    """Predict2 SFT through the raw RF DiT, before its inference x0 wrapper."""

    def __init__(self, denoiser: object, codec: object | None, conditioner: object | None, **options: object) -> None:
        super().__init__(
            denoiser,
            codec,
            conditioner,
            model_family="cosmos-predict2",
            lora_target_preset="cosmos-predict-attention-mlp",
            **options,
        )

    def forward_model(
        self,
        batch: ObjectiveBatch,
        *,
        training: bool,
        branch: str = "positive",
    ) -> torch.Tensor:
        latents, sigmas, values = self._video_batch(batch, branch=branch)
        context = values.get("context")
        condition_latents = values.get("condition_latents")
        condition_mask = values.get("condition_mask")
        indicator = values.get("condition_indicator")
        if not all(
            isinstance(value, torch.Tensor) for value in (context, condition_latents, condition_mask, indicator)
        ):
            raise TypeError("Cosmos Predict2 conditioning is incomplete")

        self.trainable_module.train(training)
        self._keep_frozen_modules_in_eval()
        model_latents = condition_mask * condition_latents + (1.0 - condition_mask) * latents
        conditional_sigma = float(getattr(self.denoiser, "sigma_conditional", 0.0001))
        conditional_timestep = conditional_sigma / (conditional_sigma + 1.0)
        frame_indicator = indicator[:, 0, :, 0, 0]
        timesteps = frame_indicator * conditional_timestep + (1.0 - frame_indicator) * sigmas[:, None]
        prediction = self.trainable_module(
            model_latents,
            timesteps,
            context,
            fps=float(values.get("fps", 16.0)),
            condition_mask=condition_mask,
            padding_mask=values.get("padding_mask"),
        )
        if not isinstance(prediction, torch.Tensor) or prediction.shape != latents.shape:
            raise ValueError("Cosmos Predict2 DiT velocity must preserve the latent shape")
        if batch.metadata.get("cosmos_replace_conditioned_target") is True and isinstance(batch.target, torch.Tensor):
            prediction = condition_mask * batch.target.to(prediction) + (1.0 - condition_mask) * prediction
        return prediction


class CosmosPredict25TrainAdapter(_CosmosVideoFlowTrainAdapter):
    """Predict2.5 full/LoRA SFT using its released RF velocity interface."""

    def __init__(self, denoiser: object, codec: object | None, conditioner: object | None, **options: object) -> None:
        super().__init__(
            denoiser,
            codec,
            conditioner,
            model_family="cosmos-predict2.5",
            lora_target_preset="cosmos-predict-attention-mlp",
            **options,
        )

    def forward_model(
        self,
        batch: ObjectiveBatch,
        *,
        training: bool,
        branch: str = "positive",
    ) -> torch.Tensor:
        latents, sigmas, values = self._video_batch(batch, branch=branch)
        initial_noise = batch.noise
        if not isinstance(initial_noise, torch.Tensor):
            initial_noise = torch.zeros_like(latents)
        values["initial_noise"] = initial_noise.to(device=latents.device, dtype=latents.dtype)
        self.trainable_module.train(training)
        self._keep_frozen_modules_in_eval()
        output = self.denoiser(
            DenoiserInput(
                latents=latents,
                timestep=sigmas * self.model_timestep_scale,
                next_timestep=torch.zeros_like(sigmas),
                conditioning=values,
                step_index=0,
                total_steps=self.num_train_timesteps,
                branch=branch,
            )
        )
        prediction = getattr(output, "sample", None)
        if not isinstance(prediction, torch.Tensor) or prediction.shape != latents.shape:
            raise ValueError("Cosmos Predict2.5 denoiser velocity must preserve the latent shape")
        condition_mask = values.get("condition_mask")
        if (
            batch.metadata.get("cosmos_replace_conditioned_target") is True
            and isinstance(batch.target, torch.Tensor)
            and isinstance(condition_mask, torch.Tensor)
        ):
            prediction = condition_mask * batch.target.to(prediction) + (1.0 - condition_mask) * prediction
        return prediction


class Cosmos3TrainAdapter:
    """Cached video flow-matching adapter for the native Cosmos3 joint graph.

    The released ``cosmos-framework`` SFT recipes train the vision branch.  The
    underlying denoiser can pack sound and action tokens, but those modalities
    do not yet have a framework-owned training cache or an author-trainer parity
    path here.
    """

    prediction_type = "flow_velocity"
    lora_target_preset = "cosmos3-generation-attention"
    conditioning_dropout_owner = "objective"
    conditioning_dropout_probability = 0.1

    def __init__(
        self,
        denoiser: object,
        conditioner: object | None = None,
        *,
        expected_latent_channels: int = 48,
        num_train_timesteps: int = COSMOS_DEFAULT_TRAIN_TIMESTEPS,
        gradient_checkpointing: bool = False,
    ) -> None:
        trainable_module = _component_module(denoiser, "model")
        if trainable_module is None or not callable(denoiser):
            raise TypeError("Cosmos3 denoiser must expose a callable nn.Module as model")
        self.denoiser = denoiser
        self.conditioner = conditioner
        self.trainable_module = trainable_module
        self.expected_latent_channels = int(expected_latent_channels)
        self.num_train_timesteps = int(num_train_timesteps)
        self.model_timestep_scale = float(self.num_train_timesteps)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        set_checkpointing = getattr(self.trainable_module, "set_gradient_checkpointing", None)
        if not callable(set_checkpointing):
            raise TypeError("Cosmos3 transformer must expose set_gradient_checkpointing")
        set_checkpointing(self.gradient_checkpointing)
        conditioner_module = _component_module(conditioner, "encoder", "model")
        self.frozen_modules = (conditioner_module,) if conditioner_module is not None else ()
        _freeze(conditioner_module)
        self.trainable_module.train()
        layers = getattr(self.trainable_module, "layers", ())
        self.fsdp_block_classes = tuple(dict.fromkeys(type(layer) for layer in layers if isinstance(layer, nn.Module)))

    def _input_ids(self, batch: TrainingBatch, *, device: torch.device) -> torch.Tensor:
        supplied = batch.conditions.get("input_ids")
        if supplied is None:
            if self.conditioner is None:
                raise RuntimeError("Cosmos3 training requires cached input_ids or its tokenizer conditioner")
            clean = batch.conditions["clean_latents"]
            video = clean["video"] if isinstance(clean, Mapping) else clean
            frames = int(batch.metadata.get("target_num_frames", (int(video.shape[2]) - 1) * 4 + 1))
            height = int(batch.metadata.get("target_height", int(video.shape[3]) * 16))
            width = int(batch.metadata.get("target_width", int(video.shape[4]) * 16))
            request = DiffusionRequest(
                prompt=batch.prompts,
                height=height,
                width=width,
                num_frames=frames,
                sampling=SamplingConfig(guidance_scale=1.0),
                inputs={"fps": float(batch.metadata.get("target_fps", 24.0))},
            )
            encoded = self.conditioner.encode(request, device=device, dtype=torch.float32)
            supplied = encoded.positive.get("input_ids")
        if not isinstance(supplied, torch.Tensor):
            raise TypeError("Cosmos3 input_ids must be a torch.Tensor")
        if supplied.ndim == 1 and batch.batch_size == 1:
            supplied = supplied.unsqueeze(0)
        if supplied.ndim != 2 or int(supplied.shape[0]) != batch.batch_size:
            raise ValueError("Cosmos3 input_ids must have shape [B,L]")
        return supplied.to(device=device, dtype=torch.long)

    @staticmethod
    def _empty_input_ids(batch: TrainingBatch, *, device: torch.device) -> torch.Tensor | None:
        supplied = batch.conditions.get("empty_input_ids")
        if supplied is None:
            return None
        if not isinstance(supplied, torch.Tensor):
            raise TypeError("Cosmos3 empty_input_ids must be a torch.Tensor")
        if supplied.ndim == 1 and batch.batch_size == 1:
            supplied = supplied.unsqueeze(0)
        if supplied.ndim != 2 or int(supplied.shape[0]) != batch.batch_size:
            raise ValueError("Cosmos3 empty_input_ids must have shape [B,L]")
        return supplied.to(device=device, dtype=torch.long)

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        if not isinstance(batch, TrainingBatch):
            raise TypeError("batch must be a TrainingBatch")
        cached = batch.conditions.get("clean_latents")
        if cached is None:
            raise ValueError("Cosmos3 video training currently consumes cached latents")
        clean_tree = {"video": cached} if isinstance(cached, torch.Tensor) else dict(cached)
        if set(clean_tree) != {"video"}:
            raise ValueError("Cosmos3 SFT currently supports only the video modality")
        device, dtype = _module_device_dtype(self.trainable_module)
        clean: dict[str, torch.Tensor] = {}
        for name, value in clean_tree.items():
            if not isinstance(value, torch.Tensor) or int(value.shape[0]) != batch.batch_size:
                raise ValueError(f"Cosmos3 {name} latent must start with the training batch size")
            if name == "video" and (value.ndim != 5 or int(value.shape[1]) != self.expected_latent_channels):
                raise ValueError("Cosmos3 video latents have incompatible shape or channels")
            clean[name] = value.detach().to(device=device, dtype=dtype)

        raw_masks = batch.conditions.get("denoise_masks", {})
        if not isinstance(raw_masks, Mapping):
            raise TypeError("Cosmos3 denoise_masks must be a mapping")
        masks: dict[str, torch.Tensor] = {}
        for name, value in clean.items():
            mask = raw_masks.get(name)
            if mask is None:
                masks[name] = torch.ones_like(value, dtype=torch.float32)
            elif isinstance(mask, torch.Tensor):
                masks[name] = torch.broadcast_to(mask.to(device=device, dtype=torch.float32), value.shape)
            else:
                raise TypeError(f"Cosmos3 {name} denoise mask must be a tensor")

        conditioning: dict[str, object] = {
            name: value
            for name, value in batch.conditions.items()
            if name not in {"clean_latents", "denoise_masks", "latent_loss_mask", "valid_latent_mask"}
        }
        conditioning["input_ids"] = self._input_ids(batch, device=device)
        empty_input_ids = self._empty_input_ids(batch, device=device)
        if empty_input_ids is not None:
            conditioning["empty_input_ids"] = empty_input_ids
        conditioning["denoise_masks"] = masks
        conditioning.setdefault("fps", float(batch.metadata.get("target_fps", 24.0)))
        valid_mask = _cached_loss_mask(
            batch,
            latent_shape=tuple(clean["video"].shape),
            device=device,
        )
        sample_weights = batch.sample_weights
        if isinstance(sample_weights, torch.Tensor):
            sample_weights = sample_weights.to(device=device, dtype=torch.float32)
        return PreparedBatch(
            sample_ids=batch.sample_ids,
            clean_latents=clean,
            conditioning=conditioning,
            loss_mask=None if valid_mask is None else {"video": valid_mask},
            sample_weights=sample_weights,
            metadata={**batch.metadata, "model_family": "cosmos3", "prediction_type": self.prediction_type},
        )

    @staticmethod
    def _state(
        name: str,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        mask: torch.Tensor,
    ) -> ModalityState:
        latent = mask * noisy + (1.0 - mask) * clean
        if name == "video":
            positions = torch.arange(int(latent.shape[2]), device=latent.device)
            return ModalityState(latent=latent, denoise_mask=mask, clean_latent=clean, positions=positions)
        squeezed = latent[0]
        squeezed_mask = mask[0]
        squeezed_clean = clean[0]
        temporal = 1 if name == "sound" else 0
        positions = torch.arange(int(squeezed.shape[temporal]), device=latent.device)
        return ModalityState(
            latent=squeezed,
            denoise_mask=squeezed_mask,
            clean_latent=squeezed_clean,
            positions=positions,
        )

    def forward_train(self, batch: ObjectiveBatch) -> Mapping[str, torch.Tensor]:
        return self.forward_model(batch, training=True)

    def forward_model(
        self,
        batch: ObjectiveBatch,
        *,
        training: bool,
        branch: str = "positive",
    ) -> Mapping[str, torch.Tensor]:
        if not isinstance(batch.model_input, Mapping) or not isinstance(batch.target, Mapping):
            raise TypeError("Cosmos3 requires modality-mapped model_input and target")
        noisy = dict(batch.model_input)
        reference = noisy["video"]
        if not isinstance(reference, torch.Tensor):
            raise TypeError("Cosmos3 video objective must be a tensor")
        sigmas = _sample_sigmas(batch, reference)
        raw_masks = batch.conditioning.get("denoise_masks")
        if not isinstance(raw_masks, Mapping):
            raise TypeError("Cosmos3 objective conditioning is missing denoise_masks")
        self.trainable_module.train(training)
        for module in self.frozen_modules:
            module.eval()
        predictions: dict[str, list[torch.Tensor]] = {name: [] for name in noisy}
        for sample_index in range(int(reference.shape[0])):
            states: dict[str, ModalityState] = {}
            for name, value in noisy.items():
                target = batch.target[name]
                mask = raw_masks[name]
                if not all(isinstance(item, torch.Tensor) for item in (value, target, mask)):
                    raise TypeError("Cosmos3 modality tensors are incomplete")
                local_sigma = sigmas[sample_index].to(value)
                sample_noisy = value[sample_index : sample_index + 1]
                sample_clean = sample_noisy - local_sigma * target[sample_index : sample_index + 1]
                states[name] = self._state(
                    name,
                    sample_noisy,
                    sample_clean,
                    mask[sample_index : sample_index + 1].to(value),
                )

            conditioning: dict[str, object] = {}
            for name, value in batch.conditioning.items():
                if name == "denoise_masks":
                    continue
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and int(value.shape[0]) == int(reference.shape[0])
                ):
                    value = value[sample_index]
                conditioning[name] = value
            empty_input_ids = conditioning.pop("empty_input_ids", None)
            caption_dropout = conditioning.pop("caption_dropout_mask", None)
            conditioning.pop("num_conditional_frames", None)
            if caption_dropout is not None:
                if not isinstance(caption_dropout, torch.Tensor) or caption_dropout.numel() != 1:
                    raise ValueError("Cosmos3 caption_dropout_mask must contain one value per sample")
                if bool(caption_dropout.item()):
                    if not isinstance(empty_input_ids, torch.Tensor):
                        raise ValueError("Cosmos3 caption dropout requires cached empty_input_ids")
                    conditioning["input_ids"] = empty_input_ids
            output = self.denoiser(
                MultiModalDenoiserInput(
                    modalities=states,
                    # Cosmos3 uses continuous RF sigmas for corruption but
                    # embeds the corresponding training timestep in [0,1000].
                    timestep=sigmas[sample_index] * self.num_train_timesteps,
                    conditioning=conditioning,
                    step_index=0,
                    total_steps=self.num_train_timesteps,
                    branch=branch,
                )
            )
            samples = dict(getattr(output, "samples", {}))
            if set(samples) != set(noisy):
                raise ValueError("Cosmos3 denoiser prediction modalities differ from the objective")
            for name, value in samples.items():
                predictions[name].append(value if name == "video" else value.unsqueeze(0))
        return {name: torch.cat(values, dim=0) for name, values in predictions.items()}


def _require_component(
    components: Mapping[ComponentKey, object],
    kind: ComponentKind,
    name: str = "main",
) -> object:
    key = ComponentKey(kind, name)
    try:
        return components[key]
    except KeyError as error:
        raise KeyError(f"Cosmos training components are missing {key}") from error


def build_cosmos_predict2_train_adapter(
    components: Mapping[ComponentKey, object],
    **options: object,
) -> CosmosPredict2TrainAdapter:
    return CosmosPredict2TrainAdapter(
        _require_component(components, ComponentKind.DENOISER),
        _require_component(components, ComponentKind.LATENT_ENCODER, "codec"),
        _require_component(components, ComponentKind.CONDITIONER),
        **options,
    )


def build_cached_cosmos_predict2_train_adapter(
    components: Mapping[ComponentKey, object],
    **options: object,
) -> CosmosPredict2TrainAdapter:
    return CosmosPredict2TrainAdapter(
        _require_component(components, ComponentKind.DENOISER),
        None,
        None,
        **options,
    )


def build_cosmos_predict25_train_adapter(
    components: Mapping[ComponentKey, object],
    **options: object,
) -> CosmosPredict25TrainAdapter:
    return CosmosPredict25TrainAdapter(
        _require_component(components, ComponentKind.DENOISER),
        _require_component(components, ComponentKind.LATENT_INITIALIZER),
        _require_component(components, ComponentKind.CONDITIONER),
        **options,
    )


def build_cached_cosmos_predict25_train_adapter(
    components: Mapping[ComponentKey, object],
    **options: object,
) -> CosmosPredict25TrainAdapter:
    return CosmosPredict25TrainAdapter(
        _require_component(components, ComponentKind.DENOISER),
        None,
        None,
        **options,
    )


def build_cached_cosmos3_train_adapter(
    components: Mapping[ComponentKey, object],
    **options: object,
) -> Cosmos3TrainAdapter:
    return Cosmos3TrainAdapter(
        _require_component(components, ComponentKind.DENOISER),
        **options,
    )


__all__ = [
    "COSMOS_DEFAULT_TRAIN_TIMESTEPS",
    "Cosmos3TrainAdapter",
    "CosmosPredict2TrainAdapter",
    "CosmosPredict25TrainAdapter",
    "build_cached_cosmos3_train_adapter",
    "build_cached_cosmos_predict2_train_adapter",
    "build_cached_cosmos_predict25_train_adapter",
    "build_cosmos_predict2_train_adapter",
    "build_cosmos_predict25_train_adapter",
]
