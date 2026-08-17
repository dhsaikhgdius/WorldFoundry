"""Training adapters for the two native HunyuanVideo transformer graphs.

The original model and HunyuanVideo 1.5 deliberately remain separate here:
the former consumes a 16-channel latent and embedded guidance, while the
latter consumes ``[noisy, condition, mask]`` channels and two text streams.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

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
    module_device_dtype as _module_device_dtype,
)

HUNYUAN_VIDEO_TIMESTEP_SCALE = 1000.0
HUNYUAN_VIDEO_MODEL_RECIPES = frozenset(
    {
        "hunyuanvideo-t2v",
        "hunyuanvideo-1.5-t2v",
    }
)

HunyuanVideoArchitecture = Literal["original", "refined"]


@dataclass(frozen=True, slots=True)
class HunyuanVideoModelContract:
    model_recipe: str
    architecture: HunyuanVideoArchitecture
    latent_channels: int
    spatial_compression: int
    temporal_compression: int = 4
    embedded_guidance_scale: float = 1.0


_MODEL_CONTRACTS = {
    "hunyuanvideo-t2v": HunyuanVideoModelContract(
        model_recipe="hunyuanvideo-t2v",
        architecture="original",
        latent_channels=16,
        spatial_compression=8,
    ),
    "hunyuanvideo-1.5-t2v": HunyuanVideoModelContract(
        model_recipe="hunyuanvideo-1.5-t2v",
        architecture="refined",
        latent_channels=32,
        spatial_compression=16,
    ),
}


def hunyuan_video_model_contract(model_recipe: str) -> HunyuanVideoModelContract:
    try:
        return _MODEL_CONTRACTS[str(model_recipe)]
    except KeyError as error:
        raise ValueError(f"unsupported HunyuanVideo training recipe: {model_recipe!r}") from error


def _batched_sigmas(sigmas: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    values = sigmas.to(device=latents.device, dtype=torch.float32)
    if values.ndim == 0:
        return values.expand(latents.shape[0])
    if values.numel() == latents.shape[0]:
        return values.reshape(latents.shape[0])
    raise ValueError("HunyuanVideo requires one sigma per sample")


class HunyuanVideoTrainAdapter:
    """Own cached conditioning, latent packing, and the native DiT call."""

    prediction_type = "flow_velocity"
    conditioning_dropout_owner = "none"

    def __init__(
        self,
        denoiser: object,
        *,
        model_recipe: str,
        codec: object | None = None,
        conditioner: object | None = None,
        model_timestep_scale: float = HUNYUAN_VIDEO_TIMESTEP_SCALE,
    ) -> None:
        contract = hunyuan_video_model_contract(model_recipe)
        trainable_module = _component_module(denoiser, "model")
        if trainable_module is None or not callable(denoiser):
            raise TypeError("HunyuanVideo denoiser must be callable and expose model")
        if codec is not None and not callable(getattr(codec, "encode", None)):
            raise TypeError("HunyuanVideo codec must expose encode")
        if conditioner is not None and not callable(getattr(conditioner, "encode", None)):
            raise TypeError("HunyuanVideo conditioner must expose encode")

        config = getattr(trainable_module, "config", None)
        if contract.architecture == "refined" and bool(getattr(config, "use_meanflow", False)):
            raise ValueError(
                "mean-flow HunyuanVideo checkpoints need both current and next timesteps; "
                "the stochastic flow-policy replay contract is single-timestep"
            )

        self.denoiser = denoiser
        self.codec = codec
        self.conditioner = conditioner
        self.trainable_module = trainable_module
        self.contract = contract
        self.model_recipe = contract.model_recipe
        self.model_timestep_scale = float(model_timestep_scale)
        self.lora_target_preset = (
            "hunyuanvideo-attention-mlp" if contract.architecture == "original" else "hunyuanvideo-1.5-attention-mlp"
        )
        blocks = tuple(getattr(trainable_module, "double_blocks", ())) + tuple(
            getattr(trainable_module, "single_blocks", ())
        )
        self.fsdp_block_classes = tuple(dict.fromkeys(type(block) for block in blocks if isinstance(block, nn.Module)))
        if not self.fsdp_block_classes:
            # Exact-signature test modules and small research variants can expose
            # a single trainable root; production checkpoints expose both stacks.
            self.fsdp_block_classes = (type(trainable_module),)

        frozen_candidates = [
            _component_module(codec, "vae", "model"),
        ]
        if isinstance(conditioner, nn.Module):
            frozen_candidates.append(conditioner)
        elif conditioner is not None:
            frozen_candidates.extend(
                value
                for name in (
                    "primary",
                    "clip",
                    "text_encoder",
                    "byt5_model",
                    "vision_encoder",
                )
                if isinstance((value := getattr(conditioner, name, None)), nn.Module)
            )
        frozen = tuple(dict.fromkeys(module for module in frozen_candidates if module is not None))
        self.frozen_modules = frozen
        for module in frozen:
            module.requires_grad_(False)
            module.eval()
        self.trainable_module.train()

    @property
    def expected_latent_channels(self) -> int:
        return self.contract.latent_channels

    def latent_shape(self, generation: Mapping[str, int]) -> tuple[int, int, int, int]:
        frames = int(generation["num_frames"])
        height = int(generation["height"])
        width = int(generation["width"])
        if (frames - 1) % self.contract.temporal_compression:
            raise ValueError("HunyuanVideo num_frames must satisfy 1 + 4k")
        if height % self.contract.spatial_compression or width % self.contract.spatial_compression:
            raise ValueError(
                f"{self.model_recipe} height and width must be divisible by {self.contract.spatial_compression}"
            )
        return (
            self.expected_latent_channels,
            (frames - 1) // self.contract.temporal_compression + 1,
            height // self.contract.spatial_compression,
            width // self.contract.spatial_compression,
        )

    def _keep_frozen_modules_in_eval(self) -> None:
        for module in self.frozen_modules:
            module.eval()

    def _encode_conditioning(
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
            if key not in {"clean_latents", "latent_loss_mask", "valid_latent_mask"}
        }
        if "text_states" not in values:
            if self.conditioner is None:
                raise RuntimeError("HunyuanVideo rollout requires cached text conditioning")
            request = DiffusionRequest(
                prompt=batch.prompts,
                height=height,
                width=width,
                num_frames=frames,
                sampling=SamplingConfig(guidance_scale=self.contract.embedded_guidance_scale),
                metadata=dict(batch.metadata),
            )
            with torch.no_grad():
                encoded = self.conditioner.encode(request, device=device, dtype=dtype)
            if not isinstance(encoded, Conditioning):
                raise TypeError("HunyuanVideo conditioner must return Conditioning")
            values = {**encoded.positive, **encoded.shared, **values}

        normalized: dict[str, object] = {}
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                target_dtype = dtype if value.is_floating_point() else value.dtype
                normalized[key] = value.detach().to(device=device, dtype=target_dtype)
            else:
                normalized[key] = value
        normalized.setdefault("embedded_guidance_scale", self.contract.embedded_guidance_scale)
        required = {"text_states", "text_mask", "text_states_2"}
        if self.contract.architecture == "refined":
            required = {"text_states", "text_mask", "byt5_text_states", "byt5_text_mask"}
        missing = sorted(name for name in required if not isinstance(normalized.get(name), torch.Tensor))
        if missing:
            raise ValueError(f"{self.model_recipe} conditioning is missing tensors: {missing}")
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
                raise TypeError(f"HunyuanVideo {name} must be a tensor")
            mask = value.detach().to(device=device, dtype=torch.float32)
            if mask.ndim == 4 and int(mask.shape[0]) == latent_shape[0]:
                mask = mask.unsqueeze(1)
            try:
                mask = torch.broadcast_to(mask, latent_shape)
            except RuntimeError as error:
                raise ValueError(f"HunyuanVideo {name} cannot broadcast to clean latents") from error
            combined = mask if combined is None else combined * mask
        return combined

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        self._keep_frozen_modules_in_eval()
        device, dtype = _module_device_dtype(self.trainable_module)
        clean_latents = batch.conditions.get("clean_latents")
        if clean_latents is None:
            if not isinstance(batch.pixel_values, torch.Tensor) or self.codec is None:
                raise ValueError("HunyuanVideo training requires cached latents or pixels with a codec")
            pixels = batch.pixel_values.to(device=device, dtype=dtype)
            with torch.no_grad():
                clean_latents = self.codec.encode(pixels)
            frames, height, width = (int(value) for value in pixels.shape[-3:])
        else:
            if batch.pixel_values is not None:
                raise ValueError("HunyuanVideo batch cannot contain pixels and cached latents together")
            if not isinstance(clean_latents, torch.Tensor) or clean_latents.ndim != 5:
                raise ValueError("HunyuanVideo clean_latents must be BCTHW")
            frames = int(
                batch.metadata.get(
                    "target_num_frames",
                    (int(clean_latents.shape[2]) - 1) * self.contract.temporal_compression + 1,
                )
            )
            height = int(
                batch.metadata.get("target_height", int(clean_latents.shape[3]) * self.contract.spatial_compression)
            )
            width = int(
                batch.metadata.get("target_width", int(clean_latents.shape[4]) * self.contract.spatial_compression)
            )
        if not isinstance(clean_latents, torch.Tensor) or clean_latents.ndim != 5:
            raise ValueError("HunyuanVideo codec output must be BCTHW")
        expected = (batch.batch_size, self.expected_latent_channels)
        if tuple(clean_latents.shape[:2]) != expected:
            raise ValueError(
                f"{self.model_recipe} clean latents must start with {expected}; got {tuple(clean_latents.shape)}"
            )
        clean_latents = clean_latents.detach().to(device=device, dtype=dtype)
        conditioning = self._encode_conditioning(
            batch,
            frames=frames,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
        )
        sample_weights = batch.sample_weights
        if isinstance(sample_weights, torch.Tensor):
            sample_weights = sample_weights.to(device=device, dtype=torch.float32)
        return PreparedBatch(
            sample_ids=batch.sample_ids,
            clean_latents=clean_latents,
            conditioning=conditioning,
            loss_mask=self._cached_loss_mask(
                batch,
                latent_shape=tuple(clean_latents.shape),
                device=device,
            ),
            sample_weights=sample_weights,
            metadata={
                **dict(batch.metadata),
                "model_family": self.model_recipe,
                "prediction_type": self.prediction_type,
                "model_timestep_scale": self.model_timestep_scale,
            },
        )

    def _refined_conditioning(
        self,
        latents: torch.Tensor,
        values: Mapping[str, object],
    ) -> dict[str, object]:
        conditioning = dict(values)
        conditioning["condition_latents"] = torch.zeros(
            latents.shape[0],
            latents.shape[1] + 1,
            *latents.shape[2:],
            device=latents.device,
            dtype=latents.dtype,
        )
        return conditioning

    def forward_train(self, batch: ObjectiveBatch) -> torch.Tensor:
        return self.forward_model(batch, training=True)

    def forward_model(
        self,
        batch: ObjectiveBatch,
        *,
        training: bool,
        branch: str = "positive",
    ) -> torch.Tensor:
        if branch != "positive":
            raise ValueError("HunyuanVideo T2V training exposes only positive conditioning")
        if not isinstance(batch.model_input, torch.Tensor) or batch.model_input.ndim != 5:
            raise TypeError("HunyuanVideo model_input must be a BCTHW tensor")
        latents = batch.model_input
        if int(latents.shape[1]) != self.expected_latent_channels:
            raise ValueError(f"{self.model_recipe} expects {self.expected_latent_channels} latent channels")
        sigmas = _batched_sigmas(batch.sigmas, latents)
        conditioning = dict(batch.conditioning)
        conditioning.setdefault("embedded_guidance_scale", self.contract.embedded_guidance_scale)
        if self.contract.architecture == "refined":
            conditioning = self._refined_conditioning(latents, conditioning)

        self.trainable_module.train(training)
        self._keep_frozen_modules_in_eval()
        timesteps = sigmas * self.model_timestep_scale
        output = self.denoiser(
            DenoiserInput(
                latents=latents,
                timestep=timesteps,
                next_timestep=torch.zeros_like(timesteps),
                conditioning=conditioning,
                step_index=0,
                total_steps=int(self.model_timestep_scale),
                branch=branch,
            )
        )
        sample = getattr(output, "sample", None)
        if not isinstance(sample, torch.Tensor) or sample.shape != latents.shape:
            raise ValueError("HunyuanVideo denoiser prediction must match the latent trajectory")
        return sample


def build_cached_hunyuan_video_train_adapter(
    components: Mapping[ComponentKey, object],
    *,
    model_recipe: str,
    **options: object,
) -> HunyuanVideoTrainAdapter:
    denoiser_key = ComponentKey(ComponentKind.DENOISER)
    if set(components) != {denoiser_key}:
        raise ValueError("cached HunyuanVideo roles must materialize only the denoiser")
    return HunyuanVideoTrainAdapter(
        components[denoiser_key],
        model_recipe=model_recipe,
        **options,
    )


def build_hunyuan_video_train_adapter(
    components: Mapping[ComponentKey, object],
    *,
    model_recipe: str,
    **options: object,
) -> HunyuanVideoTrainAdapter:
    return HunyuanVideoTrainAdapter(
        components[ComponentKey(ComponentKind.DENOISER)],
        model_recipe=model_recipe,
        codec=components.get(ComponentKey(ComponentKind.LATENT_ENCODER, "codec")),
        conditioner=components.get(ComponentKey(ComponentKind.CONDITIONER)),
        **options,
    )


__all__ = [
    "HUNYUAN_VIDEO_MODEL_RECIPES",
    "HUNYUAN_VIDEO_TIMESTEP_SCALE",
    "HunyuanVideoModelContract",
    "HunyuanVideoTrainAdapter",
    "build_cached_hunyuan_video_train_adapter",
    "build_hunyuan_video_train_adapter",
    "hunyuan_video_model_contract",
]
