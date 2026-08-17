"""Frozen native Cosmos text and video feature encoders."""

from __future__ import annotations

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DiffusionRequest,
    SamplingConfig,
)

from ..video_dataset import DecodedVideoSample
from ..video_masks import project_causal_video_mask_to_latent


def _module_target(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    reference = next(module.parameters(), None)
    if reference is None:
        reference = next(module.buffers(), None)
    if reference is None:
        return torch.device("cpu"), torch.float32
    dtype = reference.dtype if reference.is_floating_point() else torch.float32
    return reference.device, dtype


def _conditioner_module(conditioner: object) -> nn.Module | None:
    for name in ("model", "encoder", "text_encoder"):
        value = getattr(conditioner, name, None)
        if isinstance(value, nn.Module):
            return value
    return None


class CosmosTextFeatureEncoder:
    """Encode Predict contexts or Cosmos3 token IDs without retaining the VAE."""

    def __init__(
        self,
        conditioner: object,
        *,
        model_recipe: str,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> None:
        if not callable(getattr(conditioner, "encode", None)):
            raise TypeError("Cosmos cache conditioner must expose encode")
        self.conditioner = conditioner
        self.model_recipe = model_recipe
        self.cosmos3 = model_recipe.startswith("cosmos3-")
        self.predict25 = model_recipe.startswith("cosmos-predict2.5-")
        if self.cosmos3 and getattr(conditioner, "use_system_prompt", None) is not False:
            raise ValueError("Cosmos3 vision SFT tokenization requires use_system_prompt=False")
        self.device = torch.device(device)
        self.dtype = dtype
        module = _conditioner_module(conditioner)
        if module is not None:
            module.requires_grad_(False)
            module.eval()

    def encode(
        self,
        *,
        sample_id: str,
        prompt: str,
        frames: int,
        height: int,
        width: int,
        fps: float,
    ) -> dict[str, torch.Tensor]:
        request = DiffusionRequest(
            prompt=(prompt,),
            negative_prompt=("",) if self.predict25 else None,
            height=height,
            width=width,
            num_frames=frames,
            sampling=SamplingConfig(guidance_scale=2.0 if self.predict25 else 1.0),
            inputs={"fps": fps, "frame_rate": fps},
            metadata={"sample_ids": (sample_id,)},
        )
        with torch.no_grad():
            encoded = self.conditioner.encode(
                request,
                device=self.device,
                dtype=self.dtype,
            )
        if not isinstance(encoded, Conditioning):
            raise TypeError("Cosmos cache conditioner must return Conditioning")
        if self.cosmos3:
            input_ids = encoded.positive.get("input_ids")
            if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1:
                raise ValueError("Cosmos3 conditioner must return one input_ids sequence")
            empty_request = DiffusionRequest(
                prompt=("",),
                height=height,
                width=width,
                num_frames=frames,
                sampling=SamplingConfig(guidance_scale=1.0),
                inputs={
                    "fps": fps,
                    "frame_rate": fps,
                    "add_duration_template": False,
                    "add_resolution_template": False,
                },
                metadata={"sample_ids": (sample_id,)},
            )
            with torch.no_grad():
                empty = self.conditioner.encode(
                    empty_request,
                    device=self.device,
                    dtype=self.dtype,
                )
            if not isinstance(empty, Conditioning):
                raise TypeError("Cosmos3 cache conditioner must return Conditioning for the empty caption")
            empty_input_ids = empty.positive.get("input_ids")
            if not isinstance(empty_input_ids, torch.Tensor) or empty_input_ids.ndim != 1:
                raise ValueError("Cosmos3 conditioner must return one empty_input_ids sequence")
            return {
                "input_ids": input_ids.detach().cpu().long().contiguous(),
                "empty_input_ids": empty_input_ids.detach().cpu().long().contiguous(),
            }

        context = encoded.positive.get("context")
        if not isinstance(context, torch.Tensor) or context.ndim < 3 or int(context.shape[0]) != 1:
            raise ValueError("Cosmos Predict context must start with a singleton batch dimension")
        result = {"context": context[0].detach().cpu().contiguous()}
        if self.predict25:
            negative = encoded.negative.get("context")
            if not isinstance(negative, torch.Tensor) or tuple(negative.shape) != tuple(context.shape):
                raise ValueError("Cosmos Predict2.5 negative context must match positive context")
            result["negative_context"] = negative[0].detach().cpu().contiguous()
        return result


class CosmosVideoFeatureEncoder:
    """Encode BCTHW pixels through the recipe-bound Predict or Cosmos3 VAE."""

    def __init__(
        self,
        component: object,
        *,
        cosmos3: bool,
        latent_channels: int,
        temporal_compression: int,
        spatial_compression: int,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
    ) -> None:
        vae = getattr(component, "vae", None)
        if not isinstance(vae, nn.Module):
            raise TypeError("Cosmos cache codec must expose its native VAE module")
        if cosmos3:
            encode = getattr(vae, "encode", None)
        else:
            encode = getattr(component, "encode", None)
        if not callable(encode):
            raise TypeError("Cosmos cache codec must expose video encoding")
        vae.requires_grad_(False)
        vae.eval()
        self.component = component
        self.vae = vae
        self.cosmos3 = bool(cosmos3)
        self.device, self.dtype = _module_target(vae)
        self.latent_channels = int(latent_channels)
        self.temporal_compression = int(temporal_compression)
        self.spatial_compression = int(spatial_compression)
        self.tiled = bool(tiled)
        self.tile_size = tile_size
        self.tile_stride = tile_stride

    def encode(
        self,
        decoded: DecodedVideoSample,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pixels = decoded.pixel_values.unsqueeze(0).to(dtype=self.dtype)
        with torch.no_grad():
            if self.cosmos3:
                latents = self.vae.encode(
                    [pixels[0]],
                    self.device,
                    tiled=self.tiled,
                    tile_size=self.tile_size,
                    tile_stride=self.tile_stride,
                )
            else:
                latents = self.component.encode(pixels)
        expected = decoded.assignment.bucket_key.latent_shape
        expected_shape = (
            1,
            self.latent_channels,
            expected.frames,
            expected.height,
            expected.width,
        )
        if not isinstance(latents, torch.Tensor) or tuple(latents.shape) != expected_shape:
            raise ValueError(
                f"Cosmos VAE returned {getattr(latents, 'shape', None)!r}; expected {expected_shape}"
            )
        loss_mask = project_causal_video_mask_to_latent(
            decoded.valid_mask.unsqueeze(0).to(device=latents.device),
            pixel_shape=tuple(int(value) for value in pixels.shape),
            latent_shape=(expected.frames, expected.height, expected.width),
            temporal_compression=self.temporal_compression,
        )
        return (
            latents[0].detach().cpu().contiguous(),
            loss_mask[0].detach().cpu().contiguous(),
            (loss_mask[0] > 0).detach().cpu().contiguous(),
        )


__all__ = ["CosmosTextFeatureEncoder", "CosmosVideoFeatureEncoder"]
