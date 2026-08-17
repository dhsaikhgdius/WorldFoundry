"""Frozen native LTX prompt and video feature encoders."""

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


def _conditioner_module(conditioner: object) -> nn.Module | None:
    for name in ("gemma", "model", "text_encoder"):
        value = getattr(conditioner, name, None)
        if isinstance(value, nn.Module):
            return value
    return None


class LTXTextFeatureEncoder:
    """Encode the native LTX video and audio prompt projections."""

    def __init__(
        self,
        conditioner: object,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
        include_audio: bool = True,
    ) -> None:
        if not callable(getattr(conditioner, "encode", None)):
            raise TypeError("LTX cache conditioner must expose encode")
        self.conditioner = conditioner
        self.device = torch.device(device)
        self.dtype = dtype
        self.include_audio = bool(include_audio)
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
            height=height,
            width=width,
            num_frames=frames,
            sampling=SamplingConfig(guidance_scale=1.0),
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
            raise TypeError("LTX cache conditioner must return Conditioning")
        video_context = encoded.positive.get("video_context")
        audio_context = encoded.positive.get("audio_context")
        context_mask = encoded.positive.get("context_mask")
        if not isinstance(video_context, torch.Tensor) or video_context.ndim != 3 or int(video_context.shape[0]) != 1:
            raise ValueError("LTX video_context must have shape [1,sequence,features]")
        if not isinstance(context_mask, torch.Tensor) or tuple(context_mask.shape) != tuple(video_context.shape[:2]):
            raise ValueError("LTX context_mask must match video_context's batch and sequence")
        result = {
            "video_context": video_context[0].detach().cpu().contiguous(),
            "context_mask": context_mask[0].detach().cpu().contiguous(),
        }
        if self.include_audio:
            if (
                not isinstance(audio_context, torch.Tensor)
                or audio_context.ndim != 3
                or int(audio_context.shape[0]) != 1
            ):
                raise ValueError("LTX audio_context must have shape [1,sequence,features]")
            result["audio_context"] = audio_context[0].detach().cpu().contiguous()
        return result


class LTXVideoFeatureEncoder:
    """Encode normalized BCTHW pixels with the recipe's native LTX VAE."""

    def __init__(
        self,
        codec: object,
        *,
        sample_posterior: bool,
        latent_channels: int = 128,
        temporal_compression: int = 8,
        spatial_compression: int = 32,
    ) -> None:
        encode_name = "encode_posterior" if sample_posterior else "encode"
        if not callable(getattr(codec, encode_name, None)):
            raise TypeError(f"LTX cache codec must expose {encode_name}(pixels)")
        module = getattr(codec, "encoder", None)
        if not isinstance(module, nn.Module):
            raise TypeError("LTX cache codec must expose its native encoder module")
        module.requires_grad_(False)
        module.eval()
        self.codec = codec
        self.module = module
        self.sample_posterior = bool(sample_posterior)
        self.latent_channels = int(latent_channels)
        self.temporal_compression = int(temporal_compression)
        self.spatial_compression = int(spatial_compression)

    def encode(
        self,
        decoded: DecodedVideoSample,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pixels = decoded.pixel_values.unsqueeze(0)
        with torch.no_grad():
            if self.sample_posterior:
                latents = self.codec.encode_posterior(pixels)
            else:
                latents = self.codec.encode(pixels)
        expected = decoded.assignment.bucket_key.latent_shape
        expected_shape = (
            1,
            self.latent_channels,
            expected.frames,
            expected.height,
            expected.width,
        )
        if not isinstance(latents, torch.Tensor) or tuple(latents.shape) != expected_shape:
            raise ValueError(f"LTX VAE returned {getattr(latents, 'shape', None)!r}; expected {expected_shape}")
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


__all__ = ["LTXTextFeatureEncoder", "LTXVideoFeatureEncoder"]
