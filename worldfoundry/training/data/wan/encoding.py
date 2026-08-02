"""Frozen Wan text and video feature encoders."""

from __future__ import annotations

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DiffusionRequest,
    SamplingConfig,
)
from worldfoundry.training.models.wan import wan_pixel_mask_to_latent

from ..video_dataset import DecodedVideoSample
from .contracts import (
    WAN_LATENT_MEAN,
    WAN_LATENT_STD,
    require_positive_int,
)


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
        raise ValueError("Wan feature module has no parameter or buffer for placement")
    dtype = reference.dtype if reference.is_floating_point() else torch.float32
    return reference.device, dtype


def _validate_codec_normalization(codec: object) -> None:
    vae = getattr(codec, "vae", None)
    mean = getattr(vae, "mean", None)
    std = getattr(vae, "std", None)
    if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
        raise TypeError("Wan cache codec must expose tensor vae.mean and vae.std")
    expected_mean = torch.tensor(WAN_LATENT_MEAN, dtype=torch.float32)
    expected_std = torch.tensor(WAN_LATENT_STD, dtype=torch.float32)
    if tuple(mean.shape) != (16,) or not torch.equal(
        mean.detach().cpu().float(),
        expected_mean,
    ):
        raise ValueError("Wan codec channel mean differs from the official Wan2.1 normalization")
    if tuple(std.shape) != (16,) or not torch.equal(
        std.detach().cpu().float(),
        expected_std,
    ):
        raise ValueError("Wan codec channel std differs from the official Wan2.1 normalization")


class WanTextFeatureEncoder:
    """Frozen UMT5 phase of Wan cache construction."""

    def __init__(
        self,
        conditioner: object,
        *,
        text_length: int = 512,
        context_features: int = 4096,
    ) -> None:
        if not callable(getattr(conditioner, "encode", None)):
            raise TypeError("Wan cache conditioner must expose encode(request, device, dtype)")
        module = _component_module(conditioner, "text_encoder", "encoder", "model")
        if module is None:
            raise TypeError("Wan cache conditioner must expose its torch module")
        module.requires_grad_(False)
        module.eval()
        self.conditioner = conditioner
        self.module = module
        self.device, self.dtype = _module_device_dtype(module)
        self.text_length = require_positive_int(text_length, field_name="text_length")
        self.context_features = require_positive_int(
            context_features,
            field_name="context_features",
        )

    def encode(
        self,
        *,
        sample_id: str,
        prompt: str,
        frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        request = DiffusionRequest(
            prompt=(prompt,),
            height=height,
            width=width,
            num_frames=frames,
            sampling=SamplingConfig(guidance_scale=1.0),
            metadata={"sample_ids": (sample_id,)},
        )
        self.module.eval()
        with torch.no_grad():
            encoded = self.conditioner.encode(
                request,
                device=self.device,
                dtype=self.dtype,
            )
        if not isinstance(encoded, Conditioning):
            raise TypeError("Wan conditioner must return Conditioning")
        if encoded.shared:
            raise ValueError("Wan T2V cache does not accept shared image/control conditioning")
        if set(encoded.positive) != {"context"}:
            raise ValueError(
                f"Wan T2V conditioner positive keys must be exactly ['context']; got {sorted(encoded.positive)}"
            )
        context = encoded.positive["context"]
        expected = (1, self.text_length, self.context_features)
        if not isinstance(context, torch.Tensor) or tuple(context.shape) != expected:
            shape = getattr(context, "shape", None)
            raise ValueError(f"Wan text context must have shape {expected}; got {shape!r}")
        if not bool(torch.isfinite(context).all()):
            raise ValueError("Wan text context contains NaN or infinity")
        return context[0].detach().cpu().contiguous()


class WanVideoFeatureEncoder:
    """Frozen deterministic Wan VAE phase of cache construction."""

    def __init__(self, codec: object) -> None:
        if not callable(getattr(codec, "encode", None)):
            raise TypeError("Wan cache codec must expose encode(videos)")
        module = _component_module(codec, "vae", "model")
        if module is None:
            raise TypeError("Wan cache codec must expose its torch module")
        module.requires_grad_(False)
        module.eval()
        _validate_codec_normalization(codec)
        self.codec = codec
        self.module = module
        self.device, self.dtype = _module_device_dtype(module)
        self.temporal_compression = int(getattr(codec, "temporal_compression_factor", 4))
        self.spatial_compression = int(getattr(codec, "spatial_compression_factor", 8))
        if self.temporal_compression != 4 or self.spatial_compression != 8:
            raise ValueError("Wan2.1 cache requires temporal/spatial compression 4/8")

    def encode(
        self,
        decoded: DecodedVideoSample,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(decoded, DecodedVideoSample):
            raise TypeError("decoded must be a DecodedVideoSample")
        pixels = decoded.pixel_values.unsqueeze(0)
        self.module.eval()
        with torch.no_grad():
            latents = self.codec.encode(pixels)
        if not isinstance(latents, torch.Tensor) or latents.ndim != 5 or latents.shape[0] != 1:
            raise TypeError("Wan codec must return one BCTHW tensor")
        expected = decoded.assignment.bucket_key.latent_shape
        expected_shape = (
            1,
            16,
            expected.frames,
            expected.height,
            expected.width,
        )
        if tuple(latents.shape) != expected_shape:
            raise ValueError(
                "Wan codec latent shape differs from the assigned bucket: "
                f"got {tuple(latents.shape)}, expected {expected_shape}"
            )
        loss_mask = wan_pixel_mask_to_latent(
            decoded.valid_mask.unsqueeze(0).to(device=latents.device),
            pixel_shape=tuple(int(value) for value in pixels.shape),
            latent_shape=(expected.frames, expected.height, expected.width),
            temporal_compression=self.temporal_compression,
        )
        valid_mask = loss_mask > 0
        return (
            latents[0].detach().cpu().contiguous(),
            loss_mask[0].detach().cpu().contiguous(),
            valid_mask[0].detach().cpu().contiguous(),
        )


class WanFeatureEncoder:
    """Compose text and video encoders for same-lifecycle cache builds."""

    def __init__(self, codec: object, conditioner: object) -> None:
        self.video = WanVideoFeatureEncoder(codec)
        self.text = WanTextFeatureEncoder(conditioner)

    def encode(
        self,
        decoded: DecodedVideoSample,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.text.encode(
            sample_id=decoded.sample_id,
            prompt=decoded.prompt,
            frames=decoded.assignment.target_num_frames,
            height=decoded.assignment.target_height,
            width=decoded.assignment.target_width,
        )
        latents, loss_mask, valid_mask = self.video.encode(decoded)
        return latents, context, loss_mask, valid_mask


__all__ = [
    "WanFeatureEncoder",
    "WanTextFeatureEncoder",
    "WanVideoFeatureEncoder",
]
