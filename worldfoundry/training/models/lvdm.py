"""Native training bindings for the LVDM/VideoCrafter model family."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from worldfoundry.training.api.contracts import ObjectiveBatch, PreparedBatch, TrainingBatch


def module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    reference = next(module.parameters(), None)
    if reference is None:
        reference = next(module.buffers(), None)
    if reference is None:
        return torch.device("cpu"), torch.float32
    dtype = reference.dtype if reference.is_floating_point() else torch.float32
    return reference.device, dtype


def freeze_module(module: nn.Module | None) -> None:
    if module is not None:
        module.requires_grad_(False)
        module.eval()


def latent_sample(value: object, *, generator: torch.Generator | None = None) -> Tensor:
    """Resolve the distribution conventions used by native LVDM codecs."""

    distribution = getattr(value, "latent_dist", value)
    sample = getattr(distribution, "sample", None)
    if callable(sample):
        try:
            result = sample(generator=generator)
        except TypeError:
            result = sample()
    else:
        result = distribution
    if not isinstance(result, Tensor):
        raise TypeError("LVDM codec encode must return a tensor or latent distribution")
    return result


class FramewiseLVDMCodec(nn.Module):
    """Give the shared two-dimensional LVDM VAE a video-shaped train seam."""

    def __init__(self, model: nn.Module, *, scale_factor: float = 0.18215) -> None:
        super().__init__()
        if not callable(getattr(model, "encode", None)) or not callable(getattr(model, "decode", None)):
            raise TypeError("framewise LVDM codec requires encode and decode methods")
        scale = float(scale_factor)
        if not isfinite(scale) or scale <= 0.0:
            raise ValueError("scale_factor must be finite and positive")
        self.model = model
        self.scale_factor = scale

    def encode_video(
        self,
        pixels: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if pixels.ndim != 5:
            raise ValueError("LVDM video pixels must be [B,C,T,H,W]")
        batch, _, frames, _, _ = pixels.shape
        flattened = pixels.permute(0, 2, 1, 3, 4).reshape(batch * frames, *pixels.shape[1:2], *pixels.shape[3:])
        posterior = self.model.encode(flattened)
        latents = latent_sample(posterior, generator=generator)
        latents = latents.reshape(batch, frames, *latents.shape[1:]).permute(0, 2, 1, 3, 4)
        return latents * self.scale_factor

    def decode_video(self, latents: Tensor) -> Tensor:
        if latents.ndim != 5:
            raise ValueError("LVDM video latents must be [B,C,T,H,W]")
        batch, _, frames, _, _ = latents.shape
        flattened = (latents / self.scale_factor).permute(0, 2, 1, 3, 4).reshape(
            batch * frames,
            int(latents.shape[1]),
            int(latents.shape[3]),
            int(latents.shape[4]),
        )
        decoded = self.model.decode(flattened)
        sample = getattr(decoded, "sample", decoded)
        if not isinstance(sample, Tensor):
            raise TypeError("LVDM codec decode must return a tensor or sample output")
        return sample.reshape(batch, frames, *sample.shape[1:]).permute(0, 2, 1, 3, 4)


def _infer_fsdp_blocks(module: nn.Module) -> tuple[type, ...]:
    candidates: list[nn.Module] = []
    for name in ("input_blocks", "output_blocks", "blocks", "transformer_blocks"):
        values = getattr(module, name, ())
        if isinstance(values, (nn.ModuleList, nn.Sequential, list, tuple)):
            candidates.extend(value for value in values if isinstance(value, nn.Module))
    if not candidates:
        candidates.extend(value for value in module.children() if any(True for _ in value.parameters()))
    return tuple(dict.fromkeys(type(value) for value in candidates))


def _project_mask(mask: Tensor, latent_shape: tuple[int, ...]) -> Tensor:
    if mask.ndim == 4:
        mask = mask.unsqueeze(1)
    if mask.ndim != 5:
        raise ValueError("video valid_mask must be [B,T,H,W] or [B,1,T,H,W]")
    reduced = mask.float().amin(dim=1, keepdim=True)
    return F.interpolate(reduced, size=latent_shape[-3:], mode="trilinear", align_corners=False)


class LVDMUnconditionalTrainAdapter:
    """Bind the released short unconditional LVDM graph to native engines."""

    prediction_type = "epsilon"
    lora_target_preset = None

    def __init__(
        self,
        denoiser: nn.Module,
        codec: object | None,
        *,
        latent_scale: float = 0.220142075,
        latent_shift: float = 0.5837740898,
        cached_latents_are_normalized: bool = True,
        expected_latent_channels: int = 4,
    ) -> None:
        if not isinstance(denoiser, nn.Module):
            raise TypeError("LVDM denoiser must be an nn.Module")
        scale = float(latent_scale)
        shift = float(latent_shift)
        if not isfinite(scale) or scale <= 0.0 or not isfinite(shift):
            raise ValueError("LVDM latent scale/shift must be finite and scale must be positive")
        self.denoiser = denoiser
        self.codec = codec
        self.trainable_module = denoiser
        self.latent_scale = scale
        self.latent_shift = shift
        self.cached_latents_are_normalized = bool(cached_latents_are_normalized)
        self.expected_latent_channels = int(expected_latent_channels)
        diffusion_model = getattr(denoiser, "diffusion_model", denoiser)
        self.fsdp_block_classes = _infer_fsdp_blocks(diffusion_model)
        codec_module = codec if isinstance(codec, nn.Module) else getattr(codec, "model", None)
        self.frozen_modules = (codec_module,) if isinstance(codec_module, nn.Module) else ()
        for module in self.frozen_modules:
            freeze_module(module)

    @classmethod
    def from_latent_diffusion(cls, model: nn.Module, **options: object) -> "LVDMUnconditionalTrainAdapter":
        """Bind an already-loaded native LVDM checkpoint graph."""

        denoiser = getattr(model, "model", None)
        first_stage = getattr(model, "first_stage_model", None)
        if not isinstance(denoiser, nn.Module) or first_stage is None:
            raise TypeError("LVDM model must expose model and first_stage_model")
        options.setdefault("latent_scale", float(getattr(model, "scale_factor", 0.220142075)))
        options.setdefault("latent_shift", float(getattr(model, "shift_factor", 0.5837740898)))
        return cls(denoiser, first_stage, **options)

    def _encode_pixels(self, pixels: Tensor) -> Tensor:
        if self.codec is None:
            raise RuntimeError("pixel training requires an LVDM codec")
        encode_video = getattr(self.codec, "encode_video", None)
        with torch.no_grad():
            if callable(encode_video):
                latents = encode_video(pixels)
            else:
                encode = getattr(self.codec, "encode", None)
                if not callable(encode):
                    raise TypeError("LVDM codec must expose encode or encode_video")
                latents = latent_sample(encode(pixels))
        if not isinstance(latents, Tensor):
            raise TypeError("LVDM codec returned non-tensor latents")
        # FramewiseLVDMCodec already applies its VideoCrafter scale.  The short
        # LVDM 3D codec returns raw posterior samples and uses this transform.
        if not isinstance(self.codec, FramewiseLVDMCodec):
            latents = self.latent_scale * (latents + self.latent_shift)
        return latents

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        cached = batch.conditions.get("clean_latents")
        if cached is None:
            pixels = batch.pixel_values
            if not isinstance(pixels, Tensor):
                raise TypeError("LVDM training requires pixels or clean_latents")
            clean = self._encode_pixels(pixels)
            pixel_shape: tuple[int, ...] | None = tuple(int(value) for value in pixels.shape)
        else:
            if batch.pixel_values is not None:
                raise ValueError("LVDM batch cannot contain both pixels and cached latents")
            if not isinstance(cached, Tensor):
                raise TypeError("clean_latents must be a tensor")
            clean = cached
            if not self.cached_latents_are_normalized:
                clean = self.latent_scale * (clean + self.latent_shift)
            pixel_shape = None
        if clean.ndim != 5 or int(clean.shape[0]) != batch.batch_size:
            raise ValueError("LVDM clean latents must be [B,C,T,H,W]")
        if int(clean.shape[1]) != self.expected_latent_channels:
            raise ValueError(f"LVDM expected {self.expected_latent_channels} latent channels")
        device, dtype = module_device_dtype(self.trainable_module)
        clean = clean.detach().to(device=device, dtype=dtype)
        loss_mask = batch.conditions.get("latent_loss_mask")
        if loss_mask is None and batch.valid_mask is not None and batch.pixel_values is not None:
            if not isinstance(batch.valid_mask, Tensor):
                raise TypeError("valid_mask must be a tensor")
            loss_mask = _project_mask(batch.valid_mask.to(device), tuple(clean.shape))
        elif loss_mask is not None:
            if not isinstance(loss_mask, Tensor):
                raise TypeError("latent_loss_mask must be a tensor")
            loss_mask = loss_mask.to(device=device, dtype=torch.float32)
        sample_weights = batch.sample_weights
        if isinstance(sample_weights, Tensor):
            sample_weights = sample_weights.to(device=device, dtype=torch.float32)
        metadata = dict(batch.metadata)
        metadata.update(
            {
                "model_family": "lvdm-short-unconditional",
                "pixel_shape": pixel_shape,
                "latent_shape": tuple(int(value) for value in clean.shape),
                "latent_scale": self.latent_scale,
                "latent_shift": self.latent_shift,
            }
        )
        return PreparedBatch(
            sample_ids=batch.sample_ids,
            clean_latents=clean,
            conditioning={},
            loss_mask=loss_mask,
            sample_weights=sample_weights,
            metadata=metadata,
        )

    def forward_train(self, batch: ObjectiveBatch) -> Tensor:
        model_input = batch.model_input
        if isinstance(model_input, Mapping) or not isinstance(model_input, Tensor):
            raise TypeError("LVDM model_input must be one tensor")
        self.trainable_module.train()
        for module in self.frozen_modules:
            module.eval()
        output = self.denoiser(model_input, batch.timesteps)
        if isinstance(output, tuple):
            output = output[0]
        sample = getattr(output, "sample", output)
        if not isinstance(sample, Tensor) or sample.shape != model_input.shape:
            raise ValueError("LVDM denoiser output must match model_input")
        return sample


__all__ = [
    "FramewiseLVDMCodec",
    "LVDMUnconditionalTrainAdapter",
    "freeze_module",
    "latent_sample",
    "module_device_dtype",
]
