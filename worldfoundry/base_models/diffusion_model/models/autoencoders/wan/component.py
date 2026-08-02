"""Wan video decoder implemented against the native latent contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import torch

from ....components import ComponentBuildContext
from ....contracts import DiffusionRequest
from ....loaders import CheckpointSpec, ModuleLoadSpec, NativeModuleLoader
from ....optimizations import RuntimePolicy
from .model import (
    CausalConv3d,
    RMS_norm,
    Upsample,
    WanVideoVAE,
    WanVideoVAE38,
    WanVideoVAEStateDictConverter,
)


def convert_wan21_vae_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    """Map the official Wan2.1 VAE checkpoint into the native module."""

    return WanVideoVAEStateDictConverter().from_civitai(state_dict)


def _wan_residual_suffix(value: str) -> str:
    replacements = {
        "norm1": "residual.0",
        "conv1": "residual.2",
        "norm2": "residual.3",
        "conv2": "residual.6",
        "conv_shortcut": "shortcut",
    }
    head, tail = value.split(".", 1)
    return f"{replacements[head]}.{tail}"


def convert_diffusers_wan22_vae_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    """Map the official Diffusers Wan2.2 VAE layout onto the shared native VAE38."""

    converted: dict[str, object] = {}
    for key, value in state_dict.items():
        parts = key.split(".")
        if key.startswith("encoder.conv_in."):
            target = key.replace("encoder.conv_in", "model.encoder.conv1", 1)
        elif key.startswith("encoder.down_blocks."):
            block = parts[2]
            if parts[3] == "resnets":
                residual = parts[4]
                suffix = _wan_residual_suffix(".".join(parts[5:]))
                target = f"model.encoder.downsamples.{block}.downsamples.{residual}.{suffix}"
            else:
                suffix = ".".join(parts[4:])
                target = f"model.encoder.downsamples.{block}.downsamples.2.{suffix}"
        elif key.startswith("encoder.mid_block.resnets."):
            middle = "0" if parts[3] == "0" else "2"
            target = f"model.encoder.middle.{middle}.{_wan_residual_suffix('.'.join(parts[4:]))}"
        elif key.startswith("encoder.mid_block.attentions.0."):
            target = "model.encoder.middle.1." + ".".join(parts[4:])
        elif key.startswith("encoder.norm_out."):
            target = key.replace("encoder.norm_out", "model.encoder.head.0", 1)
        elif key.startswith("encoder.conv_out."):
            target = key.replace("encoder.conv_out", "model.encoder.head.2", 1)
        elif key.startswith("quant_conv."):
            target = key.replace("quant_conv", "model.conv1", 1)
        elif key.startswith("post_quant_conv."):
            target = key.replace("post_quant_conv", "model.conv2", 1)
        elif key.startswith("decoder.conv_in."):
            target = key.replace("decoder.conv_in", "model.decoder.conv1", 1)
        elif key.startswith("decoder.mid_block.resnets."):
            middle = "0" if parts[3] == "0" else "2"
            target = f"model.decoder.middle.{middle}.{_wan_residual_suffix('.'.join(parts[4:]))}"
        elif key.startswith("decoder.mid_block.attentions.0."):
            target = "model.decoder.middle.1." + ".".join(parts[4:])
        elif key.startswith("decoder.up_blocks."):
            block = parts[2]
            if parts[3] == "resnets":
                residual = parts[4]
                suffix = _wan_residual_suffix(".".join(parts[5:]))
                target = f"model.decoder.upsamples.{block}.upsamples.{residual}.{suffix}"
            else:
                suffix = ".".join(parts[4:])
                target = f"model.decoder.upsamples.{block}.upsamples.3.{suffix}"
        elif key.startswith("decoder.norm_out."):
            target = key.replace("decoder.norm_out", "model.decoder.head.0", 1)
        elif key.startswith("decoder.conv_out."):
            target = key.replace("decoder.conv_out", "model.decoder.head.2", 1)
        else:
            raise KeyError(f"unsupported Diffusers Wan VAE parameter: {key}")
        converted[target] = value
    return converted


def convert_diffusers_wan_vae_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    """Map the standard Diffusers 16-channel Wan VAE onto the native codec."""

    converted: dict[str, object] = {}
    for key, value in state_dict.items():
        parts = key.split(".")
        if key.startswith("encoder.conv_in."):
            target = key.replace("encoder.conv_in", "model.encoder.conv1", 1)
        elif key.startswith("encoder.down_blocks."):
            index = parts[2]
            suffix = ".".join(parts[3:])
            if suffix.startswith(("norm1.", "conv1.", "norm2.", "conv2.", "conv_shortcut.")):
                suffix = _wan_residual_suffix(suffix)
            target = f"model.encoder.downsamples.{index}.{suffix}"
        elif key.startswith("encoder.mid_block.resnets."):
            middle = "0" if parts[3] == "0" else "2"
            target = f"model.encoder.middle.{middle}.{_wan_residual_suffix('.'.join(parts[4:]))}"
        elif key.startswith("encoder.mid_block.attentions.0."):
            target = "model.encoder.middle.1." + ".".join(parts[4:])
        elif key.startswith("encoder.norm_out."):
            target = key.replace("encoder.norm_out", "model.encoder.head.0", 1)
        elif key.startswith("encoder.conv_out."):
            target = key.replace("encoder.conv_out", "model.encoder.head.2", 1)
        elif key.startswith("quant_conv."):
            target = key.replace("quant_conv", "model.conv1", 1)
        elif key.startswith("post_quant_conv."):
            target = key.replace("post_quant_conv", "model.conv2", 1)
        elif key.startswith("decoder.conv_in."):
            target = key.replace("decoder.conv_in", "model.decoder.conv1", 1)
        elif key.startswith("decoder.mid_block.resnets."):
            middle = "0" if parts[3] == "0" else "2"
            target = f"model.decoder.middle.{middle}.{_wan_residual_suffix('.'.join(parts[4:]))}"
        elif key.startswith("decoder.mid_block.attentions.0."):
            target = "model.decoder.middle.1." + ".".join(parts[4:])
        elif key.startswith("decoder.up_blocks."):
            block = int(parts[2])
            if parts[3] == "resnets":
                index = block * 4 + int(parts[4])
                suffix = _wan_residual_suffix(".".join(parts[5:]))
            elif parts[3] == "upsamplers" and parts[4] == "0":
                index = block * 4 + 3
                suffix = ".".join(parts[5:])
            else:
                raise KeyError(f"unsupported Diffusers Wan VAE decoder parameter: {key}")
            target = f"model.decoder.upsamples.{index}.{suffix}"
        elif key.startswith("decoder.norm_out."):
            target = key.replace("decoder.norm_out", "model.decoder.head.0", 1)
        elif key.startswith("decoder.conv_out."):
            target = key.replace("decoder.conv_out", "model.decoder.head.2", 1)
        else:
            raise KeyError(f"unsupported Diffusers Wan VAE parameter: {key}")
        if target in converted:
            raise KeyError(f"Wan VAE conversion produced duplicate parameter: {target}")
        converted[target] = value
    return converted


class WanVideoDecoder:
    """Decode Wan latents to normalized ``[B, C, T, H, W]`` video."""

    def __init__(
        self,
        vae: WanVideoVAE,
        *,
        device: torch.device,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
        chunk_duration: int = 81,
    ) -> None:
        self.vae = vae
        self.device = device
        self.tiled = tiled
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.chunk_duration = int(chunk_duration)
        if self.chunk_duration <= 0:
            raise ValueError("Wan codec chunk_duration must be positive")

    @property
    def dtype(self) -> torch.dtype:
        return next(self.vae.parameters()).dtype

    @property
    def spatial_compression_factor(self) -> int:
        return 8

    @property
    def temporal_compression_factor(self) -> int:
        return 4

    @property
    def pixel_chunk_duration(self) -> int:
        return self.chunk_duration

    @property
    def latent_chunk_duration(self) -> int:
        return self.get_latent_num_frames(self.chunk_duration)

    @property
    def latent_ch(self) -> int:
        return int(self.vae.z_dim)

    @property
    def spatial_resolution(self) -> int:
        return 512

    @property
    def name(self) -> str:
        return "wan2pt1_tokenizer"

    @staticmethod
    def get_latent_num_frames(num_pixel_frames: int) -> int:
        return 1 + (int(num_pixel_frames) - 1) // 4

    @staticmethod
    def get_pixel_num_frames(num_latent_frames: int) -> int:
        return (int(num_latent_frames) - 1) * 4 + 1

    def clear_cache(self) -> None:
        self.vae.model.clear_cache()

    def reset_dtype(self) -> None:
        """Compatibility no-op; dtype placement belongs to RuntimePolicy."""

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError(f"Wan encoder expects BCTHW pixels, got {tuple(images.shape)}")
        return self.vae.encode(
            images.to(device=self.device, dtype=self.dtype),
            self.device,
            tiled=self.tiled,
            tile_size=self.tile_size,
            tile_stride=self.tile_stride,
        )

    def decode(
        self,
        latents: torch.Tensor,
        request: DiffusionRequest | None = None,
    ) -> torch.Tensor:
        frozen_context = request.inputs.get("frozen_context_latents") if request is not None else None
        if frozen_context is not None:
            if not isinstance(frozen_context, torch.Tensor):
                raise TypeError("frozen_context_latents must be a tensor")
            context_frames = int(frozen_context.shape[-3])
            if context_frames <= 0 or context_frames >= int(latents.shape[2]):
                raise ValueError("frozen context length must be smaller than the full latent sequence")
            latents = latents[:, :, :-context_frames]
        if request is not None and bool(request.inputs.get("return_latent", False)):
            prefix = request.inputs.get("first_frame_latents")
            if prefix is None:
                return latents
            if not isinstance(prefix, torch.Tensor):
                raise TypeError("first_frame_latents must be a tensor when returning Wan latents")
            prefix = prefix.to(device=latents.device, dtype=latents.dtype)
            if prefix.shape[:2] != latents.shape[:2] or prefix.shape[-2:] != latents.shape[-2:]:
                raise ValueError("first_frame_latents must match the generated latent batch/channel/spatial shape")
            return torch.cat([prefix, latents], dim=2)
        return self.vae.decode(
            latents.to(device=self.device, dtype=self.dtype),
            self.device,
            tiled=self.tiled,
            tile_size=self.tile_size,
            tile_stride=self.tile_stride,
        )


def _load_wan_video_decoder(
    checkpoint: CheckpointSpec,
    policy: RuntimePolicy,
    *,
    module_class: type[WanVideoVAE],
    state_dict_converter=convert_wan21_vae_state_dict,
    tiled: bool = False,
    tile_size: tuple[int, int] = (34, 34),
    tile_stride: tuple[int, int] = (18, 16),
    chunk_duration: int = 81,
) -> WanVideoDecoder:
    """Load one Wan VAE variant through the shared native module loader."""

    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    vae = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=module_class,
            state_dict_converter=state_dict_converter,
            vram_module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
        ),
        checkpoint,
        policy,
    )
    if not isinstance(vae, module_class):
        raise TypeError(f"expected {module_class.__name__}, got {type(vae).__name__}")
    if len(tile_size) != 2 or len(tile_stride) != 2:
        raise ValueError("Wan VAE tile_size and tile_stride must contain two values")
    return WanVideoDecoder(
        vae,
        device=policy.device,
        tiled=bool(tiled),
        tile_size=(int(tile_size[0]), int(tile_size[1])),
        tile_stride=(int(tile_stride[0]), int(tile_stride[1])),
        chunk_duration=chunk_duration,
    )


def _build_wan_video_decoder(
    context: ComponentBuildContext,
    *,
    module_class: type[WanVideoVAE],
    state_dict_converter=convert_wan21_vae_state_dict,
) -> WanVideoDecoder:
    codec_dtype = context.component_options.get("dtype", torch.float32)
    if not isinstance(codec_dtype, torch.dtype):
        raise TypeError(f"Wan codec dtype must be a torch.dtype, got {codec_dtype!r}")
    return _load_wan_video_decoder(
        context.require_checkpoint("weights"),
        replace(context.policy, dtype=codec_dtype),
        module_class=module_class,
        state_dict_converter=state_dict_converter,
        tiled=bool(context.component_options.get("tiled", False)),
        tile_size=tuple(context.component_options.get("tile_size", (34, 34))),
        tile_stride=tuple(context.component_options.get("tile_stride", (18, 16))),
        chunk_duration=int(context.component_options.get("chunk_duration", 81)),
    )


def load_wan_video_codec(
    checkpoint_path: str | Path | CheckpointSpec,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
    chunk_duration: int = 81,
) -> WanVideoDecoder:
    """Load the shared Wan2.1 codec for native and representation consumers."""

    if isinstance(checkpoint_path, CheckpointSpec):
        checkpoint = checkpoint_path
    else:
        source = str(checkpoint_path)
        if source.startswith("hf://"):
            from worldfoundry.core.io.easy_io import resolve_checkpoint_path

            source = resolve_checkpoint_path(source)
        checkpoint = CheckpointSpec(source=source)
    return _load_wan_video_decoder(
        checkpoint,
        RuntimePolicy(device=device, dtype=dtype),
        module_class=WanVideoVAE,
        chunk_duration=chunk_duration,
    )


def build_wan_video_decoder(context: ComponentBuildContext) -> WanVideoDecoder:
    """Build the 16-channel Wan video decoder."""

    return _build_wan_video_decoder(context, module_class=WanVideoVAE)


def build_wan_video_vae38_decoder(context: ComponentBuildContext) -> WanVideoDecoder:
    """Build the 48-channel Wan2.2 video decoder."""

    return _build_wan_video_decoder(context, module_class=WanVideoVAE38)


def build_diffusers_wan_video_codec(context: ComponentBuildContext) -> WanVideoDecoder:
    """Build the shared 16-channel Wan codec from Diffusers-layout weights."""

    return _build_wan_video_decoder(
        context,
        module_class=WanVideoVAE,
        state_dict_converter=convert_diffusers_wan_vae_state_dict,
    )


__all__ = [
    "WanVideoDecoder",
    "build_diffusers_wan_video_codec",
    "build_wan_video_decoder",
    "build_wan_video_vae38_decoder",
    "convert_wan21_vae_state_dict",
    "convert_diffusers_wan22_vae_state_dict",
    "convert_diffusers_wan_vae_state_dict",
    "load_wan_video_codec",
]
