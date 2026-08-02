"""Shared Wan codec role for Cosmos Predict 2.5 initialization and decoding."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as functional

from worldfoundry.core.io.video import coerce_video_frames

from ....components import ComponentBuildContext
from ....contracts import DiffusionRequest, LatentInitialization
from ....loaders import ModuleLoadSpec, NativeModuleLoader
from ...initializers.video_conditioning import prepare_video_conditioning_pixels
from ..wan.component import convert_wan21_vae_state_dict
from ..wan.model import CausalConv3d, RMS_norm, Upsample, WanVideoVAE


class Cosmos25VideoCodec:
    """One VAE instance serving both latent initialization and final decoding."""

    def __init__(
        self,
        vae: WanVideoVAE,
        *,
        device: torch.device,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
    ) -> None:
        self.vae = vae
        self.device = device
        self.tiled = bool(tiled)
        self.tile_size = tile_size
        self.tile_stride = tile_stride

    @staticmethod
    def _resize(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return functional.interpolate(frames.float(), (height, width), mode="bilinear", align_corners=False)

    def _control_pixels(
        self,
        request: DiffusionRequest,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        control = None
        for key in ("control_video", "control", "controls"):
            if request.inputs.get(key) is not None:
                control = request.inputs[key]
                break
        if isinstance(control, Mapping):
            variant = str(request.inputs.get("controlnet_variant", "edge"))
            control = control.get(variant, next(iter(control.values()), None))
        if control is None:
            return None
        if request.batch_size != 1:
            raise ValueError("Cosmos Transfer2.5 control video currently requires batch size 1")
        frames = torch.from_numpy(coerce_video_frames(control)).permute(0, 3, 1, 2)
        if not len(frames):
            raise ValueError("Cosmos Transfer2.5 control video cannot be empty")
        frames = self._resize(frames, request.height, request.width)
        if len(frames) < request.num_frames:
            frames = torch.cat((frames, frames[-1:].expand(request.num_frames - len(frames), -1, -1, -1)))
        frames = frames[: request.num_frames]
        return frames.permute(1, 0, 2, 3).unsqueeze(0).div(127.5).sub(1).to(device=device, dtype=dtype)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Expose the shared Wan encoder through the canonical latent role."""

        if images.ndim == 4:
            images = images.unsqueeze(2)
        if images.ndim != 5:
            raise ValueError("Cosmos Predict2 Wan encoder expects BCHW or BCTHW pixels")
        return self.vae.encode(
            [images[index] for index in range(len(images))],
            self.device,
            tiled=self.tiled,
            tile_size=self.tile_size,
            tile_stride=self.tile_stride,
        )

    def initialize(
        self,
        request: DiffusionRequest,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LatentInitialization:
        if request.height % 16 or request.width % 16:
            raise ValueError("Cosmos2.5 height and width must be divisible by 16")
        if (request.num_frames - 1) % 4:
            raise ValueError("Cosmos2.5 num_frames must satisfy (num_frames - 1) % 4 == 0")
        latent_frames = (request.num_frames - 1) // 4 + 1
        shape = (request.batch_size, 16, latent_frames, request.height // 8, request.width // 8)
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        pixels, conditioned_frames = prepare_video_conditioning_pixels(
            request,
            device=device,
            dtype=dtype,
            temporal_compression=4,
            owner="Cosmos2.5",
        )
        if pixels is None:
            condition_latents = torch.zeros_like(noise)
            condition_count = 0
        else:
            condition_latents = self.vae.encode(
                [pixels[index] for index in range(len(pixels))],
                self.device,
                tiled=self.tiled,
                tile_size=self.tile_size,
                tile_stride=self.tile_stride,
            ).to(device=device, dtype=dtype)
            condition_count = (conditioned_frames - 1) // 4 + 1
        indicator = torch.zeros((request.batch_size, 1, latent_frames, 1, 1), device=device, dtype=dtype)
        indicator[:, :, :condition_count] = 1
        condition_mask = indicator.expand(-1, -1, -1, shape[-2], shape[-1])
        conditioning: dict[str, object] = {
            "condition_latents": condition_latents,
            "condition_mask": condition_mask,
            "condition_indicator": indicator,
            "initial_noise": noise.clone(),
            "padding_mask": torch.zeros(
                (request.batch_size, 1, request.height, request.width),
                device=device,
                dtype=dtype,
            ),
            "conditional_frame_timestep": float(request.inputs.get("conditional_frame_timestep", 0.0)),
        }
        control_pixels = self._control_pixels(request, device=device, dtype=dtype)
        if control_pixels is not None:
            conditioning["latent_control_input"] = self.vae.encode(
                [control_pixels[index] for index in range(len(control_pixels))],
                self.device,
                tiled=self.tiled,
                tile_size=self.tile_size,
                tile_stride=self.tile_stride,
            ).to(device=device, dtype=dtype)
            conditioning["control_context_scale"] = float(request.inputs.get("control_context_scale", 1.0))
        return LatentInitialization(latents=noise, conditioning=conditioning)

    def decode(self, latents: torch.Tensor, request: DiffusionRequest) -> torch.Tensor:
        if bool(request.inputs.get("return_latent", False)):
            return latents
        video = self.vae.decode(
            latents,
            self.device,
            tiled=self.tiled,
            tile_size=self.tile_size,
            tile_stride=self.tile_stride,
        )
        return video[:, :, : request.num_frames]


def build_cosmos25_video_codec(context: ComponentBuildContext) -> Cosmos25VideoCodec:
    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    vae = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=WanVideoVAE,
            state_dict_converter=convert_wan21_vae_state_dict,
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
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(vae, WanVideoVAE):
        raise TypeError(f"expected WanVideoVAE, got {type(vae).__name__}")
    tile_size = tuple(context.component_options.get("tile_size", (34, 34)))
    tile_stride = tuple(context.component_options.get("tile_stride", (18, 16)))
    if len(tile_size) != 2 or len(tile_stride) != 2:
        raise ValueError("Cosmos2.5 VAE tile sizes must contain two integers")
    return Cosmos25VideoCodec(
        vae,
        device=context.policy.device,
        tiled=bool(context.component_options.get("tiled", False)),
        tile_size=(int(tile_size[0]), int(tile_size[1])),
        tile_stride=(int(tile_stride[0]), int(tile_stride[1])),
    )


__all__ = ["Cosmos25VideoCodec", "build_cosmos25_video_codec"]
