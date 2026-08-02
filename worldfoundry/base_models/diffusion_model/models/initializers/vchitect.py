"""Vchitect latent initialization."""

from __future__ import annotations

import torch

from ...components import ComponentBuildContext
from ...contracts import DiffusionRequest


class VchitectLatentInitializer:
    def __init__(self, *, channels: int = 16, spatial_compression: int = 8) -> None:
        self.channels = int(channels)
        self.spatial_compression = int(spatial_compression)

    def initialize(
        self,
        request: DiffusionRequest,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if request.height % self.spatial_compression or request.width % self.spatial_compression:
            raise ValueError("Vchitect height and width must be divisible by the VAE compression")
        return torch.randn(
            request.batch_size,
            request.num_frames,
            self.channels,
            request.height // self.spatial_compression,
            request.width // self.spatial_compression,
            generator=generator,
            device=device,
            dtype=dtype,
        )


def build_vchitect_latent_initializer(context: ComponentBuildContext) -> VchitectLatentInitializer:
    return VchitectLatentInitializer(
        channels=int(context.component_options.get("channels", 16)),
        spatial_compression=int(context.component_options.get("spatial_compression", 8)),
    )


__all__ = ["VchitectLatentInitializer", "build_vchitect_latent_initializer"]
