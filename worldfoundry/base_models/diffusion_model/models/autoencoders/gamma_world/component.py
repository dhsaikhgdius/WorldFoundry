"""Multi-view layout adapter for the canonical Wan video codec."""

from __future__ import annotations

import torch
from einops import rearrange

from ....components import ComponentBuildContext
from ....contracts import DiffusionRequest
from ..wan.component import WanVideoDecoder, build_wan_video_decoder


class GammaWorldVideoCodec:
    """Reuse one Wan codec while keeping player timelines independent."""

    def __init__(self, codec: WanVideoDecoder) -> None:
        self.codec = codec

    @property
    def spatial_compression_factor(self) -> int:
        return self.codec.spatial_compression_factor

    @property
    def temporal_compression_factor(self) -> int:
        return self.codec.temporal_compression_factor

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.codec.encode(images)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, request: DiffusionRequest) -> torch.Tensor:
        if bool(request.inputs.get("return_latent", False)):
            return latents
        n_views = int(request.inputs.get("n_players", 2))
        if latents.shape[2] % n_views:
            raise ValueError(
                f"Gamma latent time dimension {latents.shape[2]} is not divisible by n_players={n_views}"
            )
        per_view = rearrange(latents, "B C (V T) H W -> (B V) C T H W", V=n_views)
        decoded = self.codec.decode(per_view)
        decoded = decoded[:, :, : request.num_frames]
        return rearrange(decoded, "(B V) C T H W -> B C T H (V W)", V=n_views)


def build_gamma_world_video_codec(context: ComponentBuildContext) -> GammaWorldVideoCodec:
    return GammaWorldVideoCodec(build_wan_video_decoder(context))


__all__ = ["GammaWorldVideoCodec", "build_gamma_world_video_codec"]
