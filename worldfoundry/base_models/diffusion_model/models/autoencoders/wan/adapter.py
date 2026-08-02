"""Framework-independent adapter surface for native Wan VAE variants."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from worldfoundry.core.nn import (
    AutoencoderKLOutput,
    DecoderOutput,
    DiagonalGaussianDistribution,
    ModuleDeviceDtypeMixin,
)


@dataclass(frozen=True, slots=True)
class WanAutoencoderConfig:
    """Runtime-visible latent geometry used by existing Wan consumers."""

    latent_channels: int
    temporal_compression_ratio: int
    spatial_compression_ratio: int


class WanAutoencoderAdapterMixin(ModuleDeviceDtypeMixin):
    """Expose canonical Wan encode/decode results without backend types."""

    @property
    def latent_channels(self) -> int:
        """Expose Diffusers-compatible latent geometry on the module."""

        return int(self.config.latent_channels)

    @property
    def temporal_compression_ratio(self) -> int:
        """Expose the temporal scale expected by existing Wan pipelines."""

        return int(self.config.temporal_compression_ratio)

    @property
    def spatial_compression_ratio(self) -> int:
        """Expose the spatial scale expected by existing Wan pipelines."""

        return int(self.config.spatial_compression_ratio)

    def encode(self, value: torch.Tensor, return_dict: bool = True):
        encoded = torch.stack(
            [
                self.model.encode(item.unsqueeze(0), self.scale).squeeze(0)
                for item in value
            ]
        )
        moments = torch.cat((encoded, torch.zeros_like(encoded)), dim=1)
        posterior = DiagonalGaussianDistribution(moments, deterministic=True)
        if return_dict:
            return AutoencoderKLOutput(latent_dist=posterior)
        return (posterior,)

    def decode(self, value: torch.Tensor, return_dict: bool = True):
        decoded = torch.stack(
            [
                self.model.decode(item.unsqueeze(0), self.scale)
                .clamp_(-1, 1)
                .squeeze(0)
                for item in value
            ]
        )
        if return_dict:
            return DecoderOutput(sample=decoded)
        return (decoded,)


__all__ = ["WanAutoencoderAdapterMixin", "WanAutoencoderConfig"]
