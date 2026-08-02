"""Checkpoint-backed audio latent statistics without signal-processing dependencies."""

import torch
from torch import nn


class PerChannelStatistics(nn.Module):
    """Normalize audio latents with learned dataset statistics."""

    def __init__(self, latent_channels: int = 128) -> None:
        super().__init__()
        self.register_buffer("std-of-means", torch.empty(latent_channels))
        self.register_buffer("mean-of-means", torch.empty(latent_channels))

    def un_normalize(self, value: torch.Tensor) -> torch.Tensor:
        return (value * self.get_buffer("std-of-means").to(value)) + self.get_buffer("mean-of-means").to(value)

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.get_buffer("mean-of-means").to(value)) / self.get_buffer("std-of-means").to(value)


__all__ = ["PerChannelStatistics"]
