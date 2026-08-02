"""Reusable checkpoint-compatible blocks for native 2D latent codecs."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from worldfoundry.core.attention import scaled_dot_product_attention


class VAE2DResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, groups: int = 32, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.nonlinearity = nn.SiLU()
        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(self.nonlinearity(self.norm1(hidden_states)))
        hidden_states = self.conv2(self.nonlinearity(self.norm2(hidden_states)))
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return residual + hidden_states


class VAE2DAttentionBlock(nn.Module):
    def __init__(self, channels: int, *, groups: int = 32, eps: float = 1e-6) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, channels, eps=eps)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.ModuleList((nn.Linear(channels, channels), nn.Dropout(0.0)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = hidden_states.shape
        residual = hidden_states
        sequence = self.group_norm(hidden_states).flatten(2).transpose(1, 2)
        query = self.to_q(sequence).unsqueeze(1)
        key = self.to_k(sequence).unsqueeze(1)
        value = self.to_v(sequence).unsqueeze(1)
        sequence = scaled_dot_product_attention(query, key, value).squeeze(1)
        sequence = self.to_out[1](self.to_out[0](sequence))
        return residual + sequence.transpose(1, 2).reshape(batch, channels, height, width)


class VAE2DMidBlock(nn.Module):
    def __init__(self, channels: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            (VAE2DResnetBlock(channels, channels, eps=eps), VAE2DResnetBlock(channels, channels, eps=eps))
        )
        self.attentions = nn.ModuleList((VAE2DAttentionBlock(channels, eps=eps),))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states)
        hidden_states = self.attentions[0](hidden_states)
        return self.resnets[1](hidden_states)


class VAE2DUpsampler(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(hidden_states, scale_factor=2.0, mode="nearest"))


class VAE2DUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, add_upsampler: bool, eps: float = 1e-6) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            (
                VAE2DResnetBlock(in_channels, out_channels, eps=eps),
                VAE2DResnetBlock(out_channels, out_channels, eps=eps),
                VAE2DResnetBlock(out_channels, out_channels, eps=eps),
            )
        )
        self.upsamplers = nn.ModuleList((VAE2DUpsampler(out_channels),)) if add_upsampler else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.resnets:
            hidden_states = block(hidden_states)
        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states)
        return hidden_states


class NativeVAE2DDecoder(nn.Module):
    """Diffusers-layout-compatible decoder implemented only with native PyTorch."""

    def __init__(
        self,
        *,
        latent_channels: int = 16,
        out_channels: int = 3,
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        norm_num_groups: int = 32,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        reversed_channels = tuple(reversed(block_out_channels))
        self.conv_in = nn.Conv2d(latent_channels, reversed_channels[0], 3, padding=1)
        self.mid_block = VAE2DMidBlock(reversed_channels[0], eps=eps)
        blocks = []
        input_channel = reversed_channels[0]
        for index, output_channel in enumerate(reversed_channels):
            blocks.append(
                VAE2DUpBlock(
                    input_channel,
                    output_channel,
                    add_upsampler=index < len(reversed_channels) - 1,
                    eps=eps,
                )
            )
            input_channel = output_channel
        self.up_blocks = nn.ModuleList(blocks)
        self.conv_norm_out = nn.GroupNorm(norm_num_groups, reversed_channels[-1], eps=eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(reversed_channels[-1], out_channels, 3, padding=1)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        hidden_states = self.mid_block(self.conv_in(latents))
        for block in self.up_blocks:
            hidden_states = block(hidden_states)
        return self.conv_out(self.conv_act(self.conv_norm_out(hidden_states)))


__all__ = [
    "NativeVAE2DDecoder",
    "VAE2DAttentionBlock",
    "VAE2DMidBlock",
    "VAE2DResnetBlock",
    "VAE2DUpBlock",
    "VAE2DUpsampler",
]
