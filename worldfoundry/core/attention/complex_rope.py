"""Complex-valued rotary tables used by video diffusion transformers."""

from __future__ import annotations

import torch
from einops import rearrange


def complex_rotary_frequencies(
    dim: int,
    end: int = 1024,
    theta: float = 10_000.0,
    *,
    subdivisions: int = 1,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a complex RoPE table, optionally at fractional position steps."""

    if dim <= 0 or dim % 2:
        raise ValueError("dim must be a positive even integer")
    if end < 0 or subdivisions <= 0:
        raise ValueError("end must be non-negative and subdivisions must be positive")
    resolved_device = torch.device("cpu") if device is None else torch.device(device)
    indices = torch.arange(0, dim, 2, dtype=torch.float64, device=resolved_device)
    inverse_frequencies = 1.0 / (float(theta) ** (indices / dim))
    positions = torch.arange(end * subdivisions, dtype=torch.float64, device=resolved_device) / subdivisions
    angles = torch.outer(positions, inverse_frequencies)
    return torch.polar(torch.ones_like(angles), angles)


def complex_rotary_frequencies_3d(
    dim: int,
    end: int = 1024,
    theta: float = 10_000.0,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build temporal, height, and width complex RoPE tables."""

    spatial_dim = dim // 3
    temporal_dim = dim - 2 * spatial_dim
    return (
        complex_rotary_frequencies(temporal_dim, end, theta, device=device),
        complex_rotary_frequencies(spatial_dim, end, theta, device=device),
        complex_rotary_frequencies(spatial_dim, end, theta, device=device),
    )


def apply_complex_rotary_embedding(
    value: torch.Tensor,
    frequencies: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """Apply a broadcast complex RoPE table to ``[B,S,H*D]`` tokens."""

    value = rearrange(value, "b s (h d) -> b s h d", h=int(num_heads))
    complex_value = torch.view_as_complex(value.to(torch.float64).reshape(*value.shape[:-1], -1, 2))
    if frequencies.device.type == "npu":
        frequencies = frequencies.to(torch.complex64)
    output = torch.view_as_real(complex_value * frequencies).flatten(2)
    return output.to(value.dtype)


__all__ = [
    "apply_complex_rotary_embedding",
    "complex_rotary_frequencies",
    "complex_rotary_frequencies_3d",
]
