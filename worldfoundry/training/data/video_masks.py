"""Mask projection for causal first-frame video latent geometries."""

from __future__ import annotations

import torch
from torch.nn import functional


def project_causal_video_mask_to_latent(
    valid_mask: torch.Tensor,
    *,
    pixel_shape: tuple[int, int, int, int, int],
    latent_shape: tuple[int, int, int],
    temporal_compression: int = 4,
) -> torch.Tensor:
    """Project pixel validity weights through a first-frame causal video codec.

    The first latent frame represents the first pixel frame. Each later latent
    frame represents ``temporal_compression`` subsequent pixel frames, so their
    validity is the mean of that temporal group. Spatial validity is projected
    with area interpolation.
    """

    compression = int(temporal_compression)
    if compression <= 0:
        raise ValueError("temporal_compression must be positive")
    mask = valid_mask
    if mask.ndim == 4 and int(mask.shape[0]) == pixel_shape[0]:
        mask = mask.unsqueeze(1)
    try:
        mask = torch.broadcast_to(mask, pixel_shape)
    except RuntimeError as error:
        raise ValueError(
            f"valid_mask shape {tuple(valid_mask.shape)} cannot broadcast to pixels {pixel_shape}"
        ) from error
    mask = mask.float().amin(dim=1, keepdim=True)

    latent_frames, latent_height, latent_width = latent_shape
    pixel_frames = int(mask.shape[2])
    expected_latent_frames = 1 + (pixel_frames - 1) // compression
    if (pixel_frames - 1) % compression or expected_latent_frames != latent_frames:
        raise ValueError("pixel valid_mask temporal geometry differs from encoded latents")
    spatial = functional.interpolate(
        mask,
        size=(pixel_frames, latent_height, latent_width),
        mode="area",
    )
    first = spatial[:, :, :1]
    if latent_frames == 1:
        return first
    remaining = spatial[:, :, 1:].reshape(
        pixel_shape[0],
        1,
        latent_frames - 1,
        compression,
        latent_height,
        latent_width,
    )
    return torch.cat((first, remaining.mean(dim=3)), dim=2)


__all__ = ["project_causal_video_mask_to_latent"]
