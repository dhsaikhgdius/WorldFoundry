"""Framework-owned latent projection for partially frozen diffusion inputs."""

from __future__ import annotations

import torch

from .base import DiffusionExtension, DiffusionRunContext


class FrozenLatentMaskExtension(DiffusionExtension):
    """Restore clean latent regions selected by a continuous denoise mask."""

    extension_id = "frozen-latent-mask"

    def after_step(
        self,
        context: DiffusionRunContext,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        clean = context.conditioning.shared.get("clean_latents")
        mask = context.conditioning.shared.get("denoise_mask")
        if not isinstance(clean, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise TypeError(
                "masked-latent execution requires tensor clean_latents and denoise_mask conditions"
            )
        clean = clean.to(device=latents.device, dtype=latents.dtype)
        mask = mask.to(device=latents.device, dtype=latents.dtype)
        try:
            return latents * mask + clean * (1.0 - mask)
        except RuntimeError as error:
            raise ValueError(
                "clean_latents and denoise_mask must broadcast to the generated latent shape"
            ) from error


__all__ = ["FrozenLatentMaskExtension"]
