"""Generic frozen-context lifecycle for suffix-conditioned diffusion."""

from __future__ import annotations

import torch

from .base import DiffusionExtension, DiffusionRunContext


class FrozenContextSuffixExtension(DiffusionExtension):
    """Restore a clean context suffix after every scheduler update."""

    extension_id = "frozen-context-suffix"

    def on_run_start(self, context: DiffusionRunContext) -> None:
        values = context.request.inputs.get("frozen_context_latents")
        if not isinstance(values, torch.Tensor):
            raise TypeError("frozen-context execution requires frozen_context_latents")
        if values.ndim == 4:
            values = values.unsqueeze(0)
        if values.ndim != 5 or int(values.shape[2]) <= 0:
            raise ValueError("frozen_context_latents must be a non-empty [B,C,T,H,W] tensor")
        context.state[self.extension_id] = values

    def after_step(
        self,
        context: DiffusionRunContext,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        clean = context.state[self.extension_id]
        if not isinstance(clean, torch.Tensor):
            raise RuntimeError("frozen context state was not initialized")
        clean = clean.to(device=latents.device, dtype=latents.dtype)
        if clean.shape[0] == 1 and latents.shape[0] > 1:
            clean = clean.expand(latents.shape[0], -1, -1, -1, -1)
        if clean.shape[:2] != latents.shape[:2] or clean.shape[-2:] != latents.shape[-2:]:
            raise ValueError("frozen context shape changed during denoising")
        if int(clean.shape[2]) >= int(latents.shape[2]):
            raise ValueError("frozen context must be shorter than the full latent sequence")
        result = latents.clone()
        result[:, :, -int(clean.shape[2]) :] = clean
        return result


__all__ = ["FrozenContextSuffixExtension"]
