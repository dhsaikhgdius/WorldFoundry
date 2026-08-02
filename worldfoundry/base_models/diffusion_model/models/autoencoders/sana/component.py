"""Sana DC-AE role for the canonical diffusion runner."""

from __future__ import annotations

import torch

from ....components import ComponentBuildContext
from ....contracts import DiffusionRequest
from ....loaders import ModuleLoadSpec, NativeModuleLoader
from .dc_ae import DCAE, dc_ae_f32c32


class SanaDCAutoencoder:
    """Encode and decode Sana image latents with explicit scale semantics."""

    spatial_compression_factor = 32
    temporal_compression_factor = 1
    latent_ch = 32

    def __init__(
        self,
        model: DCAE,
        *,
        scaling_factor: float = 0.41407,
        decoder_input_scale: float = 1.0,
    ) -> None:
        self.model = model
        self.scaling_factor = float(scaling_factor)
        self.decoder_input_scale = float(decoder_input_scale)
        if self.decoder_input_scale <= 0:
            raise ValueError("Sana decoder_input_scale must be positive")

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Sana DC-AE expects BCHW RGB images, got {tuple(images.shape)}")
        parameter = next(self.model.parameters())
        images = images.to(device=parameter.device, dtype=parameter.dtype)
        return self.model.encode(images) * self.scaling_factor

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        request: DiffusionRequest | None = None,
    ) -> torch.Tensor:
        if request is not None and bool(request.inputs.get("return_latent", False)):
            return latents
        parameter = next(self.model.parameters())
        latents = latents.to(device=parameter.device, dtype=parameter.dtype)
        latents = latents / self.decoder_input_scale
        return self.model.decode(latents / self.scaling_factor).clamp_(-1.0, 1.0)


def build_sana_dc_autoencoder(context: ComponentBuildContext) -> SanaDCAutoencoder:
    """Load the original DC-AE graph through the shared module loader."""

    config = dc_ae_f32c32("dc-ae-f32c32-sana-1.1", None)
    model = NativeModuleLoader().load(
        ModuleLoadSpec(module_class=DCAE, config={"cfg": config}),
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(model, DCAE):
        raise TypeError(f"expected DCAE, got {type(model).__name__}")
    return SanaDCAutoencoder(
        model,
        scaling_factor=config.scaling_factor or 0.41407,
        decoder_input_scale=float(context.component_options.get("decoder_input_scale", 1.0)),
    )


__all__ = ["SanaDCAutoencoder", "build_sana_dc_autoencoder"]
