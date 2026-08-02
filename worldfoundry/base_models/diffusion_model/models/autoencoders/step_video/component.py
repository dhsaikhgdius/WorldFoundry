"""StepVideo native VAE decoder component."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ....components import ComponentBuildContext
from ....contracts import DiffusionRequest
from ....loaders import ModuleLoadSpec, NativeModuleLoader
from .model import AutoencoderKL


def _vae_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    return {
        key.replace("decoder.conv_out.", "decoder.conv_out.conv."): value
        for key, value in state_dict.items()
    }


class StepVideoDecoder:
    def __init__(self, vae: AutoencoderKL) -> None:
        self.vae = vae

    def decode(self, latents: torch.Tensor, request: DiffusionRequest) -> torch.Tensor:
        if bool(request.inputs.get("return_latent", False)):
            return latents
        video = self.vae.decode(latents)
        if video.ndim != 5:
            raise ValueError("StepVideo VAE must return [B,F,C,H,W]")
        return video.permute(0, 2, 1, 3, 4).clamp(-1.0, 1.0)


def build_step_video_decoder(context: ComponentBuildContext) -> StepVideoDecoder:
    vae = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=AutoencoderKL,
            config={"z_channels": 64, "version": 2, "world_size": 1},
            state_dict_converter=_vae_state_dict,
            layer_container="decoder",
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(vae, AutoencoderKL):
        raise TypeError(f"expected StepVideo AutoencoderKL, got {type(vae).__name__}")
    return StepVideoDecoder(vae)


__all__ = ["StepVideoDecoder", "build_step_video_decoder"]
