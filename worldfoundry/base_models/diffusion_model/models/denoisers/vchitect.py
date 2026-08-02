"""Native Vchitect denoiser component."""

from __future__ import annotations

import torch

from ...components import ComponentBuildContext
from ...contracts import DenoiserInput, DenoiserOutput
from ...loaders import ModuleLoadSpec, NativeModuleLoader
from ..networks.vchitect import VchitectXLTransformerModel


VCHITECT_2B_CONFIG = {
    "sample_size": 128,
    "patch_size": 2,
    "in_channels": 16,
    "num_layers": 24,
    "attention_head_dim": 64,
    "num_attention_heads": 24,
    "joint_attention_dim": 4096,
    "caption_projection_dim": 1536,
    "pooled_projection_dim": 2048,
    "out_channels": 16,
    "pos_embed_max_size": 192,
}


class VchitectDenoiser:
    def __init__(self, model: VchitectXLTransformerModel) -> None:
        self.model = model

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        prompt = model_input.conditioning.get("prompt_embeds")
        pooled = model_input.conditioning.get("pooled_prompt_embeds")
        if not isinstance(prompt, torch.Tensor) or not isinstance(pooled, torch.Tensor):
            raise TypeError("Vchitect requires prompt_embeds and pooled_prompt_embeds tensors")
        timestep = model_input.timestep.to(device=model_input.latents.device).reshape(-1)
        if timestep.numel() == 1:
            timestep = timestep.expand(model_input.latents.shape[0])
        sample = self.model(
            model_input.latents,
            encoder_hidden_states=prompt.to(device=model_input.latents.device, dtype=model_input.latents.dtype),
            pooled_projections=pooled.to(device=model_input.latents.device, dtype=model_input.latents.dtype),
            timestep=timestep,
        )
        return DenoiserOutput(sample=sample)


def build_vchitect_denoiser(context: ComponentBuildContext) -> VchitectDenoiser:
    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=VchitectXLTransformerModel,
            config=VCHITECT_2B_CONFIG,
            layer_container="transformer_blocks",
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(model, VchitectXLTransformerModel):
        raise TypeError(f"expected VchitectXLTransformerModel, got {type(model).__name__}")
    return VchitectDenoiser(model)


__all__ = ["VCHITECT_2B_CONFIG", "VchitectDenoiser", "build_vchitect_denoiser"]
