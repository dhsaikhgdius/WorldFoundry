"""Small reusable neural-network tensor helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "AdaLayerNorm": "worldfoundry.core.nn.diffusion_transformer",
    "CheckpointMode": "worldfoundry.core.nn.activation_checkpointing",
    "AttentionBackendInfo": "worldfoundry.core.attention.native",
    "AdaZeroCallable": "worldfoundry.core.nn.transformer_ops",
    "DEFAULT_TRANSFORMER_OPS": "worldfoundry.core.nn.transformer_ops",
    "DropPath": "worldfoundry.core.nn.layers",
    "DomainAwareLinear": "worldfoundry.core.nn.layers",
    "FlowMatchScheduler": "worldfoundry.core.nn.diffusion_schedulers",
    "GatedAttentionCallable": "worldfoundry.core.nn.transformer_ops",
    "InferenceCheckpointModule": "worldfoundry.core.nn.checkpoint_compat",
    "LayerNorm2d": "worldfoundry.core.nn.layers",
    "LayerScale": "worldfoundry.core.nn.layers",
    "LitEma": "worldfoundry.core.nn.ema",
    "ModuleDeviceDtypeMixin": "worldfoundry.core.nn.module_properties",
    "SamHeadMLP": "worldfoundry.core.nn.layers",
    "SamMLPBlock": "worldfoundry.core.nn.layers",
    "SchedulerInterface": "worldfoundry.core.nn.diffusion_schedulers",
    "SACConfig": "worldfoundry.core.nn.activation_checkpointing",
    "Mlp": "worldfoundry.core.nn.layers",
    "NativeVAE2DDecoder": "worldfoundry.core.nn.vae2d",
    "PositionEmbeddingRandom": "worldfoundry.core.nn.layers",
    "PostSACallable": "worldfoundry.core.nn.transformer_ops",
    "PreAttentionCallable": "worldfoundry.core.nn.transformer_ops",
    "ProjectedTimestepEmbedding": "worldfoundry.core.nn.timestep",
    "PytorchAdaZeroFunction": "worldfoundry.core.nn.transformer_ops",
    "PytorchGatedAttention": "worldfoundry.core.nn.transformer_ops",
    "PytorchPostSAFunction": "worldfoundry.core.nn.transformer_ops",
    "PytorchPreAttention": "worldfoundry.core.nn.transformer_ops",
    "DiTModulation": "worldfoundry.core.nn.diffusion_transformer",
    "DiTFinalLayer": "worldfoundry.core.nn.diffusion_transformer",
    "ConcatenatedLinear": "worldfoundry.core.nn.diffusion_transformer",
    "ConditioningProjection": "worldfoundry.core.nn.diffusion_transformer",
    "MLPEmbedder": "worldfoundry.core.nn.diffusion_transformer",
    "TransformerMLP": "worldfoundry.core.nn.diffusion_transformer",
    "SinusoidalTimestepEmbedder": "worldfoundry.core.nn.diffusion_transformer",
    "PatchGridSpec": "worldfoundry.core.nn.patching",
    "PatchEmbed": "worldfoundry.core.nn.layers",
    "PatchEmbed_Mlp": "worldfoundry.core.nn.layers",
    "Permute": "worldfoundry.core.nn.layers",
    "PixelUnshuffle": "worldfoundry.core.nn.layers",
    "PreNormTransformerBlock": "worldfoundry.core.nn.vit_block",
    "QKVSelfAttention": "worldfoundry.core.attention.vit_qkv",
    "QKNormRopeSelfAttention": "worldfoundry.core.attention.vit_qkv",
    "RopePreNormTransformerBlock": "worldfoundry.core.nn.vit_block",
    "RMSNorm": "worldfoundry.core.nn.diffusion_transformer",
    "SwiGLU": "worldfoundry.core.nn.layers",
    "SwiGLUFFN": "worldfoundry.core.nn.layers",
    "SwiGLUFFNFused": "worldfoundry.core.nn.layers",
    "XFORMERS_AVAILABLE": "worldfoundry.core.nn.layers",
    "XFORMERS_ENABLED": "worldfoundry.core.nn.layers",
    "TransformerShapeSpec": "worldfoundry.core.nn.transformer",
    "TransformerAttentionOps": "worldfoundry.core.nn.transformer_ops",
    "TransformerOpsConfig": "worldfoundry.core.nn.transformer_ops",
    "TimestepEmbedding": "worldfoundry.core.nn.timestep",
    "DiagonalGaussianDistribution": "worldfoundry.core.nn.distributions",
    "AutoencoderKLOutput": "worldfoundry.core.nn.distributions",
    "DecoderOutput": "worldfoundry.core.nn.distributions",
    "DiracDistribution": "worldfoundry.core.nn.distributions",
    "Timesteps": "worldfoundry.core.nn.timestep",
    "add_residual": "worldfoundry.core.nn.stochastic_depth",
    "activation_layer": "worldfoundry.core.nn.diffusion_transformer",
    "apply_gate": "worldfoundry.core.nn.diffusion_transformer",
    "apply_gate_with_prefix": "worldfoundry.core.nn.diffusion_transformer",
    "apply_prenorm_transformer_residuals": "worldfoundry.core.nn.vit_block",
    "apply_rotary_embedding": "worldfoundry.core.attention.rope",
    "attention_backend_info": "worldfoundry.core.attention.native",
    "attention_head_dim": "worldfoundry.core.nn.transformer",
    "causal_attention_mask": "worldfoundry.core.nn.transformer",
    "drop_add_residual_stochastic_depth": "worldfoundry.core.nn.stochastic_depth",
    "drop_path": "worldfoundry.core.nn.layers",
    "get_branges_scales": "worldfoundry.core.nn.stochastic_depth",
    "get_same_padding": "worldfoundry.core.nn.layers",
    "get_timestep_embedding": "worldfoundry.core.nn.timestep",
    "layer_scale": "worldfoundry.core.nn.normalization",
    "list_sum": "worldfoundry.core.nn.layers",
    "merge_attention_heads": "worldfoundry.core.nn.transformer",
    "make_2tuple": "worldfoundry.core.nn.layers",
    "mlp_hidden_size": "worldfoundry.core.nn.transformer",
    "modulate_sequence": "worldfoundry.core.nn.diffusion_transformer",
    "modulate_sequence_with_prefix": "worldfoundry.core.nn.diffusion_transformer",
    "named_apply": "worldfoundry.core.nn.module_utils",
    "normalization_layer": "worldfoundry.core.nn.diffusion_transformer",
    "patchify_image": "worldfoundry.core.nn.patching",
    "rms_norm": "worldfoundry.core.nn.normalization",
    "rotary_frequencies": "worldfoundry.core.attention.rope",
    "scale_shift": "worldfoundry.core.nn.diffusion_transformer",
    "rotate_half": "worldfoundry.core.attention.rope",
    "scaled_dot_product_attention": "worldfoundry.core.attention.native",
    "sinusoidal_embedding_1d": "worldfoundry.core.nn.transformer",
    "split_attention_heads": "worldfoundry.core.nn.transformer",
    "transformer_shape_spec": "worldfoundry.core.nn.transformer",
    "to_2tuple": "worldfoundry.core.nn.layers",
    "to_3tuple": "worldfoundry.core.nn.layers",
    "unpatchify_image": "worldfoundry.core.nn.patching",
    "val2list": "worldfoundry.core.nn.layers",
    "val2tuple": "worldfoundry.core.nn.layers",
    "ceil_to_divisible": "worldfoundry.core.nn.volume",
    "chunked_interpolate": "worldfoundry.core.nn.volume",
    "pixel_shuffle_3d": "worldfoundry.core.nn.volume",
    "pixel_unshuffle_3d": "worldfoundry.core.nn.volume",
    "velocity_to_denoised": "worldfoundry.core.nn.diffusion_transformer",
    "zero_module": "worldfoundry.core.nn.layers",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


__all__ = sorted(_EXPORT_MODULES)
