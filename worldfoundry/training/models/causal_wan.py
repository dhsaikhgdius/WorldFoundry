"""Native causal Wan construction for few-step training algorithms."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.loaders import (
    CheckpointSpec,
    ModuleLoadSpec,
    NativeModuleLoader,
)
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import (
    WAN21_T2V_1P3B_CONFIG,
)
from worldfoundry.base_models.diffusion_model.optimizations import (
    AttentionBackend,
    RuntimePolicy,
)

SELF_FORCING_ODE_CHECKPOINT = CheckpointSpec(
    repo_id="gdhe17/Self-Forcing",
    revision="47f4d3cf430cf000fcad587ba02c83ed971bba69",
    files=("checkpoints/ode_init.pt",),
    file_sha256={
        "checkpoints/ode_init.pt": "b5396b8076ab3b920c9e4f4a2b52daa2c98c9983fb5e067ae5160fdf778dce21",
    },
    file_size_bytes={"checkpoints/ode_init.pt": 5_676_203_690},
)


def convert_self_forcing_causal_state_dict(
    state_dict: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    """Unwrap the official generator checkpoint into the causal Wan graph.

    The released ODE initialization stores the graph below ``generator`` and
    prefixes every graph key with ``model.``.  The additional singular wrapper
    names accepted here are the standard PyTorch/FSDP save envelopes used by
    WorldFoundry checkpoints; arbitrary partial or mixed prefixes are rejected.
    """

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise TypeError("causal Wan checkpoint must contain a non-empty mapping")
    candidate: Mapping[str, object] = state_dict
    wrappers = {"generator", "state_dict", "module", "model"}
    while len(candidate) == 1:
        name, value = next(iter(candidate.items()))
        if name not in wrappers or not isinstance(value, Mapping):
            break
        candidate = value
    if not candidate:
        raise ValueError("causal Wan checkpoint resolved to an empty state dict")
    if any(not isinstance(name, str) or not name for name in candidate):
        raise TypeError("causal Wan checkpoint parameter names must be non-empty strings")
    if any(not isinstance(value, torch.Tensor) for value in candidate.values()):
        invalid = sorted(str(name) for name, value in candidate.items() if not isinstance(value, torch.Tensor))
        raise TypeError(f"causal Wan checkpoint contains non-tensor entries: {invalid[:8]}")

    converted = {str(name): value for name, value in candidate.items()}
    for prefix in ("_orig_mod.", "module.", "model."):
        if converted and all(name.startswith(prefix) for name in converted):
            stripped = {name.removeprefix(prefix): value for name, value in converted.items()}
            if len(stripped) != len(converted):
                raise ValueError(f"causal Wan checkpoint prefix {prefix!r} creates duplicate keys")
            converted = stripped
    if any(name.startswith(("model.", "module.", "_orig_mod.")) for name in converted):
        raise ValueError("causal Wan checkpoint mixes wrapped and unwrapped parameter names")
    return converted


def causal_wan_1p3b_config() -> dict[str, object]:
    """Return the released Self-Forcing causal Wan 1.3B graph contract."""

    config = {name: value for name, value in WAN21_T2V_1P3B_CONFIG.items() if name != "has_image_input"}
    config.update(
        {
            "model_type": "t2v",
            "text_len": 512,
            "local_attn_size": -1,
            "sink_size": 0,
            "cross_attn_norm": True,
        }
    )
    return config


def load_causal_wan_1p3b(
    checkpoint: CheckpointSpec,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    gradient_checkpointing: bool = False,
) -> nn.Module:
    """Strictly load the in-tree causal graph from an audited checkpoint."""

    if not isinstance(checkpoint, CheckpointSpec):
        raise TypeError("checkpoint must be CheckpointSpec")
    if not isinstance(gradient_checkpointing, bool):
        raise TypeError("gradient_checkpointing must be a bool")
    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("causal Wan dtype must be float16, bfloat16, or float32")
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing.self_forcing import (
        CausalWanModel,
    )

    module = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=CausalWanModel,
            config=causal_wan_1p3b_config(),
            state_dict_converter=convert_self_forcing_causal_state_dict,
        ),
        checkpoint,
        RuntimePolicy(
            device=torch.device(device),
            dtype=dtype,
            attention=AttentionBackend.TORCH,
        ),
    )
    module.gradient_checkpointing = gradient_checkpointing
    module.eval()
    return module


class CausalWanTrainRole:
    """Mutable training role around one native causal Wan graph."""

    prediction_type = "flow_velocity"
    lora_target_preset = "wan-attention"
    expected_latent_channels = 16
    temporal_compression = 4
    spatial_compression = 8
    expected_text_length = 512
    expected_context_features = 4096

    def __init__(self, module: nn.Module) -> None:
        if not isinstance(module, nn.Module):
            raise TypeError("causal Wan role requires an nn.Module")
        blocks = getattr(module, "blocks", None)
        if not isinstance(blocks, nn.ModuleList) or not blocks:
            raise TypeError("causal Wan role requires a non-empty ModuleList named 'blocks'")
        self.graph = module
        self.trainable_module = module
        self.fsdp_block_classes = tuple(dict.fromkeys(type(block) for block in blocks))

    def replace_trainable_module(self, module: nn.Module) -> None:
        if not isinstance(module, nn.Module):
            raise TypeError("causal Wan trainable module must be nn.Module")
        self.trainable_module = module


def validate_causal_wan_dtype(role: CausalWanTrainRole, expected: torch.dtype) -> None:
    """Reject mixed or silently converted causal-student parameters."""

    if not isinstance(role, CausalWanTrainRole):
        raise TypeError("role must be CausalWanTrainRole")
    dtypes = {parameter.dtype for parameter in role.trainable_module.parameters() if parameter.is_floating_point()}
    if dtypes != {expected}:
        raise ValueError(
            "loaded causal Wan dtype differs from runtime.param_dtype: "
            f"loaded={sorted(map(str, dtypes))}, expected={expected}"
        )


__all__ = [
    "SELF_FORCING_ODE_CHECKPOINT",
    "CausalWanTrainRole",
    "causal_wan_1p3b_config",
    "convert_self_forcing_causal_state_dict",
    "load_causal_wan_1p3b",
    "validate_causal_wan_dtype",
]
