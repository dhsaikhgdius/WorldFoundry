"""Canonical checkpoint descriptions and native component loading."""

from importlib import import_module

from .checkpoints import CheckpointSpec
from .materialize import MaterializedCheckpoint, NativeCheckpointResolver
from .metadata import checkpoint_json_config, safetensors_json_metadata
from .module import CheckpointConfigResolver, ModuleLoadSpec, NativeModuleLoader

_WAN_EXPORTS = {
    "WanConditioningComponents",
    "WanInferenceComponents",
    "load_wan_conditioning_components",
    "load_wan_inference_components",
    "load_wan_transformer_checkpoint",
    "load_wan_vae_checkpoint",
}


def __getattr__(name):
    if name not in _WAN_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.wan_components"), name)
    globals()[name] = value
    return value

__all__ = [
    "CheckpointSpec",
    "CheckpointConfigResolver",
    "MaterializedCheckpoint",
    "ModuleLoadSpec",
    "NativeCheckpointResolver",
    "NativeModuleLoader",
    "checkpoint_json_config",
    "safetensors_json_metadata",
    "WanInferenceComponents",
    "WanConditioningComponents",
    "load_wan_conditioning_components",
    "load_wan_inference_components",
    "load_wan_transformer_checkpoint",
    "load_wan_vae_checkpoint",
]
