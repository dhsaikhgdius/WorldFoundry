"""Checkpoint loading and state-dict remapping helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "DefaultLoadPlanner": "worldfoundry.core.checkpoint.dcp",
    "DistributedCheckpointer": "worldfoundry.core.checkpoint.dcp",
    "ModelWrapper": "worldfoundry.core.checkpoint.dcp",
    "assign_state_dict_strict": "worldfoundry.core.checkpoint.assignment",
    "get_storage_reader": "worldfoundry.core.checkpoint.load",
    "load_checkpoint": "worldfoundry.core.checkpoint.load",
    "load_distributed_checkpoint": "worldfoundry.core.checkpoint.load",
    "load_safetensors_into_model_streaming": "worldfoundry.core.checkpoint.sharded_safetensors",
    "load_sharded_safetensors_parallel_with_progress": "worldfoundry.core.checkpoint.sharded_safetensors",
    "load_single_checkpoint": "worldfoundry.core.checkpoint.load",
    "load_tensor_state_dict": "worldfoundry.core.checkpoint.safe_loading",
    "load_weights_only": "worldfoundry.core.checkpoint.safe_loading",
    "remap_checkpoint_keys": "worldfoundry.core.checkpoint.remap",
    "require_mapping": "worldfoundry.core.checkpoint.safe_loading",
    "require_tensor": "worldfoundry.core.checkpoint.safe_loading",
    "safetensor_checkpoint_files": "worldfoundry.core.checkpoint.sharded_safetensors",
    "select_profile_checkpoint": "worldfoundry.core.checkpoint.selection",
    "selected_checkpoint_options": "worldfoundry.core.checkpoint.selection",
    "submodule_state_dict": "worldfoundry.core.checkpoint.remap",
    "unwrap_model": "worldfoundry.core.checkpoint.sharded_safetensors",
    "tensor_state_dict": "worldfoundry.core.checkpoint.safe_loading",
    "validate_state_dict_compatibility": "worldfoundry.core.checkpoint.assignment",
    "dcp_load_state_dict": "worldfoundry.core.checkpoint.dcp",
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
