"""High-performance VRAM, disk-offload, and lazy weight materialization helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "AutoTorchModule": "worldfoundry.core.vram.layers",
    "AutoWrappedLinear": "worldfoundry.core.vram.layers",
    "AutoWrappedModule": "worldfoundry.core.vram.layers",
    "AutoWrappedNonRecurseModule": "worldfoundry.core.vram.layers",
    "DiskMap": "worldfoundry.core.vram.disk_map",
    "LayerwiseOffloadHandle": "worldfoundry.core.vram.layerwise_offload",
    "DynamicSwapInstaller": "worldfoundry.core.vram.memory",
    "WanAutoCastLayerNorm": "worldfoundry.core.vram.layers",
    "cpu": "worldfoundry.core.vram.memory",
    "enable_layerwise_cpu_offload": "worldfoundry.core.vram.layerwise_offload",
    "enable_vram_management": "worldfoundry.core.vram.layers",
    "enable_vram_management_recursively": "worldfoundry.core.vram.layers",
    "fake_diffusers_current_device": "worldfoundry.core.vram.memory",
    "fill_vram_config": "worldfoundry.core.vram.layers",
    "get_cuda_free_memory_gb": "worldfoundry.core.vram.memory",
    "gpu": "worldfoundry.core.vram.memory",
    "gpu_complete_modules": "worldfoundry.core.vram.memory",
    "init_weights_on_device": "worldfoundry.core.vram.initialization",
    "layerwise_offload_mutation_scope": "worldfoundry.core.vram.layerwise_offload",
    "load_model_as_complete": "worldfoundry.core.vram.memory",
    "log_gpu_memory": "worldfoundry.core.vram.memory",
    "move_model_to_device_with_memory_preservation": "worldfoundry.core.vram.memory",
    "move_direct_tensors_to_device": "worldfoundry.core.vram.layers",
    "offload_model_from_device_for_memory_preservation": "worldfoundry.core.vram.memory",
    "patched_diffusers_current_device": "worldfoundry.core.vram.memory",
    "skip_model_initialization": "worldfoundry.core.vram.initialization",
    "unload_complete_models": "worldfoundry.core.vram.memory",
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
