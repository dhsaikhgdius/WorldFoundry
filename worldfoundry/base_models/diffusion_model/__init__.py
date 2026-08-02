"""Canonical WorldFoundry diffusion inference infrastructure.

The package owns its sampling contracts and execution loop.  It deliberately
does not wrap DiffSynth, Diffusers pipelines, or model-specific upstream
runtimes.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AttentionBackend": ".optimizations",
    "BuildPurpose": ".components",
    "CheckpointSpec": ".loaders",
    "ComponentBuildContext": ".components",
    "ComponentKey": ".components",
    "ComponentKind": ".components",
    "ComponentSpec": ".components",
    "Conditioning": ".contracts",
    "DenoiserInput": ".contracts",
    "DenoiserOutput": ".contracts",
    "DiffusionOutput": ".contracts",
    "DiffusionRequest": ".contracts",
    "DiffusionRunContext": ".extensions",
    "DiffusionExtension": ".extensions",
    "DiffusionExecutor": ".runners",
    "ExecutionSpec": ".components",
    "ExecutionStrategyRegistry": ".runners",
    "LatentEncoder": ".contracts",
    "NativeDiffusionPipeline": ".pipeline",
    "NativeDiffusionRecipe": ".registry",
    "NativeDiffusionRegistry": ".registry",
    "NativeDiffusionRunner": ".runners",
    "OffloadMode": ".optimizations",
    "OffloadPolicy": ".optimizations",
    "QuantizationMode": ".optimizations",
    "QuantizationPolicy": ".optimizations",
    "RunnerComponents": ".runners",
    "RuntimePolicy": ".optimizations",
    "SamplingConfig": ".contracts",
    "SchedulerStep": ".contracts",
    "default_native_diffusion_registry": ".registry",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
