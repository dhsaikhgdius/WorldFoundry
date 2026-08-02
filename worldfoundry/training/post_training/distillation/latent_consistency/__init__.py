"""Lazy public API for native latent consistency distillation."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "NativeLatentConsistencyTrainingStack": ".builder",
    "build_native_latent_consistency_training_stack": ".builder",
    "LatentConsistencyConfig": ".config",
    "LatentConsistencyDDIMSchedule": ".config",
    "LatentConsistencyLossType": ".config",
    "LatentConsistencyNoiseSchedule": ".config",
    "LatentConsistencyPredictionType": ".config",
    "build_latent_consistency_ddim_schedule": ".config",
    "LatentConsistencyPredictionAdapter": ".contracts",
    "LatentConsistencyRandomInputs": ".contracts",
    "LatentConsistencyTrainingBatch": ".contracts",
    "LATENT_CONSISTENCY_ENGINE_STATE_SCHEMA": ".engine",
    "LatentConsistencyTrainResult": ".engine",
    "NativeLatentConsistencyTrainEngine": ".engine",
    "LatentConsistencyLossResult": ".objective",
    "LatentConsistencyObjective": ".objective",
    "NativeLatentConsistencyTrainingSession": ".session",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
