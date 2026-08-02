"""Lazy public API for native Causal Consistency Distillation."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "NativeCausalConsistencyTrainingStack": ".builder",
    "build_native_causal_consistency_training_stack": ".builder",
    "CausalConsistencyConfig": ".config",
    "CausalConsistencySchedule": ".config",
    "build_causal_consistency_schedule": ".config",
    "CausalConsistencyTrainingBatch": ".contracts",
    "FrozenModuleEMA": ".ema",
    "CAUSAL_CONSISTENCY_ENGINE_STATE_SCHEMA": ".engine",
    "CausalConsistencyTrainResult": ".engine",
    "NativeCausalConsistencyTrainEngine": ".engine",
    "CausalConsistencyLossResult": ".objective",
    "CausalConsistencyObjective": ".objective",
    "NativeCausalConsistencyTrainingSession": ".session",
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
