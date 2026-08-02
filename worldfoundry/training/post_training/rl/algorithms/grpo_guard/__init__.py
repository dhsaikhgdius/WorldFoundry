"""Lazy public API for WorldFoundry-native GRPO-Guard."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "GRPO_GUARD_ENGINE_STATE_SCHEMA": ".engine",
    "GRPOGuardIterationResult": ".session",
    "GRPOGuardLoss": ".objective",
    "GRPOGuardStageAlgorithm": ".algorithm",
    "GRPOGuardStepResult": ".engine",
    "NativeGRPOGuardEngine": ".engine",
    "NativeGRPOGuardTrainingSession": ".session",
    "build_native_grpo_guard_engine": ".engine",
    "grpo_guard_policy_loss": ".objective",
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
