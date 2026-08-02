"""Lazy public API for Bagel Flow-UniGRPO."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "BAGEL_FLOW_UNIGRPO_ENGINE_STATE_SCHEMA": ".engine",
    "BagelFlowUniGRPOIterationResult": ".session",
    "BagelFlowUniGRPOLoss": ".objective",
    "BagelFlowUniGRPOStageAlgorithm": ".algorithm",
    "BagelFlowUniGRPOStepResult": ".engine",
    "NativeBagelFlowUniGRPOEngine": ".engine",
    "NativeBagelFlowUniGRPOTrainingSession": ".session",
    "bagel_flow_unigrpo_loss": ".objective",
    "build_native_bagel_flow_unigrpo_engine": ".engine",
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
