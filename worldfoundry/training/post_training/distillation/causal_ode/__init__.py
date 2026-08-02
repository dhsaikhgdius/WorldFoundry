"""Lazy public API for native Causal ODE trajectory regression."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "NativeCausalODETrainingStack": ".builder",
    "build_native_causal_ode_training_stack": ".builder",
    "CausalODEConfig": ".config",
    "warped_causal_ode_timesteps": ".config",
    "CausalODETrainingBatch": ".contracts",
    "CAUSAL_ODE_ENGINE_STATE_SCHEMA": ".engine",
    "CausalODETrainResult": ".engine",
    "NativeCausalODETrainEngine": ".engine",
    "CausalODELossResult": ".objective",
    "CausalODEObjective": ".objective",
    "PreparedCausalODEBatch": ".objective",
    "NativeCausalODETrainingSession": ".session",
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
