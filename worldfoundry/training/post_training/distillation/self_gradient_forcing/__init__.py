"""Lazy public API for native Self-Gradient-Forcing distillation."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "NativeSelfGradientForcingTrainingStack": ".builder",
    "build_native_self_gradient_forcing_training_stack": ".builder",
    "CacheTargetMode": ".config",
    "ExitStepRankMode": ".config",
    "SelfGradientForcingConfig": ".config",
    "shifted_flow_timestep": ".config",
    "SelfGradientForcingAdapter": ".contracts",
    "SelfGradientForcingReplay": ".contracts",
    "SELF_GRADIENT_FORCING_ENGINE_STATE_SCHEMA": ".engine",
    "NativeSelfGradientForcingTrainEngine": ".engine",
    "SELF_GRADIENT_FORCING_RNG_STATE_SCHEMA": ".rollout",
    "SelfGradientForcingSampler": ".rollout",
    "NativeSelfGradientForcingTrainingSession": ".session",
    "WanSelfGradientForcingAdapter": ".wan",
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
