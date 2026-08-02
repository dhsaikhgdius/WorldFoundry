"""Native causal Self-Forcing rollout and holistic DMD training."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "NativeSelfForcingDataLoader": ".batching",
    "NativeSelfForcingTrainingStack": ".builder",
    "build_native_self_forcing_training_stack": ".builder",
    "SelfForcingConfig": ".config",
    "shifted_few_step_schedule": ".config",
    "CausalChunkAdapter": ".contracts",
    "SelfForcingRollout": ".contracts",
    "DelayedSelfForcingEMA": ".ema",
    "SelfForcingRolloutSampler": ".rollout",
    "NativeSelfForcingTrainingSession": ".session",
    "WanSelfForcingChunkAdapter": ".wan",
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
