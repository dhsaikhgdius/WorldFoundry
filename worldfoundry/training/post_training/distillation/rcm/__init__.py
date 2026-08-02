"""Lazy public API for native rCM and Causal-rCM training."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "NativeRFRCMPredictionAdapter": ".adapters",
    "NativeRCMTrainingStack": ".builder",
    "build_native_causal_rcm_training_stack": ".builder",
    "build_native_rcm_training_stack": ".builder",
    "causal_rcm_config_from_algorithm": ".builder",
    "rcm_config_from_algorithm": ".builder",
    "CausalExactJVPAdapter": ".causal",
    "CausalRCMConfig": ".causal",
    "CausalRolloutRequest": ".causal",
    "CausalSelfForcingAdapter": ".causal",
    "CausalTeacherForcingAdapter": ".causal",
    "NativeCausalRCMLossAdapter": ".causal",
    "RFScoreAdapter": ".causal",
    "causal_block_pattern": ".causal",
    "CausalBlockModelAdapter": ".causal_rollout",
    "NativeCausalSelfForcingRollout": ".causal_rollout",
    "RCMConfig": ".config",
    "RCMExactJVPAdapter": ".contracts",
    "RCMLossAdapter": ".contracts",
    "RCMLossResult": ".contracts",
    "RCMPrediction": ".contracts",
    "RCMPredictionAdapter": ".contracts",
    "RCMTrainingBatch": ".contracts",
    "RCM_ENGINE_STATE_SCHEMA": ".engine",
    "NativeRCMTrainEngine": ".engine",
    "RCMTrainResult": ".engine",
    "NativeRCMLossAdapter": ".objective",
    "NativeRCMTrainingSession": ".session",
    "RCMRunSummary": ".session",
    "ProcessGroupRCMTensorSynchronizer": ".synchronization",
    "RCMTensorSynchronizer": ".synchronization",
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
