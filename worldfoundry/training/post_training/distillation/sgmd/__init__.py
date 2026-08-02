"""Lazy public API for native SGMD."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "NativeSGMDTrainingStack": ".builder",
    "build_native_sgmd_training_stack": ".builder",
    "SGMDConfig": ".config",
    "shifted_flow_sigma": ".config",
    "SGMDLossAdapter": ".contracts",
    "SGMDPredictionAdapter": ".contracts",
    "SGMDTrainingBatch": ".contracts",
    "SGMD_ENGINE_STATE_SCHEMA": ".engine",
    "NativeSGMDTrainEngine": ".engine",
    "SGMDTrainResult": ".engine",
    "expand_sigmas": ".math",
    "sgmd_classifier_free_guidance": ".math",
    "sgmd_diversity_loss_per_sample": ".math",
    "sgmd_euler_step": ".math",
    "sgmd_fake_clean_diagnostic_per_sample": ".math",
    "sgmd_fake_correction_loss_per_sample": ".math",
    "sgmd_fake_score_flow_loss_per_sample": ".math",
    "sgmd_normalized_fisher_loss_per_sample": ".math",
    "NativeSGMDLossAdapter": ".objective",
    "SGMDLossResult": ".objective",
    "SGMDRollout": ".objective",
    "sample_sgmd_score_sigmas": ".objective",
    "simulate_sgmd_student": ".objective",
    "NativeSGMDTrainingSession": ".session",
    "SGMDRunSummary": ".session",
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
