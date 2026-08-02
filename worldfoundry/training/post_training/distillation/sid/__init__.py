"""Lazy public API for native Score Identity Distillation."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "NativeSIDTrainingStack": ".builder",
    "build_native_sid_training_stack": ".builder",
    "SIDDiscriminatorAdapter": ".contracts",
    "SIDLossAdapter": ".contracts",
    "SIDPredictionAdapter": ".contracts",
    "SIDTrainingBatch": ".contracts",
    "NativeSIDTrainEngine": ".engine",
    "SID_ENGINE_STATE_SCHEMA": ".engine",
    "SIDTrainResult": ".engine",
    "sid_classifier_free_guidance": ".math",
    "sid_fake_score_adversarial_loss_per_sample": ".math",
    "sid_fake_score_flow_loss_per_sample": ".math",
    "sid_generator_adversarial_loss_per_sample": ".math",
    "sid_generator_loss_per_sample": ".math",
    "sid_score_weight": ".math",
    "NativeSIDLossAdapter": ".objective",
    "SIDConfig": ".objective",
    "SIDFewStepPrediction": ".objective",
    "SIDLossResult": ".objective",
    "sample_sid_score_sigmas": ".objective",
    "simulate_sid_student": ".objective",
    "NativeSIDTrainingSession": ".session",
    "SIDRunSummary": ".session",
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
