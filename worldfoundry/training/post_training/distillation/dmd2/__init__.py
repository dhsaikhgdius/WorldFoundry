"""Lazy public API for native DMD2 distillation."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "NativeDMD2TrainingStack": ".builder",
    "build_native_dmd2_training_stack": ".builder",
    "DMD2GuidanceAdapter": ".contracts",
    "DMD2LossAdapter": ".contracts",
    "DMD2PredictionAdapter": ".contracts",
    "DMD2TrainingBatch": ".contracts",
    "DMD2_ENGINE_STATE_SCHEMA": ".engine",
    "DMD2TrainResult": ".engine",
    "NativeDMD2TrainEngine": ".engine",
    "dmd2_distribution_gradient": ".math",
    "dmd2_generator_adversarial_loss": ".math",
    "dmd2_guidance_adversarial_loss": ".math",
    "dmd2_proxy_loss_per_sample": ".math",
    "dmd2_weighted_total": ".math",
    "DMD2Config": ".objective",
    "DMD2FewStepPrediction": ".objective",
    "DMD2LossResult": ".objective",
    "NativeDMD2LossAdapter": ".objective",
    "dmd2_teacher_guidance": ".objective",
    "sample_dmd2_score_levels": ".objective",
    "simulate_dmd2_student": ".objective",
    "DMD2RunSummary": ".session",
    "NativeDMD2TrainingSession": ".session",
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
