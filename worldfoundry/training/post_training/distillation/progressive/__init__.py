"""Native progressive DDIM distillation."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "NativeProgressiveDistillationTrainingStack": ".builder",
    "build_native_progressive_distillation_training_stack": ".builder",
    "ProgressiveDistillationConfig": ".config",
    "ProgressiveLearningRateAnneal": ".config",
    "ProgressiveLossWeight": ".config",
    "ProgressivePredictionType": ".config",
    "ProgressiveDistillationBatch": ".contracts",
    "ProgressivePredictionAdapter": ".contracts",
    "ProgressiveRandomInputs": ".contracts",
    "NativeProgressiveDistillationTrainEngine": ".engine",
    "PROGRESSIVE_DISTILLATION_ENGINE_STATE_SCHEMA": ".engine",
    "ProgressiveDistillationTrainResult": ".engine",
    "ProgressiveDistillationLossResult": ".objective",
    "ProgressiveDistillationObjective": ".objective",
    "NativeProgressiveDistillationTrainingSession": ".session",
    "ProgressiveDistillationRunSummary": ".session",
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
