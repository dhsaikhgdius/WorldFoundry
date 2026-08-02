"""Lazy public API for native Data-Forcing Distillation."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "NativeDFDTrainingStack": ".builder",
    "build_native_dfd_training_stack": ".builder",
    "DFDConfig": ".config",
    "DFDDiscriminatorAdapter": ".contracts",
    "DFDFakeScoreAdapter": ".contracts",
    "DFDLossAdapter": ".contracts",
    "DFDPredictionAdapter": ".contracts",
    "DFDTrainingBatch": ".contracts",
    "DFD_ENGINE_STATE_SCHEMA": ".engine",
    "DFDTrainResult": ".engine",
    "NativeDFDTrainEngine": ".engine",
    "data_forcing_teacher_data": ".math",
    "dfd_distribution_gradient": ".math",
    "dfd_proxy_loss_per_sample": ".math",
    "shifted_uniform_timesteps": ".math",
    "DFDLossResult": ".objective",
    "DFDStudentPrediction": ".objective",
    "NativeDFDLossAdapter": ".objective",
    "dfd_teacher_guidance": ".objective",
    "prepare_dfd_student_prediction": ".objective",
    "sample_dfd_score_timesteps": ".objective",
    "sample_dfd_student_timesteps": ".objective",
    "DFDRunSummary": ".session",
    "NativeDFDTrainingSession": ".session",
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
