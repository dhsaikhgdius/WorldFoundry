"""Distribution Matching Distillation math and native runtime."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "NativeDMDTrainingStack": ".builder",
    "build_native_dmd_training_stack": ".builder",
    "DMDLossAdapter": ".contracts",
    "DMDTrainingBatch": ".contracts",
    "DMD_ENGINE_STATE_SCHEMA": ".engine",
    "DMDTrainResult": ".engine",
    "NativeDMDTrainEngine": ".engine",
    "DMDConfig": ".objective",
    "DMDLossResult": ".objective",
    "FewStepPrediction": ".objective",
    "FewStepSchedule": ".objective",
    "FlowDMDLossAdapter": ".objective",
    "dmd_distribution_gradient": ".objective",
    "dmd_proxy_loss": ".objective",
    "dmd_teacher_guidance": ".objective",
    "sample_dmd_score_sigmas": ".objective",
    "simulate_few_step_student": ".objective",
    "DMDRunSummary": ".session",
    "NativeDMDTrainingSession": ".session",
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
