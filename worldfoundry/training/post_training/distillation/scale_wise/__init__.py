"""Scale-wise few-step distillation math and native execution."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "NativeScaleWiseTrainingStack": ".builder",
    "build_native_scale_wise_training_stack": ".builder",
    "ScaleWiseConfig": ".config",
    "ScaleWiseSchedule": ".config",
    "flow_match_solver_sigmas": ".config",
    "ScaleWiseCriticAdapter": ".contracts",
    "ScaleWiseLossAdapter": ".contracts",
    "ScaleWisePredictionAdapter": ".contracts",
    "ScaleWiseTrainingBatch": ".contracts",
    "NativeScaleWiseTrainEngine": ".engine",
    "SCALE_WISE_ENGINE_STATE_SCHEMA": ".engine",
    "ScaleWiseTrainResult": ".engine",
    "classifier_free_guidance": ".math",
    "clean_from_velocity": ".math",
    "discriminator_logistic_loss": ".math",
    "dmd_loss_per_sample": ".math",
    "fake_diffusion_loss_per_sample": ".math",
    "flow_noise": ".math",
    "generator_logistic_loss": ".math",
    "mmd_loss": ".math",
    "pool_token_features": ".math",
    "upscale_previous_latents": ".math",
    "FlowScaleWiseLossAdapter": ".objective",
    "ScaleWiseLossResult": ".objective",
    "ScaleWiseStudentSample": ".objective",
    "NativeScaleWiseTrainingSession": ".session",
    "ScaleWiseRunSummary": ".session",
    "SD3AdapterDisabledTeacherAdapter": ".sd3",
    "SD3ScaleWiseCriticAdapter": ".sd3",
    "SD3ScaleWiseCriticModule": ".sd3",
    "SD3ScaleWisePredictionAdapter": ".sd3",
    "sd3_velocity_and_features": ".sd3",
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
