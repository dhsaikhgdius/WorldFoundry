"""Lazy public API for native Adversarial Diffusion Distillation."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ADDTrainableRoles": ".adapters",
    "FeatureLayout": ".adapters",
    "MultiScaleFeatureDiscriminator": ".adapters",
    "NativeADDDiscriminatorAdapter": ".adapters",
    "ProjectionFeatureHead": ".adapters",
    "audit_add_model_graph": ".adapters",
    "NativeADDTrainingStack": ".builder",
    "build_native_add_training_stack": ".builder",
    "ADDConfig": ".config",
    "ADDDistillationWeighting": ".config",
    "ADDNoiseSchedule": ".config",
    "ADDDecoderAdapter": ".contracts",
    "ADDDiscriminatorAdapter": ".contracts",
    "ADDDiscriminatorHeadOutput": ".contracts",
    "ADDDiscriminatorOutput": ".contracts",
    "ADDLossAdapter": ".contracts",
    "ADDLossResult": ".contracts",
    "ADDPredictionAdapter": ".contracts",
    "ADDTrainingBatch": ".contracts",
    "ADD_ENGINE_STATE_SCHEMA": ".engine",
    "ADDTrainResult": ".engine",
    "NativeADDTrainEngine": ".engine",
    "add_forward_noise": ".math",
    "discriminator_hinge_loss_per_sample": ".math",
    "distillation_weights": ".math",
    "feature_r1_penalty_per_sample": ".math",
    "generator_hinge_loss_per_sample": ".math",
    "pixel_distillation_loss_per_sample": ".math",
    "sample_student_timesteps": ".math",
    "sample_teacher_timesteps": ".math",
    "schedule_coefficients": ".math",
    "NativeADDLossAdapter": ".objective",
    "ADDRunSummary": ".session",
    "NativeADDTrainingSession": ".session",
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
