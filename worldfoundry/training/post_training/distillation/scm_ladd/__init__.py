"""Continuous-time consistency with latent adversarial distillation."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "NativeSCMLADDTrainingStack": ".builder",
    "build_native_scm_ladd_training_stack": ".builder",
    "SCMLADDDiscriminatorAdapter": ".contracts",
    "SCMLADDLossAdapter": ".contracts",
    "SCMLADDLossResult": ".contracts",
    "SCMLADDTrainingBatch": ".contracts",
    "SCMVelocityPrediction": ".contracts",
    "TrigFlowPredictionAdapter": ".contracts",
    "SCM_LADD_ENGINE_STATE_SCHEMA": ".engine",
    "NativeSCMLADDTrainEngine": ".engine",
    "SCMLADDTrainResult": ".engine",
    "NativeSCMLADDLossAdapter": ".objective",
    "classifier_free_velocity": ".math",
    "flow_velocity_to_trigflow": ".math",
    "ladd_discriminator_hinge_loss": ".math",
    "ladd_generator_hinge_loss": ".math",
    "sample_trigflow_timesteps": ".math",
    "scm_adaptive_loss": ".math",
    "scm_tangent_target": ".math",
    "trigflow_clean_prediction": ".math",
    "trigflow_interpolate": ".math",
    "trigflow_to_flow_input": ".math",
    "trigflow_to_flow_time": ".math",
    "NativeSCMLADDTrainingSession": ".session",
    "SCMLADDRunSummary": ".session",
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
