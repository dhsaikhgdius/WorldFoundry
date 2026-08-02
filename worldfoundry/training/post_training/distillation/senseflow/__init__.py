"""Lazy public API for native SenseFlow distillation."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "LinearWarmupConstantScheduler": ".builder",
    "NativeSenseFlowTrainingStack": ".builder",
    "build_native_senseflow_training_stack": ".builder",
    "ISGLoss": ".config",
    "SchedulerCadence": ".config",
    "ScoreSampling": ".config",
    "SenseFlowConfig": ".config",
    "SenseFlowOptimizerConfig": ".config",
    "SenseFlowSchedule": ".config",
    "SenseFlowDiscriminatorAdapter": ".contracts",
    "SenseFlowFakeScoreAdapter": ".contracts",
    "SenseFlowGeneratorPhase": ".contracts",
    "SenseFlowLossAdapter": ".contracts",
    "SenseFlowLossResult": ".contracts",
    "SenseFlowPredictionAdapter": ".contracts",
    "SenseFlowPreparedBatch": ".contracts",
    "SenseFlowTeacherAdapter": ".contracts",
    "SenseFlowTrainingBatch": ".contracts",
    "SENSEFLOW_ENGINE_STATE_SCHEMA": ".engine",
    "NativeSenseFlowTrainEngine": ".engine",
    "SenseFlowTrainResult": ".engine",
    "IDAUpdate": ".math",
    "ISGPaths": ".math",
    "audit_ida_alignment": ".math",
    "expand_levels": ".math",
    "flow_euler_step": ".math",
    "flow_isg_paths": ".math",
    "flow_velocity_from_clean": ".math",
    "implicit_distribution_alignment_": ".math",
    "isg_loss_per_sample": ".math",
    "sample_isg_midpoint": ".math",
    "sample_score_sigmas": ".math",
    "senseflow_adversarial_time_weight": ".math",
    "senseflow_discriminator_hinge_loss": ".math",
    "senseflow_distribution_gradient": ".math",
    "senseflow_generator_hinge_loss": ".math",
    "senseflow_proxy_loss_per_sample": ".math",
    "senseflow_sigma_at_timestep": ".math",
    "NativeSenseFlowLossAdapter": ".objective",
    "SenseFlowAnchorRollout": ".rollout",
    "simulate_senseflow_anchor": ".rollout",
    "NativeSenseFlowTrainingSession": ".session",
    "SenseFlowRunSummary": ".session",
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
