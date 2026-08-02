"""Native diagonal distillation schedules, rollout, and compound DMD."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "NativeDiagonalTrainingStack": ".builder",
    "build_native_diagonal_training_stack": ".builder",
    "load_diagonal_ode_initialization": ".checkpoint",
    "load_diagonal_stage_weights": ".checkpoint",
    "DiagonalObjectiveConfig": ".config",
    "DiagonalScheduleConfig": ".config",
    "ExitStepMode": ".config",
    "RegressionLossType": ".config",
    "build_block_denoising_steps": ".config",
    "DiagonalCausalAdapter": ".contracts",
    "DiagonalFewStepPrediction": ".contracts",
    "DiagonalRollout": ".contracts",
    "DIAGONAL_ENGINE_STATE_SCHEMA": ".engine",
    "NativeDiagonalTrainEngine": ".engine",
    "DiagonalDistributionGradients": ".math",
    "DiagonalProxyLosses": ".math",
    "diagonal_distribution_gradients": ".math",
    "diagonal_flow_regression_loss": ".math",
    "diagonal_proxy_losses": ".math",
    "diagonal_regression_loss": ".math",
    "dynamic_motion_weights": ".math",
    "exponential_motion_weights": ".math",
    "hybrid_motion_weights": ".math",
    "SpatialMotionHead": ".motion",
    "register_motion_head": ".motion",
    "DIAGONAL_OBJECTIVE_STATE_SCHEMA": ".objective",
    "DiagonalDMDLossAdapter": ".objective",
    "DiagonalDMDLossResult": ".objective",
    "DIAGONAL_SAMPLER_STATE_SCHEMA": ".rollout",
    "DiagonalFixedTeacherSampler": ".rollout",
    "DiagonalRolloutSampler": ".rollout",
    "NativeDiagonalTrainingSession": ".session",
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
