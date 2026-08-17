"""Lazy public API for WorldFoundry native training and post-training."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ObjectiveBatch": ".api",
    "PreparedBatch": ".api",
    "TrainingBatch": ".api",
    "TrainingObjective": ".api",
    "TrainModelAdapter": ".api",
    "TrainStepResult": ".api",
    "IncompleteTrainingCheckpointError": ".checkpoint",
    "TrainingCheckpointCompatibilityError": ".checkpoint",
    "TrainingCheckpointer": ".checkpoint",
    "TrainingProgress": ".checkpoint",
    "TrainingState": ".checkpoint",
    "DeterministicDistributedSampler": ".data",
    "TrainingManifestDataset": ".data",
    "TrainingManifestError": ".data",
    "TrainingSample": ".data",
    "build_stateful_dataloader": ".data",
    "inspect_training_manifest": ".data",
    "load_training_manifest": ".data",
    "PowerEMA": ".ema",
    "power_ema_beta": ".ema",
    "power_ema_exponent": ".ema",
    "DMDConfig": ".post_training",
    "DMDTrainingBatch": ".post_training",
    "FewStepSchedule": ".post_training",
    "FlowDMDLossAdapter": ".post_training",
    "FlowTrajectory": ".post_training",
    "FlowTrajectorySampler": ".post_training",
    "NativeDMDTrainEngine": ".post_training",
    "NativeDiffusionOPDEngine": ".post_training",
    "NativeDiffusionOPDTrainingRun": ".post_training",
    "build_native_diffusion_opd_training_run": ".post_training",
    "build_native_diffusion_opd_training_stack": ".post_training",
    "NativeAgenticTrainingRun": ".post_training",
    "NativeAgenticTrainingSession": ".post_training",
    "materialize_agentic_training_run": ".post_training",
    "materialize_qwen3_agentic_training_run": ".post_training",
    "materialize_qwen3_post_training_run": ".post_training",
    "materialize_qwen3_token_ppo_training_run": ".post_training",
    "setup_ray_agentic_rollout": ".post_training",
    "NativeDanceGRPOEngine": ".post_training",
    "NativeFlowDPPOEngine": ".post_training",
    "NativeFlowGRPOEngine": ".post_training",
    "NativeMixGRPOEngine": ".post_training",
    "NativeTokenPPOEngine": ".post_training",
    "NativeTokenPPOTrainingRun": ".post_training",
    "materialize_token_ppo_training_run": ".post_training",
    "NativeFlowPredictionAdapter": ".post_training",
    "NativeFlowTrajectoryReplay": ".post_training",
    "POST_TRAINING_RECIPE_SCHEMA": ".recipes",
    "TRAINING_RECIPE_SCHEMA": ".recipes",
    "PostTrainingRecipe": ".recipes",
    "TrainingRecipe": ".recipes",
    "PromptSafetyAudit": ".safety",
    "ShieldGemmaPromptFilter": ".safety",
    "UnsafeTrainingPromptError": ".safety",
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
