"""DDRL transition replay and grouped policy optimization."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "DDRL_ADVANTAGE_EPSILON": ".objective",
    "DDRL_ENGINE_STATE_SCHEMA": ".engine",
    "DDRLAdvantages": ".objective",
    "DDRLDataRegularizerAdapter": ".contracts",
    "DDRLIterationResult": ".session",
    "DDRLLoss": ".objective",
    "DDRLReplayAdapter": ".contracts",
    "DDRLRewardAdapter": ".contracts",
    "DDRLRolloutAdapter": ".contracts",
    "DDRLRolloutBatch": ".contracts",
    "DDRLStepResult": ".engine",
    "DDRLTrajectory": ".contracts",
    "NativeDDRLEngine": ".engine",
    "NativeDDRLTrainingStack": ".builder",
    "NativeDDRLTrainingSession": ".session",
    "build_native_ddrl_training_stack": ".builder",
    "ddrl_group_advantages": ".objective",
    "ddrl_loss": ".objective",
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
