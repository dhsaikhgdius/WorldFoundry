"""Native DiffusionOPD public surface."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "BranchClassifierFreeGuidance": ".adapters",
    "DIFFUSION_OPD_DATA_LOADER_STATE_SCHEMA": ".batching",
    "NativeDiffusionOPDDataLoader": ".batching",
    "NativeDiffusionOPDTrainingStack": ".builder",
    "build_native_diffusion_opd_training_stack": ".builder",
    "DiffusionOPDReplayResult": ".contracts",
    "DiffusionOPDRolloutBatch": ".contracts",
    "DiffusionOPDTrajectory": ".contracts",
    "DIFFUSION_OPD_ENGINE_STATE_SCHEMA": ".engine",
    "DiffusionOPDTrainResult": ".engine",
    "NativeDiffusionOPDEngine": ".engine",
    "DiffusionOPDLoss": ".objective",
    "diffusion_opd_loss": ".objective",
    "DiffusionOPDRunSummary": ".run",
    "NativeDiffusionOPDTrainingRun": ".run",
    "build_native_diffusion_opd_training_run": ".run",
    "DiffusionOPDIterationResult": ".session",
    "NativeDiffusionOPDTrainingSession": ".session",
    "DiffusionOPDTrajectorySampler": ".trajectory",
    "NativeDiffusionOPDTrajectoryReplay": ".trajectory",
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
