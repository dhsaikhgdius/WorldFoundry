"""Paired forward-process Diffusion-DPO training."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "DIFFUSION_DPO_ENGINE_STATE_SCHEMA": ".engine",
    "DiffusionDPOBatch": ".contracts",
    "DiffusionDPOForwardSample": ".objective",
    "DiffusionDPOLoss": ".objective",
    "DiffusionDPORunSummary": ".session",
    "DiffusionDPOStepResult": ".engine",
    "NativeDiffusionDPOEngine": ".engine",
    "NativeDiffusionDPOTrainingStack": ".builder",
    "NativeDiffusionDPOTrainingSession": ".session",
    "build_native_diffusion_dpo_training_stack": ".builder",
    "diffusion_dpo_forward_process": ".objective",
    "diffusion_dpo_loss": ".objective",
    "sample_diffusion_dpo_forward_process": ".objective",
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
