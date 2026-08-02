"""Native supervised-training session lifecycles."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "FSDP2TrainingSession": ".fsdp2",
    "SingleDeviceTrainingSession": ".single_device",
    "OverfitGateError": ".statistics",
    "SingleDeviceRunSummary": ".statistics",
    "TRAINING_METRIC_SCHEMA": ".io",
    "TRAINING_RUN_SCHEMA": ".io",
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
