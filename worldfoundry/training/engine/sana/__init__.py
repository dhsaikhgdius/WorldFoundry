"""SANA model-family training execution."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "build_sana_fsdp2_session": ".sft",
    "build_sana_single_device_session": ".sft",
    "materialize_sana_cached_training_session": ".sft",
    "SanaSCMLADDTrainingRun": ".scm_ladd_run",
    "materialize_sana_scm_ladd_training_run": ".scm_ladd",
    "SanaSIDTrainingRun": ".sid_run",
    "SANA_SID_RUN_SCHEMA": ".sid_run",
    "materialize_sana_sid_training_run": ".sid",
    "SanaSIDDataLoader": ".sid_data",
    "collate_sana_sid_prompts": ".sid_data",
    "prepare_sana_sid_batch": ".sid_data",
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
