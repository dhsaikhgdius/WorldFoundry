"""Lazy public API for full-model and parameter-efficient tuning."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "AdapterApplication": ".application",
    "AdapterArtifact": ".application",
    "ExportedAdapterArtifact": ".application",
    "DEFAULT_MAX_SHARD_SIZE_BYTES": ".full_model",
    "FULL_MODEL_ARTIFACT_SCHEMA": ".full_model",
    "FULL_MODEL_INDEX_NAME": ".full_model",
    "FULL_MODEL_MANIFEST_NAME": ".full_model",
    "FullModelArtifact": ".full_model",
    "inspect_full_model": ".full_model",
    "load_full_model": ".full_model",
    "save_full_model": ".full_model",
    "SANA_ATTENTION": ".peft",
    "WAN_ATTENTION": ".peft",
    "LoraTargetAudit": ".peft",
    "PeftAdapterArtifact": ".peft",
    "PeftLoraApplication": ".peft",
    "apply_peft_lora": ".peft",
    "apply_peft_lora_with_audit": ".peft",
    "apply_peft_lora_to_adapter": ".peft",
    "audit_lora_targets": ".peft",
    "inspect_peft_adapter": ".peft",
    "load_peft_adapter": ".peft",
    "merge_peft_adapter": ".peft",
    "save_peft_adapter": ".peft",
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
