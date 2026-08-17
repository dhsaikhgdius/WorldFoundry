"""Lazy public facade for native Cosmos training execution."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "COSMOS_PREDICT25_DMD2_FEATURE_IDS": ".dmd2",
    "COSMOS_PREDICT25_DMD2_PRETRAINED_REVISION": ".dmd2",
    "COSMOS_PREDICT25_DMD2_PRETRAINED_WEIGHT_FILE": ".dmd2",
    "COSMOS_PREDICT25_DMD2_REPO_ID": ".dmd2",
    "COSMOS_PREDICT25_DMD2_RUN_SCHEMA": ".dmd2",
    "CosmosPredict25DMD2TrainingRun": ".dmd2",
    "cosmos_predict25_dmd2_lr_multiplier": ".dmd2",
    "cosmos_predict25_dmd2_pretrained_checkpoint": ".dmd2",
    "materialize_cosmos_predict25_dmd2_training_run": ".dmd2",
    "validate_cosmos_predict25_dmd2_cache": ".dmd2",
    "COSMOS3_DMD2_FLOW_SIGMAS": ".dmd2_roles",
    "COSMOS_DMD2_GENERATOR_UPDATE_INTERVAL": ".dmd2_roles",
    "COSMOS_PREDICT25_DMD2_FLOW_SIGMAS": ".dmd2_roles",
    "COSMOS_PREDICT25_DMD2_TRIGFLOW_TIMES": ".dmd2_roles",
    "Cosmos3VideoDMD2PredictionAdapter": ".dmd2_roles",
    "CosmosDMD2DiscriminatorHead": ".dmd2_roles",
    "CosmosDMD2GuidanceAdapter": ".dmd2_roles",
    "CosmosFlowDMD2PredictionAdapter": ".dmd2_roles",
    "cosmos3_dmd2_schedule": ".dmd2_roles",
    "cosmos_predict25_dmd2_schedule": ".dmd2_roles",
    "trigflow_time_to_flow_sigma": ".dmd2_roles",
    "COSMOS_PREDICT_LORA_CONDITIONAL_FRAME_PROBABILITIES": ".objective",
    "COSMOS_PREDICT_LOSS_SCALE": ".objective",
    "COSMOS3_NANO_CFG_DROPOUT": ".objective",
    "COSMOS3_NANO_CONDITIONING_CONFIG": ".objective",
    "Cosmos3VisionFlowMatchingObjective": ".objective",
    "CosmosPredictFlowMatchingObjective": ".objective",
    "COSMOS3_LORA_PRESET": ".sft",
    "COSMOS3_NANO_VISION_SFT_PRESET": ".sft",
    "COSMOS_PREDICT_LORA_PRESET": ".sft",
    "apply_cosmos_tuning": ".sft",
    "audit_cosmos_lora_targets": ".sft",
    "build_cosmos_fsdp2_session": ".sft",
    "build_cosmos_single_device_session": ".sft",
    "cosmos_training_checkpoint_overrides": ".sft",
    "materialize_cosmos_cached_training_session": ".sft",
    "validate_cosmos_cache_contract": ".sft",
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
