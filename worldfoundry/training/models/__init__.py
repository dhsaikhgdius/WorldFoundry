"""Lazy public facade for model-family training adapters."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ANYFLOW_BIDIRECTIONAL_WAN_SMALL_CHECKPOINT": ".anyflow",
    "ANYFLOW_FAR_WAN_SMALL_CHECKPOINT": ".anyflow",
    "AnyFlowArchitecture": ".anyflow",
    "AnyFlowCheckpoint": ".anyflow",
    "NativeAnyFlowModelMaterializer": ".anyflow",
    "SELF_FORCING_ODE_CHECKPOINT": ".causal_wan",
    "CausalWanTrainRole": ".causal_wan",
    "causal_wan_1p3b_config": ".causal_wan",
    "convert_self_forcing_causal_state_dict": ".causal_wan",
    "load_causal_wan_1p3b": ".causal_wan",
    "validate_causal_wan_dtype": ".causal_wan",
    "SANA_600M_512_TRAIN_FLOW_SHIFT": ".sana",
    "SANA_DEFAULT_TRAIN_TIMESTEPS": ".sana",
    "SanaTrainAdapter": ".sana",
    "build_cached_sana_train_adapter": ".sana",
    "build_sana_train_adapter": ".sana",
    "DiffusersSanaConditioner": ".sana_sid",
    "DiffusersSanaDenoiser": ".sana_sid",
    "SanaSIDPredictionAdapter": ".sana_sid",
    "build_local_diffusers_sana_sid_adapter": ".sana_sid",
    "WAN_DEFAULT_CONTEXT_FEATURES": ".wan",
    "WAN_DEFAULT_TEXT_LENGTH": ".wan",
    "WAN_DEFAULT_TRAIN_TIMESTEPS": ".wan",
    "WanTrainAdapter": ".wan",
    "build_cached_wan_train_adapter": ".wan",
    "build_wan_train_adapter": ".wan",
    "wan_pixel_mask_to_latent": ".wan",
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
