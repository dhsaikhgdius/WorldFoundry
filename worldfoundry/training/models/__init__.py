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
    "WAN22_A14B_BOUNDARY_RATIO": ".wan22",
    "WAN22_A14B_LATENT_CHANNELS": ".wan22",
    "WAN22_DUAL_ATTENTION": ".wan22",
    "Wan22DualExpertModule": ".wan22",
    "Wan22TrainAdapter": ".wan22",
    "build_wan22_train_adapter": ".wan22",
    "HUNYUAN_VIDEO_MODEL_RECIPES": ".hunyuan_video",
    "HUNYUAN_VIDEO_TIMESTEP_SCALE": ".hunyuan_video",
    "HunyuanVideoModelContract": ".hunyuan_video",
    "HunyuanVideoTrainAdapter": ".hunyuan_video",
    "build_cached_hunyuan_video_train_adapter": ".hunyuan_video",
    "build_hunyuan_video_train_adapter": ".hunyuan_video",
    "hunyuan_video_model_contract": ".hunyuan_video",
    "COSMOS_DEFAULT_TRAIN_TIMESTEPS": ".cosmos",
    "Cosmos3TrainAdapter": ".cosmos",
    "CosmosPredict2TrainAdapter": ".cosmos",
    "CosmosPredict25TrainAdapter": ".cosmos",
    "build_cached_cosmos3_train_adapter": ".cosmos",
    "build_cached_cosmos_predict2_train_adapter": ".cosmos",
    "build_cached_cosmos_predict25_train_adapter": ".cosmos",
    "build_cosmos_predict2_train_adapter": ".cosmos",
    "build_cosmos_predict25_train_adapter": ".cosmos",
    "DynamiCrafterTrainAdapter": ".dynamicrafter",
    "DynamiCrafterTrainableGraph": ".dynamicrafter",
    "dynamicrafter_objective": ".dynamicrafter",
    "FramewiseLVDMCodec": ".lvdm",
    "LVDMUnconditionalTrainAdapter": ".lvdm",
    "LTX_DEFAULT_FPS": ".ltx",
    "LTX_DEFAULT_LATENT_CHANNELS": ".ltx",
    "LTX_DEFAULT_SPATIAL_COMPRESSION": ".ltx",
    "LTX_DEFAULT_TEMPORAL_COMPRESSION": ".ltx",
    "LTXTrainAdapter": ".ltx",
    "build_cached_ltx_train_adapter": ".ltx",
    "build_ltx_train_adapter": ".ltx",
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
