"""Public facade for WorldFoundry-native training execution engines."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "AnyFlowCachedDataLoader": ".anyflow.data",
    "AnyFlowRoleBundle": ".anyflow.roles",
    "AnyFlowTrainingRun": ".anyflow.run",
    "materialize_anyflow_training_run": ".anyflow.materialize",
    "build_cosmos_fsdp2_session": ".cosmos.sft",
    "build_cosmos_single_device_session": ".cosmos.sft",
    "materialize_cosmos_cached_training_session": ".cosmos.sft",
    "VideoFlowPolicyMaterialization": ".video_policy",
    "materialize_video_flow_policy_roles": ".video_policy",
    "materialize_video_flow_policy_training_run": ".video_policy",
    "HunyuanVideoDiffusionNFTRuntime": ".hunyuan_video",
    "HunyuanVideoFlowPolicyMaterialization": ".hunyuan_video",
    "HunyuanVideoRLDataPlan": ".hunyuan_video",
    "HunyuanVideoRLProfile": ".hunyuan_video",
    "audit_hunyuan_video_lora_targets": ".hunyuan_video",
    "build_hunyuan_video_diffusion_nft_stack": ".hunyuan_video",
    "build_hunyuan_video_flow_policy_materialization": ".hunyuan_video",
    "hunyuan_video_rl_profile": ".hunyuan_video",
    "load_hunyuan_video_role_adapter": ".hunyuan_video",
    "materialize_hunyuan_video_diffusion_nft": ".hunyuan_video",
    "materialize_hunyuan_video_flow_policy": ".hunyuan_video",
    "validate_hunyuan_video_diffusion_nft_recipe": ".hunyuan_video",
    "validate_hunyuan_video_flow_policy_recipe": ".hunyuan_video",
    "FSDP2_ENGINE_STATE_SCHEMA": ".fsdp",
    "FSDP2TrainEngine": ".fsdp",
    "build_sana_fsdp2_session": ".sana.sft",
    "build_sana_single_device_session": ".sana.sft",
    "materialize_sana_cached_training_session": ".sana.sft",
    "SanaSIDTrainingRun": ".sana.sid_run",
    "SANA_SID_RUN_SCHEMA": ".sana.sid_run",
    "materialize_sana_sid_training_run": ".sana.sid",
    "TRAINING_METRIC_SCHEMA": ".sessions.io",
    "TRAINING_RUN_SCHEMA": ".sessions.io",
    "FSDP2TrainingSession": ".sessions.fsdp2",
    "OverfitGateError": ".sessions.statistics",
    "SingleDeviceRunSummary": ".sessions.statistics",
    "SingleDeviceTrainingSession": ".sessions.single_device",
    "SINGLE_DEVICE_ENGINE_STATE_SCHEMA": ".single_device",
    "SingleDeviceTrainEngine": ".single_device",
    "build_adamw": ".single_device",
    "trainable_parameters": ".single_device",
    "build_ltx_fsdp2_session": ".ltx.sft",
    "build_ltx_single_device_session": ".ltx.sft",
    "materialize_ltx_cached_training_session": ".ltx.sft",
    "LTXDiffusionNFTRuntime": ".ltx.flow_policy",
    "LTXFlowPolicyRuntime": ".ltx.flow_policy",
    "LTXFlowPredictionAdapter": ".ltx.flow_policy",
    "ltx_flow_policy_profile": ".ltx.flow_policy",
    "ltx_flow_policy_sigmas": ".ltx.flow_policy",
    "materialize_ltx_diffusion_nft_stack": ".ltx.flow_policy",
    "materialize_ltx_flow_policy_stack": ".ltx.flow_policy",
    "validate_ltx_diffusion_nft_recipe": ".ltx.flow_policy",
    "validate_ltx_flow_policy_recipe": ".ltx.flow_policy",
    "build_wan_fsdp2_session": ".wan.sft",
    "build_wan_single_device_session": ".wan.sft",
    "materialize_wan_cached_training_session": ".wan.sft",
    "DMDTrainableRoles": ".wan.dmd",
    "WAN_DMD_RUN_SCHEMA": ".wan.dmd",
    "WanDMDRoleBundle": ".wan.dmd",
    "WanDMDTrainingRun": ".wan.dmd",
    "materialize_wan_dmd_training_run": ".wan.dmd",
    "WAN_SELF_FORCING_RUN_SCHEMA": ".wan.self_forcing",
    "WanSelfForcingRoleBundle": ".wan.self_forcing",
    "WanSelfForcingTrainingRun": ".wan.self_forcing",
    "materialize_wan_self_forcing_training_run": ".wan.self_forcing",
    "validate_wan_self_forcing_recipe": ".wan.self_forcing_recipe",
    "WAN_DIFFUSION_NFT_RUN_SCHEMA": ".wan.diffusion_nft",
    "WanDiffusionNFTRoleBundle": ".wan.diffusion_nft",
    "WanDiffusionNFTRunSummary": ".wan.diffusion_nft",
    "WanDiffusionNFTTrainingRun": ".wan.diffusion_nft",
    "materialize_wan_diffusion_nft_training_run": ".wan.diffusion_nft",
    "validate_wan_diffusion_nft_recipe": ".wan.diffusion_nft",
    "WAN_FLOW_POLICY_RUN_SCHEMA": ".wan.flow_policy",
    "WanFlowPolicyDataPlan": ".wan.flow_policy",
    "WanFlowPolicyRoleBundle": ".wan.flow_policy",
    "WanFlowPolicyRunSummary": ".wan.flow_policy",
    "WanFlowPolicyTrainingRun": ".wan.flow_policy",
    "materialize_wan_flow_policy_training_run": ".wan.flow_policy",
    "validate_wan_flow_policy_recipe": ".wan.flow_policy",
    "WAN22_T2V_A14B_MODEL": ".wan22",
    "WAN22_T2V_A14B_REPOSITORY": ".wan22",
    "Wan22DiffusionNFTRuntime": ".wan22",
    "Wan22FlowPolicyDataPlan": ".wan22",
    "Wan22FlowPolicyProfile": ".wan22",
    "Wan22FlowPolicyRuntime": ".wan22",
    "Wan22RoleCheckpoints": ".wan22",
    "load_wan22_role_adapter": ".wan22",
    "materialize_wan22_diffusion_nft_stack": ".wan22",
    "materialize_wan22_flow_policy_stack": ".wan22",
    "validate_wan22_diffusion_nft_recipe": ".wan22",
    "validate_wan22_flow_policy_recipe": ".wan22",
    "wan22_flow_policy_profile": ".wan22",
    "wan22_flow_policy_sigmas": ".wan22",
    "wan22_role_checkpoints": ".wan22",
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
