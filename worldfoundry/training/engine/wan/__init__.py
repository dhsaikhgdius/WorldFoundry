"""Wan model-family training execution."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "build_wan_fsdp2_session": ".sft",
    "build_wan_single_device_session": ".sft",
    "materialize_wan_cached_training_session": ".sft",
    "DMDTrainableRoles": ".dmd",
    "WAN_DMD_RUN_SCHEMA": ".dmd",
    "WanDMDRoleBundle": ".dmd",
    "WanDMDTrainingRun": ".dmd",
    "materialize_wan_dmd_training_run": ".dmd",
    "WAN_SELF_FORCING_RUN_SCHEMA": ".self_forcing",
    "WanSelfForcingRoleBundle": ".self_forcing",
    "WanSelfForcingTrainingRun": ".self_forcing",
    "materialize_wan_self_forcing_training_run": ".self_forcing",
    "validate_wan_self_forcing_recipe": ".self_forcing_recipe",
    "WAN_DIFFUSION_NFT_RUN_SCHEMA": ".diffusion_nft",
    "WanDiffusionNFTRoleBundle": ".diffusion_nft",
    "WanDiffusionNFTRunSummary": ".diffusion_nft",
    "WanDiffusionNFTTrainingRun": ".diffusion_nft",
    "materialize_wan_diffusion_nft_training_run": ".diffusion_nft",
    "validate_wan_diffusion_nft_recipe": ".diffusion_nft",
    "WAN_FLOW_POLICY_RUN_SCHEMA": ".flow_policy",
    "WanFlowPolicyDataPlan": ".flow_policy",
    "WanFlowPolicyRoleBundle": ".flow_policy",
    "WanFlowPolicyRunSummary": ".flow_policy",
    "WanFlowPolicyTrainingRun": ".flow_policy",
    "materialize_wan_flow_policy_training_run": ".flow_policy",
    "validate_wan_flow_policy_recipe": ".flow_policy",
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
