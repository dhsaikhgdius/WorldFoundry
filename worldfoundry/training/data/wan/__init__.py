"""Lazy public facade for Wan cache preparation."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "WAN_CONDITIONING_LAYOUT": ".contracts",
    "WAN_LATENT_MEAN": ".contracts",
    "WAN_LATENT_STD": ".contracts",
    "WanCachePreparationResult": ".training_cache",
    "WanFeatureEncoder": ".encoding",
    "WanTextFeatureEncoder": ".encoding",
    "WanVideoFeatureEncoder": ".encoding",
    "build_wan_video_decoding_dataset": ".training_cache",
    "materialize_wan_rollout_conditioning_cache": ".rollout_cache",
    "materialize_wan_training_cache": ".training_cache",
    "prepare_wan_training_cache_from_audits": ".training_cache",
    "wan_cache_contract": ".contracts",
    "wan_checkpoint_asset_identity": ".contracts",
    "wan_latent_normalization": ".contracts",
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
