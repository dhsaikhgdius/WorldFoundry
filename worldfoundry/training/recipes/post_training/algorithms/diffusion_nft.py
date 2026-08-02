"""Pure recipe contracts for terminal-latent DiffusionNFT training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from ..common import (
    advantage_normalization_mode,
    frozen_float_mapping,
    mapping,
    positive_int,
    strict_mapping,
)
from ..rewards.videoalign import VIDEOALIGN_REWARD_FIELDS, VideoAlignRewardSpec

DIFFUSION_NFT_DECAY_SCHEDULES = frozenset({"copy", "linear_to_0_5", "delayed_linear_to_0_999"})
DIFFUSION_NFT_ADVANTAGE_MODES = frozenset({"all", "positive_only", "negative_only", "one_only", "binary"})

DIFFUSION_NFT_COLLECTION_FIELDS = {
    "sigmas",
    "group_size",
    "guidance_scale",
    "latent_dtype",
    "forward_batch_size",
}
DIFFUSION_NFT_OLD_POLICY_REFRESH_FIELDS = {"decay", "interval"}
DIFFUSION_NFT_ALGORITHM_FIELDS = {
    "type",
    "collection",
    "num_train_timesteps",
    "beta",
    "advantage_clip_max",
    "advantage_epsilon",
    "advantage_mode",
    "advantage_normalization",
    "reference_mse_weight",
    "reference_checkpoint",
    "reconstruction_mae_floor",
    "old_policy_refresh",
    "reward_weights",
    "reward_model",
}


@dataclass(frozen=True, slots=True)
class DiffusionNFTTerminalLatentCollectionSpec:
    """Deterministic flow collection ending at a clean terminal latent."""

    sigmas: tuple[float, ...]
    group_size: int = 2
    guidance_scale: float = 1.0
    latent_dtype: str = "bfloat16"
    forward_batch_size: int | None = None

    def __post_init__(self) -> None:
        sigmas = tuple(float(value) for value in self.sigmas)
        if (
            len(sigmas) < 2
            or any(not isfinite(value) or not 0 <= value <= 1 for value in sigmas)
            or any(left <= right for left, right in zip(sigmas, sigmas[1:]))
        ):
            raise ValueError("DiffusionNFT collection sigmas must be finite and strictly descending in [0,1]")
        if sigmas[0] != 1.0 or sigmas[-1] != 0.0:
            raise ValueError("DiffusionNFT collection sigmas must start at 1 and end at 0")
        group_size = positive_int(
            self.group_size,
            field_name="algorithm.collection.group_size",
        )
        if group_size < 2:
            raise ValueError("DiffusionNFT collection group_size must be at least two")
        guidance_scale = float(self.guidance_scale)
        if not isfinite(guidance_scale) or guidance_scale < 1:
            raise ValueError("DiffusionNFT collection guidance_scale must be finite and at least one")
        dtype = str(self.latent_dtype).lower().removeprefix("torch.")
        dtype = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(
            dtype,
            dtype,
        )
        if dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("DiffusionNFT collection latent_dtype must be bfloat16, float16, or float32")
        forward_batch_size = (
            None
            if self.forward_batch_size is None
            else positive_int(
                self.forward_batch_size,
                field_name="algorithm.collection.forward_batch_size",
            )
        )
        object.__setattr__(self, "sigmas", sigmas)
        object.__setattr__(self, "group_size", group_size)
        object.__setattr__(self, "guidance_scale", guidance_scale)
        object.__setattr__(self, "latent_dtype", dtype)
        object.__setattr__(self, "forward_batch_size", forward_batch_size)


@dataclass(frozen=True, slots=True)
class DiffusionNFTOldPolicyRefreshSpec:
    """Decay schedule and cadence for refreshing the collection policy."""

    decay: str = "copy"
    interval: int = 1

    def __post_init__(self) -> None:
        decay = str(self.decay).strip().lower().replace("-", "_")
        if decay not in DIFFUSION_NFT_DECAY_SCHEDULES:
            raise ValueError("DiffusionNFT old-policy decay must be copy, linear_to_0_5, or delayed_linear_to_0_999")
        interval = positive_int(
            self.interval,
            field_name="algorithm.old_policy_refresh.interval",
        )
        object.__setattr__(self, "decay", decay)
        object.__setattr__(self, "interval", interval)


@dataclass(frozen=True, slots=True)
class DiffusionNFTAlgorithmSpec:
    """Terminal collection, reward mapping, and forward-process update contract."""

    collection: DiffusionNFTTerminalLatentCollectionSpec
    reward_weights: Mapping[str, float]
    reward_model: VideoAlignRewardSpec
    num_train_timesteps: int = 1000
    beta: float = 1.0
    advantage_clip_max: float = 5.0
    advantage_epsilon: float = 1.0e-4
    advantage_mode: str = "all"
    advantage_normalization: str = "group-population-std"
    reference_mse_weight: float = 0.0
    reference_checkpoint: str | None = None
    reconstruction_mae_floor: float = 1.0e-5
    old_policy_refresh: DiffusionNFTOldPolicyRefreshSpec = DiffusionNFTOldPolicyRefreshSpec()
    type: str = "diffusion-nft"

    def __post_init__(self) -> None:
        resolved_type = str(self.type).strip().lower().replace("_", "-")
        if resolved_type != "diffusion-nft":
            raise ValueError("DiffusionNFT algorithm type must be 'diffusion-nft'")
        if not isinstance(self.collection, DiffusionNFTTerminalLatentCollectionSpec):
            raise TypeError("DiffusionNFT collection must be DiffusionNFTTerminalLatentCollectionSpec")
        if not isinstance(self.old_policy_refresh, DiffusionNFTOldPolicyRefreshSpec):
            raise TypeError("DiffusionNFT old_policy_refresh must be DiffusionNFTOldPolicyRefreshSpec")
        if not isinstance(self.reward_model, VideoAlignRewardSpec):
            raise TypeError("DiffusionNFT reward_model must be VideoAlignRewardSpec")
        num_train_timesteps = positive_int(
            self.num_train_timesteps,
            field_name="algorithm.num_train_timesteps",
        )
        if num_train_timesteps < 2:
            raise ValueError("DiffusionNFT num_train_timesteps must be at least two")
        reward_weights = frozen_float_mapping(
            self.reward_weights,
            field_name="reward_weights",
        )
        if set(reward_weights) != set(self.reward_model.reward_ids):
            raise ValueError("reward_weights must exactly match reward_model.reward_ids")
        for name, value, positive in (
            ("beta", self.beta, True),
            ("advantage_clip_max", self.advantage_clip_max, True),
            ("advantage_epsilon", self.advantage_epsilon, True),
            ("reference_mse_weight", self.reference_mse_weight, False),
            ("reconstruction_mae_floor", self.reconstruction_mae_floor, True),
        ):
            resolved = float(value)
            if not isfinite(resolved) or (resolved <= 0 if positive else resolved < 0):
                relation = "positive" if positive else "non-negative"
                raise ValueError(f"DiffusionNFT {name} must be finite and {relation}")
            object.__setattr__(self, name, resolved)
        if self.beta > 1:
            raise ValueError("DiffusionNFT beta must be at most one")
        advantage_mode = str(self.advantage_mode).strip().lower().replace("-", "_")
        if advantage_mode not in DIFFUSION_NFT_ADVANTAGE_MODES:
            raise ValueError(
                "DiffusionNFT advantage_mode must be all, positive_only, negative_only, one_only, or binary"
            )
        advantage_normalization = advantage_normalization_mode(
            self.advantage_normalization,
            field_name="algorithm.advantage_normalization",
        )
        reference_checkpoint = None if self.reference_checkpoint is None else str(self.reference_checkpoint).strip()
        if self.reference_mse_weight > 0 and not reference_checkpoint:
            raise ValueError("positive reference_mse_weight requires an explicit reference_checkpoint")
        if self.reference_mse_weight == 0 and reference_checkpoint is not None:
            raise ValueError("reference_checkpoint is unused when reference_mse_weight is zero")
        object.__setattr__(self, "type", resolved_type)
        object.__setattr__(self, "num_train_timesteps", num_train_timesteps)
        object.__setattr__(self, "reward_weights", reward_weights)
        object.__setattr__(self, "advantage_mode", advantage_mode)
        object.__setattr__(self, "advantage_normalization", advantage_normalization)
        object.__setattr__(self, "reference_checkpoint", reference_checkpoint)


def parse_diffusion_nft_algorithm(value: object) -> DiffusionNFTAlgorithmSpec:
    """Parse a strict DiffusionNFT section and both nested contracts."""

    payload = mapping(value, field_name="algorithm")
    missing = sorted(name for name in ("collection", "reward_weights", "reward_model") if name not in payload)
    if missing:
        raise ValueError(f"DiffusionNFT algorithm is missing required fields: {missing}")
    algorithm_payload = strict_mapping(
        payload,
        field_name="algorithm",
        allowed=DIFFUSION_NFT_ALGORITHM_FIELDS,
    )
    collection_payload = strict_mapping(
        algorithm_payload.pop("collection"),
        field_name="algorithm.collection",
        allowed=DIFFUSION_NFT_COLLECTION_FIELDS,
    )
    refresh_payload = strict_mapping(
        algorithm_payload.pop("old_policy_refresh", {}),
        field_name="algorithm.old_policy_refresh",
        allowed=DIFFUSION_NFT_OLD_POLICY_REFRESH_FIELDS,
    )
    reward_payload = strict_mapping(
        algorithm_payload.pop("reward_model"),
        field_name="algorithm.reward_model",
        allowed=VIDEOALIGN_REWARD_FIELDS,
    )
    return DiffusionNFTAlgorithmSpec(
        **algorithm_payload,
        collection=DiffusionNFTTerminalLatentCollectionSpec(**collection_payload),
        old_policy_refresh=DiffusionNFTOldPolicyRefreshSpec(**refresh_payload),
        reward_model=VideoAlignRewardSpec(**reward_payload),
    )


__all__ = [
    "DIFFUSION_NFT_ADVANTAGE_MODES",
    "DIFFUSION_NFT_DECAY_SCHEDULES",
    "DiffusionNFTAlgorithmSpec",
    "DiffusionNFTOldPolicyRefreshSpec",
    "DiffusionNFTTerminalLatentCollectionSpec",
]
