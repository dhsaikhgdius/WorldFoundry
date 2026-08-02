"""Runtime contracts for WorldFoundry-native DiffusionNFT."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Protocol, runtime_checkable

import torch

from ....shared.contracts import freeze_mapping, non_empty_ids
from ....shared.validation import non_negative_int

DIFFUSION_NFT_OLD_POLICY_SCHEDULES = frozenset({"copy", "linear_to_0_5", "delayed_linear_to_0_999"})


@dataclass(frozen=True, slots=True)
class DiffusionNFTRollout:
    """One collected rollout reduced to the terminal state DiffusionNFT needs.

    The contract deliberately contains no denoising trajectory or log
    probabilities.  DiffusionNFT re-enters the forward process from the clean
    terminal latent and learns from rewards grouped by prompt identity.
    """

    collection_id: str
    policy_revision: str
    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    clean_latents: torch.Tensor
    rewards: torch.Tensor
    conditioning: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.collection_id, str) or not self.collection_id.strip():
            raise ValueError("collection_id must be a non-empty string")
        if not isinstance(self.policy_revision, str) or not self.policy_revision.strip():
            raise ValueError("policy_revision must be a non-empty string")
        sample_ids = non_empty_ids(self.sample_ids, field_name="sample_ids", unique=True)
        group_ids = non_empty_ids(self.group_ids, field_name="group_ids", unique=False)
        if len(group_ids) != len(sample_ids):
            raise ValueError("group_ids length must match sample_ids")
        incomplete = sorted(group for group, count in Counter(group_ids).items() if count < 2)
        if incomplete:
            raise ValueError(f"every DiffusionNFT prompt group needs at least two samples: {incomplete}")
        if (
            not isinstance(self.clean_latents, torch.Tensor)
            or self.clean_latents.ndim < 2
            or int(self.clean_latents.shape[0]) != len(sample_ids)
            or not self.clean_latents.is_floating_point()
        ):
            raise TypeError("clean_latents must be a floating [B,...] torch.Tensor")
        if not bool(torch.isfinite(self.clean_latents).all()):
            raise ValueError("clean_latents must be finite")
        if (
            not isinstance(self.rewards, torch.Tensor)
            or self.rewards.ndim != 1
            or tuple(self.rewards.shape) != (len(sample_ids),)
            or not self.rewards.is_floating_point()
        ):
            raise TypeError("rewards must be a floating [B] torch.Tensor")
        if not bool(torch.isfinite(self.rewards).all()):
            raise ValueError("rewards must be finite")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "group_ids", group_ids)
        object.__setattr__(
            self,
            "conditioning",
            freeze_mapping(self.conditioning, field_name="conditioning"),
        )

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)


@dataclass(frozen=True, slots=True)
class DiffusionNFTTerminalLatents:
    """A completed collection before named rewards are scalarized."""

    collection_id: str
    policy_revision: str
    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    clean_latents: torch.Tensor
    transition_count: int
    conditioning: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.collection_id, str) or not self.collection_id.strip():
            raise ValueError("collection_id must be a non-empty string")
        if not isinstance(self.policy_revision, str) or not self.policy_revision.strip():
            raise ValueError("policy_revision must be a non-empty string")
        sample_ids = non_empty_ids(self.sample_ids, field_name="sample_ids", unique=True)
        group_ids = non_empty_ids(self.group_ids, field_name="group_ids", unique=False)
        if len(group_ids) != len(sample_ids):
            raise ValueError("group_ids length must match sample_ids")
        incomplete = sorted(group for group, count in Counter(group_ids).items() if count < 2)
        if incomplete:
            raise ValueError(f"every DiffusionNFT prompt group needs at least two samples: {incomplete}")
        if (
            not isinstance(self.clean_latents, torch.Tensor)
            or self.clean_latents.ndim < 2
            or int(self.clean_latents.shape[0]) != len(sample_ids)
            or not self.clean_latents.is_floating_point()
        ):
            raise TypeError("clean_latents must be a floating [B,...] torch.Tensor")
        if not bool(torch.isfinite(self.clean_latents).all()):
            raise ValueError("clean_latents must be finite")
        transition_count = non_negative_int(
            self.transition_count,
            field_name="transition_count",
        )
        if transition_count == 0:
            raise ValueError("transition_count must be positive")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "group_ids", group_ids)
        object.__setattr__(self, "transition_count", transition_count)
        object.__setattr__(
            self,
            "conditioning",
            freeze_mapping(self.conditioning, field_name="conditioning"),
        )
        object.__setattr__(
            self,
            "metadata",
            freeze_mapping(self.metadata, field_name="metadata"),
        )

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)

    def with_rewards(self, rewards: torch.Tensor) -> DiffusionNFTRollout:
        return DiffusionNFTRollout(
            collection_id=self.collection_id,
            policy_revision=self.policy_revision,
            sample_ids=self.sample_ids,
            group_ids=self.group_ids,
            clean_latents=self.clean_latents,
            rewards=rewards,
            conditioning=self.conditioning,
        )


@runtime_checkable
class DiffusionNFTRewardAdapter(Protocol):
    """Score a terminal-latent population into named tensor components."""

    reward_ids: tuple[str, ...]

    def score(
        self,
        terminal_latents: DiffusionNFTTerminalLatents,
    ) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class OldPolicyRefresh:
    """How the behavior-policy anchor follows the optimized policy.

    ``retention`` is the coefficient on the previous old-policy parameter:
    ``old <- retention * old + (1 - retention) * policy``.  This state is
    intentionally separate from any export/evaluation EMA.
    """

    schedule: str = "copy"
    update_interval: int = 1

    def __post_init__(self) -> None:
        schedule = str(self.schedule)
        if schedule not in DIFFUSION_NFT_OLD_POLICY_SCHEDULES:
            raise ValueError("old-policy schedule must be copy, linear_to_0_5, or delayed_linear_to_0_999")
        interval = non_negative_int(self.update_interval, field_name="update_interval")
        if interval == 0:
            raise ValueError("old-policy update_interval must be positive")
        object.__setattr__(self, "schedule", schedule)
        object.__setattr__(self, "update_interval", interval)

    def should_refresh(self, optimizer_step: int) -> bool:
        step = non_negative_int(optimizer_step, field_name="optimizer_step")
        return step > 0 and step % self.update_interval == 0

    def retention(self, optimizer_step: int) -> float:
        step = non_negative_int(optimizer_step, field_name="optimizer_step")
        if self.schedule == "copy":
            return 0.0
        if self.schedule == "linear_to_0_5":
            return min(step * 0.001, 0.5)
        if step < 75:
            return 0.0
        return min((step - 75) * 0.0075, 0.999)

    def state_dict(self) -> dict[str, object]:
        return {"schedule": self.schedule, "update_interval": self.update_interval}

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, object]) -> OldPolicyRefresh:
        if not isinstance(state_dict, Mapping) or set(state_dict) != {
            "schedule",
            "update_interval",
        }:
            raise ValueError("old-policy refresh state fields differ from the active schema")
        return cls(
            schedule=str(state_dict["schedule"]),
            update_interval=state_dict["update_interval"],
        )


def validate_mix_beta(value: float) -> float:
    """Resolve the positive/negative prediction displacement coefficient."""

    beta = float(value)
    if not isfinite(beta) or not 0 < beta <= 1:
        raise ValueError("DiffusionNFT beta must be finite and in (0,1]")
    return beta


__all__ = [
    "DIFFUSION_NFT_OLD_POLICY_SCHEDULES",
    "DiffusionNFTRewardAdapter",
    "DiffusionNFTRollout",
    "DiffusionNFTTerminalLatents",
    "OldPolicyRefresh",
    "validate_mix_beta",
]
