"""Named reward components supplied by an injected evaluator or HTTP service."""

from __future__ import annotations

from dataclasses import dataclass

REMOTE_REWARD_FIELDS = {"type", "reward_ids"}


@dataclass(frozen=True, slots=True)
class RemoteRewardSpec:
    """The component names that a non-VideoAlign reward service must return."""

    reward_ids: tuple[str, ...]
    type: str = "remote"

    def __post_init__(self) -> None:
        reward_type = str(self.type).strip().lower().replace("_", "-")
        if reward_type != "remote":
            raise ValueError("remote reward type must be 'remote'")
        reward_ids = tuple(str(value).strip() for value in self.reward_ids)
        if not reward_ids or any(not value for value in reward_ids):
            raise ValueError("remote reward_ids must contain non-empty names")
        if len(set(reward_ids)) != len(reward_ids):
            raise ValueError("remote reward_ids must be unique")
        object.__setattr__(self, "type", reward_type)
        object.__setattr__(self, "reward_ids", reward_ids)


__all__ = ["REMOTE_REWARD_FIELDS", "RemoteRewardSpec"]
