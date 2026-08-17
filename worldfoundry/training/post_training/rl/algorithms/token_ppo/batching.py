"""Stateful sample batching for native token PPO runs."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from ....shared.contracts import freeze_mapping
from .contracts import TokenPPORolloutRequest

TOKEN_PPO_DATA_LOADER_STATE_SCHEMA = "worldfoundry-token-ppo-data-loader"


@dataclass(frozen=True, slots=True)
class TokenPPOSample:
    """One model-owned conditioning payload with a stable sample id."""

    sample_id: str
    conditioning: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sample_id = str(self.sample_id).strip()
        if not sample_id:
            raise ValueError("sample_id must be non-empty")
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(
            self,
            "conditioning",
            freeze_mapping(self.conditioning, field_name="conditioning"),
        )


class NativeTokenPPODataLoader(Iterator[TokenPPORolloutRequest]):
    """Emit exact-resume PPO rollout requests at the active policy revision."""

    def __init__(
        self,
        samples: Sequence[TokenPPOSample],
        *,
        batch_size: int,
        policy_revision: Callable[[], str],
        sampling_temperature: float,
        shuffle: bool = True,
        shuffle_seed: int = 42,
        tail_policy: str = "drop",
    ) -> None:
        values = tuple(samples)
        if not values or not all(isinstance(sample, TokenPPOSample) for sample in values):
            raise ValueError("token PPO data loader requires TokenPPOSample values")
        if len({sample.sample_id for sample in values}) != len(values):
            raise ValueError("token PPO sample ids must be unique")
        if isinstance(batch_size, bool) or int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if not callable(policy_revision):
            raise TypeError("policy_revision must be callable")
        resolved_tail = str(tail_policy).strip().lower().replace("_", "-")
        if resolved_tail not in {"drop", "pad", "uneven"}:
            raise ValueError("tail_policy must be drop, pad, or uneven")
        if resolved_tail == "drop" and len(values) < int(batch_size):
            raise ValueError("drop tail policy requires at least one complete batch")
        self.samples = values
        self.batch_size = int(batch_size)
        self.policy_revision = policy_revision
        self.sampling_temperature = float(sampling_temperature)
        self.shuffle = bool(shuffle)
        self.shuffle_seed = int(shuffle_seed)
        self.tail_policy = resolved_tail
        self.epoch = 0
        self.cursor = 0
        self.completed_batches = 0
        self.order = self._order_for_epoch(0)

    def _order_for_epoch(self, epoch: int) -> list[int]:
        order = list(range(len(self.samples)))
        if self.shuffle:
            random.Random(self.shuffle_seed + epoch).shuffle(order)
        return order

    def _next_epoch(self) -> None:
        self.epoch += 1
        self.cursor = 0
        self.order = self._order_for_epoch(self.epoch)

    def _take_indices(self) -> tuple[int, ...]:
        remaining = len(self.order) - self.cursor
        if remaining == 0 or (self.tail_policy == "drop" and remaining < self.batch_size):
            self._next_epoch()
            remaining = len(self.order)
        take = min(remaining, self.batch_size)
        indices = list(self.order[self.cursor : self.cursor + take])
        self.cursor += take
        if self.tail_policy == "pad" and take < self.batch_size:
            while len(indices) < self.batch_size:
                self._next_epoch()
                needed = self.batch_size - len(indices)
                epoch_take = min(needed, len(self.order))
                indices.extend(self.order[:epoch_take])
                self.cursor = epoch_take
        return tuple(indices)

    def __iter__(self) -> NativeTokenPPODataLoader:
        return self

    def __next__(self) -> TokenPPORolloutRequest:
        indices = self._take_indices()
        batch_index = self.completed_batches
        selected = tuple(self.samples[index] for index in indices)
        revision = str(self.policy_revision()).strip()
        if not revision:
            raise ValueError("active PPO policy revision cannot be empty")
        self.completed_batches += 1
        sample_ids = tuple(
            f"{sample.sample_id}::batch-{batch_index:08d}-row-{row:04d}" for row, sample in enumerate(selected)
        )
        return TokenPPORolloutRequest(
            sample_ids=sample_ids,
            policy_revision=revision,
            sampling_temperature=self.sampling_temperature,
            conditioning={
                "base_sample_ids": tuple(sample.sample_id for sample in selected),
                "samples": tuple(dict(sample.conditioning) for sample in selected),
            },
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": TOKEN_PPO_DATA_LOADER_STATE_SCHEMA,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "completed_batches": self.completed_batches,
            "order": list(self.order),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        expected = {"schema", "epoch", "cursor", "completed_batches", "order"}
        if not isinstance(state_dict, Mapping) or set(state_dict) != expected:
            raise ValueError("token PPO data-loader state fields differ")
        if state_dict["schema"] != TOKEN_PPO_DATA_LOADER_STATE_SCHEMA:
            raise ValueError("unsupported token PPO data-loader state schema")
        epoch = int(state_dict["epoch"])
        cursor = int(state_dict["cursor"])
        completed_batches = int(state_dict["completed_batches"])
        order = [int(index) for index in state_dict["order"]]  # type: ignore[union-attr]
        if (
            epoch < 0
            or completed_batches < 0
            or not 0 <= cursor <= len(self.samples)
            or sorted(order) != list(range(len(self.samples)))
        ):
            raise ValueError("saved token PPO data-loader position is invalid")
        self.epoch = epoch
        self.cursor = cursor
        self.completed_batches = completed_batches
        self.order = order


__all__ = [
    "TOKEN_PPO_DATA_LOADER_STATE_SCHEMA",
    "NativeTokenPPODataLoader",
    "TokenPPOSample",
]
