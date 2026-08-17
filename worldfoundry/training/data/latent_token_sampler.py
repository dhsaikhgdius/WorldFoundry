"""Shape-compatible latent-token batching with exact checkpoint replay."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence, Sized
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .sampler import SamplerStateMismatchError, distributed_context_from_environment, stable_permutation
from .video_bucketing import VideoBucketKey

LATENT_TOKEN_BATCH_SAMPLER_SCHEMA = "worldfoundry-latent-token-batch-sampler"


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


def _json_mapping(value: Mapping[str, object], *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    try:
        payload = json.loads(json.dumps(dict(value), sort_keys=True))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} must contain JSON values") from error
    return MappingProxyType(payload)


def _group_name(key: VideoBucketKey, contract: Mapping[str, object]) -> str:
    return json.dumps(
        {"bucket_key": key.to_dict(), "batch_contract": dict(contract)},
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True, slots=True)
class MicrobatchTokenStats:
    bucket_key: VideoBucketKey
    sample_count: int
    latent_tokens: int

    def __post_init__(self) -> None:
        samples = _positive_int(self.sample_count, field_name="sample_count")
        tokens = _positive_int(self.latent_tokens, field_name="latent_tokens")
        if tokens != samples * self.bucket_key.token_count:
            raise ValueError("latent_tokens does not match the bucket token count")
        object.__setattr__(self, "sample_count", samples)
        object.__setattr__(self, "latent_tokens", tokens)


class LatentTokenBatchSampler:
    """Build homogeneous microbatches within a per-rank latent-token budget."""

    def __init__(
        self,
        data_source: Sized,
        *,
        bucket_keys: Sequence[VideoBucketKey] | None = None,
        batch_contracts: Sequence[Mapping[str, object]] | None = None,
        sample_ids: Sequence[str] | None = None,
        max_latent_tokens: int,
        seed: int = 42,
        shuffle: bool = True,
        rank: int | None = None,
        world_size: int | None = None,
        tail_policy: str = "drop",
        epoch: int = 0,
    ) -> None:
        if not isinstance(data_source, Sized) or not callable(getattr(data_source, "__getitem__", None)):
            raise TypeError("data_source must be a sized map-style dataset")
        dataset_size = len(data_source)
        if dataset_size <= 0:
            raise ValueError("data_source cannot be empty")
        keys = tuple(getattr(data_source, "bucket_keys", ()) if bucket_keys is None else bucket_keys)
        if len(keys) != dataset_size or not all(isinstance(key, VideoBucketKey) for key in keys):
            raise ValueError("bucket_keys must contain one VideoBucketKey per dataset item")
        raw_contracts = getattr(data_source, "batch_contracts", None) if batch_contracts is None else batch_contracts
        if raw_contracts is None:
            contracts = tuple(MappingProxyType(key.to_dict()) for key in keys)
        else:
            contracts = tuple(_json_mapping(value, field_name="batch_contract") for value in raw_contracts)
            if len(contracts) != dataset_size:
                raise ValueError("batch_contracts must contain one mapping per dataset item")
        raw_sample_ids = getattr(data_source, "sample_ids", None) if sample_ids is None else sample_ids
        if raw_sample_ids is None:
            raise ValueError("data_source must expose sample_ids")
        resolved_sample_ids = tuple(str(value) for value in raw_sample_ids)
        if len(resolved_sample_ids) != dataset_size or len(set(resolved_sample_ids)) != dataset_size:
            raise ValueError("sample_ids must be unique and match the dataset size")
        token_budget = _positive_int(max_latent_tokens, field_name="max_latent_tokens")
        if any(key.token_count > token_budget for key in keys):
            raise ValueError("max_latent_tokens is smaller than a per-sample bucket")
        inferred_rank, inferred_world_size = distributed_context_from_environment()
        resolved_rank = inferred_rank if rank is None else int(rank)
        resolved_world_size = inferred_world_size if world_size is None else int(world_size)
        if resolved_world_size <= 0 or not 0 <= resolved_rank < resolved_world_size:
            raise ValueError(f"invalid rank/world_size: {resolved_rank}/{resolved_world_size}")
        policy = str(tail_policy).strip().lower().replace("_", "-")
        if policy not in {"drop", "pad"}:
            raise ValueError("tail_policy must be 'drop' or 'pad'")
        if isinstance(epoch, bool) or int(epoch) < 0:
            raise ValueError("epoch must be non-negative")

        self.data_source = data_source
        self.dataset_size = dataset_size
        self.sample_ids = resolved_sample_ids
        self.bucket_keys = keys
        self.batch_contracts = contracts
        self.max_latent_tokens = token_budget
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.rank = resolved_rank
        self.world_size = resolved_world_size
        self.tail_policy = policy
        self._epoch = int(epoch)
        self._position = 0
        self._cached_epoch: int | None = None
        self._global_batches: tuple[tuple[int, ...], ...] = ()
        self._local_batches: tuple[tuple[int, ...], ...] = ()
        self._consumed_sample_count = 0
        self._consumed_latent_tokens = 0
        self._build_batches()

    def _build_batches(self) -> None:
        order = (
            stable_permutation(self.dataset_size, seed=self.seed, epoch=self._epoch)
            if self.shuffle
            else tuple(range(self.dataset_size))
        )
        queues: OrderedDict[str, list[int]] = OrderedDict()
        keys_by_group: dict[str, VideoBucketKey] = {}
        capacities: dict[str, int] = {}
        batches: list[tuple[int, ...]] = []
        for sample_index in order:
            key = self.bucket_keys[sample_index]
            group = _group_name(key, self.batch_contracts[sample_index])
            queue = queues.setdefault(group, [])
            keys_by_group[group] = key
            capacities[group] = self.max_latent_tokens // key.token_count
            queue.append(sample_index)
            if len(queue) == capacities[group]:
                batches.append(tuple(queue))
                queue.clear()
        for queue in queues.values():
            if queue:
                batches.append(tuple(queue))

        self.global_batch_count = len(batches)
        if self.tail_policy == "drop":
            effective = (len(batches) // self.world_size) * self.world_size
            batches = batches[:effective]
        elif batches and len(batches) % self.world_size:
            original = tuple(batches)
            while len(batches) % self.world_size:
                batches.append(original[(len(batches) - len(original)) % len(original)])
        self.global_effective_batch_count = len(batches)
        if not batches:
            raise ValueError("tail_policy='drop' leaves every rank without a batch")
        local: list[tuple[int, ...]] = []
        for group_start in range(0, len(batches), self.world_size):
            group = batches[group_start : group_start + self.world_size]
            local_position = len(local)
            local.append(group[(self.rank + local_position) % self.world_size])
        self._cached_epoch = self._epoch
        self._global_batches = tuple(batches)
        self._local_batches = tuple(local)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def position(self) -> int:
        return self._position

    @property
    def consumed_sample_count(self) -> int:
        return self._consumed_sample_count

    @property
    def consumed_latent_tokens(self) -> int:
        return self._consumed_latent_tokens

    @property
    def global_dropped_batch_count(self) -> int:
        return self.global_batch_count - self.global_effective_batch_count if self.tail_policy == "drop" else 0

    @property
    def global_padded_batch_count(self) -> int:
        return self.global_effective_batch_count - self.global_batch_count if self.tail_policy == "pad" else 0

    def __len__(self) -> int:
        return len(self._local_batches)

    def __iter__(self) -> Iterator[list[int]]:
        if self._position >= len(self):
            self._move_to_epoch(self._epoch + 1)
        while self._position < len(self):
            batch = self._local_batches[self._position]
            stats = self.describe_batch(batch)
            self._position += 1
            self._consumed_sample_count += stats.sample_count
            self._consumed_latent_tokens += stats.latent_tokens
            yield list(batch)

    def describe_batch(self, indices: Sequence[int]) -> MicrobatchTokenStats:
        values = tuple(int(index) for index in indices)
        if not values:
            raise ValueError("batch indices cannot be empty")
        keys = {self.bucket_keys[index] for index in values}
        contracts = {_group_name(self.bucket_keys[index], self.batch_contracts[index]) for index in values}
        if len(keys) != 1 or len(contracts) != 1:
            raise ValueError("one microbatch cannot mix bucket keys or batch contracts")
        key = next(iter(keys))
        tokens = len(values) * key.token_count
        if tokens > self.max_latent_tokens:
            raise ValueError("microbatch exceeds max_latent_tokens")
        return MicrobatchTokenStats(key, len(values), tokens)

    def _move_to_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._position = 0
        self._consumed_sample_count = 0
        self._consumed_latent_tokens = 0
        self._build_batches()

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or int(epoch) < 0:
            raise ValueError("epoch must be non-negative")
        if int(epoch) != self._epoch:
            self._move_to_epoch(int(epoch))

    def state_dict(self) -> dict[str, object]:
        next_batch = None if self._position >= len(self) else self._local_batches[self._position]
        return {
            "schema": LATENT_TOKEN_BATCH_SAMPLER_SCHEMA,
            "sample_ids": list(self.sample_ids),
            "bucket_keys": [key.to_dict() for key in self.bucket_keys],
            "batch_contracts": [dict(contract) for contract in self.batch_contracts],
            "max_latent_tokens": self.max_latent_tokens,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "tail_policy": self.tail_policy,
            "rank": self.rank,
            "world_size": self.world_size,
            "epoch": self._epoch,
            "position": self._position,
            "local_length": len(self),
            "global_batch_count": self.global_batch_count,
            "global_effective_batch_count": self.global_effective_batch_count,
            "consumed_sample_count": self._consumed_sample_count,
            "consumed_latent_tokens": self._consumed_latent_tokens,
            "next_batch_indices": None if next_batch is None else list(next_batch),
            "next_batch_sample_ids": None if next_batch is None else [self.sample_ids[index] for index in next_batch],
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("batch sampler state must be a mapping")
        required = set(self.state_dict())
        if set(state_dict) != required:
            raise SamplerStateMismatchError("batch sampler state fields differ from the active schema")
        saved_epoch = int(state_dict["epoch"])
        saved_position = int(state_dict["position"])
        current_epoch = self._epoch
        current_position = self._position
        current_samples = self._consumed_sample_count
        current_tokens = self._consumed_latent_tokens
        try:
            self._move_to_epoch(saved_epoch)
            invariants = {
                "schema": LATENT_TOKEN_BATCH_SAMPLER_SCHEMA,
                "sample_ids": list(self.sample_ids),
                "bucket_keys": [key.to_dict() for key in self.bucket_keys],
                "batch_contracts": [dict(contract) for contract in self.batch_contracts],
                "max_latent_tokens": self.max_latent_tokens,
                "seed": self.seed,
                "shuffle": self.shuffle,
                "tail_policy": self.tail_policy,
                "rank": self.rank,
                "world_size": self.world_size,
                "local_length": len(self),
                "global_batch_count": self.global_batch_count,
                "global_effective_batch_count": self.global_effective_batch_count,
            }
            mismatches = [key for key, value in invariants.items() if state_dict[key] != value]
            if mismatches:
                raise SamplerStateMismatchError(f"batch sampler exact-resume invariants differ: {mismatches}")
            if not 0 <= saved_position <= len(self):
                raise SamplerStateMismatchError("batch sampler position is out of range")
            self._position = saved_position
            expected_samples = sum(len(batch) for batch in self._local_batches[:saved_position])
            expected_tokens = sum(self.describe_batch(batch).latent_tokens for batch in self._local_batches[:saved_position])
            next_batch = None if saved_position >= len(self) else self._local_batches[saved_position]
            expected_next_indices = None if next_batch is None else list(next_batch)
            expected_next_ids = None if next_batch is None else [self.sample_ids[index] for index in next_batch]
            if (
                state_dict["consumed_sample_count"] != expected_samples
                or state_dict["consumed_latent_tokens"] != expected_tokens
                or state_dict["next_batch_indices"] != expected_next_indices
                or state_dict["next_batch_sample_ids"] != expected_next_ids
            ):
                raise SamplerStateMismatchError("saved batch sampler position does not match deterministic replay")
            self._consumed_sample_count = expected_samples
            self._consumed_latent_tokens = expected_tokens
        except Exception:
            self._move_to_epoch(current_epoch)
            self._position = current_position
            self._consumed_sample_count = current_samples
            self._consumed_latent_tokens = current_tokens
            raise


__all__ = ["LATENT_TOKEN_BATCH_SAMPLER_SCHEMA", "LatentTokenBatchSampler", "MicrobatchTokenStats"]
