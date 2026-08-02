"""Shape-compatible latent-token batching with exact checkpoint replay."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence, Sized
from dataclasses import dataclass
from typing import Any

from worldfoundry.core.io.integrity import canonical_sha256 as _core_canonical_sha256

from .sampler import SamplerStateMismatchError, distributed_context_from_environment, stable_permutation
from .video_bucketing import VideoBucketKey

LATENT_TOKEN_BATCH_SAMPLER_SCHEMA = "worldfoundry-latent-token-batch-sampler"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _canonical_sha256(value: object) -> str:
    try:
        return _core_canonical_sha256(value)
    except (TypeError, ValueError) as error:
        raise TypeError("bucket metadata must be JSON serializable without NaN or infinity") from error


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return resolved


def _positive_int(value: object, *, field_name: str) -> int:
    resolved = _non_negative_int(value, field_name=field_name)
    if resolved == 0:
        raise ValueError(f"{field_name} must be positive")
    return resolved


def _sha256(value: object, *, field_name: str) -> str:
    resolved = str(value).strip().lower()
    if _SHA256_PATTERN.fullmatch(resolved) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return resolved


@dataclass(frozen=True, slots=True)
class MicrobatchTokenStats:
    bucket_key: VideoBucketKey
    sample_count: int
    latent_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.bucket_key, VideoBucketKey):
            raise TypeError("bucket_key must be a VideoBucketKey")
        samples = _positive_int(self.sample_count, field_name="sample_count")
        tokens = _positive_int(self.latent_tokens, field_name="latent_tokens")
        if tokens != samples * self.bucket_key.token_count:
            raise ValueError("latent_tokens does not match sample_count and bucket token count")
        object.__setattr__(self, "sample_count", samples)
        object.__setattr__(self, "latent_tokens", tokens)


class LatentTokenBatchSampler:
    """Shape-compatible, token-budgeted batches with exact local resume.

    Every rank deterministically simulates the same global bucket queues.
    Completed global batches are committed in groups of ``world_size`` and one
    batch from each group is assigned to each rank.  ``drop`` removes an
    incomplete final group; ``pad`` repeats complete batches to finish it.
    Consequently all ranks execute the same number of collectives without
    mixing incompatible shapes inside a microbatch.
    """

    def __init__(
        self,
        data_source: Sized,
        *,
        bucket_keys: Sequence[VideoBucketKey] | None = None,
        batch_contract_digests: Sequence[str] | None = None,
        dataset_digest: str | None = None,
        data_content_digest: str | None = None,
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
        keys_source = getattr(data_source, "bucket_keys", None) if bucket_keys is None else bucket_keys
        if keys_source is None:
            raise ValueError("bucket_keys are required when data_source does not expose them")
        keys = tuple(keys_source)
        if len(keys) != dataset_size or not all(isinstance(key, VideoBucketKey) for key in keys):
            raise ValueError("bucket_keys must contain one VideoBucketKey per dataset item")
        contracts_source = (
            getattr(data_source, "batch_contract_digests", None)
            if batch_contract_digests is None
            else batch_contract_digests
        )
        if contracts_source is None:
            contracts = tuple(key.digest for key in keys)
        else:
            contracts = tuple(_sha256(value, field_name="batch_contract_digest") for value in contracts_source)
            if len(contracts) != dataset_size:
                raise ValueError("batch_contract_digests must contain one SHA-256 digest per dataset item")
        batch_groups = tuple(
            _canonical_sha256(
                {
                    "bucket_key": key.to_dict(),
                    "batch_contract_digest": contract,
                }
            )
            for key, contract in zip(keys, contracts)
        )

        sample_ids_source = getattr(data_source, "sample_ids", None)
        if sample_ids_source is None:
            raise ValueError("data_source must expose stable sample_ids for exact resume")
        sample_ids = tuple(str(value) for value in sample_ids_source)
        if len(sample_ids) != dataset_size or any(not value.strip() for value in sample_ids):
            raise ValueError("data_source.sample_ids must contain one non-empty id per item")
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("data_source.sample_ids must be unique")

        digest = dataset_digest if dataset_digest is not None else getattr(data_source, "dataset_digest", None)
        digest = _sha256(digest, field_name="dataset_digest")
        content_digest = data_content_digest
        if content_digest is None:
            content_digest = getattr(data_source, "index_sha256", digest)
        content_digest = _sha256(content_digest, field_name="data_content_digest")
        token_budget = _positive_int(max_latent_tokens, field_name="max_latent_tokens")
        oversized = sorted({key.token_count for key in keys if key.token_count > token_budget})
        if oversized:
            raise ValueError(
                f"max_latent_tokens={token_budget} is smaller than per-sample bucket token counts {oversized}"
            )
        if isinstance(seed, bool):
            raise TypeError("seed must be an integer, not bool")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a bool")
        if isinstance(epoch, bool) or int(epoch) < 0:
            raise ValueError("epoch must be a non-negative integer")

        inferred_rank, inferred_world_size = distributed_context_from_environment()
        resolved_rank = inferred_rank if rank is None else int(rank)
        resolved_world_size = inferred_world_size if world_size is None else int(world_size)
        if resolved_world_size <= 0 or not 0 <= resolved_rank < resolved_world_size:
            raise ValueError(f"invalid rank/world_size: {resolved_rank}/{resolved_world_size}")
        policy = str(tail_policy).strip().lower().replace("_", "-")
        if policy not in {"drop", "pad"}:
            raise ValueError("tail_policy must be 'drop' or 'pad' for collective-safe batching")

        self.data_source = data_source
        self.dataset_size = dataset_size
        self.sample_ids = sample_ids
        self.bucket_keys = keys
        self.batch_contract_digests = contracts
        self.batch_group_digests = batch_groups
        self.dataset_digest = digest
        self.data_content_digest = content_digest
        self.max_latent_tokens = token_budget
        self.seed = int(seed)
        self.shuffle = shuffle
        self.rank = resolved_rank
        self.world_size = resolved_world_size
        self.tail_policy = policy
        self.assignment_digest = _canonical_sha256(
            {
                "sample_ids": list(sample_ids),
                "bucket_keys": [key.to_dict() for key in keys],
                "batch_contract_digests": list(contracts),
            }
        )
        self.config_digest = _canonical_sha256(
            {
                "dataset_digest": self.dataset_digest,
                "data_content_digest": self.data_content_digest,
                "assignment_digest": self.assignment_digest,
                "max_latent_tokens": self.max_latent_tokens,
                "seed": self.seed,
                "shuffle": self.shuffle,
                "world_size": self.world_size,
                "tail_policy": self.tail_policy,
                "rank_assignment": "rotating-global-batch-groups",
            }
        )
        self._bucket_by_digest = {group_digest: key for group_digest, key in zip(batch_groups, keys)}
        if len(self._bucket_by_digest) != len(set(batch_groups)):
            raise RuntimeError("batch-group digest collision detected")
        self._capacities = {
            digest_key: self.max_latent_tokens // key.token_count for digest_key, key in self._bucket_by_digest.items()
        }
        counts = Counter(batch_groups)
        self.global_batch_count = sum(
            math.ceil(count / self._capacities[digest_key]) for digest_key, count in counts.items()
        )
        if self.tail_policy == "drop":
            self.global_effective_batch_count = (self.global_batch_count // self.world_size) * self.world_size
        else:
            self.global_effective_batch_count = math.ceil(self.global_batch_count / self.world_size) * self.world_size
        if self.global_effective_batch_count == 0:
            raise ValueError(
                "tail_policy='drop' would leave every rank without a batch; use 'pad' or a smaller world_size"
            )
        self._epoch = int(epoch)
        self._cached_epoch: int | None = None
        self._cached_permutation: tuple[int, ...] = ()
        self._reset_runtime()

    def _reset_runtime(self) -> None:
        self._position = 0
        self._source_position = 0
        self._queues: dict[str, list[int]] = {}
        self._queue_order: list[str] = []
        self._pending_global_batches: list[tuple[int, ...]] = []
        self._padding_source: list[tuple[int, ...]] = []
        self._source_exhausted = False
        self._flush_complete = False
        self._consumed_sample_count = 0
        self._consumed_latent_tokens = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def position(self) -> int:
        return self._position

    @property
    def remaining(self) -> int:
        return len(self) - self._position

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
        return self.global_effective_batch_count // self.world_size

    def _permutation(self) -> tuple[int, ...]:
        if self._cached_epoch == self._epoch:
            return self._cached_permutation
        if self.shuffle:
            order = stable_permutation(self.dataset_size, seed=self.seed, epoch=self._epoch)
        else:
            order = tuple(range(self.dataset_size))
        self._cached_epoch = self._epoch
        self._cached_permutation = order
        return order

    def _append_global_batch(self, indices: Sequence[int]) -> None:
        batch = tuple(int(index) for index in indices)
        self.describe_batch(batch)
        self._pending_global_batches.append(batch)
        if len(self._padding_source) < self.world_size:
            self._padding_source.append(batch)

    def _fill_pending_batches(self) -> None:
        order = self._permutation()
        while self._source_position < len(order) and len(self._pending_global_batches) < self.world_size:
            sample_index = order[self._source_position]
            self._source_position += 1
            bucket_digest = self.batch_group_digests[sample_index]
            if bucket_digest not in self._queues:
                self._queues[bucket_digest] = []
                self._queue_order.append(bucket_digest)
            queue = self._queues[bucket_digest]
            queue.append(sample_index)
            if len(queue) == self._capacities[bucket_digest]:
                self._append_global_batch(queue)
                queue.clear()

        if self._source_position == len(order):
            self._source_exhausted = True
        if self._source_exhausted and not self._flush_complete:
            for bucket_digest in self._queue_order:
                queue = self._queues[bucket_digest]
                if queue:
                    self._append_global_batch(queue)
                    queue.clear()
            self._flush_complete = True

    def _next_local_indices(self) -> tuple[int, ...]:
        while len(self._pending_global_batches) < self.world_size and not self._flush_complete:
            self._fill_pending_batches()
        if len(self._pending_global_batches) >= self.world_size:
            group = self._pending_global_batches[: self.world_size]
            del self._pending_global_batches[: self.world_size]
            return group[(self.rank + self._position) % self.world_size]
        if self._pending_global_batches:
            if self.tail_policy == "drop":
                self._pending_global_batches.clear()
                raise StopIteration
            if not self._padding_source:
                raise RuntimeError("cannot pad a non-empty dataset without a completed global batch")
            group = list(self._pending_global_batches)
            self._pending_global_batches.clear()
            padding_index = 0
            while len(group) < self.world_size:
                group.append(self._padding_source[padding_index % len(self._padding_source)])
                padding_index += 1
            return group[(self.rank + self._position) % self.world_size]
        raise StopIteration

    def __iter__(self) -> Iterator[list[int]]:
        if self._position >= len(self):
            self._move_to_epoch(self._epoch + 1)
        while self._position < len(self):
            indices = self._next_local_indices()
            stats = self.describe_batch(indices)
            self._position += 1
            self._consumed_sample_count += stats.sample_count
            self._consumed_latent_tokens += stats.latent_tokens
            yield list(indices)

    def describe_batch(self, indices: Sequence[int]) -> MicrobatchTokenStats:
        values = tuple(int(index) for index in indices)
        if not values:
            raise ValueError("batch indices cannot be empty")
        if any(index < 0 or index >= self.dataset_size for index in values):
            raise ValueError("batch contains an out-of-range dataset index")
        keys = {self.bucket_keys[index] for index in values}
        if len(keys) != 1:
            raise ValueError("one microbatch cannot mix video bucket keys")
        contracts = {self.batch_contract_digests[index] for index in values}
        if len(contracts) != 1:
            raise ValueError("one microbatch cannot mix video batch contracts")
        key = next(iter(keys))
        tokens = len(values) * key.token_count
        if tokens > self.max_latent_tokens:
            raise ValueError("microbatch exceeds max_latent_tokens")
        return MicrobatchTokenStats(bucket_key=key, sample_count=len(values), latent_tokens=tokens)

    def _move_to_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._cached_epoch = None
        self._cached_permutation = ()
        self._reset_runtime()

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or int(epoch) < 0:
            raise ValueError("epoch must be a non-negative integer")
        resolved = int(epoch)
        if resolved != self._epoch:
            self._move_to_epoch(resolved)

    def advance_epoch(self, *, force: bool = False) -> None:
        if not force and self._position < len(self):
            raise RuntimeError("cannot advance an unfinished batch-sampler epoch without force=True")
        self._move_to_epoch(self._epoch + 1)

    def _runtime_snapshot(self) -> dict[str, object]:
        return {
            "source_position": self._source_position,
            "queues": {key: list(value) for key, value in self._queues.items()},
            "queue_order": list(self._queue_order),
            "pending": [list(batch) for batch in self._pending_global_batches],
            "padding_source": [list(batch) for batch in self._padding_source],
            "source_exhausted": self._source_exhausted,
            "flush_complete": self._flush_complete,
        }

    def _restore_runtime_snapshot(self, state: Mapping[str, object]) -> None:
        self._source_position = int(state["source_position"])
        raw_queues = state["queues"]
        assert isinstance(raw_queues, Mapping)
        self._queues = {str(key): [int(item) for item in value] for key, value in raw_queues.items()}
        self._queue_order = [str(value) for value in state["queue_order"]]
        self._pending_global_batches = [tuple(int(item) for item in batch) for batch in state["pending"]]
        self._padding_source = [tuple(int(item) for item in batch) for batch in state["padding_source"]]
        self._source_exhausted = bool(state["source_exhausted"])
        self._flush_complete = bool(state["flush_complete"])

    def _peek_next_indices(self) -> tuple[int, ...] | None:
        if self._position >= len(self):
            return None
        snapshot = self._runtime_snapshot()
        try:
            return self._next_local_indices()
        finally:
            self._restore_runtime_snapshot(snapshot)

    @staticmethod
    def _serialized_batch(indices: Sequence[int], sample_ids: Sequence[str]) -> dict[str, object]:
        values = [int(index) for index in indices]
        return {"sample_indices": values, "sample_ids": [sample_ids[index] for index in values]}

    def state_dict(self) -> dict[str, object]:
        permutation = self._permutation()
        next_indices = self._peek_next_indices()
        bucket_queues = [
            {
                "bucket_digest": bucket_digest,
                **self._serialized_batch(self._queues[bucket_digest], self.sample_ids),
            }
            for bucket_digest in self._queue_order
            if self._queues[bucket_digest]
        ]
        return {
            "schema": LATENT_TOKEN_BATCH_SAMPLER_SCHEMA,
            "dataset_digest": self.dataset_digest,
            "data_content_digest": self.data_content_digest,
            "dataset_size": self.dataset_size,
            "assignment_digest": self.assignment_digest,
            "config_digest": self.config_digest,
            "max_latent_tokens": self.max_latent_tokens,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "tail_policy": self.tail_policy,
            "rank": self.rank,
            "world_size": self.world_size,
            "rank_assignment": "rotating-global-batch-groups",
            "epoch": self._epoch,
            "epoch_permutation_sha256": _canonical_sha256([self.sample_ids[index] for index in permutation]),
            "source_position": self._source_position,
            "position": self._position,
            "local_length": len(self),
            "global_batch_count": self.global_batch_count,
            "global_effective_batch_count": self.global_effective_batch_count,
            "bucket_queue_order": list(self._queue_order),
            "bucket_queues": bucket_queues,
            "pending_global_batches": [
                self._serialized_batch(batch, self.sample_ids) for batch in self._pending_global_batches
            ],
            "padding_source_batches": [
                self._serialized_batch(batch, self.sample_ids) for batch in self._padding_source
            ],
            "source_exhausted": self._source_exhausted,
            "flush_complete": self._flush_complete,
            "consumed_sample_count": self._consumed_sample_count,
            "consumed_latent_tokens": self._consumed_latent_tokens,
            "data_rng": {
                "algorithm": "splitmix64-fisher-yates",
                "seed": self.seed,
                "epoch": self._epoch,
                "source_position": self._source_position,
            },
            "next_batch_indices": None if next_indices is None else list(next_indices),
            "next_batch_sample_ids": (
                None if next_indices is None else [self.sample_ids[index] for index in next_indices]
            ),
        }

    def _fresh(self, *, epoch: int) -> LatentTokenBatchSampler:
        sampler = LatentTokenBatchSampler(
            self.data_source,
            bucket_keys=self.bucket_keys,
            batch_contract_digests=self.batch_contract_digests,
            dataset_digest=self.dataset_digest,
            data_content_digest=self.data_content_digest,
            max_latent_tokens=self.max_latent_tokens,
            seed=self.seed,
            shuffle=self.shuffle,
            rank=self.rank,
            world_size=self.world_size,
            tail_policy=self.tail_policy,
            epoch=epoch,
        )
        return sampler

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Transactionally restore and replay-audit every serialized queue."""

        if not isinstance(state_dict, Mapping):
            raise TypeError("batch sampler state_dict must be a mapping")
        active_fields = set(self.state_dict())
        unknown = sorted(set(state_dict) - active_fields)
        missing = sorted(active_fields - set(state_dict))
        if unknown or missing:
            raise SamplerStateMismatchError(
                f"batch sampler state fields mismatch; missing={missing}, unknown={unknown}"
            )
        invariants = {
            "schema": LATENT_TOKEN_BATCH_SAMPLER_SCHEMA,
            "dataset_digest": self.dataset_digest,
            "data_content_digest": self.data_content_digest,
            "dataset_size": self.dataset_size,
            "assignment_digest": self.assignment_digest,
            "config_digest": self.config_digest,
            "max_latent_tokens": self.max_latent_tokens,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "tail_policy": self.tail_policy,
            "rank": self.rank,
            "world_size": self.world_size,
            "rank_assignment": "rotating-global-batch-groups",
            "local_length": len(self),
            "global_batch_count": self.global_batch_count,
            "global_effective_batch_count": self.global_effective_batch_count,
        }
        mismatches = {
            key: (state_dict[key], expected) for key, expected in invariants.items() if state_dict[key] != expected
        }
        if mismatches:
            detail = ", ".join(
                f"{key}=saved:{saved!r}/active:{active!r}" for key, (saved, active) in mismatches.items()
            )
            raise SamplerStateMismatchError(f"batch sampler exact-resume invariants differ: {detail}")
        epoch = int(state_dict["epoch"])
        position = int(state_dict["position"])
        if epoch < 0 or not 0 <= position <= len(self):
            raise SamplerStateMismatchError("saved batch sampler epoch/position is out of range")

        reconstructed = self._fresh(epoch=epoch)
        iterator = iter(reconstructed)
        try:
            for _ in range(position):
                next(iterator)
        except StopIteration as error:
            raise SamplerStateMismatchError("saved position exceeds the reconstructed epoch") from error
        expected_state = reconstructed.state_dict()
        differing = sorted(key for key in expected_state if state_dict[key] != expected_state[key])
        if differing:
            raise SamplerStateMismatchError(
                f"saved batch sampler queues/order do not match deterministic replay: {differing}"
            )

        self._epoch = reconstructed._epoch
        self._cached_epoch = reconstructed._cached_epoch
        self._cached_permutation = reconstructed._cached_permutation
        self._position = reconstructed._position
        self._source_position = reconstructed._source_position
        self._queues = {key: list(value) for key, value in reconstructed._queues.items()}
        self._queue_order = list(reconstructed._queue_order)
        self._pending_global_batches = list(reconstructed._pending_global_batches)
        self._padding_source = list(reconstructed._padding_source)
        self._source_exhausted = reconstructed._source_exhausted
        self._flush_complete = reconstructed._flush_complete
        self._consumed_sample_count = reconstructed._consumed_sample_count
        self._consumed_latent_tokens = reconstructed._consumed_latent_tokens


__all__ = [
    "LATENT_TOKEN_BATCH_SAMPLER_SCHEMA",
    "LatentTokenBatchSampler",
    "MicrobatchTokenStats",
]
