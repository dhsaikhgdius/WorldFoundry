"""Topology-aware deterministic sampling with exact rank-local resume state."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Mapping, Sized
from typing import Any

SAMPLER_STATE_SCHEMA = "worldfoundry-training-sampler"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MASK_64 = (1 << 64) - 1
_GOLDEN_GAMMA = 0x9E3779B97F4A7C15


class SamplerStateMismatchError(ValueError):
    """Raised when exact resume invariants do not match the active sampler."""


def distributed_context_from_environment(
    environment: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    """Resolve global rank/world-size without importing ``torch.distributed``."""

    values = os.environ if environment is None else environment
    rank_raw = values.get("RANK")
    world_size_raw = values.get("WORLD_SIZE")
    if rank_raw is None and world_size_raw is None:
        return 0, 1
    if rank_raw is None or world_size_raw is None:
        raise ValueError("RANK and WORLD_SIZE must either both be set or both be absent")
    try:
        rank = int(rank_raw)
        world_size = int(world_size_raw)
    except ValueError as error:
        raise ValueError("RANK and WORLD_SIZE must be integers") from error
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive; got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"RANK must satisfy 0 <= RANK < WORLD_SIZE; got {rank}/{world_size}")
    return rank, world_size


def _splitmix64(state: int) -> tuple[int, int]:
    state = (state + _GOLDEN_GAMMA) & _MASK_64
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK_64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK_64
    return state, (value ^ (value >> 31)) & _MASK_64


def _stable_permutation(size: int, *, seed: int, epoch: int) -> list[int]:
    values = list(range(size))
    state = ((seed & _MASK_64) ^ (((epoch + 1) * _GOLDEN_GAMMA) & _MASK_64)) & _MASK_64
    for upper in range(size - 1, 0, -1):
        state, random_bits = _splitmix64(state)
        selected = random_bits % (upper + 1)
        values[upper], values[selected] = values[selected], values[upper]
    return values


def stable_permutation(size: int, *, seed: int, epoch: int) -> tuple[int, ...]:
    """Return the repository's stable, dependency-free epoch permutation.

    This public seam lets the map sampler, token-budget batch sampler, and
    streaming shard cursor share exactly one ordering algorithm.  Returning an
    immutable tuple also prevents callers from accidentally changing resume
    state after it has been derived.
    """

    if isinstance(size, bool) or int(size) < 0:
        raise ValueError("size must be a non-negative integer")
    if isinstance(seed, bool):
        raise TypeError("seed must be an integer, not bool")
    if isinstance(epoch, bool) or int(epoch) < 0:
        raise ValueError("epoch must be a non-negative integer")
    return tuple(_stable_permutation(int(size), seed=int(seed), epoch=int(epoch)))


class DeterministicDistributedSampler:
    """A dependency-free map-style sampler with JSON-serializable state.

    State is rank-local and exact only for the same dataset digest, seed,
    topology, and tail policy.  This is intentional: model/optimizer tensors
    may be elastically resharded later, while data order must not silently make
    the same claim.

    The sampler exposes the conventional ``state_dict``/``load_state_dict``
    seam recognized by ``torchdata.stateful_dataloader.StatefulDataLoader``.
    A plain multi-worker ``torch.utils.data.DataLoader`` may prefetch sampler
    indices ahead of delivered batches, so its sampler state alone is not an
    exact worker-level checkpoint.
    """

    def __init__(
        self,
        data_source: Sized,
        *,
        dataset_digest: str | None = None,
        seed: int = 42,
        shuffle: bool = True,
        rank: int | None = None,
        world_size: int | None = None,
        tail_policy: str = "drop",
        epoch: int = 0,
    ) -> None:
        if not isinstance(data_source, Sized):
            raise TypeError("data_source must implement __len__")
        dataset_size = len(data_source)
        if dataset_size <= 0:
            raise ValueError("data_source cannot be empty")

        inferred_rank, inferred_world_size = distributed_context_from_environment()
        resolved_rank = inferred_rank if rank is None else int(rank)
        resolved_world_size = inferred_world_size if world_size is None else int(world_size)
        if resolved_world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 <= resolved_rank < resolved_world_size:
            raise ValueError(f"rank must satisfy 0 <= rank < world_size; got {resolved_rank}/{resolved_world_size}")

        digest = dataset_digest
        if digest is None:
            digest = getattr(data_source, "dataset_digest", None)
        digest = str(digest or "").lower()
        if _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError("dataset_digest must be a 64-character lowercase hexadecimal digest")
        if isinstance(seed, bool):
            raise TypeError("seed must be an integer, not bool")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a bool")
        if isinstance(epoch, bool) or int(epoch) < 0:
            raise ValueError("epoch must be a non-negative integer")
        policy = str(tail_policy).strip().lower().replace("_", "-")
        if policy not in {"drop", "pad", "uneven"}:
            raise ValueError("tail_policy must be 'drop', 'pad', or 'uneven'")
        if policy == "drop" and dataset_size < resolved_world_size:
            raise ValueError(
                "tail_policy='drop' would leave every rank empty because dataset size is smaller than world_size"
            )

        self.data_source = data_source
        self.dataset_size = dataset_size
        self.dataset_digest = digest
        self.seed = int(seed)
        self.shuffle = shuffle
        self.rank = resolved_rank
        self.world_size = resolved_world_size
        self.tail_policy = policy
        self._epoch = int(epoch)
        self._position = 0
        self._cached_epoch: int | None = None
        self._cached_indices: tuple[int, ...] = ()

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
    def global_effective_size(self) -> int:
        remainder = self.dataset_size % self.world_size
        if self.tail_policy == "drop":
            return self.dataset_size - remainder
        if self.tail_policy == "pad" and remainder:
            return self.dataset_size + self.world_size - remainder
        return self.dataset_size

    def _indices_for_epoch(self) -> tuple[int, ...]:
        if self._cached_epoch == self._epoch:
            return self._cached_indices
        if self.shuffle:
            global_indices = _stable_permutation(self.dataset_size, seed=self.seed, epoch=self._epoch)
        else:
            global_indices = list(range(self.dataset_size))

        if self.tail_policy == "drop":
            global_indices = global_indices[: self.global_effective_size]
        elif self.tail_policy == "pad" and len(global_indices) < self.global_effective_size:
            missing = self.global_effective_size - len(global_indices)
            global_indices.extend(global_indices[index % self.dataset_size] for index in range(missing))

        local_indices = tuple(global_indices[self.rank :: self.world_size])
        self._cached_epoch = self._epoch
        self._cached_indices = local_indices
        return local_indices

    @property
    def epoch_indices(self) -> tuple[int, ...]:
        """Return this rank's immutable index order for diagnostics/tests."""

        return self._indices_for_epoch()

    def __len__(self) -> int:
        if self.tail_policy in {"drop", "pad"}:
            return self.global_effective_size // self.world_size
        quotient, remainder = divmod(self.dataset_size, self.world_size)
        return quotient + int(self.rank < remainder)

    def __iter__(self) -> Iterator[int]:
        indices = self._indices_for_epoch()
        if self._position >= len(indices):
            self._move_to_epoch(self._epoch + 1)
            indices = self._indices_for_epoch()
        while self._position < len(indices):
            index = indices[self._position]
            # Advance before yielding so a state captured immediately after the
            # consumer receives the index points at the next unseen sample.
            self._position += 1
            yield index

    def _move_to_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._position = 0
        self._cached_epoch = None
        self._cached_indices = ()

    def set_epoch(self, epoch: int) -> None:
        """Select an epoch; selecting the active epoch preserves resume cursor."""

        if isinstance(epoch, bool) or int(epoch) < 0:
            raise ValueError("epoch must be a non-negative integer")
        resolved = int(epoch)
        if resolved != self._epoch:
            self._move_to_epoch(resolved)

    def advance_epoch(self, *, force: bool = False) -> None:
        if not force and self._position < len(self):
            raise RuntimeError("cannot advance an unfinished sampler epoch without force=True")
        self._move_to_epoch(self._epoch + 1)

    def _next_identity(self) -> tuple[int | None, str | None]:
        indices = self._indices_for_epoch()
        if self._position >= len(indices):
            return None, None
        next_index = indices[self._position]
        sample_ids = getattr(self.data_source, "sample_ids", None)
        if sample_ids is None:
            return next_index, None
        return next_index, str(sample_ids[next_index])

    def state_dict(self) -> dict[str, object]:
        next_index, next_sample_id = self._next_identity()
        return {
            "schema": SAMPLER_STATE_SCHEMA,
            "dataset_digest": self.dataset_digest,
            "dataset_size": self.dataset_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "tail_policy": self.tail_policy,
            "rank": self.rank,
            "world_size": self.world_size,
            "epoch": self._epoch,
            "position": self._position,
            "local_length": len(self),
            "global_effective_size": self.global_effective_size,
            "next_index": next_index,
            "next_sample_id": next_sample_id,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore exact state, rejecting topology/config/data mismatches."""

        if not isinstance(state_dict, Mapping):
            raise TypeError("sampler state_dict must be a mapping")
        required = {
            "schema",
            "dataset_digest",
            "dataset_size",
            "seed",
            "shuffle",
            "tail_policy",
            "rank",
            "world_size",
            "epoch",
            "position",
            "local_length",
            "global_effective_size",
            "next_index",
            "next_sample_id",
        }
        unknown = sorted(set(state_dict) - required)
        missing = sorted(required - set(state_dict))
        if unknown or missing:
            raise SamplerStateMismatchError(f"sampler state fields mismatch; missing={missing}, unknown={unknown}")
        if state_dict["schema"] != SAMPLER_STATE_SCHEMA:
            raise SamplerStateMismatchError(f"unsupported sampler state schema: {state_dict['schema']!r}")

        invariants = {
            "dataset_digest": self.dataset_digest,
            "dataset_size": self.dataset_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "tail_policy": self.tail_policy,
            "rank": self.rank,
            "world_size": self.world_size,
        }
        mismatches = {
            key: (state_dict[key], expected) for key, expected in invariants.items() if state_dict[key] != expected
        }
        if mismatches:
            detail = ", ".join(
                f"{key}=saved:{saved!r}/active:{active!r}" for key, (saved, active) in mismatches.items()
            )
            raise SamplerStateMismatchError(f"sampler exact-resume invariants differ: {detail}")

        epoch = int(state_dict["epoch"])
        position = int(state_dict["position"])
        if epoch < 0:
            raise SamplerStateMismatchError("sampler epoch cannot be negative")

        previous = (self._epoch, self._position, self._cached_epoch, self._cached_indices)
        try:
            self._move_to_epoch(epoch)
            if int(state_dict["local_length"]) != len(self):
                raise SamplerStateMismatchError("saved sampler local_length does not match active geometry")
            if int(state_dict["global_effective_size"]) != self.global_effective_size:
                raise SamplerStateMismatchError("saved global_effective_size does not match active geometry")
            if not 0 <= position <= len(self):
                raise SamplerStateMismatchError(f"sampler position must be in [0, {len(self)}]; got {position}")
            self._position = position

            expected_next_index, expected_next_sample_id = self._next_identity()
            if (
                state_dict["next_index"] != expected_next_index
                or state_dict["next_sample_id"] != expected_next_sample_id
            ):
                raise SamplerStateMismatchError("saved next sample identity does not match the reconstructed order")
        except Exception:  # noqa: BLE001 - restore must be transactional for every validation failure.
            self._epoch, self._position, self._cached_epoch, self._cached_indices = previous
            raise


__all__ = [
    "DeterministicDistributedSampler",
    "SAMPLER_STATE_SCHEMA",
    "SamplerStateMismatchError",
    "distributed_context_from_environment",
    "stable_permutation",
]
