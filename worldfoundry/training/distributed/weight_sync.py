"""Full-model and LoRA synchronization from trainers to rollout workers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor


class WeightKind(StrEnum):
    FULL = "full"
    LORA = "lora"


@dataclass(frozen=True, slots=True)
class WeightUpdateHeader:
    """Description sent before the tensors of one staged revision update."""

    revision: int
    kind: WeightKind
    tensor_names: tuple[str, ...]
    bucket_count: int

    def __post_init__(self) -> None:
        if self.revision < 0:
            raise ValueError("weight revision must be non-negative")
        if not self.tensor_names or len(set(self.tensor_names)) != len(self.tensor_names):
            raise ValueError("weight update requires unique tensor names")
        if self.bucket_count <= 0:
            raise ValueError("weight update requires at least one bucket")


@dataclass(frozen=True, slots=True)
class WeightBucket:
    """One bounded transfer unit belonging to a weight update."""

    revision: int
    index: int
    tensors: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        tensors = dict(self.tensors)
        if self.revision < 0 or self.index < 0:
            raise ValueError("weight bucket revision and index must be non-negative")
        if not tensors:
            raise ValueError("weight bucket cannot be empty")
        if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
            raise TypeError("weight buckets can only contain tensors")
        object.__setattr__(self, "tensors", MappingProxyType(tensors))


@dataclass(frozen=True, slots=True)
class WeightSyncReport:
    revision: int
    kind: WeightKind
    tensor_count: int
    byte_count: int
    bucket_count: int
    receiver_count: int
    transmitted: bool


@runtime_checkable
class WeightUpdateReceiver(Protocol):
    def begin_weight_update(self, header: WeightUpdateHeader) -> None: ...

    def write_weight_bucket(self, bucket: WeightBucket) -> None: ...

    def validate_weight_update(self, revision: int) -> None: ...

    def commit_weight_update(self, revision: int) -> None: ...

    def abort_weight_update(self, revision: int) -> None: ...


def _is_lora_tensor(name: str) -> bool:
    lowered = name.lower()
    return "lora_" in lowered or ".modules_to_save." in lowered


def _materialize_tensor(value: torch.Tensor, *, transmit: bool) -> torch.Tensor | None:
    if isinstance(value, DTensor):
        value = value.full_tensor()
    if not transmit:
        return None
    return value.detach().to(device="cpu", copy=True).contiguous()


def materialize_weight_tensors(
    module: nn.Module,
    *,
    kind: WeightKind | str,
    source_rank: int = 0,
    name_transform: Callable[[str], str] | None = None,
) -> dict[str, torch.Tensor]:
    """Materialize a deterministic CPU state on the transmitting rank.

    Every distributed rank must call this function. DTensor ``full_tensor`` is
    collective even though only ``source_rank`` keeps and transmits the result.
    """

    if not isinstance(module, nn.Module):
        raise TypeError("weight synchronization source must be an nn.Module")
    resolved_kind = WeightKind(str(kind).strip().lower())
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    if distributed and not 0 <= source_rank < dist.get_world_size():
        raise ValueError("source_rank is outside the active process group")
    transmit = rank == source_rank
    selected: dict[str, torch.Tensor] = {}
    for name, value in sorted(module.state_dict().items()):
        if not isinstance(value, torch.Tensor):
            continue
        if resolved_kind is WeightKind.LORA and not _is_lora_tensor(name):
            continue
        materialized = _materialize_tensor(value, transmit=transmit)
        if materialized is not None:
            target_name = name_transform(name) if name_transform is not None else name
            if target_name in selected:
                raise ValueError(f"weight name transform produced duplicate key {target_name!r}")
            selected[target_name] = materialized
    if transmit and not selected:
        raise ValueError(f"no {resolved_kind.value} tensors were selected for synchronization")
    return selected


def build_weight_buckets(
    tensors: Mapping[str, torch.Tensor],
    *,
    revision: int,
    max_bucket_bytes: int,
) -> tuple[WeightBucket, ...]:
    """Partition tensors without splitting an individual tensor."""

    limit = int(max_bucket_bytes)
    if limit <= 0:
        raise ValueError("max_bucket_bytes must be positive")
    buckets: list[WeightBucket] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0
    for name, tensor in sorted(tensors.items()):
        size = tensor.numel() * tensor.element_size()
        if current and current_bytes + size > limit:
            buckets.append(WeightBucket(revision=revision, index=len(buckets), tensors=current))
            current = {}
            current_bytes = 0
        current[name] = tensor
        current_bytes += size
    if current:
        buckets.append(WeightBucket(revision=revision, index=len(buckets), tensors=current))
    if not buckets:
        raise ValueError("cannot build buckets from an empty tensor mapping")
    return tuple(buckets)


class ModuleWeightReceiver:
    """Stage every bucket in CPU memory before loading it into a module."""

    def __init__(self, module: nn.Module) -> None:
        if not isinstance(module, nn.Module):
            raise TypeError("module must be an nn.Module")
        self.module = module
        self.last_revision = -1
        self._header: WeightUpdateHeader | None = None
        self._buckets: dict[int, Mapping[str, torch.Tensor]] = {}

    def begin_weight_update(self, header: WeightUpdateHeader) -> None:
        if not isinstance(header, WeightUpdateHeader):
            raise TypeError("header must be WeightUpdateHeader")
        if header.revision <= self.last_revision:
            raise ValueError(f"weight revision {header.revision} is not newer than {self.last_revision}")
        if self._header is not None:
            raise RuntimeError("another weight update is already active")
        self._header = header
        self._buckets = {}

    def write_weight_bucket(self, bucket: WeightBucket) -> None:
        if self._header is None or bucket.revision != self._header.revision:
            raise ValueError("weight bucket does not belong to the active update")
        if bucket.index >= self._header.bucket_count:
            raise ValueError("weight bucket index exceeds the declared bucket count")
        if bucket.index in self._buckets:
            raise ValueError(f"weight bucket {bucket.index} was already received")
        self._buckets[bucket.index] = {
            name: tensor.detach().to(device="cpu", copy=True) for name, tensor in bucket.tensors.items()
        }

    def _validated_state(self, revision: int) -> dict[str, torch.Tensor]:
        header = self._header
        if header is None or revision != header.revision:
            raise ValueError("no matching weight update is active")
        if set(self._buckets) != set(range(header.bucket_count)):
            raise ValueError("weight update is missing one or more buckets")
        state: dict[str, torch.Tensor] = {}
        for index in range(header.bucket_count):
            state.update(self._buckets[index])
        if tuple(sorted(state)) != tuple(sorted(header.tensor_names)):
            raise ValueError("received tensor names differ from the update header")

        current = self.module.state_dict()
        current_names = set(current)
        received_names = set(state)
        if header.kind is WeightKind.FULL and received_names != current_names:
            missing = sorted(current_names - received_names)
            unexpected = sorted(received_names - current_names)
            raise ValueError(f"full weight update differs from the module: missing={missing}, unexpected={unexpected}")
        unknown = received_names - current_names
        if unknown:
            raise ValueError(f"weight update contains unknown tensors: {sorted(unknown)}")
        mismatched_shapes = {
            name: (tuple(state[name].shape), tuple(current[name].shape))
            for name in state
            if tuple(state[name].shape) != tuple(current[name].shape)
        }
        if mismatched_shapes:
            raise ValueError(f"weight update contains incompatible tensor shapes: {mismatched_shapes}")
        return state

    def validate_weight_update(self, revision: int) -> None:
        """Validate a staged update without changing live module weights."""

        self._validated_state(revision)

    def commit_weight_update(self, revision: int) -> None:
        header = self._header
        if header is None:
            raise ValueError("no matching weight update is active")
        state = self._validated_state(revision)
        self.module.load_state_dict(state, strict=header.kind is WeightKind.FULL)
        self.last_revision = revision
        self._header = None
        self._buckets = {}

    def abort_weight_update(self, revision: int) -> None:
        if self._header is not None and self._header.revision == revision:
            self._header = None
            self._buckets = {}


class NativeWeightSynchronizer:
    """Collectively gather once, validate every receiver, then commit a revision."""

    def __init__(self, *, max_bucket_bytes: int = 256 * 1024 * 1024, source_rank: int = 0) -> None:
        if max_bucket_bytes <= 0:
            raise ValueError("max_bucket_bytes must be positive")
        self.max_bucket_bytes = int(max_bucket_bytes)
        self.source_rank = int(source_rank)

    def sync(
        self,
        module: nn.Module,
        receivers: Sequence[WeightUpdateReceiver],
        *,
        revision: int,
        kind: WeightKind | str,
        name_transform: Callable[[str], str] | None = None,
    ) -> WeightSyncReport:
        if revision < 0:
            raise ValueError("weight revision must be non-negative")
        resolved_kind = WeightKind(str(kind).strip().lower())
        tensors = materialize_weight_tensors(
            module,
            kind=resolved_kind,
            source_rank=self.source_rank,
            name_transform=name_transform,
        )
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        transmit = rank == self.source_rank
        if not transmit:
            return WeightSyncReport(
                revision=revision,
                kind=resolved_kind,
                tensor_count=0,
                byte_count=0,
                bucket_count=0,
                receiver_count=len(receivers),
                transmitted=False,
            )

        buckets = build_weight_buckets(
            tensors,
            revision=revision,
            max_bucket_bytes=self.max_bucket_bytes,
        )
        header = WeightUpdateHeader(
            revision=revision,
            kind=resolved_kind,
            tensor_names=tuple(sorted(tensors)),
            bucket_count=len(buckets),
        )
        started: list[WeightUpdateReceiver] = []
        try:
            for receiver in receivers:
                receiver.begin_weight_update(header)
                started.append(receiver)
            for bucket in buckets:
                for receiver in receivers:
                    receiver.write_weight_bucket(bucket)
            for receiver in receivers:
                receiver.validate_weight_update(revision)
            for receiver in receivers:
                receiver.commit_weight_update(revision)
        except Exception:
            for receiver in started:
                receiver.abort_weight_update(revision)
            raise

        return WeightSyncReport(
            revision=revision,
            kind=resolved_kind,
            tensor_count=len(tensors),
            byte_count=sum(value.numel() * value.element_size() for value in tensors.values()),
            bucket_count=len(buckets),
            receiver_count=len(receivers),
            transmitted=True,
        )


__all__ = [
    "ModuleWeightReceiver",
    "NativeWeightSynchronizer",
    "WeightBucket",
    "WeightKind",
    "WeightSyncReport",
    "WeightUpdateHeader",
    "WeightUpdateReceiver",
    "build_weight_buckets",
    "materialize_weight_tensors",
]
