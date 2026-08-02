"""Tensor synchronization for replicated context-parallel rCM trajectories."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.distributed as dist


@runtime_checkable
class RCMTensorSynchronizer(Protocol):
    """Synchronize a freshly sampled tensor across one replicated group."""

    def synchronize_tensor(self, value: torch.Tensor) -> torch.Tensor: ...


class ProcessGroupRCMTensorSynchronizer:
    """Broadcast random tensors from the first rank of a process group."""

    def __init__(self, process_group: object | None = None) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("process-group rCM synchronization requires initialized distributed")
        ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(process_group))
        if not ranks:
            raise ValueError("rCM synchronization process group is empty")
        self.process_group = process_group
        self.source_rank = min(ranks)
        self.world_size = len(ranks)

    def synchronize_tensor(self, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError("rCM synchronization requires a torch.Tensor")
        if self.world_size > 1:
            dist.broadcast(
                value,
                src=self.source_rank,
                group=self.process_group,
            )
        return value


def synchronize_rcm_tensor(
    value: torch.Tensor,
    synchronizer: RCMTensorSynchronizer | None,
) -> torch.Tensor:
    """Apply an optional synchronization boundary and validate its output."""

    if synchronizer is None:
        return value
    synchronized = synchronizer.synchronize_tensor(value)
    if not isinstance(synchronized, torch.Tensor):
        raise TypeError("rCM synchronizer must return a torch.Tensor")
    if synchronized.shape != value.shape or synchronized.device != value.device:
        raise ValueError("rCM synchronization must preserve tensor shape and device")
    if synchronized.dtype != value.dtype:
        raise ValueError("rCM synchronization must preserve tensor dtype")
    return synchronized


__all__ = [
    "ProcessGroupRCMTensorSynchronizer",
    "RCMTensorSynchronizer",
    "synchronize_rcm_tensor",
]
