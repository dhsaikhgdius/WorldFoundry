"""Dynamic data-parallel contracts shared by native post-training engines."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn


@dataclass(frozen=True, slots=True)
class PostTrainingParallelContext:
    """The active data-parallel group, derived at process start."""

    rank: int
    world_size: int
    process_group: object | None = None

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or isinstance(self.world_size, bool):
            raise TypeError("parallel rank/world_size must be integers")
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("parallel rank/world_size are inconsistent")
        if self.world_size > 1 and (not dist.is_available() or not dist.is_initialized()):
            raise RuntimeError("multi-rank post-training requires an initialized process group")

    @classmethod
    def current(cls, process_group: object | None = None) -> PostTrainingParallelContext:
        if dist.is_available() and dist.is_initialized():
            return cls(
                rank=dist.get_rank(process_group),
                world_size=dist.get_world_size(process_group),
                process_group=process_group,
            )
        return cls(rank=0, world_size=1)

    def audit_synchronized_module(self, module: nn.Module, *, role: str) -> None:
        """Reject unsynchronized trainable modules in a multi-rank process."""

        if self.world_size == 1:
            return
        from torch.nn.parallel import DistributedDataParallel

        if isinstance(module, DistributedDataParallel):
            return
        try:
            from torch.distributed.fsdp import FSDPModule
            from torch.distributed.tensor import DTensor
        except ImportError:
            FSDPModule = ()  # type: ignore[assignment,misc]
            DTensor = ()  # type: ignore[assignment,misc]
        if isinstance(module, FSDPModule) or all(isinstance(parameter, DTensor) for parameter in module.parameters()):
            return
        raise TypeError(f"multi-rank {role} must be wrapped by DDP or FSDP2 before engine creation")

    def broadcast_from_coordinator(self, value: torch.Tensor) -> torch.Tensor:
        """Broadcast one tensor from rank zero of this data-parallel group."""

        if self.world_size == 1:
            return value
        source = 0 if self.process_group is None else dist.get_global_rank(self.process_group, 0)
        dist.broadcast(value, src=source, group=self.process_group)
        return value

    def scale_local_mean(self, local_mean: torch.Tensor, local_weight: torch.Tensor | float | int) -> torch.Tensor:
        """Scale a local mean for gradient averaging over uneven rank weights."""

        if local_mean.numel() != 1:
            raise ValueError("local_mean must be scalar")
        weight = torch.as_tensor(local_weight, device=local_mean.device, dtype=torch.float32)
        if weight.numel() != 1 or not bool(torch.isfinite(weight)) or not bool(weight > 0):
            raise ValueError("local loss weight must be finite and positive")
        if self.world_size == 1:
            return local_mean
        global_weight = weight.detach().clone()
        dist.all_reduce(global_weight, op=dist.ReduceOp.SUM, group=self.process_group)
        return local_mean * (weight / global_weight * self.world_size)

    def global_standard_deviation(
        self,
        values: torch.Tensor,
        *,
        correction: int,
    ) -> torch.Tensor:
        """Compute one data-parallel standard deviation without gathering samples."""

        if not isinstance(values, torch.Tensor) or values.ndim != 1 or not values.is_floating_point():
            raise TypeError("values must be a one-dimensional floating tensor")
        if values.numel() == 0 or not bool(torch.isfinite(values).all()):
            raise ValueError("values must be non-empty and finite")
        if isinstance(correction, bool) or correction not in {0, 1}:
            raise ValueError("correction must be zero or one")
        values_fp64 = values.detach().to(dtype=torch.float64)
        count = torch.tensor(float(values.numel()), device=values.device, dtype=torch.float64)
        statistics = torch.stack((count, values_fp64.sum(), values_fp64.square().sum()))
        if self.world_size > 1:
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM, group=self.process_group)
        global_count, global_sum, global_square_sum = statistics.unbind()
        if not bool(global_count > correction):
            raise ValueError("global standard deviation has insufficient samples")
        centered_square_sum = global_square_sum - global_sum.square() / global_count
        variance = (centered_square_sum / (global_count - correction)).clamp_min(0.0)
        return variance.sqrt().to(dtype=torch.float32)

    def audit_local_group_ownership(self, group_ids: tuple[str, ...]) -> None:
        """Require every GRPO prompt group to be complete on exactly one rank."""

        if self.world_size == 1:
            return
        local = tuple(sorted(set(group_ids)))
        gathered: list[object] = [None] * self.world_size
        dist.all_gather_object(gathered, local, group=self.process_group)
        owners: dict[str, int] = {}
        duplicated: set[str] = set()
        for rank, values in enumerate(gathered):
            if not isinstance(values, (tuple, list)) or any(not isinstance(value, str) for value in values):
                raise RuntimeError("failed to gather post-training group ownership")
            for value in values:
                if value in owners and owners[value] != rank:
                    duplicated.add(value)
                owners[value] = rank
        if duplicated:
            raise ValueError(f"Flow-GRPO groups cannot be split across data-parallel ranks: {sorted(duplicated)}")


__all__ = ["PostTrainingParallelContext"]
