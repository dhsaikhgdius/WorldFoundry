"""Run summaries and distributed parameter-change statistics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


@dataclass(frozen=True, slots=True)
class SingleDeviceRunSummary:
    optimizer_steps: int
    microbatches: int
    sample_count: int
    latent_token_count: int
    initial_loss: float
    final_loss: float
    best_loss: float
    loss_reduction_fraction: float
    wall_time_seconds: float
    samples_per_second: float
    latent_tokens_per_second: float
    changed_parameter_tensors: int
    parameter_delta_l2: float
    parameter_delta_max_abs: float
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    overfit_gate_passed: bool | None

    def to_dict(self) -> dict[str, object]:
        return {
            "optimizer_steps": self.optimizer_steps,
            "microbatches": self.microbatches,
            "sample_count": self.sample_count,
            "latent_token_count": self.latent_token_count,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "loss_reduction_fraction": self.loss_reduction_fraction,
            "wall_time_seconds": self.wall_time_seconds,
            "samples_per_second": self.samples_per_second,
            "latent_tokens_per_second": self.latent_tokens_per_second,
            "changed_parameter_tensors": self.changed_parameter_tensors,
            "parameter_delta_l2": self.parameter_delta_l2,
            "parameter_delta_max_abs": self.parameter_delta_max_abs,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "overfit_gate_passed": self.overfit_gate_passed,
        }


class OverfitGateError(RuntimeError):
    """Raised after a completed run fails its explicit loss/parameter gate."""

    def __init__(self, summary: SingleDeviceRunSummary) -> None:
        self.summary = summary
        super().__init__(
            "one-batch overfit gate failed: "
            f"initial_loss={summary.initial_loss:.8g}, "
            f"final_loss={summary.final_loss:.8g}, "
            f"changed_parameter_tensors={summary.changed_parameter_tensors}"
        )


def _local_parameter(parameter: torch.nn.Parameter) -> torch.Tensor:
    value = parameter.detach()
    return value.to_local() if isinstance(value, DTensor) else value


def parameter_snapshot(
    parameters: Sequence[torch.nn.Parameter],
) -> tuple[torch.Tensor, ...]:
    """Capture local FP32 parameter shards before a run."""

    return tuple(_local_parameter(parameter).float().cpu().clone() for parameter in parameters)


def parameter_delta(
    parameters: Sequence[torch.nn.Parameter],
    before: Sequence[torch.Tensor],
    *,
    device: torch.device,
    distributed: bool,
) -> tuple[int, float, float]:
    """Measure changed tensors and global L2/max deltas after a run."""

    changed_flags: list[int] = []
    squared_sum = 0.0
    maximum = 0.0
    for parameter, reference in zip(parameters, before):
        delta = _local_parameter(parameter).float().cpu() - reference
        changed_flags.append(int(bool(torch.count_nonzero(delta))))
        squared_sum += float(delta.double().square().sum())
        if delta.numel():
            maximum = max(maximum, float(delta.abs().max()))
    if not distributed:
        return sum(changed_flags), math.sqrt(squared_sum), maximum

    flags = torch.tensor(changed_flags, device=device, dtype=torch.int64)
    dist.all_reduce(flags, op=dist.ReduceOp.MAX)
    squared = torch.tensor(squared_sum, device=device, dtype=torch.float64)
    dist.all_reduce(squared, op=dist.ReduceOp.SUM)
    largest = torch.tensor(maximum, device=device, dtype=torch.float64)
    dist.all_reduce(largest, op=dist.ReduceOp.MAX)
    return int(flags.sum()), math.sqrt(float(squared)), float(largest)


__all__ = [
    "OverfitGateError",
    "SingleDeviceRunSummary",
    "parameter_delta",
    "parameter_snapshot",
]
