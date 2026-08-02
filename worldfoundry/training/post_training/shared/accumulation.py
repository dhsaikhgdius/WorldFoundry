"""Weighted microbatch and data-parallel reduction for native post-training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from typing import Protocol

import torch
import torch.distributed as dist
from torch import nn

from .distributed import PostTrainingParallelContext


class WeightedLossAdapter(Protocol):
    """Prepare an exact microbatch denominator before forward collectives."""

    def loss_denominator(self, batch: object, *, role: str) -> object: ...


class WeightedLossResult(Protocol):
    loss: torch.Tensor
    metrics: Mapping[str, object]


def declared_loss_weight(
    loss_adapter: WeightedLossAdapter,
    batch: object,
    *,
    role: str,
    device: torch.device,
) -> torch.Tensor:
    """Resolve a microbatch weight before forward/backward collectives."""

    value = loss_adapter.loss_denominator(batch, role=role)
    weight = torch.as_tensor(value, device=device, dtype=torch.float32)
    if weight.numel() != 1 or not bool(torch.isfinite(weight)) or not bool(weight > 0):
        raise ValueError(f"{role} loss_denominator must be one finite positive scalar")
    return weight.detach().reshape(())


def global_denominator(
    weights: Sequence[torch.Tensor],
    parallel_context: PostTrainingParallelContext,
) -> torch.Tensor:
    """Reduce one optimizer update's denominator before backward."""

    if not weights:
        raise ValueError("accumulation weights cannot be empty")
    global_weight = sum(
        weights,
        torch.zeros((), device=weights[0].device, dtype=torch.float32),
    )
    if parallel_context.world_size > 1:
        dist.all_reduce(
            global_weight,
            op=dist.ReduceOp.SUM,
            group=parallel_context.process_group,
        )
    if not bool(torch.isfinite(global_weight)) or not bool(global_weight > 0):
        raise FloatingPointError("global loss denominator must be finite and positive")
    return global_weight


def check_reported_weight(
    result: WeightedLossResult,
    expected: torch.Tensor,
    *,
    role: str,
) -> None:
    """Reject objectives whose declared and realized reductions diverge."""

    reported = result.metrics.get("loss_denominator")
    if reported is None:
        raise TypeError(f"{role} loss must report loss_denominator")
    value = torch.as_tensor(reported, device=expected.device, dtype=torch.float32)
    if value.numel() != 1 or not torch.equal(value.detach().reshape(()), expected):
        raise RuntimeError(f"{role} loss denominator changed between preparation and reduction")


@contextmanager
def _fsdp_accumulation_context(module: nn.Module, *, final_microbatch: bool):
    module.set_requires_gradient_sync(final_microbatch)  # type: ignore[attr-defined]
    module.set_reshard_after_backward(final_microbatch)  # type: ignore[attr-defined]
    try:
        yield
    except BaseException:
        module.set_requires_gradient_sync(True)  # type: ignore[attr-defined]
        module.set_reshard_after_backward(True)  # type: ignore[attr-defined]
        raise


def accumulation_context(module: nn.Module, *, final_microbatch: bool):
    """Suppress intermediate DDP/FSDP2 reductions during accumulation."""

    try:
        from torch.distributed.fsdp import FSDPModule
    except ImportError:
        FSDPModule = ()  # type: ignore[assignment,misc]
    if isinstance(module, FSDPModule):
        return _fsdp_accumulation_context(module, final_microbatch=final_microbatch)
    if final_microbatch:
        return nullcontext()
    from torch.nn.parallel import DistributedDataParallel

    if isinstance(module, DistributedDataParallel):
        return module.no_sync()
    return nullcontext()


def global_loss_statistics(
    results: Sequence[WeightedLossResult],
    weights: Sequence[torch.Tensor],
    parallel_context: PostTrainingParallelContext,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return detached global numerator, denominator, and weighted mean."""

    if not results or len(results) != len(weights):
        raise ValueError("loss statistics require aligned non-empty results and weights")
    device = results[0].loss.device
    numerator = torch.zeros((), device=device, dtype=torch.float32)
    denominator = torch.zeros((), device=device, dtype=torch.float32)
    for result, weight in zip(results, weights, strict=True):
        numerator = numerator + result.loss.detach().float() * weight.to(device=device)
        denominator = denominator + weight.to(device=device)
    statistics = torch.stack((numerator, denominator))
    if parallel_context.world_size > 1:
        dist.all_reduce(
            statistics,
            op=dist.ReduceOp.SUM,
            group=parallel_context.process_group,
        )
    global_numerator, global_weight = statistics.unbind()
    if not bool(torch.isfinite(global_numerator)) or not bool(torch.isfinite(global_weight)):
        raise FloatingPointError("global loss statistics must be finite")
    if not bool(global_weight > 0):
        raise FloatingPointError("global loss denominator must be positive")
    return global_numerator, global_weight, global_numerator / global_weight


def role_metrics(
    results: Sequence[WeightedLossResult],
    *,
    global_numerator: torch.Tensor,
    global_denominator: torch.Tensor,
) -> dict[str, object]:
    """Preserve final diagnostics and expose the committed reduction."""

    metrics = dict(results[-1].metrics)
    metrics.update(
        {
            "loss_numerator": global_numerator,
            "loss_denominator": global_denominator,
            "microbatches": len(results),
        }
    )
    return metrics


__all__ = [
    "accumulation_context",
    "check_reported_weight",
    "declared_loss_weight",
    "global_denominator",
    "global_loss_statistics",
    "role_metrics",
]
