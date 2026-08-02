"""Shared, architecture-neutral FlowMap regression used by AnyFlow students."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor

from ...shared.distributed import PostTrainingParallelContext
from .config import AnyFlowMapConfig
from .contracts import AnyFlowTrainingBatch
from .math import (
    allocate_flowmap_intervals,
    balance_flowmap_losses,
    beta08_train_weight,
    flowmap_central_difference,
    flowmap_interpolate,
    flowmap_target,
    fused_guidance_prediction,
    shift_flowmap_time,
)

FlowMapPrediction = Callable[
    [Tensor, Tensor, Tensor, Mapping[str, object], bool, str],
    Tensor,
]


@dataclass(frozen=True, slots=True)
class FlowMapRegressionResult:
    loss: Tensor
    metrics: Mapping[str, object]


def _global_layout(
    local_batch: int,
    reference: Tensor,
    context: PostTrainingParallelContext,
) -> tuple[int, int]:
    if context.world_size == 1:
        return 0, local_batch
    local = torch.tensor([local_batch], device=reference.device, dtype=torch.int64)
    gathered = [torch.empty_like(local) for _ in range(context.world_size)]
    dist.all_gather(gathered, local, group=context.process_group)
    counts = tuple(int(value.item()) for value in gathered)
    if any(value <= 0 for value in counts):
        raise ValueError("AnyFlow requires a positive local batch on every rank")
    return sum(counts[: context.rank]), sum(counts)


def _global_diffusion_mean(
    losses: Tensor,
    mask: Tensor,
    context: PostTrainingParallelContext,
) -> Tensor:
    local_sum = losses.detach()[mask].double().sum()
    local_count = torch.tensor(
        float(mask.sum().item()),
        device=losses.device,
        dtype=torch.float64,
    )
    statistics = torch.stack((local_sum, local_count))
    if context.world_size > 1:
        dist.all_reduce(
            statistics,
            op=dist.ReduceOp.SUM,
            group=context.process_group,
        )
    total, count = statistics.unbind()
    if not bool(count > 0):
        raise ValueError("AnyFlow global batch contains no diffusion anchor; increase batch size or diffusion_ratio")
    return (total / count).to(dtype=losses.dtype)


def flowmap_regression_loss(
    clean: Tensor,
    batch: AnyFlowTrainingBatch,
    config: AnyFlowMapConfig,
    prediction: FlowMapPrediction,
    *,
    parallel_context: PostTrainingParallelContext,
    generator: torch.Generator | None,
    condition_first_frame: bool = False,
) -> FlowMapRegressionResult:
    """Compute the released interval-mixture/JVP-target FlowMap objective."""

    if not isinstance(clean, Tensor) or clean.ndim != 5 or not clean.is_floating_point():
        raise TypeError("AnyFlow FlowMap clean latents must be floating BCTHW")
    if int(clean.shape[0]) != batch.batch_size:
        raise ValueError("AnyFlow FlowMap clean latents and batch IDs differ")
    batch_size, frames = int(clean.shape[0]), int(clean.shape[2])
    noise = torch.randn(
        clean.shape,
        device=clean.device,
        dtype=clean.dtype,
        generator=generator,
    )
    first = torch.rand(
        (batch_size,),
        device=clean.device,
        dtype=clean.dtype,
        generator=generator,
    )
    second = torch.rand(
        (batch_size,),
        device=clean.device,
        dtype=clean.dtype,
        generator=generator,
    )
    global_start, global_batch = _global_layout(
        batch_size,
        clean,
        parallel_context,
    )
    raw_t, raw_r, diffusion_mask = allocate_flowmap_intervals(
        first,
        second,
        global_start_index=global_start,
        global_batch_size=global_batch,
        diffusion_ratio=config.diffusion_ratio,
        consistency_ratio=config.consistency_ratio,
    )
    shifted_t = shift_flowmap_time(raw_t, config.timestep_shift)
    shifted_r = shift_flowmap_time(raw_r, config.timestep_shift)
    model_t = shifted_t[:, None].expand(batch_size, frames).clone() * float(config.num_train_timesteps)
    model_r = shifted_r[:, None].expand(batch_size, frames) * float(config.num_train_timesteps)
    if condition_first_frame:
        model_t[:, 0] = 0
    normalized_t = model_t / float(config.num_train_timesteps)
    noisy = flowmap_interpolate(clean, noise, normalized_t)
    conditional = prediction(
        noisy,
        model_t,
        model_r,
        batch.conditioning,
        True,
        "positive",
    )
    if not isinstance(conditional, Tensor) or conditional.shape != noisy.shape:
        raise ValueError("AnyFlow FlowMap prediction must preserve the latent shape")
    guidance = config.fused_guidance_scale
    if guidance == 1.0:
        predicted = conditional
    else:
        with torch.no_grad():
            unconditional = prediction(
                noisy,
                model_t,
                model_r,
                batch.unconditional_conditioning,
                False,
                "negative",
            )
        if not isinstance(unconditional, Tensor) or unconditional.shape != noisy.shape:
            raise ValueError("AnyFlow FlowMap prediction must preserve the latent shape")
        predicted = fused_guidance_prediction(
            conditional,
            unconditional,
            guidance,
        )

    epsilon = config.central_difference_epsilon
    state_delta = (noise - clean) * (epsilon / float(config.num_train_timesteps))
    with torch.no_grad():
        plus = prediction(
            noisy + state_delta,
            model_t + epsilon,
            model_r,
            batch.conditioning,
            False,
            "positive",
        )
        minus = prediction(
            noisy - state_delta,
            model_t - epsilon,
            model_r,
            batch.conditioning,
            False,
            "positive",
        )
        derivative = flowmap_central_difference(
            plus,
            minus,
            epsilon=epsilon,
            guidance_scale=guidance,
        )
        target = flowmap_target(clean, noise, derivative, model_t, model_r)

    per_sample = (predicted.float() - target.float()).square().flatten(1).mean(1)
    weights = beta08_train_weight(
        model_t,
        num_train_timesteps=config.num_train_timesteps,
        shift=config.timestep_shift,
    ).mean(dim=1)
    weighted = per_sample * weights
    diffusion_mean = _global_diffusion_mean(
        weighted,
        diffusion_mask,
        parallel_context,
    )
    balanced = balance_flowmap_losses(
        weighted,
        diffusion_mask,
        diffusion_mean,
    )
    loss = balanced.mean()
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError("non-finite AnyFlow FlowMap regression loss")
    return FlowMapRegressionResult(
        loss=loss,
        metrics={
            "diffusion_samples": diffusion_mask.sum().detach(),
            "global_diffusion_mean": diffusion_mean.detach(),
            "raw_t_mean": raw_t.detach().mean(),
            "raw_r_mean": raw_r.detach().mean(),
            "first_frame_conditioned": condition_first_frame,
        },
    )


__all__ = ["FlowMapRegressionResult", "flowmap_regression_loss"]
