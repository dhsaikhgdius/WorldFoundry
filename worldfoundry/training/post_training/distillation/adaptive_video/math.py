"""Pure tensor equations for adaptive video distillation.

Key formulas:
  - Temporal variance regularization: L_temp = -log(Var_frames(x) + eps), applied when L >= cutoff
  - Adaptive regression weight: w_slot = 1 - sigmoid(sensitivity * (L_slot - EMA_slot))
  - Slot EMA update: EMA <- decay * EMA + (1 - decay) * mean(L_slot)

References:
  - DMD (base distribution matching): https://arxiv.org/abs/2311.18828
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "adaptive video distillation requires the 'train-core' extra"
        ) from error
    return torch


@dataclass(frozen=True, slots=True)
class TemporalRegularizationResult:
    applied_loss: object
    raw_loss: object
    motion_metric: object


@dataclass(frozen=True, slots=True)
class AdaptiveRegressionObservation:
    loss_sums: object
    sample_counts: object


@dataclass(frozen=True, slots=True)
class AdaptiveRegressionWeightResult:
    weights: object
    tentative_ema: object
    observation: AdaptiveRegressionObservation


def temporal_variance_regularization(
    generated_video: object,
    *,
    frame_axis: int = 1,
    epsilon: float = 1.0e-6,
    cutoff: float = 0.8,
) -> TemporalRegularizationResult:
    """Penalize videos whose across-frame variance is below the cutoff."""

    torch = _require_torch()
    if not torch.is_tensor(generated_video) or generated_video.ndim < 3:
        raise TypeError("generated_video must be a [B,F,...] torch.Tensor")
    axis = int(frame_axis)
    if axis < 0:
        axis += generated_video.ndim
    if axis <= 0 or axis >= generated_video.ndim:
        raise ValueError("frame_axis must select a non-batch video dimension")
    if int(generated_video.shape[axis]) < 2:
        raise ValueError("temporal regularization requires at least two frames")
    resolved_epsilon = float(epsilon)
    resolved_cutoff = float(cutoff)
    if not isfinite(resolved_epsilon) or resolved_epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not isfinite(resolved_cutoff):
        raise ValueError("cutoff must be finite")
    variance = torch.var(generated_video.double(), dim=axis, unbiased=False)
    motion_metric = variance.mean()
    raw_loss = -torch.log(motion_metric + resolved_epsilon)
    applied_loss = torch.where(
        raw_loss >= resolved_cutoff,
        raw_loss,
        torch.zeros_like(raw_loss),
    )
    return TemporalRegularizationResult(
        applied_loss=applied_loss,
        raw_loss=raw_loss,
        motion_metric=motion_metric,
    )


def adaptive_regression_weights(
    per_sample_losses: object,
    schedule_indices: object,
    running_ema: object,
    initialized: object,
    *,
    decay: float,
    sensitivity: float,
) -> AdaptiveRegressionWeightResult:
    """Compute slot-wise weights without mutating the running EMA state."""

    torch = _require_torch()
    if not torch.is_tensor(per_sample_losses) or per_sample_losses.ndim != 1:
        raise TypeError("per_sample_losses must be a one-dimensional tensor")
    if not per_sample_losses.is_floating_point() or not bool(
        torch.isfinite(per_sample_losses).all()
    ):
        raise ValueError("per_sample_losses must be finite floating-point values")
    if not torch.is_tensor(schedule_indices) or schedule_indices.shape != per_sample_losses.shape:
        raise ValueError("schedule_indices must match per_sample_losses")
    if schedule_indices.dtype == torch.bool or schedule_indices.is_floating_point():
        raise TypeError("schedule_indices must use an integer dtype")
    if not torch.is_tensor(running_ema) or running_ema.ndim != 1:
        raise TypeError("running_ema must be a one-dimensional tensor")
    if not torch.is_tensor(initialized) or initialized.shape != running_ema.shape:
        raise ValueError("initialized must match running_ema")
    if initialized.dtype != torch.bool:
        raise TypeError("initialized must be boolean")
    resolved_decay = float(decay)
    resolved_sensitivity = float(sensitivity)
    if not isfinite(resolved_decay) or not 0.0 <= resolved_decay < 1.0:
        raise ValueError("decay must be in [0,1)")
    if not isfinite(resolved_sensitivity) or resolved_sensitivity <= 0:
        raise ValueError("sensitivity must be finite and positive")
    slot_count = int(running_ema.numel())
    indices = schedule_indices.to(device=per_sample_losses.device, dtype=torch.long)
    if not bool(((indices >= 0) & (indices < slot_count)).all()):
        raise ValueError("schedule_indices fall outside the running EMA")
    ema = running_ema.to(
        device=per_sample_losses.device,
        dtype=per_sample_losses.dtype,
    )
    flags = initialized.to(device=per_sample_losses.device)
    weights = torch.empty_like(per_sample_losses)
    tentative = ema.clone()
    sums = torch.zeros_like(ema)
    counts = torch.zeros(slot_count, device=ema.device, dtype=torch.int64)
    for index in torch.unique(indices, sorted=True).tolist():
        slot = int(index)
        mask = indices == slot
        current = per_sample_losses[mask].mean()
        candidate = (
            resolved_decay * ema[slot] + (1.0 - resolved_decay) * current
            if bool(flags[slot])
            else current
        )
        tentative[slot] = candidate
        weights[mask] = 1.0 - torch.sigmoid(
            resolved_sensitivity * (current.detach() - candidate.detach())
        )
        sums[slot] = per_sample_losses[mask].detach().sum()
        counts[slot] = mask.sum()
    return AdaptiveRegressionWeightResult(
        weights=weights,
        tentative_ema=tentative,
        observation=AdaptiveRegressionObservation(
            loss_sums=sums,
            sample_counts=counts,
        ),
    )


__all__ = [
    "AdaptiveRegressionObservation",
    "AdaptiveRegressionWeightResult",
    "TemporalRegularizationResult",
    "adaptive_regression_weights",
    "temporal_variance_regularization",
]
