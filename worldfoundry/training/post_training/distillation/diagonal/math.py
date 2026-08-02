"""Pure tensor equations used by diagonal distribution matching.

Key formulas:
  - Spatial DMD gradient: g_s = (x0_fake - x0_real) / mean|x_gen - x0_real|
  - Motion DMD gradient: g_m = Delta(x0_fake) - Delta(x0_real), normalized by |Delta(x_gen)-Delta(x_real)|
  - Motion weights: w = 0.7 * dynamic_cumulative_error + 0.3 * 1.2^frame / mean
  - Spatial proxy: L_s = 0.5 * ||x_gen - stop(x_gen - g_s)||^2
  - Motion proxy: L_m = mean(w * ||Delta(x_gen) - stop(Delta(x_gen) - g_m)||^2)

References:
  - Diagonal Distillation: https://arxiv.org/abs/2603.09488
  - DMD: https://arxiv.org/abs/2311.18828
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _resolved_frame_dim(reference: Tensor, frame_dim: int) -> int:
    if not isinstance(reference, Tensor) or reference.ndim < 3:
        raise TypeError("video values must be batched torch.Tensor objects with at least three dimensions")
    if isinstance(frame_dim, bool) or not isinstance(frame_dim, int):
        raise TypeError("frame_dim must be an integer")
    resolved = frame_dim % reference.ndim
    if resolved == 0:
        raise ValueError("frame_dim cannot resolve to the batch dimension")
    return resolved


def _frames_first(reference: Tensor, frame_dim: int) -> tuple[Tensor, int]:
    resolved = _resolved_frame_dim(reference, frame_dim)
    return reference.movedim(resolved, 1), resolved


@dataclass(frozen=True, slots=True)
class DiagonalDistributionGradients:
    spatial: Tensor
    motion: Tensor
    spatial_normalizer: Tensor
    motion_normalizer: Tensor


@dataclass(frozen=True, slots=True)
class DiagonalProxyLosses:
    spatial: Tensor
    motion: Tensor
    spatial_target: Tensor
    motion_target: Tensor
    motion_weights: Tensor


def diagonal_distribution_gradients(
    generated_clean: Tensor,
    fake_score_clean: Tensor,
    real_score_clean: Tensor,
    *,
    frame_dim: int = 2,
    normalization_epsilon: float = 0.0,
) -> DiagonalDistributionGradients:
    """Compute released spatial and temporal-difference DMD gradients."""

    if not all(isinstance(value, Tensor) for value in (generated_clean, fake_score_clean, real_score_clean)):
        raise TypeError("diagonal DMD inputs must be torch.Tensor values")
    if generated_clean.shape != fake_score_clean.shape or generated_clean.shape != real_score_clean.shape:
        raise ValueError("diagonal DMD inputs must share a shape")
    generated, resolved = _frames_first(generated_clean, frame_dim)
    fake = fake_score_clean.movedim(resolved, 1)
    real = real_score_clean.movedim(resolved, 1)
    epsilon = float(normalization_epsilon)
    if not isfinite(epsilon) or epsilon < 0:
        raise ValueError("normalization_epsilon must be finite and non-negative")

    spatial_normalizer = (generated.float() - real.float()).abs().mean(
        dim=tuple(range(1, generated.ndim)),
        keepdim=True,
    )
    spatial_denominator = spatial_normalizer.clamp_min(epsilon) if epsilon > 0 else spatial_normalizer
    spatial = (fake.float() - real.float()) / spatial_denominator
    spatial = torch.nan_to_num(spatial)

    if int(generated.shape[1]) > 1:
        fake_motion = fake[:, 1:] - fake[:, :-1]
        real_motion = real[:, 1:] - real[:, :-1]
        generated_motion = generated[:, 1:] - generated[:, :-1]
        motion = fake_motion.float() - real_motion.float()
        motion_normalizer = (generated_motion.float() - real_motion.float()).abs().mean(
            dim=tuple(range(1, generated_motion.ndim)),
            keepdim=True,
        )
        motion_denominator = motion_normalizer.clamp_min(epsilon) if epsilon > 0 else motion_normalizer
        motion = torch.nan_to_num(motion / motion_denominator)
    else:
        motion = spatial[:, :0]
        motion_normalizer = spatial_normalizer[:, :0]
    return DiagonalDistributionGradients(
        spatial=spatial.movedim(1, resolved),
        motion=motion.movedim(1, resolved),
        spatial_normalizer=spatial_normalizer.movedim(1, resolved),
        motion_normalizer=motion_normalizer.movedim(1, resolved),
    )


def dynamic_motion_weights(
    motion_gradient: Tensor,
    generated_clean: Tensor,
    *,
    frame_dim: int = 2,
) -> Tensor:
    """Released cumulative-error weight, detached per sample and frame pair."""

    motion, motion_dim = _frames_first(motion_gradient, frame_dim)
    generated, generated_dim = _frames_first(generated_clean, frame_dim)
    if motion_dim != generated_dim:
        raise RuntimeError("resolved frame dimensions unexpectedly differ")
    if int(generated.shape[1]) != int(motion.shape[1]) + 1:
        raise ValueError("motion_gradient must have one fewer frame than generated_clean")
    if motion.shape != generated[:, 1:].shape:
        raise ValueError("motion_gradient shape differs from generated next-frame values")
    per_frame = F.mse_loss(motion, generated[:, 1:], reduction="none").mean(
        dim=tuple(range(2, motion.ndim))
    )
    cumulative = torch.cumsum(per_frame, dim=1)
    denominator = cumulative[:, -1:].clamp_min(1.0e-6)
    return (1.0 + cumulative / denominator).detach()


def exponential_motion_weights(
    frame_pairs: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Released normalized ``1.2 ** frame_index`` temporal prior."""

    if isinstance(frame_pairs, bool) or int(frame_pairs) <= 0:
        raise ValueError("frame_pairs must be a positive integer")
    indices = torch.arange(int(frame_pairs), device=device, dtype=dtype)
    weights = torch.pow(torch.tensor(1.2, device=device, dtype=dtype), indices)
    return weights / weights.mean()


def hybrid_motion_weights(
    motion_gradient: Tensor,
    generated_clean: Tensor,
    *,
    frame_dim: int = 2,
) -> Tensor:
    """Combine the released 0.7 dynamic and 0.3 exponential weights."""

    dynamic = dynamic_motion_weights(
        motion_gradient,
        generated_clean,
        frame_dim=frame_dim,
    )
    exponential = exponential_motion_weights(
        int(dynamic.shape[1]),
        device=dynamic.device,
        dtype=dynamic.dtype,
    ).view(1, -1)
    return 0.7 * dynamic + 0.3 * exponential.expand_as(dynamic)


def diagonal_proxy_losses(
    generated_clean: Tensor,
    gradients: DiagonalDistributionGradients,
    *,
    gradient_mask: Tensor | None = None,
    frame_dim: int = 2,
) -> DiagonalProxyLosses:
    """Construct exact spatial and weighted motion DMD proxy losses."""

    if not isinstance(gradients, DiagonalDistributionGradients):
        raise TypeError("gradients must be DiagonalDistributionGradients")
    if generated_clean.shape != gradients.spatial.shape:
        raise ValueError("spatial gradient must match generated_clean")
    generated, resolved = _frames_first(generated_clean, frame_dim)
    spatial_gradient = gradients.spatial.movedim(resolved, 1)
    spatial_target = (generated.double() - spatial_gradient.double()).detach()
    if gradient_mask is not None:
        if not isinstance(gradient_mask, Tensor) or gradient_mask.dtype != torch.bool:
            raise TypeError("gradient_mask must be a boolean torch.Tensor")
        if gradient_mask.shape != generated_clean.shape:
            raise ValueError("gradient_mask must match generated_clean")
        mask = gradient_mask.movedim(resolved, 1)
        if not bool(mask.any()):
            raise ValueError("gradient_mask must select at least one value")
        spatial_loss = 0.5 * F.mse_loss(
            generated.double()[mask],
            spatial_target[mask],
            reduction="mean",
        )
    else:
        spatial_loss = 0.5 * F.mse_loss(generated.double(), spatial_target, reduction="mean")

    motion_gradient = gradients.motion.movedim(resolved, 1)
    if int(generated.shape[1]) > 1:
        weights = hybrid_motion_weights(
            gradients.motion,
            generated_clean,
            frame_dim=frame_dim,
        )
        predicted_motion = (generated[:, 1:] - generated[:, :-1]).double()
        motion_target = (predicted_motion - motion_gradient.double()).detach()
        weight_shape = (int(weights.shape[0]), int(weights.shape[1])) + (1,) * (generated.ndim - 2)
        motion_loss = ((predicted_motion - motion_target).square() * weights.view(weight_shape).double()).mean()
    else:
        weights = generated.new_empty((int(generated.shape[0]), 0), dtype=torch.float32)
        motion_target = generated.double()[:, :0]
        motion_loss = spatial_loss * 0.0
    return DiagonalProxyLosses(
        spatial=spatial_loss,
        motion=motion_loss,
        spatial_target=spatial_target.movedim(1, resolved),
        motion_target=motion_target.movedim(1, resolved),
        motion_weights=weights,
    )


def diagonal_regression_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    gradient_mask: Tensor | None,
    loss_type: str,
    epsilon: float,
    cauchy_scale: float,
) -> Tensor:
    """Released optional fixed-teacher regression reductions."""

    if prediction.shape != target.shape:
        raise ValueError("regression prediction and target must share a shape")
    if gradient_mask is not None:
        if gradient_mask.dtype != torch.bool or gradient_mask.shape != prediction.shape:
            raise ValueError("regression gradient_mask must be boolean and match prediction")
        if not bool(gradient_mask.any()):
            raise ValueError("regression gradient_mask must select at least one value")
        prediction = prediction[gradient_mask]
        target = target[gradient_mask]
    if loss_type == "mse":
        return F.mse_loss(prediction, target, reduction="mean")
    difference = prediction - target
    if loss_type == "charbonnier":
        return torch.sqrt(difference.square() + float(epsilon) ** 2).mean()
    if loss_type == "cauchy":
        return torch.log1p((difference / float(cauchy_scale)).square()).mean()
    raise ValueError(f"unsupported diagonal regression loss: {loss_type!r}")


def diagonal_flow_regression_loss(
    generated_clean: Tensor,
    regression_target: Tensor,
    student_head: nn.Module,
    teacher_head: nn.Module,
    *,
    gradient_mask: Tensor | None,
    frame_dim: int = 2,
) -> Tensor:
    """EMA motion-head regression on adjacent-frame deltas."""

    generated, resolved = _frames_first(generated_clean, frame_dim)
    target = regression_target.movedim(resolved, 1)
    if generated.shape != target.shape:
        raise ValueError("flow regression target must match generated_clean")
    if int(generated.shape[1]) < 2:
        return generated.sum() * 0.0
    student_delta = generated[:, 1:] - generated[:, :-1]
    teacher_delta = target[:, 1:] - target[:, :-1]
    try:
        student_parameter = next(student_head.parameters())
        teacher_parameter = next(teacher_head.parameters())
    except StopIteration as error:
        raise ValueError("motion heads must expose parameters") from error
    student_input = student_delta.to(
        device=student_parameter.device,
        dtype=student_parameter.dtype,
    )
    teacher_input = teacher_delta.to(
        device=teacher_parameter.device,
        dtype=teacher_parameter.dtype,
    )
    student_feature = student_head(student_input)
    with torch.no_grad():
        teacher_feature = teacher_head(teacher_input)
    if student_feature.shape != student_input.shape or teacher_feature.shape != teacher_input.shape:
        raise ValueError("motion heads must preserve adjacent-frame delta shape")
    if gradient_mask is None:
        return F.mse_loss(student_feature, teacher_feature, reduction="mean")
    if gradient_mask.dtype != torch.bool or gradient_mask.shape != generated_clean.shape:
        raise ValueError("flow regression gradient_mask must be boolean and match generated_clean")
    mask = gradient_mask.movedim(resolved, 1)
    pair_mask = (mask[:, 1:] & mask[:, :-1]).to(device=student_feature.device)
    if not bool(pair_mask.any()):
        return generated.sum() * 0.0
    return F.mse_loss(student_feature[pair_mask], teacher_feature[pair_mask], reduction="mean")


__all__ = [
    "DiagonalDistributionGradients",
    "DiagonalProxyLosses",
    "diagonal_distribution_gradients",
    "diagonal_flow_regression_loss",
    "diagonal_proxy_losses",
    "diagonal_regression_loss",
    "dynamic_motion_weights",
    "exponential_motion_weights",
    "hybrid_motion_weights",
]
