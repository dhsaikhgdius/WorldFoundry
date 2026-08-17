"""Closed-form teacher/student transition-mean objective for DiffusionOPD."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class DiffusionOPDLoss:
    """Reduced objective and its per-sample/per-step values."""

    loss: torch.Tensor
    per_sample_step: torch.Tensor


def diffusion_opd_loss(
    student_transition_means: torch.Tensor,
    teacher_transition_means: torch.Tensor,
    transition_scales: torch.Tensor,
    *,
    add_kl_coefficient: bool,
) -> DiffusionOPDLoss:
    """Match frozen teacher means, optionally normalized by shared variance."""

    if student_transition_means.ndim < 3 or student_transition_means.shape != teacher_transition_means.shape:
        raise ValueError("DiffusionOPD means must share shape [B,K,...latent]")
    try:
        torch.broadcast_shapes(
            tuple(transition_scales.shape),
            tuple(student_transition_means.shape),
        )
    except RuntimeError as error:
        raise ValueError("DiffusionOPD transition scales must broadcast to means") from error
    student = student_transition_means.float()
    teacher = teacher_transition_means.detach().to(device=student.device, dtype=torch.float32)
    if add_kl_coefficient:
        scales = transition_scales.detach().to(device=student.device, dtype=torch.float32)
        if not bool((scales > 0).all()):
            raise ValueError("DiffusionOPD KL transition scales must be positive")
        elementwise = (student - teacher).square() / (2.0 * scales.square())
    else:
        elementwise = 0.5 * (student - teacher).square()
    latent_dims = tuple(range(2, elementwise.ndim))
    per_sample_step = elementwise.mean(dim=latent_dims)
    return DiffusionOPDLoss(
        loss=per_sample_step.mean(),
        per_sample_step=per_sample_step,
    )


__all__ = ["DiffusionOPDLoss", "diffusion_opd_loss"]
