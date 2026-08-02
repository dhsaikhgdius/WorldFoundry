"""Formula-level Data-Forcing Distillation operations.

Key formulas:
  - Data forcing (teacher input): x_teacher = x_gen + stop(x_real - x_gen)  when enabled
  - DFD gradient: g = (x0_fake - x0_teacher) / (mean|x_gen - x0_teacher| + eps)
  - DMD proxy: L = 0.5 * ||x_gen - stop(x_gen - g)||^2
  - Shifted RF sampler: t' = shift * u / (u * (shift - 1) + 1), u ~ Uniform[min, max)

References:
  - Data-Forcing Distillation (DFD): https://arxiv.org/abs/2606.18478
  - DMD2 / FastGen: https://arxiv.org/abs/2405.14867
"""

from __future__ import annotations

from math import isfinite

import torch


def data_forcing_teacher_data(
    generated: torch.Tensor,
    real: torch.Tensor,
    *,
    enabled: bool,
) -> torch.Tensor:
    """Use real values with the released identity-gradient substitution."""

    if not isinstance(generated, torch.Tensor) or not isinstance(real, torch.Tensor):
        raise TypeError("DFD teacher-data inputs must be tensors")
    if generated.ndim < 2 or generated.shape != real.shape:
        raise ValueError("DFD teacher-data inputs must share a [B,...] shape")
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be bool")
    return generated + (real.to(generated) - generated).detach() if enabled else generated


def dfd_distribution_gradient(
    generated: torch.Tensor,
    fake_score_clean: torch.Tensor,
    teacher_clean: torch.Tensor,
    *,
    epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FastGen's ``(fake-teacher)/(mean|generated-teacher|+eps)`` field."""

    values = (generated, fake_score_clean, teacher_clean)
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError("DFD distribution-gradient inputs must be tensors")
    if generated.ndim < 2 or any(value.shape != generated.shape for value in values[1:]):
        raise ValueError("DFD distribution-gradient inputs must share a [B,...] shape")
    resolved_epsilon = float(epsilon)
    if not isfinite(resolved_epsilon) or resolved_epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    axes = tuple(range(1, generated.ndim))
    generated_fp32 = generated.float()
    teacher_fp32 = teacher_clean.float()
    normalizer = (generated_fp32 - teacher_fp32).abs().mean(
        dim=axes,
        keepdim=True,
    )
    gradient = (fake_score_clean.float() - teacher_fp32) / (normalizer + resolved_epsilon)
    if not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError("DFD distribution gradient is non-finite")
    return gradient, normalizer.reshape(int(generated.shape[0]))


def dfd_proxy_loss_per_sample(
    generated: torch.Tensor,
    distribution_gradient: torch.Tensor,
) -> torch.Tensor:
    """Half-MSE proxy whose clean-sample gradient is the DFD score discrepancy."""

    if not isinstance(generated, torch.Tensor) or not isinstance(
        distribution_gradient,
        torch.Tensor,
    ):
        raise TypeError("DFD proxy inputs must be tensors")
    if generated.ndim < 2 or generated.shape != distribution_gradient.shape:
        raise ValueError("DFD proxy inputs must share a [B,...] shape")
    generated_fp32 = generated.float()
    pseudo_target = (generated_fp32 - distribution_gradient.float()).detach()
    return 0.5 * (generated_fp32 - pseudo_target).square().flatten(1).mean(1)


def shifted_uniform_timesteps(
    uniform: torch.Tensor,
    *,
    minimum: float,
    maximum: float,
    shift: float,
) -> torch.Tensor:
    """Transform uniform variates exactly as FastGen's RF ``shifted`` sampler."""

    if not isinstance(uniform, torch.Tensor) or uniform.ndim != 1:
        raise TypeError("uniform must be a rank-one tensor")
    if not uniform.is_floating_point() or not bool(((uniform >= 0) & (uniform < 1)).all()):
        raise ValueError("uniform values must be floating point in [0,1)")
    minimum_value = float(minimum)
    maximum_value = float(maximum)
    shift_value = float(shift)
    if not 0.0 <= minimum_value < maximum_value <= 1.0:
        raise ValueError("timestep bounds must satisfy 0 <= min < max <= 1")
    if not isfinite(shift_value) or shift_value < 1.0:
        raise ValueError("shift must be finite and at least one")
    timestep = uniform * (maximum_value - minimum_value) + minimum_value
    timestep = timestep * shift_value / (timestep * (shift_value - 1.0) + 1.0)
    return timestep.clamp(min=minimum_value, max=maximum_value)


__all__ = [
    "data_forcing_teacher_data",
    "dfd_distribution_gradient",
    "dfd_proxy_loss_per_sample",
    "shifted_uniform_timesteps",
]
