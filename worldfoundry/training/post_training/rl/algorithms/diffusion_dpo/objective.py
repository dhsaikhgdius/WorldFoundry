"""Paired forward process and preference objective for Diffusion-DPO.

Key formulas:
  - Flow forward: x_t = (1 - t) * x_0 + t * eps; target velocity v = eps - x_0
  - Pair MSE margin: delta = MSE_w(chosen) - MSE_w(rejected)
  - DPO logit: logit = -0.5 * beta * (delta_policy - delta_ref)
  - Loss: L = -log sigmoid(logit)

References:
  - Diffusion-DPO: https://arxiv.org/abs/2311.12908
  - DPO: https://arxiv.org/abs/2305.18290
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from ....shared.validation import positive_float


@dataclass(frozen=True, slots=True)
class DiffusionDPOForwardSample:
    noisy_latents: torch.Tensor
    target_velocity: torch.Tensor
    times: torch.Tensor
    noise: torch.Tensor


@dataclass(frozen=True, slots=True)
class DiffusionDPOLoss:
    loss: torch.Tensor
    logits: torch.Tensor
    current_mse: torch.Tensor
    reference_mse: torch.Tensor
    current_pair_margin: torch.Tensor
    reference_pair_margin: torch.Tensor
    preference_accuracy: torch.Tensor


def _floating_tensor(
    value: object,
    *,
    field_name: str,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{field_name} must be a floating torch.Tensor")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{field_name} must have shape {shape}")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{field_name} must be finite")
    return value


def _paired_batch_size(value: torch.Tensor, *, field_name: str) -> int:
    if value.ndim < 2 or int(value.shape[0]) == 0:
        raise ValueError(f"{field_name} must have non-empty shape [B,...]")
    batch_size = int(value.shape[0])
    if batch_size % 2:
        raise ValueError(f"{field_name} batch size must be even")
    return batch_size


def diffusion_dpo_forward_process(
    clean_latents: torch.Tensor,
    times: torch.Tensor,
    noise: torch.Tensor,
) -> DiffusionDPOForwardSample:
    """Build paired ``x_t=(1-t)x_0+t*epsilon`` and ``epsilon-x_0`` targets."""

    clean = _floating_tensor(clean_latents, field_name="clean_latents")
    batch_size = _paired_batch_size(clean, field_name="clean_latents")
    shape = tuple(clean.shape)
    noise_tensor = _floating_tensor(noise, field_name="noise", shape=shape)
    time_tensor = _floating_tensor(times, field_name="times", shape=(batch_size,))
    if noise_tensor.device != clean.device or time_tensor.device != clean.device:
        raise ValueError("forward-process tensors must share a device")
    if bool((time_tensor < 0).any()) or bool((time_tensor > 1).any()):
        raise ValueError("forward-process times must be in [0,1]")
    if not torch.equal(time_tensor[0::2], time_tensor[1::2]):
        raise ValueError("each chosen/rejected pair must share one timestep")
    if not torch.equal(noise_tensor[0::2], noise_tensor[1::2]):
        raise ValueError("each chosen/rejected pair must share one noise sample")

    clean = clean.detach()
    noise_tensor = noise_tensor.detach().to(dtype=clean.dtype)
    time_tensor = time_tensor.detach().to(dtype=torch.float32)
    expanded_times = time_tensor.to(dtype=clean.dtype).reshape((batch_size,) + (1,) * (clean.ndim - 1))
    noisy_latents = (1 - expanded_times) * clean + expanded_times * noise_tensor
    target_velocity = noise_tensor - clean
    return DiffusionDPOForwardSample(
        noisy_latents=noisy_latents,
        target_velocity=target_velocity,
        times=time_tensor,
        noise=noise_tensor,
    )


def sample_diffusion_dpo_forward_process(
    clean_latents: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> DiffusionDPOForwardSample:
    """Sample one timestep and one noise tensor, then repeat both within each pair."""

    clean = _floating_tensor(clean_latents, field_name="clean_latents")
    batch_size = _paired_batch_size(clean, field_name="clean_latents")
    pair_count = batch_size // 2
    pair_times = torch.rand(
        (pair_count,),
        device=clean.device,
        dtype=torch.float32,
        generator=generator,
    )
    pair_noise = torch.randn(
        (pair_count, *tuple(clean.shape[1:])),
        device=clean.device,
        dtype=clean.dtype,
        generator=generator,
    )
    return diffusion_dpo_forward_process(
        clean,
        pair_times.repeat_interleave(2),
        pair_noise.repeat_interleave(2, dim=0),
    )


def diffusion_dpo_loss(
    *,
    target_velocity: torch.Tensor,
    policy_prediction: torch.Tensor,
    reference_prediction: torch.Tensor,
    beta: float,
) -> DiffusionDPOLoss:
    """Compute the adjacent-pair flow-matching preference loss."""

    target = _floating_tensor(target_velocity, field_name="target_velocity")
    batch_size = _paired_batch_size(target, field_name="target_velocity")
    shape = tuple(target.shape)
    policy = _floating_tensor(policy_prediction, field_name="policy_prediction", shape=shape)
    reference = _floating_tensor(
        reference_prediction,
        field_name="reference_prediction",
        shape=shape,
    )
    if policy.device != target.device or reference.device != target.device:
        raise ValueError("Diffusion-DPO objective tensors must share a device")
    preference_beta = positive_float(beta, field_name="beta")
    compute_dtype = torch.float64 if torch.float64 in {target.dtype, policy.dtype, reference.dtype} else torch.float32
    target_for_loss = target.detach().to(dtype=compute_dtype)
    current_error = policy.to(dtype=compute_dtype) - target_for_loss
    reference_error = reference.detach().to(dtype=compute_dtype) - target_for_loss
    reduce_dims = tuple(range(1, target.ndim))
    current_mse = current_error.square().mean(dim=reduce_dims)
    reference_mse = reference_error.square().mean(dim=reduce_dims)

    current_pair_margin = current_mse[0:batch_size:2] - current_mse[1:batch_size:2]
    reference_pair_margin = reference_mse[0:batch_size:2] - reference_mse[1:batch_size:2]
    logits = -0.5 * preference_beta * (current_pair_margin - reference_pair_margin)
    loss = -functional.logsigmoid(logits).mean()
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError("non-finite Diffusion-DPO objective")
    return DiffusionDPOLoss(
        loss=loss,
        logits=logits,
        current_mse=current_mse,
        reference_mse=reference_mse,
        current_pair_margin=current_pair_margin,
        reference_pair_margin=reference_pair_margin,
        preference_accuracy=(logits.detach() > 0).float().mean(),
    )


__all__ = [
    "DiffusionDPOForwardSample",
    "DiffusionDPOLoss",
    "diffusion_dpo_forward_process",
    "diffusion_dpo_loss",
    "sample_diffusion_dpo_forward_process",
]
