"""Forward-process objective for WorldFoundry-native DiffusionNFT.

Key formulas:
  - Flow forward: x_t = (1 - t) * x_0 + t * eps; target v = eps - x_0
  - Group advantage: A_i = (r_i - mu_g) / (std_g + eps), clipped to [-clip, clip]
  - Reward probability: p_i = 0.5 + 0.5 * A_i / clip_max  in [0, 1]
  - NFT loss: weighted flow-matching with implicit pos/neg from p_i and old policy

References:
  - DiffusionNFT: https://arxiv.org/abs/2509.16117
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch

from ...objectives.group_advantages import (
    normalize_data_parallel_grouped_advantages,
    normalize_grouped_advantages,
)
from .contracts import validate_mix_beta


@dataclass(frozen=True, slots=True)
class DiffusionNFTForwardSample:
    """A clean rollout state re-noised through the flow forward process."""

    noisy_latents: torch.Tensor
    target_velocity: torch.Tensor
    times: torch.Tensor


@dataclass(frozen=True, slots=True)
class DiffusionNFTRewardWeights:
    """Prompt-group advantages and their optimality-probability mapping."""

    advantages: torch.Tensor
    reward_probabilities: torch.Tensor
    group_means: torch.Tensor
    group_stds: torch.Tensor
    group_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiffusionNFTLoss:
    """Scalar objective plus unreduced terms used for correctness metrics."""

    loss: torch.Tensor
    policy_loss: torch.Tensor
    reference_mse: torch.Tensor | None
    positive_reconstruction: torch.Tensor
    negative_reconstruction: torch.Tensor
    flow_matching_mse: torch.Tensor
    old_policy_mse: torch.Tensor


def _floating_batch_tensor(
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


def diffusion_nft_forward_process(
    clean_latents: torch.Tensor,
    times: torch.Tensor,
    noise: torch.Tensor,
) -> DiffusionNFTForwardSample:
    """Construct ``x_t=(1-t)x_0+t*epsilon`` and target ``epsilon-x_0``."""

    clean = _floating_batch_tensor(clean_latents, field_name="clean_latents")
    if clean.ndim < 2 or int(clean.shape[0]) == 0:
        raise ValueError("clean_latents must have non-empty shape [B,...]")
    noise_tensor = _floating_batch_tensor(
        noise,
        field_name="noise",
        shape=tuple(clean.shape),
    )
    time_tensor = _floating_batch_tensor(
        times,
        field_name="times",
        shape=(int(clean.shape[0]),),
    )
    if noise_tensor.device != clean.device or time_tensor.device != clean.device:
        raise ValueError("forward-process tensors must share a device")
    if bool((time_tensor < 0).any()) or bool((time_tensor > 1).any()):
        raise ValueError("forward-process times must be in [0,1]")
    noise_tensor = noise_tensor.to(dtype=clean.dtype)
    time_tensor = time_tensor.to(dtype=torch.float32)
    expanded = time_tensor.to(dtype=clean.dtype).reshape((int(clean.shape[0]),) + (1,) * (clean.ndim - 1))
    noisy = (1 - expanded) * clean + expanded * noise_tensor
    target = noise_tensor - clean
    return DiffusionNFTForwardSample(
        noisy_latents=noisy,
        target_velocity=target,
        times=time_tensor,
    )


def diffusion_nft_reward_weights(
    rewards: torch.Tensor,
    group_ids: tuple[str, ...],
    *,
    advantage_clip_max: float,
    epsilon: float = 1.0e-4,
    normalization: str = "group-population-std",
    advantage_mode: str = "all",
    parallel_context: object | None = None,
) -> DiffusionNFTRewardWeights:
    """Normalize rewards per prompt and map clipped advantages into ``[0,1]``."""

    clip_max = float(advantage_clip_max)
    if not isfinite(clip_max) or clip_max <= 0:
        raise ValueError("advantage_clip_max must be finite and positive")
    if parallel_context is None:
        grouped = normalize_grouped_advantages(
            rewards,
            group_ids,
            epsilon=epsilon,
            clip_max=clip_max,
            normalization=normalization,
        )
    else:
        grouped = normalize_data_parallel_grouped_advantages(
            rewards,
            group_ids,
            parallel_context=parallel_context,
            epsilon=epsilon,
            clip_max=clip_max,
            normalization=normalization,
        )
    advantages = grouped.advantages.detach()
    mode = str(advantage_mode).strip().lower().replace("-", "_")
    mapped_advantages = advantages
    if mode == "positive_only":
        mapped_advantages = advantages.clamp(min=0)
    elif mode == "negative_only":
        mapped_advantages = advantages.clamp(max=0)
    elif mode == "one_only":
        mapped_advantages = torch.where(
            advantages > 0,
            torch.ones_like(advantages),
            torch.zeros_like(advantages),
        )
    elif mode == "binary":
        mapped_advantages = advantages.sign()
    elif mode != "all":
        raise ValueError("advantage_mode must be all, positive_only, negative_only, one_only, or binary")
    reward_probabilities = (0.5 + 0.5 * mapped_advantages / clip_max).clamp(0, 1)
    return DiffusionNFTRewardWeights(
        advantages=advantages,
        reward_probabilities=reward_probabilities,
        group_means=grouped.group_means,
        group_stds=grouped.group_stds,
        group_order=grouped.group_order,
    )


def diffusion_nft_loss(
    *,
    clean_latents: torch.Tensor,
    noisy_latents: torch.Tensor,
    times: torch.Tensor,
    target_velocity: torch.Tensor,
    policy_prediction: torch.Tensor,
    old_policy_prediction: torch.Tensor,
    reward_probabilities: torch.Tensor,
    beta: float,
    advantage_clip_max: float,
    reference_prediction: torch.Tensor | None = None,
    reference_mse_weight: float = 0.0,
    reconstruction_mae_floor: float = 1.0e-5,
) -> DiffusionNFTLoss:
    """Compute DiffusionNFT's reward-weighted implicit positive/negative loss."""

    clean = _floating_batch_tensor(clean_latents, field_name="clean_latents")
    if clean.ndim < 2 or int(clean.shape[0]) == 0:
        raise ValueError("clean_latents must have non-empty shape [B,...]")
    shape = tuple(clean.shape)
    noisy = _floating_batch_tensor(noisy_latents, field_name="noisy_latents", shape=shape)
    target = _floating_batch_tensor(target_velocity, field_name="target_velocity", shape=shape)
    policy = _floating_batch_tensor(policy_prediction, field_name="policy_prediction", shape=shape)
    old = _floating_batch_tensor(
        old_policy_prediction,
        field_name="old_policy_prediction",
        shape=shape,
    ).detach()
    time_tensor = _floating_batch_tensor(
        times,
        field_name="times",
        shape=(int(clean.shape[0]),),
    )
    probabilities = _floating_batch_tensor(
        reward_probabilities,
        field_name="reward_probabilities",
        shape=(int(clean.shape[0]),),
    ).detach()
    if any(value.device != clean.device for value in (noisy, target, policy, old, time_tensor, probabilities)):
        raise ValueError("DiffusionNFT objective tensors must share a device")
    if bool((time_tensor < 0).any()) or bool((time_tensor > 1).any()):
        raise ValueError("times must be in [0,1]")
    if bool((probabilities < 0).any()) or bool((probabilities > 1).any()):
        raise ValueError("reward_probabilities must be in [0,1]")
    mix_beta = validate_mix_beta(beta)
    clip_max = float(advantage_clip_max)
    if not isfinite(clip_max) or clip_max <= 0:
        raise ValueError("advantage_clip_max must be finite and positive")
    mae_floor = float(reconstruction_mae_floor)
    if not isfinite(mae_floor) or mae_floor <= 0:
        raise ValueError("reconstruction_mae_floor must be finite and positive")
    reference_weight = float(reference_mse_weight)
    if not isfinite(reference_weight) or reference_weight < 0:
        raise ValueError("reference_mse_weight must be finite and non-negative")
    if reference_prediction is None and reference_weight != 0:
        raise ValueError("positive reference_mse_weight requires reference_prediction")

    old = old.to(dtype=policy.dtype)
    expanded_times = time_tensor.to(dtype=policy.dtype).reshape((int(clean.shape[0]),) + (1,) * (clean.ndim - 1))
    positive_velocity = (1 - mix_beta) * old + mix_beta * policy
    negative_velocity = (1 + mix_beta) * old - mix_beta * policy
    positive_clean = noisy.to(dtype=policy.dtype) - expanded_times * positive_velocity
    negative_clean = noisy.to(dtype=policy.dtype) - expanded_times * negative_velocity
    clean_for_loss = clean.to(dtype=policy.dtype)
    reduce_dims = tuple(range(1, clean.ndim))
    positive_error = positive_clean - clean_for_loss
    negative_error = negative_clean - clean_for_loss
    positive_mae = (
        positive_error.detach()
        .double()
        .abs()
        .mean(dim=reduce_dims, keepdim=True)
        .clamp_min(mae_floor)
        .to(dtype=policy.dtype)
    )
    negative_mae = (
        negative_error.detach()
        .double()
        .abs()
        .mean(dim=reduce_dims, keepdim=True)
        .clamp_min(mae_floor)
        .to(dtype=policy.dtype)
    )
    positive_reconstruction = (positive_error.square() / positive_mae).mean(dim=reduce_dims)
    negative_reconstruction = (negative_error.square() / negative_mae).mean(dim=reduce_dims)
    reward_weight = probabilities.to(dtype=policy.dtype)
    per_sample = (
        clip_max * (reward_weight * positive_reconstruction + (1 - reward_weight) * negative_reconstruction) / mix_beta
    )
    policy_loss = per_sample.mean()

    reference_mse: torch.Tensor | None = None
    if reference_prediction is not None:
        reference = _floating_batch_tensor(
            reference_prediction,
            field_name="reference_prediction",
            shape=shape,
        )
        if reference.device != clean.device:
            raise ValueError("reference_prediction must share the objective device")
        reference_mse = (policy - reference.detach().to(dtype=policy.dtype)).square().mean(dim=reduce_dims).mean()
    loss = policy_loss if reference_mse is None else policy_loss + reference_weight * reference_mse
    flow_matching_mse = (policy - target.detach().to(dtype=policy.dtype)).square().mean(dim=reduce_dims).mean()
    old_policy_mse = (policy - old).square().mean(dim=reduce_dims).mean()
    if not bool(torch.isfinite(loss.detach()).all()):
        raise FloatingPointError("non-finite DiffusionNFT objective")
    return DiffusionNFTLoss(
        loss=loss,
        policy_loss=policy_loss,
        reference_mse=reference_mse,
        positive_reconstruction=positive_reconstruction,
        negative_reconstruction=negative_reconstruction,
        flow_matching_mse=flow_matching_mse,
        old_policy_mse=old_policy_mse,
    )


__all__ = [
    "DiffusionNFTForwardSample",
    "DiffusionNFTLoss",
    "DiffusionNFTRewardWeights",
    "diffusion_nft_forward_process",
    "diffusion_nft_loss",
    "diffusion_nft_reward_weights",
]
