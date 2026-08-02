"""Pure IDA, ISG, DMD, and hinge-GAN equations used by SenseFlow.

Key formulas:
  - Flow velocity from clean: v = (x_t - x_0) / sigma
  - ISG paths (Eq. 12): x_mid = Euler(x, v_teacher); x_target = Euler(x_mid, v_student_mid)
  - ISG loss: ||Euler(x, v_student) - x_target|| (MSE or Charbonnier)
  - IDA (Eq. 9): phi <- decay * phi + (1 - decay) * theta
  - DMD gradient: g = (x0_fake - x0_teacher) / mean|x_gen - x0_teacher|
  - DMD proxy: L = 0.5 * ||x_gen - stop(x_gen - g)||^2
  - Hinge GAN: L_G = -D(fake); L_D = relu(1-D(real)) + relu(1+D(fake))

References:
  - SenseFlow (IDA + ISG + VFM-GAN): https://arxiv.org/abs/2506.00523
  - DMD: https://arxiv.org/abs/2311.18828
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor, nn

from worldfoundry.core.distributed.device_mesh_collectives import get_local_tensor_if_dtensor
from worldfoundry.training.objectives.flow_matching import flow_shift_sigmas


def _matching_latents(*values: Tensor) -> None:
    if not values or not all(isinstance(value, Tensor) for value in values):
        raise TypeError("SenseFlow latent inputs must be torch.Tensor values")
    if values[0].ndim < 2 or any(value.shape != values[0].shape for value in values[1:]):
        raise ValueError("SenseFlow latent inputs must share a non-empty [B,...] shape")


def expand_levels(levels: Tensor, reference: Tensor) -> Tensor:
    if not isinstance(levels, Tensor) or not isinstance(reference, Tensor):
        raise TypeError("levels and reference must be torch.Tensor values")
    if reference.ndim < 2:
        raise ValueError("reference must include batch and feature dimensions")
    if levels.ndim == 0:
        levels = levels.expand(int(reference.shape[0]))
    if levels.ndim != 1 or levels.shape[0] != reference.shape[0]:
        raise ValueError("levels must be scalar or have shape [B]")
    return levels.reshape((int(levels.shape[0]),) + (1,) * (reference.ndim - 1))


def flow_velocity_from_clean(sample: Tensor, clean: Tensor, sigmas: Tensor) -> Tensor:
    """Recover the FM-OT velocity from a clean prediction: ``v=(x_t-x_0)/sigma``."""

    _matching_latents(sample, clean)
    expanded = expand_levels(sigmas.float(), sample)
    if not bool((expanded > 0).all()):
        raise ValueError("velocity recovery requires strictly positive sigmas")
    return (sample.float() - clean.float()) / expanded


def flow_euler_step(
    sample: Tensor,
    velocity: Tensor,
    sigmas: Tensor,
    next_sigmas: Tensor,
) -> Tensor:
    """Move from ``sigma`` to ``sigma_next`` with one flow Euler step."""

    _matching_latents(sample, velocity)
    sigma = expand_levels(sigmas.float(), sample)
    sigma_next = expand_levels(next_sigmas.float(), sample)
    return (sample.float() + (sigma_next - sigma) * velocity.float()).to(sample.dtype)


def sample_isg_midpoint(
    current_timestep: int,
    next_timestep: int,
    *,
    margin: int,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    """Sample the released inclusive integer interval after excluding both margins."""

    for name, value in (("current_timestep", current_timestep), ("next_timestep", next_timestep)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if isinstance(margin, bool) or not isinstance(margin, int) or margin < 0:
        raise ValueError("margin must be a non-negative integer")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be torch.Generator")
    lower = next_timestep + margin
    upper = current_timestep - margin
    if lower > upper:
        raise ValueError("the ISG segment has no midpoint after margins")
    return torch.randint(
        lower,
        upper + 1,
        (),
        device=device,
        generator=generator,
        dtype=torch.int64,
    )


def senseflow_sigma_at_timestep(
    timestep: Tensor,
    *,
    num_train_timesteps: int,
    flow_shift: float,
    timestep_index_offset: int,
) -> Tensor:
    """Map an author-code scheduler index to its exact effective flow sigma."""

    if not isinstance(timestep, Tensor) or timestep.numel() != 1:
        raise TypeError("timestep must be one scalar tensor")
    if (
        isinstance(num_train_timesteps, bool)
        or not isinstance(num_train_timesteps, int)
        or num_train_timesteps < 2
    ):
        raise ValueError("num_train_timesteps must be an integer >= 2")
    if (
        isinstance(timestep_index_offset, bool)
        or not isinstance(timestep_index_offset, int)
        or timestep_index_offset < 0
    ):
        raise ValueError("timestep_index_offset must be a non-negative integer")
    resolved_shift = float(flow_shift)
    if not isfinite(resolved_shift) or resolved_shift <= 0:
        raise ValueError("flow_shift must be finite and positive")
    base = (timestep.float() + float(timestep_index_offset)) / float(num_train_timesteps)
    if not bool(((base >= 0) & (base <= 1)).all()):
        raise ValueError("timestep maps outside the scheduler sigma grid")
    return flow_shift_sigmas(base, resolved_shift)


@dataclass(frozen=True, slots=True)
class ISGPaths:
    teacher_midpoint: Tensor
    target_next: Tensor
    direct_next: Tensor


def flow_isg_paths(
    anchor_sample: Tensor,
    teacher_velocity: Tensor,
    midpoint_student_velocity: Tensor,
    anchor_student_velocity: Tensor,
    *,
    anchor_sigmas: Tensor,
    midpoint_sigmas: Tensor,
    next_sigmas: Tensor,
) -> ISGPaths:
    """Construct the two paths in SenseFlow Eq. 12 using Euler transitions."""

    _matching_latents(
        anchor_sample,
        teacher_velocity,
        midpoint_student_velocity,
        anchor_student_velocity,
    )
    teacher_midpoint = flow_euler_step(
        anchor_sample,
        teacher_velocity,
        anchor_sigmas,
        midpoint_sigmas,
    )
    target_next = flow_euler_step(
        teacher_midpoint,
        midpoint_student_velocity,
        midpoint_sigmas,
        next_sigmas,
    )
    direct_next = flow_euler_step(
        anchor_sample,
        anchor_student_velocity,
        anchor_sigmas,
        next_sigmas,
    )
    return ISGPaths(
        teacher_midpoint=teacher_midpoint,
        target_next=target_next,
        direct_next=direct_next,
    )


def isg_loss_per_sample(
    direct_next: Tensor,
    target_next: Tensor,
    *,
    loss_type: str,
    epsilon: float = 1.0e-3,
) -> Tensor:
    """Paper MSE or the released executable's Charbonnier ISG reduction."""

    _matching_latents(direct_next, target_next)
    if loss_type not in {"charbonnier", "mse"}:
        raise ValueError("loss_type must be 'charbonnier' or 'mse'")
    resolved_epsilon = float(epsilon)
    if not isfinite(resolved_epsilon) or resolved_epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    difference = direct_next.float() - target_next.detach().float()
    if loss_type == "mse":
        per_element = difference.square()
    else:
        per_element = torch.sqrt(difference.square() + resolved_epsilon**2) - resolved_epsilon
    return per_element.flatten(1).mean(1)


def senseflow_distribution_gradient(
    generated_clean: Tensor,
    fake_clean: Tensor,
    teacher_clean: Tensor,
    *,
    normalization_epsilon: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Released normalized field ``(x_fake-x_real)/mean(abs(x_gen-x_real))``."""

    _matching_latents(generated_clean, fake_clean, teacher_clean)
    epsilon = float(normalization_epsilon)
    if not isfinite(epsilon) or epsilon < 0:
        raise ValueError("normalization_epsilon must be finite and non-negative")
    axes = tuple(range(1, generated_clean.ndim))
    normalizer = (generated_clean.float() - teacher_clean.float()).abs().mean(
        dim=axes,
        keepdim=True,
    )
    denominator = normalizer.clamp_min(epsilon) if epsilon > 0 else normalizer
    gradient = torch.nan_to_num((fake_clean.float() - teacher_clean.float()) / denominator)
    return gradient, normalizer.flatten(1)[:, 0]


def senseflow_proxy_loss_per_sample(generated_clean: Tensor, gradient: Tensor) -> Tensor:
    """Half-MSE proxy whose gradient with respect to the generator output is ``gradient``."""

    _matching_latents(generated_clean, gradient)
    generated = generated_clean.float()
    target = (generated - gradient.float()).detach()
    return 0.5 * (generated - target).square().flatten(1).mean(1)


def sample_score_sigmas(
    reference: Tensor,
    *,
    sampling: str,
    minimum_timestep_fraction: float,
    maximum_timestep_fraction: float,
    flow_shift: float,
    generator: torch.Generator,
    num_train_timesteps: int = 1000,
) -> Tensor:
    """Sample per-example DMD/fake-score levels with released distributions."""

    if not isinstance(reference, Tensor) or reference.ndim < 2:
        raise TypeError("reference must be a [B,...] tensor")
    minimum = float(minimum_timestep_fraction)
    maximum = float(maximum_timestep_fraction)
    if not 0 <= minimum < maximum <= 1:
        raise ValueError("score timestep fractions must satisfy 0 <= min < max <= 1")
    resolved_shift = float(flow_shift)
    if not isfinite(resolved_shift) or resolved_shift <= 0:
        raise ValueError("flow_shift must be finite and positive")
    if (
        isinstance(num_train_timesteps, bool)
        or not isinstance(num_train_timesteps, int)
        or num_train_timesteps < 2
    ):
        raise ValueError("num_train_timesteps must be an integer >= 2")
    train_steps = num_train_timesteps
    lower = int(minimum * train_steps)
    upper = min(int(maximum * train_steps), train_steps - 1)
    if sampling == "uniform-schedule-index":
        indices = torch.randint(
            lower,
            upper + 1,
            (int(reference.shape[0]),),
            device=reference.device,
            generator=generator,
        )
        base = (indices.float() + 1.0) / float(train_steps)
    elif sampling == "logit-normal-scheduler-index":
        density = torch.randn(
            (int(reference.shape[0]),),
            device=reference.device,
            dtype=torch.float32,
            generator=generator,
        ).sigmoid()
        indices = (density * train_steps).long().clamp(min=lower, max=upper)
        base = (float(train_steps) - indices.float()) / float(train_steps)
    else:
        raise ValueError("unsupported SenseFlow score sampling mode")
    return flow_shift_sigmas(base, resolved_shift)


def _per_sample_logits(logits: Tensor, *, field_name: str) -> Tensor:
    if not isinstance(logits, Tensor) or logits.ndim < 1:
        raise TypeError(f"{field_name} must have a batch dimension")
    values = logits.float().reshape(int(logits.shape[0]), -1)
    if values.shape[1] == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError(f"{field_name} must contain finite logits")
    return values


def senseflow_generator_hinge_loss(fake_logits: Tensor) -> Tensor:
    return -_per_sample_logits(fake_logits, field_name="fake_logits").mean(1)


def senseflow_discriminator_hinge_loss(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
    real = _per_sample_logits(real_logits, field_name="real_logits")
    fake = _per_sample_logits(fake_logits, field_name="fake_logits")
    if real.shape[0] != fake.shape[0]:
        raise ValueError("real and fake discriminator batches must have equal size")
    return torch.relu(1.0 - real).mean(1) + torch.relu(1.0 + fake).mean(1)


def senseflow_adversarial_time_weight(adversarial_scales: Tensor) -> Tensor:
    """Square the model-author clean-progress scale used by VFM-GAN."""

    if not isinstance(adversarial_scales, Tensor) or adversarial_scales.ndim != 1:
        raise TypeError("adversarial_scales must have shape [B]")
    if not bool(torch.isfinite(adversarial_scales).all()) or not bool(
        ((adversarial_scales >= 0) & (adversarial_scales <= 1)).all()
    ):
        raise ValueError("adversarial_scales must be finite and lie in [0,1]")
    return adversarial_scales.float().square()


def _parameter_inventory(module: nn.Module, *, role: str) -> dict[str, nn.Parameter]:
    if not isinstance(module, nn.Module):
        raise TypeError(f"{role} must be an nn.Module")
    inventory = dict(module.named_parameters())
    if not inventory:
        raise ValueError(f"{role} has no parameters")
    return inventory


def audit_ida_alignment(student: nn.Module, fake_score: nn.Module) -> tuple[str, ...]:
    """Require name-, shard-, shape-, dtype-, and device-aligned IDA parameters."""

    if student is fake_score:
        raise ValueError("IDA student and fake score modules must be distinct")
    source = _parameter_inventory(student, role="IDA student")
    target = _parameter_inventory(fake_score, role="IDA fake score")
    if tuple(source) != tuple(target):
        raise ValueError("IDA student and fake-score parameter inventories differ")
    if any(
        source[name].requires_grad != target[name].requires_grad
        for name in source
    ):
        raise ValueError("IDA student and fake-score trainable parameter masks differ")
    tracked = tuple(name for name, parameter in source.items() if parameter.requires_grad)
    if not tracked:
        raise ValueError("IDA student has no trainable parameters")
    for name in tracked:
        source_local = get_local_tensor_if_dtensor(source[name])
        target_local = get_local_tensor_if_dtensor(target[name])
        if source_local.shape != target_local.shape:
            raise ValueError(f"IDA parameter {name!r} local shapes differ")
        if source_local.dtype != target_local.dtype:
            raise ValueError(f"IDA parameter {name!r} dtypes differ")
        if source_local.device != target_local.device:
            raise ValueError(f"IDA parameter {name!r} local shards are not colocated")
        if source_local is target_local:
            raise ValueError(f"IDA parameter {name!r} unexpectedly shares storage")
    return tracked


@dataclass(frozen=True, slots=True)
class IDAUpdate:
    parameter_count: int
    mean_absolute_shift: Tensor


@torch.no_grad()
def implicit_distribution_alignment_(
    student: nn.Module,
    fake_score: nn.Module,
    *,
    decay: float,
) -> IDAUpdate:
    """Apply SenseFlow Eq. 9: ``phi <- decay*phi + (1-decay)*theta``."""

    resolved_decay = float(decay)
    if not isfinite(resolved_decay) or not 0 <= resolved_decay <= 1:
        raise ValueError("IDA decay must lie in [0,1]")
    tracked = audit_ida_alignment(student, fake_score)
    source = dict(student.named_parameters())
    target = dict(fake_score.named_parameters())
    total_shift = None
    total_values = 0
    for name in tracked:
        source_local = get_local_tensor_if_dtensor(source[name]).detach()
        target_local = get_local_tensor_if_dtensor(target[name])
        previous = target_local.detach().clone()
        target_local.mul_(resolved_decay).add_(source_local, alpha=1.0 - resolved_decay)
        shift = (target_local.float() - previous.float()).abs().sum()
        total_shift = shift if total_shift is None else total_shift + shift
        total_values += target_local.numel()
    assert total_shift is not None and total_values > 0
    return IDAUpdate(
        parameter_count=len(tracked),
        mean_absolute_shift=total_shift / float(total_values),
    )


__all__ = [
    "IDAUpdate",
    "ISGPaths",
    "audit_ida_alignment",
    "expand_levels",
    "flow_euler_step",
    "flow_isg_paths",
    "flow_velocity_from_clean",
    "implicit_distribution_alignment_",
    "isg_loss_per_sample",
    "sample_isg_midpoint",
    "sample_score_sigmas",
    "senseflow_sigma_at_timestep",
    "senseflow_adversarial_time_weight",
    "senseflow_discriminator_hinge_loss",
    "senseflow_distribution_gradient",
    "senseflow_generator_hinge_loss",
    "senseflow_proxy_loss_per_sample",
]
