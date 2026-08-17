"""Shared optimizer construction and parameter-inventory contracts."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Literal

import torch
from torch import nn

from worldfoundry.training.optimizers import CAME
from worldfoundry.training.recipes.spec import LRSchedulerSpec


def trainable_parameters(module: nn.Module) -> tuple[nn.Parameter, ...]:
    """Return an immutable, non-empty trainable parameter inventory."""

    if not isinstance(module, nn.Module):
        raise TypeError("trainable module must be an nn.Module")
    parameters = tuple(parameter for parameter in module.parameters() if parameter.requires_grad)
    if not parameters:
        raise ValueError("trainable module has no parameters with requires_grad=True")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("trainable module exposes duplicate parameter objects")
    return parameters


def build_adamw(
    parameters: Iterable[nn.Parameter],
    *,
    learning_rate: float,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    epsilon: float = 1.0e-8,
    fused: bool | Literal["auto"] = "auto",
) -> torch.optim.AdamW:
    """Build AdamW, selecting PyTorch's fused CUDA path when safe."""

    values = tuple(parameters)
    if not values or not all(isinstance(parameter, nn.Parameter) for parameter in values):
        raise ValueError("optimizer parameters must be a non-empty nn.Parameter iterable")
    if not all(parameter.requires_grad for parameter in values):
        raise ValueError("optimizer parameters must all require gradients")
    if len({id(parameter) for parameter in values}) != len(values):
        raise ValueError("optimizer parameters cannot contain duplicates")
    if fused not in {True, False, "auto"}:
        raise ValueError("fused must be true, false, or 'auto'")
    use_fused = all(parameter.device.type == "cuda" for parameter in values) if fused == "auto" else fused
    try:
        return torch.optim.AdamW(
            values,
            lr=float(learning_rate),
            betas=tuple(float(value) for value in betas),
            eps=float(epsilon),
            weight_decay=float(weight_decay),
            fused=use_fused,
        )
    except (RuntimeError, TypeError) as error:
        if fused != "auto" or not use_fused:
            raise
        # Older/backend-specific torch builds can expose the argument without
        # providing a usable fused kernel. Auto mode remains portable.
        try:
            return torch.optim.AdamW(
                values,
                lr=float(learning_rate),
                betas=tuple(float(value) for value in betas),
                eps=float(epsilon),
                weight_decay=float(weight_decay),
                fused=False,
            )
        except Exception:  # noqa: BLE001 - preserve the original capability failure.
            raise error


def build_came(
    parameters: Iterable[nn.Parameter],
    *,
    learning_rate: float,
    weight_decay: float = 0.0,
    betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
    epsilons: tuple[float, float] = (1.0e-30, 1.0e-16),
    update_clip_threshold: float = 1.0,
) -> CAME:
    """Build the CAME variant used by official SANA training."""

    values = tuple(parameters)
    if not values or not all(isinstance(parameter, nn.Parameter) for parameter in values):
        raise ValueError("optimizer parameters must be a non-empty nn.Parameter iterable")
    if not all(parameter.requires_grad for parameter in values):
        raise ValueError("optimizer parameters must all require gradients")
    if len({id(parameter) for parameter in values}) != len(values):
        raise ValueError("optimizer parameters cannot contain duplicates")
    return CAME(
        values,
        lr=learning_rate,
        eps=epsilons,
        clip_threshold=update_clip_threshold,
        betas=betas,
        weight_decay=weight_decay,
    )


def warmup_cosine_multiplier(step: int, spec: LRSchedulerSpec) -> float:
    """Return the recipe-defined warmup and cosine LR multiplier."""

    current = int(step)
    assert spec.peak_factor is not None
    if current < spec.warmup_steps:
        progress = current / spec.warmup_steps
        return spec.start_factor + (spec.peak_factor - spec.start_factor) * progress
    progress = min((current - spec.warmup_steps) / (spec.total_steps - spec.warmup_steps), 1.0)
    return spec.end_factor + 0.5 * (spec.peak_factor - spec.end_factor) * (
        1.0 + math.cos(math.pi * progress)
    )


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    spec: LRSchedulerSpec | None,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build the optional scheduler declared by a native training recipe."""

    if spec is None:
        return None
    if spec.type == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=spec.start_factor,
            end_factor=spec.end_factor,
            total_iters=spec.total_steps,
        )
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: warmup_cosine_multiplier(step, spec),
    )


def _optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    """Return the unique parameter-object identities owned by an optimizer."""

    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
        if isinstance(parameter, nn.Parameter)
    }


def audit_optimizer_parameters(
    optimizer: torch.optim.Optimizer,
    parameters: Iterable[nn.Parameter],
    *,
    role: str,
) -> tuple[nn.Parameter, ...]:
    """Require an optimizer to own exactly one validated parameter inventory."""

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError(f"{role} optimizer must be a torch.optim.Optimizer")
    values = tuple(parameters)
    if not values or not all(isinstance(parameter, nn.Parameter) for parameter in values):
        raise ValueError(f"{role} parameters must be a non-empty nn.Parameter iterable")
    expected = {id(parameter) for parameter in values}
    actual = _optimizer_parameter_ids(optimizer)
    if actual != expected:
        raise ValueError(
            f"{role} optimizer parameter audit failed: "
            f"missing={len(expected - actual)}, unexpected={len(actual - expected)}"
        )
    return values


__all__ = [
    "audit_optimizer_parameters",
    "build_adamw",
    "build_came",
    "build_lr_scheduler",
    "trainable_parameters",
    "warmup_cosine_multiplier",
]
