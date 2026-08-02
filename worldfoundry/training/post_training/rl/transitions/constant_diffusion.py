"""Constant-diffusion stochastic transition for rectified-flow policies."""

from __future__ import annotations

from math import isfinite

from .flow_sde import (
    FlowSDETransition,
    flow_sigma_for_sample,
    gaussian_transition_log_prob,
)


def constant_diffusion_flow_transition(
    velocity: object,
    sample: object,
    sigma: object,
    sigma_next: object,
    *,
    eta: float,
    generator: object | None = None,
    next_sample: object | None = None,
    trajectory_dtype: object | None = None,
) -> FlowSDETransition:
    """Sample or replay one constant-diffusion flow transition in FP32."""

    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("constant-diffusion flow transitions require the 'train-core' extra") from error
    if not torch.is_tensor(velocity) or not torch.is_tensor(sample):
        raise TypeError("velocity and sample must be torch.Tensor values")
    if velocity.shape != sample.shape or sample.ndim < 2:
        raise ValueError("velocity and sample must share shape [B,...]")
    resolved_eta = float(eta)
    if not isfinite(resolved_eta) or resolved_eta < 0:
        raise ValueError("eta must be finite and non-negative")
    if trajectory_dtype is None:
        trajectory_dtype = sample.dtype
    if trajectory_dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }:
        raise ValueError("trajectory_dtype must be a floating torch dtype")

    current = sample.float()
    prediction = velocity.float()
    sigma_value = flow_sigma_for_sample(sigma, current, field_name="sigma")
    next_sigma_value = flow_sigma_for_sample(
        sigma_next,
        current,
        field_name="sigma_next",
    )
    if not bool(torch.isfinite(sigma_value).all() and torch.isfinite(next_sigma_value).all()):
        raise ValueError("sigmas must be finite")
    if not bool((sigma_value > 0).all() and (sigma_value <= 1).all()):
        raise ValueError("sigma must be in (0,1]")
    if not bool((next_sigma_value >= 0).all() and (next_sigma_value < sigma_value).all()):
        raise ValueError("sigma_next must be in [0,sigma)")

    dt = next_sigma_value - sigma_value
    diffusion_correction = resolved_eta**2 / (2.0 * sigma_value)
    mean = (
        current * (1.0 + diffusion_correction * dt)
        + prediction * (1.0 + diffusion_correction * (1.0 - sigma_value)) * dt
    )
    scale = torch.full_like(sigma_value, resolved_eta) * torch.sqrt(-dt)

    if next_sample is None:
        if resolved_eta < 1.0e-7:
            sampled = mean
        else:
            noise = torch.randn(
                mean.shape,
                device=mean.device,
                dtype=mean.dtype,
                generator=generator,
            )
            sampled = mean + scale * noise
    else:
        if not torch.is_tensor(next_sample) or next_sample.shape != sample.shape:
            raise ValueError("next_sample must match sample shape")
        sampled = next_sample.float()

    stored = sampled.to(dtype=trajectory_dtype)
    log_prob = None if resolved_eta < 1.0e-7 else gaussian_transition_log_prob(stored.float(), mean, scale)
    return FlowSDETransition(
        next_sample=stored,
        mean=mean,
        scale=scale,
        log_prob=log_prob,
    )


__all__ = ["constant_diffusion_flow_transition"]
