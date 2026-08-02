"""Pure rectified-flow corruption math and a minimal training objective.

The optional discrete/shifted path matches the training-time noise convention
used by the original SANA implementation: sample an integer index, convert it
to a base flow sigma, then apply the rational flow shift before corruption.
Model-specific conversion from effective sigma to a denoiser timestep remains
the adapter's responsibility.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, prod

from worldfoundry.training.api.contracts import ObjectiveBatch, PreparedBatch, TrainStepResult


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("flow-matching training requires the 'train-core' extra (PyTorch)") from error
    return torch


def _same_shape(left: object, right: object) -> bool:
    return tuple(getattr(left, "shape", ())) == tuple(getattr(right, "shape", ()))


def _sigma_for_reference(sigmas: object, reference: object) -> object:
    ndim = getattr(sigmas, "ndim", None)
    reference_ndim = getattr(reference, "ndim", None)
    if ndim is None or reference_ndim is None:
        return sigmas
    if ndim == 0:
        return sigmas
    if ndim == 1:
        if int(sigmas.shape[0]) != int(reference.shape[0]):
            raise ValueError(
                f"one sigma per sample is required; got {tuple(sigmas.shape)} for {tuple(reference.shape)}"
            )
        return sigmas.reshape((int(sigmas.shape[0]),) + (1,) * (int(reference_ndim) - 1))
    return sigmas


def flow_interpolate(clean: object, noise: object, sigmas: object) -> object:
    """Return ``x_sigma = (1-sigma) * clean + sigma * noise``.

    The function uses only tensor arithmetic and therefore works for both
    PyTorch tensors and NumPy arrays in golden-math tests.
    """

    if not _same_shape(clean, noise):
        raise ValueError(
            f"clean/noise shapes differ: {getattr(clean, 'shape', None)} vs {getattr(noise, 'shape', None)}"
        )
    sigma = _sigma_for_reference(sigmas, clean)
    return clean + sigma * (noise - clean)


def flow_velocity_target(clean: object, noise: object) -> object:
    """Return the rectified-flow velocity target ``noise - clean``."""

    if not _same_shape(clean, noise):
        raise ValueError(
            f"clean/noise shapes differ: {getattr(clean, 'shape', None)} vs {getattr(noise, 'shape', None)}"
        )
    return noise - clean


def flow_shift_sigmas(sigmas: object, shift: float) -> object:
    """Apply ``shift*sigma / (1 + (shift-1)*sigma)``.

    This is a pure tensor/array operation.  A shift of one is exactly the
    identity, while positive shifts preserve the endpoints and monotonicity.
    """

    resolved_shift = float(shift)
    if not isfinite(resolved_shift) or resolved_shift <= 0:
        raise ValueError("flow shift must be finite and positive")
    return resolved_shift * sigmas / (1 + (resolved_shift - 1) * sigmas)


def flow_match_solver_sigmas(
    *,
    num_train_timesteps: int = 1000,
    num_solver_steps: int = 28,
    shift: float = 3.0,
) -> tuple[float, ...]:
    """Build a FlowMatch-Euler inference grid with its terminal zero.

    This follows the scheduler lifecycle used by SD3: shift the training grid,
    interpolate between its extrema, shift the solver grid, then append zero.
    """

    if isinstance(num_train_timesteps, bool) or int(num_train_timesteps) < 2:
        raise ValueError("num_train_timesteps must be at least two")
    if isinstance(num_solver_steps, bool) or int(num_solver_steps) < 2:
        raise ValueError("num_solver_steps must be at least two")
    resolved_shift = float(shift)
    if not isfinite(resolved_shift) or resolved_shift <= 0.0:
        raise ValueError("shift must be finite and positive")
    train_steps = int(num_train_timesteps)
    solver_steps = int(num_solver_steps)

    def shifted(value: float) -> float:
        return resolved_shift * value / (1.0 + (resolved_shift - 1.0) * value)

    sigma_max = shifted(1.0)
    sigma_min = shifted(1.0 / train_steps)
    denominator = solver_steps - 1
    interpolated = tuple(
        sigma_max + (sigma_min - sigma_max) * index / denominator
        for index in range(solver_steps)
    )
    return (*tuple(shifted(value) for value in interpolated), 0.0)


def flow_clean_from_velocity(noisy: object, velocity: object, sigmas: object) -> object:
    """Recover ``x_0`` from ``x_sigma`` and velocity."""

    if not _same_shape(noisy, velocity):
        raise ValueError("noisy and velocity shapes must match")
    sigma = _sigma_for_reference(sigmas, noisy)
    return noisy - sigma * velocity


def flow_noise_from_velocity(noisy: object, velocity: object, sigmas: object) -> object:
    """Recover sampled noise from ``x_sigma`` and velocity."""

    if not _same_shape(noisy, velocity):
        raise ValueError("noisy and velocity shapes must match")
    sigma = _sigma_for_reference(sigmas, noisy)
    return noisy + (1 - sigma) * velocity


@dataclass(frozen=True, slots=True)
class FlowMatchingLoss:
    """Token-weighted MSE and its explicit reduction terms."""

    loss: object
    numerator: object
    denominator: object
    per_sample: object


def _expanded_mask(mask: object, target: object) -> object:
    torch = _require_torch()
    if not torch.is_tensor(mask):
        raise TypeError("loss_mask must be a torch.Tensor")
    if mask.ndim + 1 == target.ndim and int(mask.shape[0]) == int(target.shape[0]):
        mask = mask.unsqueeze(1)
    try:
        mask = torch.broadcast_to(mask, target.shape)
    except RuntimeError as error:
        raise ValueError(f"loss_mask shape {tuple(mask.shape)} cannot broadcast to {tuple(target.shape)}") from error
    mask = mask.to(device=target.device, dtype=torch.float32)
    if not bool(torch.isfinite(mask).all()):
        raise ValueError("loss_mask must be finite")
    if not bool((mask >= 0).all()):
        raise ValueError("loss_mask cannot contain negative weights")
    return mask


def flow_matching_mse(
    prediction: object,
    target: object,
    *,
    loss_mask: object | None = None,
    sample_weights: object | None = None,
) -> FlowMatchingLoss:
    """Compute FP32 MSE with token masks and per-sample weights.

    The denominator is the sum of effective element weights, not a mean of
    rank-local means.  Callers can all-reduce ``numerator`` and ``denominator``
    independently for variable-token distributed batches.
    """

    torch = _require_torch()
    if not torch.is_tensor(prediction) or not torch.is_tensor(target):
        raise TypeError("prediction and target must be torch.Tensor values")
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shapes differ: {tuple(prediction.shape)} vs {tuple(target.shape)}")
    if prediction.ndim < 2:
        raise ValueError("flow-matching tensors must include batch and feature dimensions")

    squared = (prediction.float() - target.float()).square()
    batch_size = int(squared.shape[0])
    effective = torch.ones_like(squared)
    if loss_mask is not None:
        effective = effective * _expanded_mask(loss_mask, squared)
    if sample_weights is not None:
        if not torch.is_tensor(sample_weights):
            raise TypeError("sample_weights must be a torch.Tensor")
        if tuple(sample_weights.shape) != (batch_size,):
            raise ValueError(f"sample_weights must have shape ({batch_size},); got {tuple(sample_weights.shape)}")
        weights = sample_weights.to(device=squared.device, dtype=torch.float32)
        if not bool(torch.isfinite(weights).all()):
            raise ValueError("sample_weights must be finite")
        if not bool((weights >= 0).all()):
            raise ValueError("sample_weights cannot be negative")
        effective = effective * weights.reshape((batch_size,) + (1,) * (squared.ndim - 1))

    flat_squared = squared.reshape(batch_size, -1)
    flat_effective = effective.reshape(batch_size, -1)
    per_numerator = (flat_squared * flat_effective).sum(dim=1)
    per_denominator = flat_effective.sum(dim=1)
    denominator = per_denominator.sum()
    if not bool(denominator.detach() > 0):
        raise ValueError("flow-matching loss has no positive-weight elements")
    numerator = per_numerator.sum()
    per_sample = torch.where(
        per_denominator > 0,
        per_numerator / per_denominator.clamp_min(1.0e-12),
        torch.zeros_like(per_numerator),
    )
    return FlowMatchingLoss(
        loss=numerator / denominator,
        numerator=numerator,
        denominator=denominator,
        per_sample=per_sample,
    )


def flow_matching_denominator(
    target: object,
    *,
    loss_mask: object | None = None,
    sample_weights: object | None = None,
) -> object:
    """Compute the data-dependent MSE denominator without a model forward."""

    torch = _require_torch()
    if not torch.is_tensor(target) or target.ndim < 2:
        raise TypeError("target must be a torch.Tensor with batch and feature dimensions")
    batch_size = int(target.shape[0])
    if loss_mask is None:
        elements = prod(int(size) for size in target.shape[1:])
        per_sample = torch.full(
            (batch_size,),
            float(elements),
            device=target.device,
            dtype=torch.float32,
        )
    else:
        expanded = _expanded_mask(loss_mask, target)
        per_sample = expanded.reshape(batch_size, -1).sum(dim=1)
    if sample_weights is not None:
        if not torch.is_tensor(sample_weights) or tuple(sample_weights.shape) != (batch_size,):
            raise ValueError(f"sample_weights must have shape ({batch_size},)")
        weights = sample_weights.to(device=target.device, dtype=torch.float32)
        if not bool(torch.isfinite(weights).all()) or not bool((weights >= 0).all()):
            raise ValueError("sample_weights must be finite and non-negative")
        per_sample = per_sample * weights
    denominator = per_sample.sum()
    if not bool(denominator > 0):
        raise ValueError("flow-matching loss has no positive-weight elements")
    return denominator


@dataclass(frozen=True, slots=True)
class FlowMatchingConfig:
    timestep_sampler: str = "logit_normal"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    min_sigma: float = 0.0
    max_sigma: float = 1.0
    flow_shift: float = 1.0
    num_train_timesteps: int | None = None

    def __post_init__(self) -> None:
        sampler = str(self.timestep_sampler).strip().lower().replace("_", "-")
        if sampler not in {"uniform", "logit-normal"}:
            raise ValueError(f"unsupported flow timestep sampler: {sampler!r}")
        if float(self.logit_std) <= 0:
            raise ValueError("logit_std must be positive")
        minimum = float(self.min_sigma)
        maximum = float(self.max_sigma)
        if not 0.0 <= minimum < maximum <= 1.0:
            raise ValueError("flow sigma range must satisfy 0 <= min < max <= 1")
        flow_shift = float(self.flow_shift)
        if not isfinite(flow_shift) or flow_shift <= 0:
            raise ValueError("flow_shift must be finite and positive")
        num_train_timesteps = self.num_train_timesteps
        if num_train_timesteps is not None:
            if isinstance(num_train_timesteps, bool):
                raise TypeError("num_train_timesteps must be an integer, not bool")
            num_train_timesteps = int(num_train_timesteps)
            if num_train_timesteps < 2:
                raise ValueError("num_train_timesteps must be at least two")
        object.__setattr__(self, "timestep_sampler", sampler)
        object.__setattr__(self, "logit_mean", float(self.logit_mean))
        object.__setattr__(self, "logit_std", float(self.logit_std))
        object.__setattr__(self, "min_sigma", minimum)
        object.__setattr__(self, "max_sigma", maximum)
        object.__setattr__(self, "flow_shift", flow_shift)
        object.__setattr__(self, "num_train_timesteps", num_train_timesteps)


class FlowMatchingObjective:
    """Minimal velocity-prediction objective for a model adapter."""

    prediction_type = "flow_velocity"

    def __init__(self, config: FlowMatchingConfig | None = None) -> None:
        self.config = config or FlowMatchingConfig()

    def _sample_noise_levels(
        self,
        batch_size: int,
        *,
        device: object,
        generator: object | None = None,
    ) -> tuple[object, object]:
        torch = _require_torch()
        if isinstance(batch_size, bool) or int(batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.config.timestep_sampler == "uniform":
            unit = torch.rand(batch_size, device=device, dtype=torch.float32, generator=generator)
        else:
            logits = torch.randn(batch_size, device=device, dtype=torch.float32, generator=generator)
            unit = torch.sigmoid(logits * self.config.logit_std + self.config.logit_mean)

        if self.config.num_train_timesteps is None:
            base_sigmas = self.config.min_sigma + unit * (self.config.max_sigma - self.config.min_sigma)
            timesteps = base_sigmas
        else:
            steps = self.config.num_train_timesteps
            # ``rand``/``sigmoid`` are in [0, 1), so the upper clamp is a
            # defensive guard for reduced-precision or future samplers.
            timesteps = torch.floor(unit * steps).to(torch.long).clamp_(0, steps - 1)
            discrete_unit = timesteps.to(torch.float32) / float(steps)
            base_sigmas = self.config.min_sigma + discrete_unit * (self.config.max_sigma - self.config.min_sigma)
        return flow_shift_sigmas(base_sigmas, self.config.flow_shift), timesteps

    def sample_sigmas(self, batch_size: int, *, device: object, generator: object | None = None):
        """Sample effective corruption sigmas.

        Use :meth:`corrupt` when raw discrete timestep indices are also needed.
        """

        sigmas, _ = self._sample_noise_levels(batch_size, device=device, generator=generator)
        return sigmas

    def corrupt(self, batch: PreparedBatch, *, generator: object | None = None) -> ObjectiveBatch:
        torch = _require_torch()
        clean_tree = batch.clean_latents
        first = next(iter(clean_tree.values())) if isinstance(clean_tree, Mapping) else clean_tree
        if not torch.is_tensor(first):
            raise TypeError("clean_latents must contain torch.Tensor values")
        sigmas, timesteps = self._sample_noise_levels(
            batch.batch_size,
            device=first.device,
            generator=generator,
        )

        def corrupt_one(clean):
            if not torch.is_tensor(clean):
                raise TypeError("clean_latents must contain torch.Tensor values")
            noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
            local_sigmas = sigmas.to(device=clean.device, dtype=clean.dtype)
            return flow_interpolate(clean, noise, local_sigmas), flow_velocity_target(clean, noise), noise

        if isinstance(clean_tree, Mapping):
            triples = {key: corrupt_one(clean) for key, clean in clean_tree.items()}
            model_input = {key: values[0] for key, values in triples.items()}
            target = {key: values[1] for key, values in triples.items()}
            noise = {key: values[2] for key, values in triples.items()}
        else:
            model_input, target, noise = corrupt_one(clean_tree)

        metadata = dict(batch.metadata)
        metadata["objective"] = "flow_matching"
        metadata["prediction_type"] = self.prediction_type
        metadata["flow_shift"] = self.config.flow_shift
        metadata["num_train_timesteps"] = self.config.num_train_timesteps
        return ObjectiveBatch(
            sample_ids=batch.sample_ids,
            model_input=model_input,
            target=target,
            sigmas=sigmas,
            timesteps=timesteps,
            conditioning=batch.conditioning,
            noise=noise,
            loss_mask=batch.loss_mask,
            sample_weights=batch.sample_weights,
            metadata=metadata,
        )

    def compute_loss(self, prediction: object, batch: ObjectiveBatch) -> TrainStepResult:
        torch = _require_torch()
        target_tree = batch.target
        if isinstance(target_tree, Mapping):
            if not isinstance(prediction, Mapping) or set(prediction) != set(target_tree):
                raise ValueError("prediction keys must match target keys")
            mask_tree = batch.loss_mask
            results: dict[str, FlowMatchingLoss] = {}
            for key, target in target_tree.items():
                mask = mask_tree.get(key) if isinstance(mask_tree, Mapping) else mask_tree
                results[key] = flow_matching_mse(
                    prediction[key],
                    target,
                    loss_mask=mask,
                    sample_weights=batch.sample_weights,
                )
        else:
            if isinstance(prediction, Mapping):
                raise ValueError("a tensor target requires a tensor prediction")
            results = {
                "main": flow_matching_mse(
                    prediction,
                    target_tree,
                    loss_mask=batch.loss_mask,
                    sample_weights=batch.sample_weights,
                )
            }

        numerator = sum(
            (result.numerator for result in results.values()),
            torch.zeros((), device=next(iter(results.values())).loss.device),
        )
        denominator = sum(
            (result.denominator for result in results.values()),
            torch.zeros((), device=next(iter(results.values())).loss.device),
        )
        loss = numerator / denominator
        component_losses = {f"flow_matching/{key}": result.loss for key, result in results.items()}
        losses = {"flow_matching": loss, **component_losses}
        target_values = target_tree.values() if isinstance(target_tree, Mapping) else (target_tree,)
        latent_tokens = sum(
            int(value.shape[0]) * prod(int(size) for size in value.shape[2:]) for value in target_values
        )
        return TrainStepResult(
            loss=loss,
            losses=losses,
            metrics={
                "loss_numerator": numerator.detach(),
                "loss_denominator": denominator.detach(),
                "sigma_mean": batch.sigmas.float().mean().detach(),
                "sigma_min": batch.sigmas.float().min().detach(),
                "sigma_max": batch.sigmas.float().max().detach(),
            },
            sample_count=batch.batch_size,
            latent_token_count=latent_tokens,
            diagnostics={
                "prediction_type": self.prediction_type,
                "timestep_sampler": self.config.timestep_sampler,
                "flow_shift": self.config.flow_shift,
                "num_train_timesteps": self.config.num_train_timesteps,
            },
        )

    def loss_denominator(self, batch: ObjectiveBatch) -> object:
        """Return the exact weight denominator used for microbatch accumulation."""

        target_tree = batch.target
        if isinstance(target_tree, Mapping):
            mask_tree = batch.loss_mask
            denominators = []
            for key, target in target_tree.items():
                mask = mask_tree.get(key) if isinstance(mask_tree, Mapping) else mask_tree
                denominators.append(
                    flow_matching_denominator(
                        target,
                        loss_mask=mask,
                        sample_weights=batch.sample_weights,
                    )
                )
            return sum(denominators, denominators[0].new_zeros(()))
        return flow_matching_denominator(
            target_tree,
            loss_mask=batch.loss_mask,
            sample_weights=batch.sample_weights,
        )

    def prepared_loss_denominator(self, batch: PreparedBatch) -> object:
        """Return the denominator before allocating corruption tensors."""

        clean_tree = batch.clean_latents
        if isinstance(clean_tree, Mapping):
            mask_tree = batch.loss_mask
            denominators = []
            for key, clean in clean_tree.items():
                mask = mask_tree.get(key) if isinstance(mask_tree, Mapping) else mask_tree
                denominators.append(
                    flow_matching_denominator(
                        clean,
                        loss_mask=mask,
                        sample_weights=batch.sample_weights,
                    )
                )
            return sum(denominators, denominators[0].new_zeros(()))
        return flow_matching_denominator(
            clean_tree,
            loss_mask=batch.loss_mask,
            sample_weights=batch.sample_weights,
        )


__all__ = [
    "FlowMatchingConfig",
    "FlowMatchingLoss",
    "FlowMatchingObjective",
    "flow_clean_from_velocity",
    "flow_match_solver_sigmas",
    "flow_interpolate",
    "flow_matching_denominator",
    "flow_matching_mse",
    "flow_noise_from_velocity",
    "flow_shift_sigmas",
    "flow_velocity_target",
]
