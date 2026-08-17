"""Minimal, correctness-first single-device native training stepper."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import replace

import torch
from torch import nn

from worldfoundry.core.gradient import clip_grad_norm_
from worldfoundry.training.api.contracts import (
    TrainingBatch,
    TrainingObjective,
    TrainModelAdapter,
    TrainStepResult,
)
from worldfoundry.training.optimization import (
    audit_optimizer_parameters,
    build_adamw,
    trainable_parameters,
)

SINGLE_DEVICE_ENGINE_STATE_SCHEMA = "worldfoundry-training-engine-single"


class SingleDeviceTrainEngine:
    """Run finite-gated optimizer steps on one device.

    Multi-microbatch steps use the objective's data-dependent denominator to
    scale every backward call.  This preserves token/mask/sample weighting
    without retaining all forward activation graphs at once.
    """

    def __init__(
        self,
        adapter: TrainModelAdapter,
        objective: TrainingObjective,
        optimizer: torch.optim.Optimizer,
        *,
        max_grad_norm: float | None = 1.0,
        autocast_dtype: torch.dtype | None = None,
        train_batch_end: Callable[[], None] | None = None,
        optimizer_step_end: Callable[[], None] | None = None,
    ) -> None:
        module = getattr(adapter, "trainable_module", None)
        if not isinstance(module, nn.Module):
            raise TypeError("adapter.trainable_module must be an nn.Module")
        if getattr(adapter, "prediction_type", None) != getattr(objective, "prediction_type", None):
            raise ValueError(
                "adapter/objective prediction types differ: "
                f"{getattr(adapter, 'prediction_type', None)!r} vs "
                f"{getattr(objective, 'prediction_type', None)!r}"
            )
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer")
        resolved_max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        if resolved_max_grad_norm is not None and (
            not torch.isfinite(torch.tensor(resolved_max_grad_norm)) or resolved_max_grad_norm <= 0
        ):
            raise ValueError("max_grad_norm must be finite and positive")
        if autocast_dtype not in {None, torch.float16, torch.bfloat16}:
            raise ValueError("autocast_dtype must be float16, bfloat16, or None")
        if train_batch_end is not None and not callable(train_batch_end):
            raise TypeError("train_batch_end must be callable or None")
        if optimizer_step_end is not None and not callable(optimizer_step_end):
            raise TypeError("optimizer_step_end must be callable or None")

        parameters = audit_optimizer_parameters(
            optimizer,
            trainable_parameters(module),
            role="single-device",
        )
        devices = {parameter.device for parameter in parameters}
        if len(devices) != 1:
            raise ValueError(f"single-device engine found parameters on multiple devices: {devices}")
        device = next(iter(devices))
        if device.type == "cpu" and autocast_dtype is torch.float16:
            raise ValueError("CPU autocast does not support the float16 training path")

        self.adapter = adapter
        self.objective = objective
        self.optimizer = optimizer
        self.parameters = parameters
        self.device = device
        self.max_grad_norm = resolved_max_grad_norm
        self.autocast_dtype = autocast_dtype
        self.grad_scaler = (
            torch.amp.GradScaler("cuda")
            if device.type == "cuda" and autocast_dtype is torch.float16
            else None
        )
        self.train_batch_end = train_batch_end
        self.optimizer_step_end = optimizer_step_end
        self.global_step = 0
        self._phase = "idle"
        self._poisoned = False

    @property
    def is_poisoned(self) -> bool:
        return self._poisoned

    def _ensure_ready(self) -> None:
        if self._poisoned:
            raise RuntimeError(
                f"{type(self).__name__} is poisoned after optimizer.step began; "
                "restore the complete training state into a fresh engine"
            )
        if self._phase != "idle":
            raise RuntimeError(f"{type(self).__name__} already has an active training step")

    def _begin_training_step(self) -> None:
        self._ensure_ready()
        self._phase = "pre-step"

    def _mark_optimizer_step_started(self) -> None:
        if self._phase != "pre-step":
            raise RuntimeError("optimizer.step can only begin after pre-step work")
        # Keep this phase until the complete TrainStepResult has been built.
        # Once optimizer.step is called, even an exception in metrics/reduction
        # code leaves the parameter/optimizer commit status unknowable.
        self._phase = "optimizer-step"

    def _complete_training_step(self) -> None:
        if self._phase != "optimizer-step":
            raise RuntimeError("training step completed without an optimizer commit")
        self._phase = "idle"

    def _abort_training_step(self, *, force_poison: bool = False) -> None:
        if self._phase == "optimizer-step" or force_poison:
            self._poisoned = True
        try:
            self.optimizer.zero_grad(set_to_none=True)
        except BaseException:
            # A failed cleanup cannot establish a safe retry boundary even if
            # optimizer.step had not started.
            self._poisoned = True
        self._phase = "idle"

    def _autocast(self):
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    def _backward(self, loss: torch.Tensor) -> None:
        if self.grad_scaler is None:
            loss.backward()
            return
        self.grad_scaler.scale(loss).backward()

    def _unscale_gradients(self) -> None:
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)

    def _active_amp_grad_scaler(self) -> torch.amp.GradScaler | None:
        """Return the scaler only when torch-AMP overflow skipping applies.

        A real, enabled ``torch.amp.GradScaler`` owns non-finite gradients:
        ``step()`` skips the update and ``update()`` lowers the scale.
        Duck-typed test scalers and disabled scalers keep the legacy
        fail-stop clip semantics instead.
        """

        scaler = self.grad_scaler
        if isinstance(scaler, torch.amp.GradScaler) and scaler.is_enabled():
            return scaler
        return None

    def _step_optimizer(self) -> bool:
        """Run one optimizer step; report whether parameters were updated."""

        if self.grad_scaler is None:
            self.optimizer.step()
            return True
        scaler = self._active_amp_grad_scaler()
        if scaler is None:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            return True
        scale_before = scaler.get_scale()
        scaler.step(self.optimizer)
        scaler.update()
        # GradScaler enforces backoff_factor < 1 and only applies it when
        # unscale_ recorded non-finite gradients, so the scale strictly
        # decreases if and only if this optimizer step was skipped.
        return not scaler.get_scale() < scale_before

    def train_step(
        self,
        batch: TrainingBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> TrainStepResult:
        self._begin_training_step()
        try:
            self.optimizer.zero_grad(set_to_none=True)
            prepared = self.adapter.prepare_batch(batch)
            objective_batch = self.objective.corrupt(prepared, generator=generator)
            with self._autocast():
                prediction = self.adapter.forward_train(objective_batch)
            # Objective reduction owns explicit FP32 math and stays outside the
            # autocast region even when model forward uses BF16/FP16.
            result = self.objective.compute_loss(prediction, objective_batch)
            if result.skipped:
                raise RuntimeError("single-device engine cannot optimizer-step a pre-skipped result")
            if not isinstance(result.loss, torch.Tensor) or result.loss.numel() != 1:
                raise TypeError("training objective must return one scalar tensor loss")
            if not bool(torch.isfinite(result.loss.detach()).all()):
                raise FloatingPointError("non-finite training loss; optimizer step was not applied")

            self._backward(result.loss)
            self._unscale_gradients()
            grad_norm = clip_grad_norm_(
                self.parameters,
                float("inf") if self.max_grad_norm is None else self.max_grad_norm,
                # AMP fp16 overflow is a recoverable event: the scaler must
                # reach step()/update() to skip this step and lower the
                # scale, so clipping cannot raise on the non-finite norm.
                error_if_nonfinite=self._active_amp_grad_scaler() is None,
            )
            self._mark_optimizer_step_started()
            stepped = self._step_optimizer()
            self.global_step += 1
            if self.optimizer_step_end is not None:
                self.optimizer_step_end()
            if self.train_batch_end is not None:
                self.train_batch_end()

            metrics: dict[str, object] = dict(result.metrics)
            if stepped:
                metrics["grad_norm"] = grad_norm.detach()
            else:
                # Durable metrics use canonical JSON, which rejects the
                # non-finite norm; the explicit flag records the skip.
                metrics["optimizer_step_skipped"] = True
            metrics["global_step"] = torch.tensor(self.global_step, device=result.loss.device, dtype=torch.int64)
            diagnostics: dict[str, object] = dict(result.diagnostics)
            diagnostics.update(
                {
                    "engine": "single-device",
                    "device": str(self.device),
                    "autocast_dtype": None if self.autocast_dtype is None else str(self.autocast_dtype),
                    "grad_scaling": self.grad_scaler is not None,
                    "max_grad_norm": self.max_grad_norm,
                }
            )
            completed = replace(result, metrics=metrics, diagnostics=diagnostics, skipped=not stepped)
            self._complete_training_step()
            return completed
        except BaseException:
            self._abort_training_step()
            raise

    def train_accumulation(
        self,
        batches: Sequence[TrainingBatch],
        *,
        generator: torch.Generator | None = None,
    ) -> TrainStepResult:
        """Apply one token-weighted optimizer step over several microbatches."""

        self._ensure_ready()
        microbatches = tuple(batches)
        if not microbatches:
            raise ValueError("gradient accumulation requires at least one microbatch")
        if not all(isinstance(batch, TrainingBatch) for batch in microbatches):
            raise TypeError("gradient accumulation batches must be TrainingBatch values")
        if len(microbatches) == 1:
            return self.train_step(microbatches[0], generator=generator)
        denominator_fn = getattr(self.objective, "prepared_loss_denominator", None)
        if not callable(denominator_fn):
            raise TypeError("multi-microbatch accumulation requires objective.prepared_loss_denominator")

        self._begin_training_step()
        try:
            self.optimizer.zero_grad(set_to_none=True)
            prepared_batches = tuple(self.adapter.prepare_batch(batch) for batch in microbatches)
            denominators = tuple(denominator_fn(batch).detach().float() for batch in prepared_batches)
            total_denominator = sum(
                denominators,
                torch.zeros((), device=denominators[0].device, dtype=torch.float32),
            )
            if not bool(torch.isfinite(total_denominator)) or not bool(total_denominator > 0):
                raise FloatingPointError("gradient accumulation denominator must be finite and positive")

            total_numerator = torch.zeros_like(total_denominator)
            component_numerators: dict[str, torch.Tensor] = {}
            sigma_sum = torch.zeros_like(total_denominator)
            sigma_min: torch.Tensor | None = None
            sigma_max: torch.Tensor | None = None
            sample_count = 0
            latent_token_count = 0
            first_diagnostics: Mapping[str, object] | None = None

            for index, (prepared, denominator) in enumerate(zip(prepared_batches, denominators)):
                objective_batch = self.objective.corrupt(prepared, generator=generator)
                with self._autocast():
                    prediction = self.adapter.forward_train(objective_batch)
                result = self.objective.compute_loss(prediction, objective_batch)
                if result.skipped:
                    raise RuntimeError("single-device accumulation cannot include a skipped result")
                if not isinstance(result.loss, torch.Tensor) or result.loss.numel() != 1:
                    raise TypeError("training objective must return one scalar tensor loss")
                if not bool(torch.isfinite(result.loss.detach()).all()):
                    raise FloatingPointError("non-finite accumulated training loss")
                reported_denominator = result.metrics.get("loss_denominator")
                if not isinstance(reported_denominator, torch.Tensor):
                    raise TypeError("accumulated objective must report tensor loss_denominator")
                if not torch.equal(reported_denominator.detach().float(), denominator):
                    raise RuntimeError("objective denominator changed between preparation and loss reduction")

                self._backward(result.loss * (denominator / total_denominator))
                if index + 1 < len(prepared_batches) and self.train_batch_end is not None:
                    self.train_batch_end()
                numerator = result.metrics.get("loss_numerator")
                if not isinstance(numerator, torch.Tensor):
                    raise TypeError("accumulated objective must report tensor loss_numerator")
                total_numerator += numerator.detach().float()
                for name, value in result.losses.items():
                    if isinstance(value, torch.Tensor):
                        component_numerators[name] = (
                            component_numerators.get(
                                name,
                                torch.zeros_like(total_denominator),
                            )
                            + value.detach().float() * denominator
                        )
                sigma_mean = result.metrics.get("sigma_mean")
                current_min = result.metrics.get("sigma_min")
                current_max = result.metrics.get("sigma_max")
                if all(isinstance(value, torch.Tensor) for value in (sigma_mean, current_min, current_max)):
                    sigma_sum += sigma_mean.detach().float() * result.sample_count
                    sigma_min = (
                        current_min.detach().float()
                        if sigma_min is None
                        else torch.minimum(
                            sigma_min,
                            current_min.detach().float(),
                        )
                    )
                    sigma_max = (
                        current_max.detach().float()
                        if sigma_max is None
                        else torch.maximum(
                            sigma_max,
                            current_max.detach().float(),
                        )
                    )
                sample_count += result.sample_count
                latent_token_count += result.latent_token_count
                if first_diagnostics is None:
                    first_diagnostics = result.diagnostics

            self._unscale_gradients()
            grad_norm = clip_grad_norm_(
                self.parameters,
                float("inf") if self.max_grad_norm is None else self.max_grad_norm,
                # See train_step: the AMP scaler owns non-finite gradients.
                error_if_nonfinite=self._active_amp_grad_scaler() is None,
            )
            self._mark_optimizer_step_started()
            stepped = self._step_optimizer()
            self.global_step += 1
            if self.optimizer_step_end is not None:
                self.optimizer_step_end()
            if self.train_batch_end is not None:
                self.train_batch_end()

            loss = total_numerator / total_denominator
            losses = {name: numerator / total_denominator for name, numerator in component_numerators.items()}
            metrics: dict[str, object] = {
                "loss_numerator": total_numerator,
                "loss_denominator": total_denominator,
                "global_step": torch.tensor(self.global_step, device=loss.device, dtype=torch.int64),
                "microbatch_count": torch.tensor(len(microbatches), device=loss.device, dtype=torch.int64),
            }
            if stepped:
                metrics["grad_norm"] = grad_norm.detach()
            else:
                metrics["optimizer_step_skipped"] = True
            if sample_count and sigma_min is not None and sigma_max is not None:
                metrics.update(
                    {
                        "sigma_mean": sigma_sum / sample_count,
                        "sigma_min": sigma_min,
                        "sigma_max": sigma_max,
                    }
                )
            diagnostics = dict(first_diagnostics or {})
            diagnostics.update(
                {
                    "engine": "single-device",
                    "device": str(self.device),
                    "autocast_dtype": None if self.autocast_dtype is None else str(self.autocast_dtype),
                    "grad_scaling": self.grad_scaler is not None,
                    "max_grad_norm": self.max_grad_norm,
                    "gradient_accumulation": "token-weighted",
                }
            )
            completed = TrainStepResult(
                loss=loss,
                losses=losses,
                metrics=metrics,
                sample_count=sample_count,
                latent_token_count=latent_token_count,
                skipped=not stepped,
                diagnostics=diagnostics,
            )
            self._complete_training_step()
            return completed
        except BaseException:
            self._abort_training_step()
            raise

    def state_dict(self) -> dict[str, object]:
        self._ensure_ready()
        return {
            "schema": SINGLE_DEVICE_ENGINE_STATE_SCHEMA,
            "global_step": self.global_step,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        self._ensure_ready()
        if not isinstance(state_dict, Mapping):
            raise TypeError("engine state_dict must be a mapping")
        expected = {"schema", "global_step"}
        if set(state_dict) != expected:
            raise ValueError(f"engine state fields must be {sorted(expected)}; got {sorted(state_dict)}")
        if state_dict["schema"] != SINGLE_DEVICE_ENGINE_STATE_SCHEMA:
            raise ValueError(f"unsupported engine state schema: {state_dict['schema']!r}")
        step = state_dict["global_step"]
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("engine global_step must be an integer, not bool")
        if step < 0:
            raise ValueError("engine global_step must be a non-negative integer")
        # Validate the complete payload before clearing gradients or changing
        # the committed optimizer-step counter.
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step = step


__all__ = [
    "SINGLE_DEVICE_ENGINE_STATE_SCHEMA",
    "SingleDeviceTrainEngine",
    "build_adamw",
    "trainable_parameters",
]
