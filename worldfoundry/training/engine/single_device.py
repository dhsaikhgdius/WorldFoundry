"""Minimal, correctness-first single-device native training stepper."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
        max_grad_norm: float = 1.0,
        autocast_dtype: torch.dtype | None = None,
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
        resolved_max_grad_norm = float(max_grad_norm)
        if not torch.isfinite(torch.tensor(resolved_max_grad_norm)) or resolved_max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")
        if autocast_dtype not in {None, torch.float16, torch.bfloat16}:
            raise ValueError("autocast_dtype must be float16, bfloat16, or None")

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

            result.loss.backward()
            grad_norm = clip_grad_norm_(
                self.parameters,
                self.max_grad_norm,
                error_if_nonfinite=True,
            )
            self._mark_optimizer_step_started()
            self.optimizer.step()
            self.global_step += 1

            metrics: dict[str, object] = dict(result.metrics)
            metrics["grad_norm"] = grad_norm.detach()
            metrics["global_step"] = torch.tensor(self.global_step, device=result.loss.device, dtype=torch.int64)
            diagnostics: dict[str, object] = dict(result.diagnostics)
            diagnostics.update(
                {
                    "engine": "single-device",
                    "device": str(self.device),
                    "autocast_dtype": None if self.autocast_dtype is None else str(self.autocast_dtype),
                    "max_grad_norm": self.max_grad_norm,
                }
            )
            completed = replace(result, metrics=metrics, diagnostics=diagnostics)
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

            for prepared, denominator in zip(prepared_batches, denominators):
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

                (result.loss * (denominator / total_denominator)).backward()
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

            grad_norm = clip_grad_norm_(
                self.parameters,
                self.max_grad_norm,
                error_if_nonfinite=True,
            )
            self._mark_optimizer_step_started()
            self.optimizer.step()
            self.global_step += 1

            loss = total_numerator / total_denominator
            losses = {name: numerator / total_denominator for name, numerator in component_numerators.items()}
            metrics: dict[str, object] = {
                "loss_numerator": total_numerator,
                "loss_denominator": total_denominator,
                "grad_norm": grad_norm.detach(),
                "global_step": torch.tensor(self.global_step, device=loss.device, dtype=torch.int64),
                "microbatch_count": torch.tensor(len(microbatches), device=loss.device, dtype=torch.int64),
            }
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
