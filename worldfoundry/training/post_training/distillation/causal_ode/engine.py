"""Atomic optimizer engine for native Causal ODE regression."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from worldfoundry.core.gradient import clip_grad_norm_
from worldfoundry.training.optimization import audit_optimizer_parameters, trainable_parameters

from ...shared.accumulation import accumulation_context, global_denominator, global_loss_statistics, role_metrics
from ...shared.distributed import PostTrainingParallelContext
from ...shared.validation import non_negative_int, positive_float, validate_stateful_or_none
from .contracts import CausalODETrainingBatch
from .objective import CausalODELossResult, CausalODEObjective, PreparedCausalODEBatch

CAUSAL_ODE_ENGINE_STATE_SCHEMA = "worldfoundry-causal-ode-engine"


def _microbatches(
    batch: CausalODETrainingBatch | Sequence[CausalODETrainingBatch],
    *,
    expected: int,
) -> tuple[CausalODETrainingBatch, ...]:
    if isinstance(batch, CausalODETrainingBatch):
        batches = (batch,)
    elif isinstance(batch, Sequence):
        batches = tuple(batch)
    else:
        raise TypeError("Causal ODE batch must be one batch or a sequence of microbatches")
    if len(batches) != expected:
        raise ValueError(
            f"Causal ODE optimizer iteration requires {expected} microbatches; got {len(batches)}"
        )
    if not all(isinstance(value, CausalODETrainingBatch) for value in batches):
        raise TypeError("every Causal ODE microbatch must be CausalODETrainingBatch")
    return batches


@dataclass(frozen=True, slots=True)
class CausalODETrainResult:
    loss: Tensor
    trajectory_indices: tuple[Tensor, ...]
    metrics: Mapping[str, object]


class NativeCausalODETrainEngine:
    """Own trajectory-index RNG, weighted accumulation, and one student commit."""

    def __init__(
        self,
        *,
        student_module: nn.Module,
        objective: CausalODEObjective,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: float = 10.0,
        gradient_accumulation_steps: int = 1,
        scheduler: object | None = None,
        parallel_context: PostTrainingParallelContext | None = None,
        seed: int = 0,
    ) -> None:
        if not isinstance(student_module, nn.Module):
            raise TypeError("student_module must be nn.Module")
        if not isinstance(objective, CausalODEObjective):
            raise TypeError("objective must be CausalODEObjective")
        if objective.student.module is not student_module:
            raise ValueError("objective student adapter does not own student_module")
        accumulation = non_negative_int(
            gradient_accumulation_steps,
            field_name="gradient_accumulation_steps",
        )
        if accumulation == 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        validate_stateful_or_none(scheduler, field_name="scheduler")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")

        parameters = trainable_parameters(student_module)
        audit_optimizer_parameters(optimizer, parameters, role="Causal ODE student")
        context = parallel_context or PostTrainingParallelContext.current()
        context.audit_synchronized_module(student_module, role="Causal ODE student")
        rng_device = parameters[0].device
        rng = torch.Generator(device=rng_device)
        rng.manual_seed((seed + context.rank) % (2**63 - 1))

        self.student_module = student_module
        self.objective = objective
        self.optimizer = optimizer
        self.student_parameters = parameters
        self.max_grad_norm = positive_float(max_grad_norm, field_name="max_grad_norm")
        self.gradient_accumulation_steps = accumulation
        self.scheduler = scheduler
        self.parallel_context = context
        self._rng = rng
        self._rng_device = str(rng_device)
        self.global_step = 0
        self.optimizer_steps = 0
        self._phase = "idle"
        self._poisoned = False

    @property
    def config_digest(self) -> str:
        return self.objective.config_digest

    def train_step(
        self,
        batch: CausalODETrainingBatch | Sequence[CausalODETrainingBatch],
    ) -> CausalODETrainResult:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("Causal ODE engine has a partially committed update; restore a checkpoint")
        batches = _microbatches(batch, expected=self.gradient_accumulation_steps)
        prepared: list[PreparedCausalODEBatch] = []
        weights: list[Tensor] = []
        for microbatch in batches:
            trajectories = microbatch.ode_trajectories
            if not isinstance(trajectories, Tensor) or str(trajectories.device) != self._rng_device:
                raise ValueError("Causal ODE batch device differs from the checkpointed RNG device")
            indices = self.objective.sample_trajectory_indices(microbatch, generator=self._rng)
            item = self.objective.prepare(microbatch, indices)
            prepared.append(item)
            weights.append(
                torch.tensor(
                    item.loss_denominator,
                    device=self.student_parameters[0].device,
                    dtype=torch.float32,
                )
            )
        total_weight = global_denominator(weights, self.parallel_context)
        results: list[CausalODELossResult] = []
        self.optimizer.zero_grad(set_to_none=True)
        optimizer_mutated = False
        try:
            self._phase = "student-backward"
            self.student_module.train()
            for index, (item, weight) in enumerate(zip(prepared, weights, strict=True)):
                with accumulation_context(
                    self.student_module,
                    final_microbatch=index + 1 == len(prepared),
                ):
                    result = self.objective.loss(item)
                    reported = torch.as_tensor(
                        result.metrics.get("loss_denominator"),
                        device=weight.device,
                        dtype=torch.float32,
                    )
                    if reported.numel() != 1 or not torch.equal(reported.reshape(()), weight):
                        raise RuntimeError("Causal ODE declared and realized loss denominators differ")
                    gradient_weight = weight / total_weight * float(self.parallel_context.world_size)
                    (result.loss * gradient_weight).backward()
                results.append(result)
            grad_norm = clip_grad_norm_(
                self.student_parameters,
                self.max_grad_norm,
                error_if_nonfinite=True,
            )
            optimizer_mutated = True
            self.optimizer.step()
            self.optimizer_steps += 1
            if self.scheduler is not None:
                self.scheduler.step()
            self.global_step += 1
            self._phase = "idle"
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            if optimizer_mutated:
                self._poisoned = True
            else:
                self._phase = "idle"
            raise

        numerator, denominator, loss = global_loss_statistics(
            results,
            weights,
            self.parallel_context,
        )
        metrics = {
            "global_step": self.global_step,
            "optimizer_steps": self.optimizer_steps,
            "accumulated_microbatches": len(prepared),
            "grad_norm": grad_norm.detach(),
            **role_metrics(
                results,
                global_numerator=numerator,
                global_denominator=denominator,
            ),
        }
        return CausalODETrainResult(
            loss=loss.detach().float(),
            trajectory_indices=tuple(item.trajectory_indices.detach() for item in prepared),
            metrics=metrics,
        )

    def state_dict(self) -> dict[str, object]:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("cannot checkpoint a partially committed Causal ODE update")
        return {
            "schema": CAUSAL_ODE_ENGINE_STATE_SCHEMA,
            "global_step": self.global_step,
            "optimizer_steps": self.optimizer_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "config_digest": self.config_digest,
            "data_parallel_size": self.parallel_context.world_size,
            "rng_device": self._rng_device,
            "rng_state": self._rng.get_state().clone(),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("Causal ODE engine state must be a mapping")
        expected = {
            "schema",
            "global_step",
            "optimizer_steps",
            "gradient_accumulation_steps",
            "config_digest",
            "data_parallel_size",
            "rng_device",
            "rng_state",
        }
        if set(state_dict) != expected:
            raise ValueError("Causal ODE engine state fields differ from the active schema")
        if state_dict["schema"] != CAUSAL_ODE_ENGINE_STATE_SCHEMA:
            raise ValueError(f"unsupported Causal ODE engine schema: {state_dict['schema']!r}")
        active = {
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "config_digest": self.config_digest,
            "data_parallel_size": self.parallel_context.world_size,
            "rng_device": self._rng_device,
        }
        for name, value in active.items():
            if state_dict[name] != value:
                raise ValueError(f"saved Causal ODE {name} differs from the active engine")
        global_step = non_negative_int(state_dict["global_step"], field_name="global_step")
        optimizer_steps = non_negative_int(
            state_dict["optimizer_steps"],
            field_name="optimizer_steps",
        )
        if optimizer_steps != global_step:
            raise ValueError("saved Causal ODE optimizer counter violates its update cadence")
        rng_state = state_dict["rng_state"]
        if not isinstance(rng_state, Tensor) or rng_state.dtype != torch.uint8 or rng_state.ndim != 1:
            raise ValueError("saved Causal ODE RNG state is invalid")
        self._rng.set_state(rng_state.detach().cpu())
        self.global_step = global_step
        self.optimizer_steps = optimizer_steps
        self._phase = "idle"
        self._poisoned = False


__all__ = ["CAUSAL_ODE_ENGINE_STATE_SCHEMA", "CausalODETrainResult", "NativeCausalODETrainEngine"]
