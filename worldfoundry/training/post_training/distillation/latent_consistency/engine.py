"""Atomic, distributed, and resumable latent consistency training engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from worldfoundry.core.gradient import clip_grad_norm_
from worldfoundry.training.optimization import (
    audit_optimizer_parameters,
    trainable_parameters,
)

from ...shared.accumulation import (
    accumulation_context,
    global_denominator,
    global_loss_statistics,
    role_metrics,
)
from ...shared.distributed import PostTrainingParallelContext
from ...shared.validation import non_negative_int, positive_float, validate_stateful_or_none
from ..causal_consistency.ema import FrozenModuleEMA
from .contracts import (
    LatentConsistencyRandomInputs,
    LatentConsistencyTrainingBatch,
)
from .objective import LatentConsistencyLossResult, LatentConsistencyObjective

LATENT_CONSISTENCY_ENGINE_STATE_SCHEMA = "worldfoundry-latent-consistency-engine"


def _ema_module(module: nn.Module) -> nn.Module:
    """Unwrap DDP so a replicated student can update an unwrapped frozen target."""

    from torch.nn.parallel import DistributedDataParallel

    return module.module if isinstance(module, DistributedDataParallel) else module


def _microbatches(
    value: LatentConsistencyTrainingBatch | Sequence[LatentConsistencyTrainingBatch],
    *,
    expected: int,
) -> tuple[LatentConsistencyTrainingBatch, ...]:
    if isinstance(value, LatentConsistencyTrainingBatch):
        batches = (value,)
    elif isinstance(value, Sequence):
        batches = tuple(value)
    else:
        raise TypeError("latent consistency batch must be one batch or a sequence")
    if len(batches) != expected:
        raise ValueError(f"latent consistency optimizer iteration requires {expected} microbatches; got {len(batches)}")
    if not all(isinstance(batch, LatentConsistencyTrainingBatch) for batch in batches):
        raise TypeError("every latent consistency microbatch must use the native batch contract")
    return batches


@dataclass(frozen=True, slots=True)
class LatentConsistencyTrainResult:
    loss: Tensor
    metrics: Mapping[str, object]


class NativeLatentConsistencyTrainEngine:
    """Commit one online-student optimizer update followed by one EMA update."""

    def __init__(
        self,
        *,
        student_module: nn.Module,
        teacher_module: nn.Module,
        ema_target_module: nn.Module,
        objective: LatentConsistencyObjective,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        scheduler: object | None = None,
        parallel_context: PostTrainingParallelContext | None = None,
        seed: int = 0,
        initialize_ema_target: bool = True,
    ) -> None:
        modules = (student_module, teacher_module, ema_target_module)
        if not all(isinstance(module, nn.Module) for module in modules):
            raise TypeError("all latent consistency roles must be nn.Module values")
        if len({id(module) for module in modules}) != 3:
            raise ValueError("latent consistency roles must be distinct modules")
        inventories = tuple({id(parameter) for parameter in module.parameters()} for module in modules)
        if any(
            inventories[left] & inventories[right]
            for left in range(len(inventories))
            for right in range(left + 1, len(inventories))
        ):
            raise ValueError("latent consistency roles cannot share parameters")
        if not isinstance(objective, LatentConsistencyObjective):
            raise TypeError("objective must be LatentConsistencyObjective")
        if (
            objective.student_module is not student_module
            or objective.teacher_module is not teacher_module
            or objective.ema_target_module is not ema_target_module
        ):
            raise ValueError("objective adapters and engine role modules differ")
        accumulation = non_negative_int(
            gradient_accumulation_steps,
            field_name="gradient_accumulation_steps",
        )
        if accumulation == 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(initialize_ema_target, bool):
            raise TypeError("initialize_ema_target must be bool")
        validate_stateful_or_none(scheduler, field_name="scheduler")

        teacher_module.requires_grad_(False)
        ema_target_module.requires_grad_(False)
        teacher_module.eval()
        ema_target_module.eval()
        parameters = trainable_parameters(student_module)
        audit_optimizer_parameters(
            optimizer,
            parameters,
            role="latent consistency student",
        )
        ema_source = _ema_module(student_module)
        ema_target = _ema_module(ema_target_module)
        if initialize_ema_target:
            try:
                ema_target.load_state_dict(ema_source.state_dict(), strict=True)
            except RuntimeError as error:
                raise ValueError("student and EMA target state inventories must match exactly") from error
        ema_updater = FrozenModuleEMA(
            ema_source,
            ema_target,
            decay=objective.config.ema_decay,
            initialize_target=False,
        )
        context = parallel_context or PostTrainingParallelContext.current()
        context.audit_synchronized_module(
            student_module,
            role="latent consistency student",
        )
        rng_device = parameters[0].device
        rng = torch.Generator(device=rng_device)
        rng.manual_seed((seed + context.rank) % (2**63 - 1))

        self.student_module = student_module
        self.teacher_module = teacher_module
        self.ema_target_module = ema_target_module
        self.objective = objective
        self.optimizer = optimizer
        self.student_parameters = parameters
        self.max_grad_norm = positive_float(max_grad_norm, field_name="max_grad_norm")
        self.gradient_accumulation_steps = accumulation
        self.scheduler = scheduler
        self.parallel_context = context
        self.ema_updater = ema_updater
        self._rng = rng
        self._rng_device = str(rng_device)
        self.global_step = 0
        self.optimizer_steps = 0
        self._phase = "idle"
        self._poisoned = False

    def sample_random_inputs(
        self,
        batch: LatentConsistencyTrainingBatch,
    ) -> LatentConsistencyRandomInputs:
        """Sample noise, DDIM pair, and guidance once per training example."""

        if not isinstance(batch, LatentConsistencyTrainingBatch):
            raise TypeError("batch must be LatentConsistencyTrainingBatch")
        clean = batch.clean_latents
        if not isinstance(clean, Tensor) or str(clean.device) != self._rng_device:
            raise ValueError("latent consistency batch device differs from the checkpointed RNG device")
        noise = torch.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.dtype,
            generator=self._rng,
        )
        indices = torch.randint(
            0,
            self.objective.pair_count,
            (batch.batch_size,),
            device=clean.device,
            dtype=torch.int64,
            generator=self._rng,
        )
        guidance = torch.rand(
            (batch.batch_size,),
            device=clean.device,
            dtype=torch.float32,
            generator=self._rng,
        )
        config = self.objective.config
        guidance = (
            guidance * (config.guidance_coefficient_max - config.guidance_coefficient_min)
            + config.guidance_coefficient_min
        )
        return LatentConsistencyRandomInputs(
            noise=noise,
            timestep_indices=indices,
            guidance_coefficients=guidance,
        )

    def train_step(
        self,
        batch: LatentConsistencyTrainingBatch | Sequence[LatentConsistencyTrainingBatch],
    ) -> LatentConsistencyTrainResult:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("latent consistency engine has a partially committed update; restore a checkpoint")
        batches = _microbatches(
            batch,
            expected=self.gradient_accumulation_steps,
        )
        weights: list[Tensor] = []
        for microbatch in batches:
            clean = microbatch.clean_latents
            if not isinstance(clean, Tensor) or str(clean.device) != self._rng_device:
                raise ValueError("latent consistency batch device differs from the checkpointed RNG device")
            weights.append(
                torch.tensor(
                    self.objective.loss_denominator(microbatch),
                    device=self.student_parameters[0].device,
                    dtype=torch.float32,
                )
            )
        total_weight = global_denominator(weights, self.parallel_context)
        results: list[LatentConsistencyLossResult] = []
        self.optimizer.zero_grad(set_to_none=True)
        optimizer_mutated = False
        try:
            self._phase = "student-backward"
            self.student_module.train()
            self.teacher_module.eval()
            self.ema_target_module.eval()
            for index, (microbatch, weight) in enumerate(zip(batches, weights, strict=True)):
                random_inputs = self.sample_random_inputs(microbatch)
                with accumulation_context(
                    self.student_module,
                    final_microbatch=index + 1 == len(batches),
                ):
                    result = self.objective.loss(
                        microbatch,
                        random_inputs=random_inputs,
                    )
                    reported = torch.as_tensor(
                        result.metrics.get("loss_denominator"),
                        device=weight.device,
                        dtype=torch.float32,
                    )
                    if reported.numel() != 1 or not torch.equal(
                        reported.detach().reshape(()),
                        weight,
                    ):
                        raise RuntimeError("latent consistency declared and realized loss denominators differ")
                    gradient_weight = weight / total_weight * float(self.parallel_context.world_size)
                    (result.loss * gradient_weight).backward()
                results.append(result)
            if any(parameter.grad is not None for parameter in self.teacher_module.parameters()):
                raise RuntimeError("latent consistency teacher received gradients")
            if any(parameter.grad is not None for parameter in self.ema_target_module.parameters()):
                raise RuntimeError("latent consistency EMA target received gradients")
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
            self.ema_updater.update()
            self.optimizer.zero_grad(set_to_none=True)
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
            "accumulated_microbatches": len(batches),
            "grad_norm": grad_norm.detach(),
            **role_metrics(
                results,
                global_numerator=numerator,
                global_denominator=denominator,
            ),
        }
        return LatentConsistencyTrainResult(
            loss=loss.detach().float(),
            metrics=metrics,
        )

    def state_dict(self) -> dict[str, object]:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("cannot checkpoint a partially committed latent consistency update")
        return {
            "schema": LATENT_CONSISTENCY_ENGINE_STATE_SCHEMA,
            "global_step": self.global_step,
            "optimizer_steps": self.optimizer_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "data_parallel_size": self.parallel_context.world_size,
            "rng_device": self._rng_device,
            "rng_state": self._rng.get_state().clone(),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("latent consistency engine state must be a mapping")
        expected = {
            "schema",
            "global_step",
            "optimizer_steps",
            "gradient_accumulation_steps",
            "data_parallel_size",
            "rng_device",
            "rng_state",
        }
        if set(state_dict) != expected:
            raise ValueError("latent consistency engine state fields differ from the active schema")
        if state_dict["schema"] != LATENT_CONSISTENCY_ENGINE_STATE_SCHEMA:
            raise ValueError(f"unsupported latent consistency engine schema: {state_dict['schema']!r}")
        active = {
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "data_parallel_size": self.parallel_context.world_size,
            "rng_device": self._rng_device,
        }
        for name, value in active.items():
            if state_dict[name] != value:
                raise ValueError(f"saved latent consistency {name} differs from the active engine")
        global_step = non_negative_int(
            state_dict["global_step"],
            field_name="global_step",
        )
        optimizer_steps = non_negative_int(
            state_dict["optimizer_steps"],
            field_name="optimizer_steps",
        )
        if optimizer_steps != global_step:
            raise ValueError("saved latent consistency optimizer counter violates its cadence")
        rng_state = state_dict["rng_state"]
        if not isinstance(rng_state, Tensor) or rng_state.dtype != torch.uint8 or rng_state.ndim != 1:
            raise ValueError("saved latent consistency RNG state is invalid")
        self._rng.set_state(rng_state.detach().cpu())
        self.global_step = global_step
        self.optimizer_steps = optimizer_steps
        self._phase = "idle"
        self._poisoned = False


__all__ = [
    "LATENT_CONSISTENCY_ENGINE_STATE_SCHEMA",
    "LatentConsistencyTrainResult",
    "NativeLatentConsistencyTrainEngine",
]
