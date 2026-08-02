"""Atomic, resumable engine for native Causal Consistency Distillation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn

from worldfoundry.core.gradient import clip_grad_norm_
from worldfoundry.training.optimization import audit_optimizer_parameters, trainable_parameters

from ...shared.accumulation import accumulation_context, global_denominator, global_loss_statistics, role_metrics
from ...shared.distributed import PostTrainingParallelContext
from ...shared.validation import non_negative_int, positive_float, validate_stateful_or_none
from .contracts import CausalConsistencyTrainingBatch
from .ema import FrozenModuleEMA
from .objective import CausalConsistencyLossResult, CausalConsistencyObjective

CAUSAL_CONSISTENCY_ENGINE_STATE_SCHEMA = "worldfoundry-causal-consistency-engine"


def _microbatches(
    batch: CausalConsistencyTrainingBatch | Sequence[CausalConsistencyTrainingBatch],
    *,
    expected: int,
) -> tuple[CausalConsistencyTrainingBatch, ...]:
    if isinstance(batch, CausalConsistencyTrainingBatch):
        batches = (batch,)
    elif isinstance(batch, Sequence):
        batches = tuple(batch)
    else:
        raise TypeError("causal consistency batch must be one batch or a sequence")
    if len(batches) != expected:
        raise ValueError(
            "causal consistency optimizer iteration requires "
            f"{expected} microbatches; got {len(batches)}"
        )
    if not all(isinstance(value, CausalConsistencyTrainingBatch) for value in batches):
        raise TypeError("every causal consistency microbatch must have the native batch contract")
    return batches


@dataclass(frozen=True, slots=True)
class CausalConsistencyTrainResult:
    loss: Tensor
    pair_index: int
    metrics: Mapping[str, object]


class NativeCausalConsistencyTrainEngine:
    """Share one rank-synchronized pair across an accumulated optimizer step."""

    def __init__(
        self,
        *,
        student_module: nn.Module,
        teacher_module: nn.Module,
        ema_student_module: nn.Module,
        objective: CausalConsistencyObjective,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: float = 10.0,
        gradient_accumulation_steps: int = 1,
        scheduler: object | None = None,
        parallel_context: PostTrainingParallelContext | None = None,
        seed: int = 0,
        initialize_ema_target: bool = True,
    ) -> None:
        modules = (student_module, teacher_module, ema_student_module)
        if not all(isinstance(module, nn.Module) for module in modules):
            raise TypeError("all causal consistency roles must be nn.Module values")
        if len({id(module) for module in modules}) != 3:
            raise ValueError("causal consistency roles must be distinct modules")
        if not isinstance(objective, CausalConsistencyObjective):
            raise TypeError("objective must be CausalConsistencyObjective")
        if (
            objective.student.module is not student_module
            or objective.teacher.module is not teacher_module
            or objective.ema_student.module is not ema_student_module
        ):
            raise ValueError("objective adapters and engine role modules differ")
        accumulation = non_negative_int(
            gradient_accumulation_steps,
            field_name="gradient_accumulation_steps",
        )
        if accumulation == 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        validate_stateful_or_none(scheduler, field_name="scheduler")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(initialize_ema_target, bool):
            raise TypeError("initialize_ema_target must be bool")

        teacher_module.requires_grad_(False)
        ema_student_module.requires_grad_(False)
        teacher_module.eval()
        ema_student_module.eval()
        parameters = trainable_parameters(student_module)
        audit_optimizer_parameters(optimizer, parameters, role="causal consistency student")
        if any(parameter.requires_grad for parameter in teacher_module.parameters()):
            raise ValueError("causal consistency teacher must be frozen")
        if any(parameter.requires_grad for parameter in ema_student_module.parameters()):
            raise ValueError("causal consistency EMA target must be frozen")
        student_ids = {id(parameter) for parameter in student_module.parameters()}
        if student_ids & {id(parameter) for parameter in teacher_module.parameters()}:
            raise ValueError("student and causal teacher cannot share parameters")
        if student_ids & {id(parameter) for parameter in ema_student_module.parameters()}:
            raise ValueError("student and EMA target cannot share parameters")

        context = parallel_context or PostTrainingParallelContext.current()
        context.audit_synchronized_module(student_module, role="causal consistency student")
        noise_device = parameters[0].device
        pair_rng = torch.Generator(device="cpu")
        pair_rng.manual_seed(seed % (2**63 - 1))
        noise_rng = torch.Generator(device=noise_device)
        noise_rng.manual_seed((seed + context.rank) % (2**63 - 1))
        ema_updater = FrozenModuleEMA(
            student_module,
            ema_student_module,
            decay=objective.config.ema_decay,
            initialize_target=initialize_ema_target,
        )

        self.student_module = student_module
        self.teacher_module = teacher_module
        self.ema_student_module = ema_student_module
        self.objective = objective
        self.optimizer = optimizer
        self.student_parameters = parameters
        self.max_grad_norm = positive_float(max_grad_norm, field_name="max_grad_norm")
        self.gradient_accumulation_steps = accumulation
        self.scheduler = scheduler
        self.parallel_context = context
        self.ema_updater = ema_updater
        self._pair_rng = pair_rng
        self._noise_rng = noise_rng
        self._noise_rng_device = str(noise_device)
        self.global_step = 0
        self.optimizer_steps = 0
        self.last_pair_index = -1
        self._phase = "idle"
        self._poisoned = False

    @property
    def config_digest(self) -> str:
        return self.objective.config_digest

    def sample_pair_index(self) -> int:
        """Sample on rank zero and broadcast one pair for the whole iteration."""

        candidate = torch.randint(
            self.objective.pair_count,
            (1,),
            generator=self._pair_rng,
            device="cpu",
            dtype=torch.int64,
        )
        value = candidate.to(device=self.student_parameters[0].device)
        if self.parallel_context.world_size > 1:
            if self.parallel_context.rank != 0:
                value.fill_(-1)
            dist.broadcast(value, src=0, group=self.parallel_context.process_group)
        pair_index = int(value.item())
        if not 0 <= pair_index < self.objective.pair_count:
            raise RuntimeError("rank-synchronized causal consistency pair index is invalid")
        return pair_index

    def train_step(
        self,
        batch: CausalConsistencyTrainingBatch | Sequence[CausalConsistencyTrainingBatch],
    ) -> CausalConsistencyTrainResult:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError(
                "causal consistency engine has a partially committed update; restore a checkpoint"
            )
        batches = _microbatches(batch, expected=self.gradient_accumulation_steps)
        pair_index = self.sample_pair_index()
        weights: list[Tensor] = []
        for microbatch in batches:
            clean = microbatch.clean_latents
            if not isinstance(clean, Tensor) or str(clean.device) != self._noise_rng_device:
                raise ValueError(
                    "causal consistency batch device differs from the checkpointed noise RNG device"
                )
            weights.append(
                torch.tensor(
                    self.objective.loss_denominator(microbatch),
                    device=self.student_parameters[0].device,
                    dtype=torch.float32,
                )
            )
        total_weight = global_denominator(weights, self.parallel_context)
        results: list[CausalConsistencyLossResult] = []
        self.optimizer.zero_grad(set_to_none=True)
        optimizer_mutated = False
        try:
            self._phase = "student-backward"
            self.student_module.train()
            self.teacher_module.eval()
            self.ema_student_module.eval()
            for index, (microbatch, weight) in enumerate(zip(batches, weights, strict=True)):
                clean = microbatch.clean_latents
                assert isinstance(clean, Tensor)
                noise = torch.randn(
                    clean.shape,
                    device=clean.device,
                    dtype=clean.dtype,
                    generator=self._noise_rng,
                )
                with accumulation_context(
                    self.student_module,
                    final_microbatch=index + 1 == len(batches),
                ):
                    result = self.objective.loss(
                        microbatch,
                        pair_index=pair_index,
                        noise=noise,
                    )
                    reported = torch.as_tensor(
                        result.metrics.get("loss_denominator"),
                        device=weight.device,
                        dtype=torch.float32,
                    )
                    if reported.numel() != 1 or not torch.equal(reported.reshape(()), weight):
                        raise RuntimeError(
                            "causal consistency declared and realized loss denominators differ"
                        )
                    gradient_weight = weight / total_weight * float(self.parallel_context.world_size)
                    (result.loss * gradient_weight).backward()
                results.append(result)
            if any(parameter.grad is not None for parameter in self.teacher_module.parameters()):
                raise RuntimeError("causal consistency teacher received gradients")
            if any(parameter.grad is not None for parameter in self.ema_student_module.parameters()):
                raise RuntimeError("causal consistency EMA target received gradients")
            grad_norm = clip_grad_norm_(
                self.student_parameters,
                self.max_grad_norm,
                error_if_nonfinite=True,
            )
            optimizer_mutated = True
            self.optimizer.step()
            self.optimizer_steps += 1
            self.ema_updater.update()
            if self.scheduler is not None:
                self.scheduler.step()
            self.global_step += 1
            self.last_pair_index = pair_index
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
            "pair_index": pair_index,
            "accumulated_microbatches": len(batches),
            "grad_norm": grad_norm.detach(),
            **role_metrics(
                results,
                global_numerator=numerator,
                global_denominator=denominator,
            ),
        }
        return CausalConsistencyTrainResult(
            loss=loss.detach().float(),
            pair_index=pair_index,
            metrics=metrics,
        )

    def state_dict(self) -> dict[str, object]:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("cannot checkpoint a partially committed causal consistency update")
        return {
            "schema": CAUSAL_CONSISTENCY_ENGINE_STATE_SCHEMA,
            "global_step": self.global_step,
            "optimizer_steps": self.optimizer_steps,
            "last_pair_index": self.last_pair_index,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "config_digest": self.config_digest,
            "data_parallel_size": self.parallel_context.world_size,
            "noise_rng_device": self._noise_rng_device,
            "pair_rng_state": self._pair_rng.get_state().clone(),
            "noise_rng_state": self._noise_rng.get_state().clone(),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("causal consistency engine state must be a mapping")
        expected = {
            "schema",
            "global_step",
            "optimizer_steps",
            "last_pair_index",
            "gradient_accumulation_steps",
            "config_digest",
            "data_parallel_size",
            "noise_rng_device",
            "pair_rng_state",
            "noise_rng_state",
        }
        if set(state_dict) != expected:
            raise ValueError("causal consistency engine state fields differ from the active schema")
        if state_dict["schema"] != CAUSAL_CONSISTENCY_ENGINE_STATE_SCHEMA:
            raise ValueError(
                f"unsupported causal consistency engine schema: {state_dict['schema']!r}"
            )
        active = {
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "config_digest": self.config_digest,
            "data_parallel_size": self.parallel_context.world_size,
            "noise_rng_device": self._noise_rng_device,
        }
        for name, value in active.items():
            if state_dict[name] != value:
                raise ValueError(f"saved causal consistency {name} differs from the active engine")
        global_step = non_negative_int(state_dict["global_step"], field_name="global_step")
        optimizer_steps = non_negative_int(
            state_dict["optimizer_steps"],
            field_name="optimizer_steps",
        )
        if optimizer_steps != global_step:
            raise ValueError("saved causal consistency optimizer counter violates its cadence")
        last_pair = int(state_dict["last_pair_index"])
        if global_step == 0:
            if last_pair != -1:
                raise ValueError("unstarted causal consistency state cannot have a pair index")
        elif not 0 <= last_pair < self.objective.pair_count:
            raise ValueError("saved causal consistency pair index is invalid")
        states = (state_dict["pair_rng_state"], state_dict["noise_rng_state"])
        if any(
            not isinstance(value, Tensor) or value.dtype != torch.uint8 or value.ndim != 1
            for value in states
        ):
            raise ValueError("saved causal consistency RNG state is invalid")
        pair_state, noise_state = states
        assert isinstance(pair_state, Tensor) and isinstance(noise_state, Tensor)
        self._pair_rng.set_state(pair_state.detach().cpu())
        self._noise_rng.set_state(noise_state.detach().cpu())
        self.global_step = global_step
        self.optimizer_steps = optimizer_steps
        self.last_pair_index = last_pair
        self._phase = "idle"
        self._poisoned = False


__all__ = [
    "CAUSAL_CONSISTENCY_ENGINE_STATE_SCHEMA",
    "CausalConsistencyTrainResult",
    "NativeCausalConsistencyTrainEngine",
]
