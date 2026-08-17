"""Student-then-fake-score optimizer state machine for native SGMD."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from worldfoundry.core.gradient import clip_grad_norm_
from worldfoundry.training.optimization import audit_optimizer_parameters, trainable_parameters

from ...shared.accumulation import (
    accumulation_context,
    check_reported_weight,
    declared_loss_weight,
    global_denominator,
    global_loss_statistics,
    role_metrics,
)
from ...shared.distributed import PostTrainingParallelContext
from ...shared.validation import non_negative_int, positive_float, validate_stateful_or_none
from .contracts import SGMDLossAdapter, SGMDTrainingBatch
from .objective import SGMDLossResult

SGMD_ENGINE_STATE_SCHEMA = "worldfoundry-sgmd-engine"


def _finite_loss(result: object, *, role: str) -> SGMDLossResult:
    if not isinstance(result, SGMDLossResult):
        raise TypeError(f"{role} loss adapter must return SGMDLossResult")
    if not isinstance(result.loss, torch.Tensor) or result.loss.numel() != 1:
        raise TypeError(f"{role} loss must be one scalar tensor")
    if not bool(torch.isfinite(result.loss.detach()).all()):
        raise FloatingPointError(f"non-finite {role} loss")
    return result


def _microbatches(
    value: SGMDTrainingBatch | Sequence[SGMDTrainingBatch],
    *,
    expected: int,
    role: str,
) -> tuple[SGMDTrainingBatch, ...]:
    if isinstance(value, SGMDTrainingBatch):
        batches = (value,)
    elif isinstance(value, Sequence):
        batches = tuple(value)
    else:
        raise TypeError(f"{role} batch must be SGMDTrainingBatch or a sequence")
    if len(batches) != expected:
        raise ValueError(
            f"SGMD {role} optimizer iteration requires exactly {expected} microbatches; "
            f"got {len(batches)}"
        )
    if not all(isinstance(batch, SGMDTrainingBatch) for batch in batches):
        raise TypeError(f"every SGMD {role} microbatch must be SGMDTrainingBatch")
    return batches


@dataclass(frozen=True, slots=True)
class SGMDTrainResult:
    student_loss: torch.Tensor
    fake_score_loss: torch.Tensor
    student_target_indices: tuple[int, ...]
    fake_score_target_indices: tuple[int, ...]
    metrics: Mapping[str, object]


class NativeSGMDTrainEngine:
    """Own SGMD role isolation, exact update order, and commit boundaries."""

    def __init__(
        self,
        *,
        student_module: nn.Module,
        teacher_module: nn.Module,
        fake_score_module: nn.Module,
        loss_adapter: SGMDLossAdapter,
        student_optimizer: torch.optim.Optimizer,
        fake_score_optimizer: torch.optim.Optimizer,
        student_max_grad_norm: float = 10.0,
        fake_score_max_grad_norm: float = 10.0,
        gradient_accumulation_steps: int = 1,
        student_scheduler: object | None = None,
        fake_score_scheduler: object | None = None,
        student_ema: object | None = None,
        parallel_context: PostTrainingParallelContext | None = None,
    ) -> None:
        modules = (student_module, teacher_module, fake_score_module)
        if not all(isinstance(module, nn.Module) for module in modules):
            raise TypeError("all SGMD roles must be nn.Module values")
        if len({id(module) for module in modules}) != 3:
            raise ValueError("SGMD student, teacher, and fake score must be distinct modules")
        if not isinstance(loss_adapter, SGMDLossAdapter):
            raise TypeError("loss_adapter must implement SGMDLossAdapter")
        if any(parameter.requires_grad for parameter in teacher_module.parameters()):
            raise ValueError("SGMD teacher parameters must be frozen")
        accumulation = non_negative_int(
            gradient_accumulation_steps,
            field_name="gradient_accumulation_steps",
        )
        if accumulation == 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if loss_adapter.num_student_steps <= 0:
            raise ValueError("SGMD loss adapter must expose a positive student step count")
        minimum = int(loss_adapter.minimum_student_target_index)
        if not 0 <= minimum < int(loss_adapter.num_student_steps):
            raise ValueError("SGMD minimum student target index is invalid")
        validate_stateful_or_none(student_scheduler, field_name="student_scheduler")
        validate_stateful_or_none(fake_score_scheduler, field_name="fake_score_scheduler")
        validate_stateful_or_none(student_ema, field_name="student_ema")

        student_parameters = trainable_parameters(student_module)
        fake_parameters = trainable_parameters(fake_score_module)
        if {id(value) for value in student_parameters} & {id(value) for value in fake_parameters}:
            raise ValueError("SGMD student and fake score cannot share trainable parameters")
        audit_optimizer_parameters(student_optimizer, student_parameters, role="SGMD student")
        audit_optimizer_parameters(fake_score_optimizer, fake_parameters, role="SGMD fake score")

        self.student_module = student_module
        self.teacher_module = teacher_module
        self.fake_score_module = fake_score_module
        self.loss_adapter = loss_adapter
        self.student_optimizer = student_optimizer
        self.fake_score_optimizer = fake_score_optimizer
        self.student_parameters = student_parameters
        self.fake_score_parameters = fake_parameters
        self.student_max_grad_norm = positive_float(
            student_max_grad_norm,
            field_name="student_max_grad_norm",
        )
        self.fake_score_max_grad_norm = positive_float(
            fake_score_max_grad_norm,
            field_name="fake_score_max_grad_norm",
        )
        self.gradient_accumulation_steps = accumulation
        self.student_scheduler = student_scheduler
        self.fake_score_scheduler = fake_score_scheduler
        self.student_ema = student_ema
        self.parallel_context = parallel_context or PostTrainingParallelContext.current()
        self.parallel_context.audit_synchronized_module(student_module, role="SGMD student")
        self.parallel_context.audit_synchronized_module(fake_score_module, role="SGMD fake score")
        self.global_step = 0
        self.student_optimizer_steps = 0
        self.fake_score_optimizer_steps = 0
        self._phase = "idle"
        self._poisoned = False
        self.teacher_module.eval()
        self.fake_score_module.eval()

    def _target_index(
        self,
        *,
        minimum: int,
        generator: torch.Generator | None,
    ) -> int:
        value = torch.randint(
            int(minimum),
            int(self.loss_adapter.num_student_steps),
            (1,),
            device=self.student_parameters[0].device,
            generator=generator,
            dtype=torch.int64,
        )
        if self.parallel_context.world_size > 1:
            if self.parallel_context.rank != 0:
                value.zero_()
            # broadcast_from_coordinator translates group-local rank zero to
            # its global rank; a raw src=0 is wrong for non-world subgroups.
            self.parallel_context.broadcast_from_coordinator(value)
        return int(value.item())

    def train_step(
        self,
        student_batch: SGMDTrainingBatch | Sequence[SGMDTrainingBatch],
        fake_score_batch: SGMDTrainingBatch | Sequence[SGMDTrainingBatch],
        *,
        generator: torch.Generator | None = None,
    ) -> SGMDTrainResult:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("SGMD engine has a partially committed iteration; restore the last checkpoint")
        student_batches = _microbatches(
            student_batch,
            expected=self.gradient_accumulation_steps,
            role="student",
        )
        fake_batches = _microbatches(
            fake_score_batch,
            expected=self.gradient_accumulation_steps,
            role="fake-score",
        )
        student_results: list[SGMDLossResult] = []
        fake_results: list[SGMDLossResult] = []
        student_weights: list[torch.Tensor] = []
        fake_weights: list[torch.Tensor] = []
        student_targets: list[int] = []
        fake_targets: list[int] = []
        self.student_optimizer.zero_grad(set_to_none=True)
        self.fake_score_optimizer.zero_grad(set_to_none=True)
        optimizer_mutated = False
        try:
            self._phase = "student-backward"
            self.student_module.train()
            self.teacher_module.eval()
            self.fake_score_module.eval()
            student_weights = [
                declared_loss_weight(
                    self.loss_adapter,
                    batch,
                    role="student",
                    device=self.student_parameters[0].device,
                )
                for batch in student_batches
            ]
            total_student_weight = global_denominator(
                student_weights,
                self.parallel_context,
            )
            for index, (batch, weight) in enumerate(
                zip(student_batches, student_weights, strict=True)
            ):
                target = self._target_index(
                    minimum=self.loss_adapter.minimum_student_target_index,
                    generator=generator,
                )
                student_targets.append(target)
                with accumulation_context(
                    self.student_module,
                    final_microbatch=index + 1 == len(student_batches),
                ):
                    result = _finite_loss(
                        self.loss_adapter.student_loss(
                            batch,
                            target_index=target,
                            generator=generator,
                        ),
                        role="SGMD student",
                    )
                    check_reported_weight(result, weight, role="SGMD student")
                    gradient_weight = (
                        weight / total_student_weight * float(self.parallel_context.world_size)
                    )
                    (result.loss * gradient_weight).backward()
                student_results.append(result)
            if any(parameter.grad is not None for parameter in self.fake_score_parameters):
                raise RuntimeError("SGMD student phase produced fake-score parameter gradients")
            student_grad_norm = clip_grad_norm_(
                self.student_parameters,
                self.student_max_grad_norm,
                error_if_nonfinite=True,
            )
            optimizer_mutated = True
            self.student_optimizer.step()
            self.student_optimizer_steps += 1
            if self.student_scheduler is not None:
                self.student_scheduler.step()
            if self.student_ema is not None:
                self.student_ema.update(self.student_module)
            self._phase = "student-committed"

            self._phase = "fake-score-backward"
            self.student_module.train()
            self.teacher_module.eval()
            self.fake_score_module.eval()
            self.student_optimizer.zero_grad(set_to_none=True)
            self.fake_score_optimizer.zero_grad(set_to_none=True)
            fake_weights = [
                declared_loss_weight(
                    self.loss_adapter,
                    batch,
                    role="fake-score",
                    device=self.fake_score_parameters[0].device,
                )
                for batch in fake_batches
            ]
            total_fake_weight = global_denominator(fake_weights, self.parallel_context)
            for index, (batch, weight) in enumerate(zip(fake_batches, fake_weights, strict=True)):
                target = self._target_index(minimum=0, generator=generator)
                fake_targets.append(target)
                with accumulation_context(
                    self.fake_score_module,
                    final_microbatch=index + 1 == len(fake_batches),
                ):
                    result = _finite_loss(
                        self.loss_adapter.fake_score_loss(
                            batch,
                            target_index=target,
                            generator=generator,
                        ),
                        role="SGMD fake score",
                    )
                    check_reported_weight(result, weight, role="SGMD fake score")
                    gradient_weight = weight / total_fake_weight * float(
                        self.parallel_context.world_size
                    )
                    (result.loss * gradient_weight).backward()
                fake_results.append(result)
            if any(parameter.grad is not None for parameter in self.student_parameters):
                raise RuntimeError("SGMD fake-score phase produced student parameter gradients")
            fake_grad_norm = clip_grad_norm_(
                self.fake_score_parameters,
                self.fake_score_max_grad_norm,
                error_if_nonfinite=True,
            )
            self.fake_score_optimizer.step()
            self.fake_score_optimizer_steps += 1
            if self.fake_score_scheduler is not None:
                self.fake_score_scheduler.step()
            self._phase = "fake-score-committed"
            self.global_step += 1
            self._phase = "idle"
        except Exception:
            self.student_optimizer.zero_grad(set_to_none=True)
            self.fake_score_optimizer.zero_grad(set_to_none=True)
            if optimizer_mutated:
                self._poisoned = True
            else:
                self._phase = "idle"
            raise

        student_numerator, student_denominator, student_loss = global_loss_statistics(
            student_results,
            student_weights,
            self.parallel_context,
        )
        fake_numerator, fake_denominator, fake_loss = global_loss_statistics(
            fake_results,
            fake_weights,
            self.parallel_context,
        )
        metrics = {
            "global_step": torch.tensor(
                self.global_step,
                device=student_loss.device,
                dtype=torch.int64,
            ),
            "student_optimizer_steps": self.student_optimizer_steps,
            "fake_score_optimizer_steps": self.fake_score_optimizer_steps,
            "student_target_indices": tuple(student_targets),
            "fake_score_target_indices": tuple(fake_targets),
            "microbatches_per_role": self.gradient_accumulation_steps,
            "student_grad_norm": student_grad_norm.detach(),
            "fake_score_grad_norm": fake_grad_norm.detach(),
            "student": role_metrics(
                student_results,
                global_numerator=student_numerator,
                global_denominator=student_denominator,
            ),
            "fake_score": role_metrics(
                fake_results,
                global_numerator=fake_numerator,
                global_denominator=fake_denominator,
            ),
        }
        return SGMDTrainResult(
            student_loss=student_loss.detach().float(),
            fake_score_loss=fake_loss.detach().float(),
            student_target_indices=tuple(student_targets),
            fake_score_target_indices=tuple(fake_targets),
            metrics=metrics,
        )

    def state_dict(self) -> dict[str, object]:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("cannot checkpoint a partially committed SGMD iteration")
        return {
            "schema": SGMD_ENGINE_STATE_SCHEMA,
            "global_step": self.global_step,
            "student_optimizer_steps": self.student_optimizer_steps,
            "fake_score_optimizer_steps": self.fake_score_optimizer_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "data_parallel_size": self.parallel_context.world_size,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("SGMD engine state must be a mapping")
        expected = {
            "schema",
            "global_step",
            "student_optimizer_steps",
            "fake_score_optimizer_steps",
            "gradient_accumulation_steps",
            "data_parallel_size",
        }
        if set(state_dict) != expected:
            raise ValueError("SGMD engine state fields differ from the active schema")
        if state_dict["schema"] != SGMD_ENGINE_STATE_SCHEMA:
            raise ValueError(f"unsupported SGMD engine schema: {state_dict['schema']!r}")
        if int(state_dict["gradient_accumulation_steps"]) != self.gradient_accumulation_steps:
            raise ValueError("saved SGMD accumulation differs from the active engine")
        if int(state_dict["data_parallel_size"]) != self.parallel_context.world_size:
            raise ValueError("saved SGMD data-parallel size differs from the active engine")
        global_step = non_negative_int(state_dict["global_step"], field_name="global_step")
        student_steps = non_negative_int(
            state_dict["student_optimizer_steps"],
            field_name="student_optimizer_steps",
        )
        fake_steps = non_negative_int(
            state_dict["fake_score_optimizer_steps"],
            field_name="fake_score_optimizer_steps",
        )
        if student_steps != global_step or fake_steps != global_step:
            raise ValueError("saved SGMD optimizer counters violate the one-to-one cadence")
        self.global_step = global_step
        self.student_optimizer_steps = student_steps
        self.fake_score_optimizer_steps = fake_steps
        self._phase = "idle"
        self._poisoned = False
        self.student_optimizer.zero_grad(set_to_none=True)
        self.fake_score_optimizer.zero_grad(set_to_none=True)


__all__ = ["SGMD_ENGINE_STATE_SCHEMA", "SGMDTrainResult", "NativeSGMDTrainEngine"]
