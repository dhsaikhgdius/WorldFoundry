"""Alternating native optimizer state machine for sCM-LADD."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
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
from .contracts import SCMLADDLossAdapter, SCMLADDLossResult, SCMLADDTrainingBatch

SCM_LADD_ENGINE_STATE_SCHEMA = "worldfoundry-scm-ladd-engine"


def _finite_loss(result: object, *, role: str) -> SCMLADDLossResult:
    if not isinstance(result, SCMLADDLossResult):
        raise TypeError(f"{role} loss adapter must return SCMLADDLossResult")
    if not isinstance(result.loss, torch.Tensor) or result.loss.numel() != 1:
        raise TypeError(f"{role} loss must be one scalar tensor")
    if not bool(torch.isfinite(result.loss.detach()).all()):
        raise FloatingPointError(f"non-finite {role} loss")
    return result


@contextmanager
def _frozen_parameters(module: nn.Module) -> Iterator[None]:
    parameters = tuple(module.parameters())
    states = tuple(parameter.requires_grad for parameter in parameters)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(parameters, states):
            parameter.requires_grad_(requires_grad)


@dataclass(frozen=True, slots=True)
class SCMLADDTrainResult:
    phase: str
    loss: torch.Tensor
    metrics: Mapping[str, object]


class NativeSCMLADDTrainEngine:
    """Own student/teacher/discriminator roles and exact G→D phase order."""

    def __init__(
        self,
        *,
        student_module: nn.Module,
        teacher_module: nn.Module,
        discriminator_module: nn.Module,
        discriminator_feature_module: nn.Module,
        loss_adapter: SCMLADDLossAdapter,
        student_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
        student_max_grad_norm: float,
        discriminator_max_grad_norm: float,
        gradient_accumulation_steps: int = 1,
        student_scheduler: object | None = None,
        discriminator_scheduler: object | None = None,
        student_ema: object | None = None,
        parallel_context: PostTrainingParallelContext | None = None,
    ) -> None:
        modules = (student_module, teacher_module, discriminator_module, discriminator_feature_module)
        if not all(isinstance(module, nn.Module) for module in modules):
            raise TypeError("all SCM-LADD roles must be nn.Module values")
        if len({id(student_module), id(teacher_module), id(discriminator_module)}) != 3:
            raise ValueError("SCM-LADD student, teacher, and discriminator modules must be distinct")
        if discriminator_feature_module is not teacher_module:
            raise ValueError("the LADD feature backbone must be the frozen SCM teacher module")
        if not isinstance(loss_adapter, SCMLADDLossAdapter):
            raise TypeError("loss_adapter must implement SCMLADDLossAdapter")
        if any(parameter.requires_grad for parameter in teacher_module.parameters()):
            raise ValueError("SCM-LADD teacher parameters must be frozen")
        student_parameters = trainable_parameters(student_module)
        discriminator_parameters = trainable_parameters(discriminator_module)
        if {id(parameter) for parameter in student_parameters} & {
            id(parameter) for parameter in discriminator_parameters
        }:
            raise ValueError("student and discriminator cannot share trainable parameters")
        audit_optimizer_parameters(student_optimizer, student_parameters, role="SCM-LADD student")
        audit_optimizer_parameters(
            discriminator_optimizer,
            discriminator_parameters,
            role="SCM-LADD discriminator",
        )
        validate_stateful_or_none(student_scheduler, field_name="student_scheduler")
        validate_stateful_or_none(discriminator_scheduler, field_name="discriminator_scheduler")
        validate_stateful_or_none(student_ema, field_name="student_ema")
        accumulation = non_negative_int(
            gradient_accumulation_steps,
            field_name="gradient_accumulation_steps",
        )
        if accumulation == 0:
            raise ValueError("gradient_accumulation_steps must be positive")

        self.student_module = student_module
        self.teacher_module = teacher_module
        self.discriminator_module = discriminator_module
        self.discriminator_feature_module = discriminator_feature_module
        self.loss_adapter = loss_adapter
        self.student_optimizer = student_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.student_parameters = student_parameters
        self.discriminator_parameters = discriminator_parameters
        self.student_max_grad_norm = positive_float(student_max_grad_norm, field_name="student_max_grad_norm")
        self.discriminator_max_grad_norm = positive_float(
            discriminator_max_grad_norm,
            field_name="discriminator_max_grad_norm",
        )
        self.gradient_accumulation_steps = accumulation
        self.student_scheduler = student_scheduler
        self.discriminator_scheduler = discriminator_scheduler
        self.student_ema = student_ema
        self.parallel_context = parallel_context or PostTrainingParallelContext.current()
        self.parallel_context.audit_synchronized_module(student_module, role="SCM-LADD student")
        self.parallel_context.audit_synchronized_module(discriminator_module, role="SCM-LADD discriminator")
        self.global_step = 0
        self.student_optimizer_steps = 0
        self.discriminator_optimizer_steps = 0
        self.next_phase = "generator"
        self._active_phase = "idle"
        self._poisoned = False
        self.teacher_module.eval()
        self.discriminator_feature_module.eval()

    @property
    def config_digest(self) -> str:
        return str(self.loss_adapter.config_digest)

    def train_step(
        self,
        batch: SCMLADDTrainingBatch | Sequence[SCMLADDTrainingBatch],
        *,
        generator: torch.Generator | None = None,
    ) -> SCMLADDTrainResult:
        if self._poisoned or self._active_phase != "idle":
            raise RuntimeError("SCM-LADD engine has a partially committed update; restore the last checkpoint")
        if isinstance(batch, SCMLADDTrainingBatch):
            batches = (batch,)
        elif isinstance(batch, Sequence):
            batches = tuple(batch)
        else:
            raise TypeError("batch must be SCMLADDTrainingBatch or a sequence of microbatches")
        if len(batches) != self.gradient_accumulation_steps:
            raise ValueError(
                "SCM-LADD optimizer update requires exactly "
                f"{self.gradient_accumulation_steps} microbatches; got {len(batches)}"
            )
        if not all(isinstance(value, SCMLADDTrainingBatch) for value in batches):
            raise TypeError("every SCM-LADD microbatch must be SCMLADDTrainingBatch")
        phase = self.next_phase
        parameters = self.student_parameters if phase == "generator" else self.discriminator_parameters
        active_module = self.student_module if phase == "generator" else self.discriminator_module
        role = f"SCM-LADD {phase}"
        weights = [
            declared_loss_weight(
                self.loss_adapter,
                microbatch,
                role=phase,
                device=parameters[0].device,
            )
            for microbatch in batches
        ]
        total_weight = global_denominator(weights, self.parallel_context)
        results: list[SCMLADDLossResult] = []
        self.student_optimizer.zero_grad(set_to_none=True)
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        optimizer_step_started = False
        try:
            if phase == "generator":
                self._active_phase = "generator-backward"
                self.student_module.train()
                self.discriminator_module.eval()
                self.teacher_module.eval()
                self.discriminator_feature_module.eval()
                with _frozen_parameters(self.discriminator_module):
                    for index, (microbatch, weight) in enumerate(zip(batches, weights, strict=True)):
                        final_microbatch = index + 1 == len(batches)
                        with accumulation_context(
                            active_module,
                            final_microbatch=final_microbatch,
                        ):
                            result = _finite_loss(
                                self.loss_adapter.generator_loss(
                                    microbatch,
                                    training_iteration=self.global_step + 1,
                                    generator=generator,
                                ),
                                role=role,
                            )
                            check_reported_weight(result, weight, role=role)
                            gradient_weight = (
                                weight / total_weight * float(self.parallel_context.world_size)
                            )
                            (result.loss * gradient_weight).backward()
                        results.append(result)
                numerator, denominator, loss = global_loss_statistics(
                    results,
                    weights,
                    self.parallel_context,
                )
                grad_norm = clip_grad_norm_(
                    self.student_parameters,
                    self.student_max_grad_norm,
                    error_if_nonfinite=True,
                )
                optimizer_step_started = True
                self.student_optimizer.step()
                self.student_optimizer_steps += 1
                self._active_phase = "generator-committed"
                if self.student_scheduler is not None:
                    self.student_scheduler.step()
                if self.student_ema is not None:
                    self.student_ema.update(self.student_module)
                self.next_phase = "discriminator"
            else:
                self._active_phase = "discriminator-backward"
                self.student_module.eval()
                self.discriminator_module.train()
                self.teacher_module.eval()
                self.discriminator_feature_module.eval()
                for index, (microbatch, weight) in enumerate(zip(batches, weights, strict=True)):
                    final_microbatch = index + 1 == len(batches)
                    with accumulation_context(
                        active_module,
                        final_microbatch=final_microbatch,
                    ):
                        result = _finite_loss(
                            self.loss_adapter.discriminator_loss(microbatch, generator=generator),
                            role=role,
                        )
                        check_reported_weight(result, weight, role=role)
                        gradient_weight = weight / total_weight * float(self.parallel_context.world_size)
                        (result.loss * gradient_weight).backward()
                    results.append(result)
                numerator, denominator, loss = global_loss_statistics(
                    results,
                    weights,
                    self.parallel_context,
                )
                grad_norm = clip_grad_norm_(
                    self.discriminator_parameters,
                    self.discriminator_max_grad_norm,
                    error_if_nonfinite=True,
                )
                optimizer_step_started = True
                self.discriminator_optimizer.step()
                self.discriminator_optimizer_steps += 1
                self._active_phase = "discriminator-committed"
                if self.discriminator_scheduler is not None:
                    self.discriminator_scheduler.step()
                self.next_phase = "generator"
            self.global_step += 1
            self._active_phase = "idle"
        except Exception:
            self.student_optimizer.zero_grad(set_to_none=True)
            self.discriminator_optimizer.zero_grad(set_to_none=True)
            if optimizer_step_started:
                self._poisoned = True
            else:
                self._active_phase = "idle"
            raise
        committed_metrics = role_metrics(
            results,
            global_numerator=numerator,
            global_denominator=denominator,
        )
        metrics = {
            **committed_metrics,
            "global_step": torch.tensor(self.global_step, device=loss.device, dtype=torch.int64),
            "student_optimizer_steps": torch.tensor(
                self.student_optimizer_steps,
                device=loss.device,
                dtype=torch.int64,
            ),
            "discriminator_optimizer_steps": torch.tensor(
                self.discriminator_optimizer_steps,
                device=loss.device,
                dtype=torch.int64,
            ),
            "accumulated_microbatches": len(batches),
            "grad_norm": grad_norm.detach(),
        }
        return SCMLADDTrainResult(phase=phase, loss=loss.detach().float(), metrics=metrics)

    def state_dict(self) -> dict[str, object]:
        if self._poisoned or self._active_phase != "idle":
            raise RuntimeError("cannot checkpoint a partially committed SCM-LADD update")
        return {
            "schema": SCM_LADD_ENGINE_STATE_SCHEMA,
            "global_step": self.global_step,
            "student_optimizer_steps": self.student_optimizer_steps,
            "discriminator_optimizer_steps": self.discriminator_optimizer_steps,
            "next_phase": self.next_phase,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "config_digest": self.config_digest,
            "data_parallel_size": self.parallel_context.world_size,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("SCM-LADD engine state must be a mapping")
        expected = {
            "schema",
            "global_step",
            "student_optimizer_steps",
            "discriminator_optimizer_steps",
            "next_phase",
            "gradient_accumulation_steps",
            "config_digest",
            "data_parallel_size",
        }
        if set(state_dict) != expected:
            raise ValueError("SCM-LADD engine state fields differ from the active schema")
        if state_dict["schema"] != SCM_LADD_ENGINE_STATE_SCHEMA:
            raise ValueError(f"unsupported SCM-LADD engine schema: {state_dict['schema']!r}")
        if state_dict["config_digest"] != self.config_digest:
            raise ValueError("saved SCM-LADD recipe differs from the active engine")
        if int(state_dict["gradient_accumulation_steps"]) != self.gradient_accumulation_steps:
            raise ValueError("saved SCM-LADD accumulation cadence differs from the active engine")
        if int(state_dict["data_parallel_size"]) != self.parallel_context.world_size:
            raise ValueError("saved SCM-LADD data-parallel size differs from the active engine")
        global_step = non_negative_int(state_dict["global_step"], field_name="global_step")
        student_steps = non_negative_int(
            state_dict["student_optimizer_steps"],
            field_name="student_optimizer_steps",
        )
        discriminator_steps = non_negative_int(
            state_dict["discriminator_optimizer_steps"],
            field_name="discriminator_optimizer_steps",
        )
        expected_student = (global_step + 1) // 2
        expected_discriminator = global_step // 2
        expected_phase = "generator" if global_step % 2 == 0 else "discriminator"
        if student_steps != expected_student or discriminator_steps != expected_discriminator:
            raise ValueError("saved SCM-LADD optimizer counters violate G-to-D alternation")
        if state_dict["next_phase"] != expected_phase:
            raise ValueError("saved SCM-LADD phase violates G-to-D alternation")
        self.global_step = global_step
        self.student_optimizer_steps = student_steps
        self.discriminator_optimizer_steps = discriminator_steps
        self.next_phase = expected_phase
        self._active_phase = "idle"
        self._poisoned = False
        self.student_optimizer.zero_grad(set_to_none=True)
        self.discriminator_optimizer.zero_grad(set_to_none=True)


__all__ = ["SCM_LADD_ENGINE_STATE_SCHEMA", "NativeSCMLADDTrainEngine", "SCMLADDTrainResult"]
