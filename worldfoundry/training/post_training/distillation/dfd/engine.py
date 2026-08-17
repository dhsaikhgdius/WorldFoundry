"""Alternating optimizer state machine for native Data-Forcing Distillation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from worldfoundry.core.gradient import clip_grad_norm_, get_total_norm
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
from .contracts import DFDLossAdapter, DFDTrainingBatch
from .objective import DFDLossResult

DFD_ENGINE_STATE_SCHEMA = "worldfoundry-dfd-engine"


def _finite_result(result: object, *, role: str) -> DFDLossResult:
    if not isinstance(result, DFDLossResult):
        raise TypeError(f"{role} loss adapter must return DFDLossResult")
    if not isinstance(result.loss, torch.Tensor) or result.loss.numel() != 1:
        raise TypeError(f"{role} loss must be one scalar tensor")
    if not bool(torch.isfinite(result.loss.detach()).all()):
        raise FloatingPointError(f"non-finite {role} loss")
    return result


def _microbatches(
    value: DFDTrainingBatch | Sequence[DFDTrainingBatch],
    *,
    expected: int,
) -> tuple[DFDTrainingBatch, ...]:
    if isinstance(value, DFDTrainingBatch):
        batches = (value,)
    elif isinstance(value, Sequence):
        batches = tuple(value)
    else:
        raise TypeError("DFD batch must be DFDTrainingBatch or a sequence")
    if len(batches) != expected:
        raise ValueError(
            f"DFD optimizer iteration requires exactly {expected} microbatches; got {len(batches)}"
        )
    if not all(isinstance(batch, DFDTrainingBatch) for batch in batches):
        raise TypeError("every DFD microbatch must be DFDTrainingBatch")
    return batches


def _unclipped_finite_norm(parameters: tuple[nn.Parameter, ...]) -> torch.Tensor:
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    return get_total_norm(gradients, error_if_nonfinite=True)


@dataclass(frozen=True, slots=True)
class DFDTrainResult:
    phase: Literal["student", "guidance"]
    loss: torch.Tensor
    data_forcing_decisions: tuple[bool, ...]
    metrics: Mapping[str, object]


class NativeDFDTrainEngine:
    """Execute the released one-student/four-guidance alternating cadence."""

    def __init__(
        self,
        *,
        student_module: nn.Module,
        teacher_module: nn.Module,
        fake_score_module: nn.Module,
        loss_adapter: DFDLossAdapter,
        student_optimizer: torch.optim.Optimizer,
        fake_score_optimizer: torch.optim.Optimizer,
        discriminator_module: nn.Module | None = None,
        discriminator_optimizer: torch.optim.Optimizer | None = None,
        student_max_grad_norm: float = 10.0,
        fake_score_max_grad_norm: float = 10.0,
        discriminator_max_grad_norm: float | None = None,
        gradient_accumulation_steps: int = 1,
        student_scheduler: object | None = None,
        fake_score_scheduler: object | None = None,
        discriminator_scheduler: object | None = None,
        student_ema: object | None = None,
        parallel_context: PostTrainingParallelContext | None = None,
    ) -> None:
        modules = (student_module, teacher_module, fake_score_module)
        if not all(isinstance(module, nn.Module) for module in modules):
            raise TypeError("DFD student, teacher, and fake score must be nn.Module values")
        if len({id(module) for module in modules}) != 3:
            raise ValueError("DFD student, teacher, and fake score must be distinct modules")
        if not isinstance(loss_adapter, DFDLossAdapter):
            raise TypeError("loss_adapter must implement DFDLossAdapter")
        if any(parameter.requires_grad for parameter in teacher_module.parameters()):
            raise ValueError("DFD teacher parameters must be frozen")
        if (discriminator_module is None) != (discriminator_optimizer is None):
            raise ValueError("DFD discriminator module and optimizer must be supplied together")
        if discriminator_module is not None and not isinstance(discriminator_module, nn.Module):
            raise TypeError("discriminator_module must be an nn.Module")
        if discriminator_module is not None and id(discriminator_module) in {
            id(student_module),
            id(teacher_module),
            id(fake_score_module),
        }:
            raise ValueError("DFD discriminator must be a distinct module")
        if (float(loss_adapter.data_forcing_probability) < 0.0) or (
            float(loss_adapter.data_forcing_probability) > 1.0
        ):
            raise ValueError("DFD loss adapter exposes an invalid forcing probability")
        frequency = non_negative_int(
            loss_adapter.student_update_frequency,
            field_name="student_update_frequency",
        )
        if frequency == 0:
            raise ValueError("student_update_frequency must be positive")
        accumulation = non_negative_int(
            gradient_accumulation_steps,
            field_name="gradient_accumulation_steps",
        )
        if accumulation == 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        for value, name in (
            (student_scheduler, "student_scheduler"),
            (fake_score_scheduler, "fake_score_scheduler"),
            (discriminator_scheduler, "discriminator_scheduler"),
            (student_ema, "student_ema"),
        ):
            validate_stateful_or_none(value, field_name=name)
        if discriminator_module is None and discriminator_scheduler is not None:
            raise ValueError("discriminator_scheduler requires a discriminator")

        student_parameters = trainable_parameters(student_module)
        fake_parameters = trainable_parameters(fake_score_module)
        discriminator_parameters = (
            () if discriminator_module is None else trainable_parameters(discriminator_module)
        )
        inventories = (
            {id(parameter) for parameter in student_parameters},
            {id(parameter) for parameter in fake_parameters},
            {id(parameter) for parameter in discriminator_parameters},
        )
        if any(
            inventories[left] & inventories[right]
            for left in range(len(inventories))
            for right in range(left + 1, len(inventories))
        ):
            raise ValueError("DFD trainable roles cannot share parameters")
        audit_optimizer_parameters(student_optimizer, student_parameters, role="DFD student")
        audit_optimizer_parameters(fake_score_optimizer, fake_parameters, role="DFD fake score")
        if discriminator_optimizer is not None:
            audit_optimizer_parameters(
                discriminator_optimizer,
                discriminator_parameters,
                role="DFD discriminator",
            )

        self.student_module = student_module
        self.teacher_module = teacher_module
        self.fake_score_module = fake_score_module
        self.discriminator_module = discriminator_module
        self.loss_adapter = loss_adapter
        self.student_optimizer = student_optimizer
        self.fake_score_optimizer = fake_score_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.student_parameters = student_parameters
        self.fake_score_parameters = fake_parameters
        self.discriminator_parameters = discriminator_parameters
        self.student_max_grad_norm = positive_float(
            student_max_grad_norm,
            field_name="student_max_grad_norm",
        )
        self.fake_score_max_grad_norm = positive_float(
            fake_score_max_grad_norm,
            field_name="fake_score_max_grad_norm",
        )
        if discriminator_module is None:
            if discriminator_max_grad_norm is not None:
                raise ValueError(
                    "discriminator_max_grad_norm requires a discriminator"
                )
            self.discriminator_max_grad_norm = None
        else:
            self.discriminator_max_grad_norm = positive_float(
                (
                    fake_score_max_grad_norm
                    if discriminator_max_grad_norm is None
                    else discriminator_max_grad_norm
                ),
                field_name="discriminator_max_grad_norm",
            )
        self.gradient_accumulation_steps = accumulation
        self.student_update_frequency = frequency
        self.student_scheduler = student_scheduler
        self.fake_score_scheduler = fake_score_scheduler
        self.discriminator_scheduler = discriminator_scheduler
        self.student_ema = student_ema
        self.parallel_context = parallel_context or PostTrainingParallelContext.current()
        self.parallel_context.audit_synchronized_module(student_module, role="DFD student")
        self.parallel_context.audit_synchronized_module(fake_score_module, role="DFD fake score")
        if discriminator_module is not None:
            self.parallel_context.audit_synchronized_module(
                discriminator_module,
                role="DFD discriminator",
            )
        self.global_step = 0
        self.student_optimizer_steps = 0
        self.fake_score_optimizer_steps = 0
        self.discriminator_optimizer_steps = 0
        self._phase = "idle"
        self._poisoned = False
        self.teacher_module.eval()

    @property
    def next_phase(self) -> Literal["student", "guidance"]:
        return (
            "student"
            if self.global_step % self.student_update_frequency == 0
            else "guidance"
        )

    def _data_forcing_decision(
        self,
        *,
        generator: torch.Generator | None,
    ) -> bool:
        probability = float(self.loss_adapter.data_forcing_probability)
        device = self.student_parameters[0].device
        decision = torch.zeros((), device=device, dtype=torch.int64)
        if self.parallel_context.rank == 0:
            if probability >= 1.0:
                decision.fill_(1)
            elif probability > 0.0:
                sampled = torch.rand((), device=device, generator=generator)
                decision.copy_((sampled < probability).to(dtype=torch.int64))
        if self.parallel_context.world_size > 1:
            # broadcast_from_coordinator translates group-local rank zero to
            # its global rank; a raw src=0 is wrong for non-world subgroups.
            self.parallel_context.broadcast_from_coordinator(decision)
        return bool(decision.item())

    def _zero_all(self) -> None:
        self.student_optimizer.zero_grad(set_to_none=True)
        self.fake_score_optimizer.zero_grad(set_to_none=True)
        if self.discriminator_optimizer is not None:
            self.discriminator_optimizer.zero_grad(set_to_none=True)

    def train_step(
        self,
        batch: DFDTrainingBatch | Sequence[DFDTrainingBatch],
        *,
        generator: torch.Generator | None = None,
    ) -> DFDTrainResult:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("DFD engine has a partially committed iteration; restore the last checkpoint")
        batches = _microbatches(batch, expected=self.gradient_accumulation_steps)
        phase = self.next_phase
        results: list[DFDLossResult] = []
        decisions: list[bool] = []
        self._zero_all()
        optimizer_mutated = False
        try:
            self._phase = f"{phase}-backward"
            self.student_module.train()
            self.teacher_module.eval()
            if phase == "student":
                self.fake_score_module.eval()
                if self.discriminator_module is not None:
                    self.discriminator_module.eval()
                active_modules = (self.student_module,)
                active_parameters = self.student_parameters
            else:
                self.fake_score_module.train()
                if self.discriminator_module is not None:
                    self.discriminator_module.train()
                    active_modules = (self.fake_score_module, self.discriminator_module)
                    active_parameters = self.fake_score_parameters + self.discriminator_parameters
                else:
                    active_modules = (self.fake_score_module,)
                    active_parameters = self.fake_score_parameters
            weights = [
                declared_loss_weight(
                    self.loss_adapter,
                    microbatch,
                    role=phase,
                    device=active_parameters[0].device,
                )
                for microbatch in batches
            ]
            total_weight = global_denominator(weights, self.parallel_context)
            for index, (microbatch, weight) in enumerate(zip(batches, weights, strict=True)):
                final = index + 1 == len(batches)
                with ExitStack() as stack:
                    for module in active_modules:
                        stack.enter_context(
                            accumulation_context(module, final_microbatch=final)
                        )
                    if phase == "student":
                        decision = self._data_forcing_decision(generator=generator)
                        decisions.append(decision)
                        result = _finite_result(
                            self.loss_adapter.student_loss(
                                microbatch,
                                data_forcing=decision,
                                generator=generator,
                            ),
                            role="DFD student",
                        )
                    else:
                        result = _finite_result(
                            self.loss_adapter.guidance_loss(
                                microbatch,
                                generator=generator,
                            ),
                            role="DFD guidance",
                        )
                    check_reported_weight(result, weight, role=f"DFD {phase}")
                    gradient_weight = (
                        weight / total_weight * float(self.parallel_context.world_size)
                    )
                    (result.loss * gradient_weight).backward()
                results.append(result)

            if phase == "student":
                if any(parameter.grad is not None for parameter in self.fake_score_parameters):
                    raise RuntimeError("DFD student phase produced fake-score parameter gradients")
                if any(parameter.grad is not None for parameter in self.discriminator_parameters):
                    raise RuntimeError("DFD student phase produced discriminator parameter gradients")
                grad_norm = clip_grad_norm_(
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
            else:
                if any(parameter.grad is not None for parameter in self.student_parameters):
                    raise RuntimeError("DFD guidance phase produced student parameter gradients")
                grad_norm = _unclipped_finite_norm(active_parameters)
                clip_grad_norm_(
                    self.fake_score_parameters,
                    self.fake_score_max_grad_norm,
                    error_if_nonfinite=True,
                )
                if self.discriminator_parameters:
                    assert self.discriminator_max_grad_norm is not None
                    clip_grad_norm_(
                        self.discriminator_parameters,
                        self.discriminator_max_grad_norm,
                        error_if_nonfinite=True,
                    )
                optimizer_mutated = True
                self.fake_score_optimizer.step()
                self.fake_score_optimizer_steps += 1
                if self.discriminator_optimizer is not None:
                    self.discriminator_optimizer.step()
                    self.discriminator_optimizer_steps += 1
                if self.fake_score_scheduler is not None:
                    self.fake_score_scheduler.step()
                if self.discriminator_scheduler is not None:
                    self.discriminator_scheduler.step()
            self._phase = f"{phase}-committed"
            self.global_step += 1
            self._phase = "idle"
        except Exception:
            self._zero_all()
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
            "phase": phase,
            "student_optimizer_steps": self.student_optimizer_steps,
            "fake_score_optimizer_steps": self.fake_score_optimizer_steps,
            "discriminator_optimizer_steps": self.discriminator_optimizer_steps,
            "data_forcing_decisions": tuple(decisions),
            "grad_norm": grad_norm.detach(),
            "role": role_metrics(
                results,
                global_numerator=numerator,
                global_denominator=denominator,
            ),
        }
        return DFDTrainResult(
            phase=phase,
            loss=loss.detach().float(),
            data_forcing_decisions=tuple(decisions),
            metrics=metrics,
        )

    @staticmethod
    def _expected_student_steps(global_step: int, frequency: int) -> int:
        return 0 if global_step == 0 else (global_step - 1) // frequency + 1

    def state_dict(self) -> dict[str, object]:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("cannot checkpoint a partially committed DFD iteration")
        return {
            "schema": DFD_ENGINE_STATE_SCHEMA,
            "global_step": self.global_step,
            "student_optimizer_steps": self.student_optimizer_steps,
            "fake_score_optimizer_steps": self.fake_score_optimizer_steps,
            "discriminator_optimizer_steps": self.discriminator_optimizer_steps,
            "student_update_frequency": self.student_update_frequency,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "student_max_grad_norm": self.student_max_grad_norm,
            "fake_score_max_grad_norm": self.fake_score_max_grad_norm,
            "discriminator_max_grad_norm": self.discriminator_max_grad_norm,
            "adversarial_enabled": self.discriminator_module is not None,
            "data_parallel_size": self.parallel_context.world_size,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("DFD engine state must be a mapping")
        expected = {
            "schema",
            "global_step",
            "student_optimizer_steps",
            "fake_score_optimizer_steps",
            "discriminator_optimizer_steps",
            "student_update_frequency",
            "gradient_accumulation_steps",
            "student_max_grad_norm",
            "fake_score_max_grad_norm",
            "discriminator_max_grad_norm",
            "adversarial_enabled",
            "data_parallel_size",
        }
        if set(state_dict) != expected:
            raise ValueError("DFD engine state fields differ from the active schema")
        if state_dict["schema"] != DFD_ENGINE_STATE_SCHEMA:
            raise ValueError(f"unsupported DFD engine schema: {state_dict['schema']!r}")
        if int(state_dict["student_update_frequency"]) != self.student_update_frequency:
            raise ValueError("saved DFD update frequency differs from the active engine")
        if int(state_dict["gradient_accumulation_steps"]) != self.gradient_accumulation_steps:
            raise ValueError("saved DFD accumulation differs from the active engine")
        for name, active in (
            ("student_max_grad_norm", self.student_max_grad_norm),
            ("fake_score_max_grad_norm", self.fake_score_max_grad_norm),
            ("discriminator_max_grad_norm", self.discriminator_max_grad_norm),
        ):
            saved = state_dict[name]
            if saved is None or active is None:
                if saved is not active:
                    raise ValueError(f"saved DFD {name} differs from the active engine")
            elif float(saved) != active:
                raise ValueError(f"saved DFD {name} differs from the active engine")
        if bool(state_dict["adversarial_enabled"]) != (self.discriminator_module is not None):
            raise ValueError("saved DFD adversarial topology differs from the active engine")
        if int(state_dict["data_parallel_size"]) != self.parallel_context.world_size:
            raise ValueError("saved DFD data-parallel size differs from the active engine")
        global_step = non_negative_int(state_dict["global_step"], field_name="global_step")
        student_steps = non_negative_int(
            state_dict["student_optimizer_steps"],
            field_name="student_optimizer_steps",
        )
        fake_steps = non_negative_int(
            state_dict["fake_score_optimizer_steps"],
            field_name="fake_score_optimizer_steps",
        )
        discriminator_steps = non_negative_int(
            state_dict["discriminator_optimizer_steps"],
            field_name="discriminator_optimizer_steps",
        )
        expected_student = self._expected_student_steps(
            global_step,
            self.student_update_frequency,
        )
        expected_guidance = global_step - expected_student
        if student_steps != expected_student or fake_steps != expected_guidance:
            raise ValueError("saved DFD optimizer counters violate the alternating cadence")
        expected_discriminator = expected_guidance if self.discriminator_module is not None else 0
        if discriminator_steps != expected_discriminator:
            raise ValueError("saved DFD discriminator counter violates the active topology")
        self.global_step = global_step
        self.student_optimizer_steps = student_steps
        self.fake_score_optimizer_steps = fake_steps
        self.discriminator_optimizer_steps = discriminator_steps
        self._phase = "idle"
        self._poisoned = False
        self._zero_all()


__all__ = ["DFD_ENGINE_STATE_SCHEMA", "DFDTrainResult", "NativeDFDTrainEngine"]
