"""Exact warmup/alternation state machine for native rCM."""

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
from .contracts import RCMLossAdapter, RCMLossResult, RCMTrainingBatch

RCM_ENGINE_STATE_SCHEMA = "worldfoundry-rcm-engine"


def _finite_loss(result: object, *, role: str) -> RCMLossResult:
    if not isinstance(result, RCMLossResult):
        raise TypeError(f"{role} objective must return RCMLossResult")
    if not isinstance(result.loss, torch.Tensor) or result.loss.numel() != 1:
        raise TypeError(f"{role} loss must be one scalar tensor")
    if not bool(torch.isfinite(result.loss.detach())):
        raise FloatingPointError(f"non-finite {role} loss")
    return result


@contextmanager
def _frozen_parameters(module: nn.Module | None) -> Iterator[None]:
    if module is None:
        yield
        return
    parameters = tuple(module.parameters())
    states = tuple(parameter.requires_grad for parameter in parameters)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(parameters, states, strict=True):
            parameter.requires_grad_(requires_grad)


@dataclass(frozen=True, slots=True)
class RCMTrainResult:
    phase: str
    loss: torch.Tensor
    metrics: Mapping[str, object]


class NativeRCMTrainEngine:
    """Commit either one joint student update or one fake-score update."""

    def __init__(
        self,
        *,
        student_module: nn.Module,
        teacher_module: nn.Module,
        fake_score_module: nn.Module | None,
        loss_adapter: RCMLossAdapter,
        student_optimizer: torch.optim.Optimizer,
        fake_score_optimizer: torch.optim.Optimizer | None,
        tangent_warmup_steps: int,
        student_update_frequency: int,
        dmd_enabled: bool,
        student_max_grad_norm: float,
        fake_score_max_grad_norm: float | None = None,
        gradient_accumulation_steps: int = 1,
        student_scheduler: object | None = None,
        fake_score_scheduler: object | None = None,
        student_ema: object | None = None,
        parallel_context: PostTrainingParallelContext | None = None,
    ) -> None:
        if not isinstance(student_module, nn.Module) or not isinstance(teacher_module, nn.Module):
            raise TypeError("rCM student and teacher must be nn.Module values")
        if fake_score_module is not None and not isinstance(fake_score_module, nn.Module):
            raise TypeError("rCM fake score must be an nn.Module")
        role_modules = [student_module, teacher_module]
        if fake_score_module is not None:
            role_modules.append(fake_score_module)
        if len({id(module) for module in role_modules}) != len(role_modules):
            raise ValueError("rCM student, teacher, and fake-score modules must be distinct")
        if not isinstance(loss_adapter, RCMLossAdapter):
            raise TypeError("loss_adapter must implement RCMLossAdapter")
        if any(parameter.requires_grad for parameter in teacher_module.parameters()):
            raise ValueError("rCM teacher parameters must be frozen")
        if not isinstance(dmd_enabled, bool):
            raise TypeError("dmd_enabled must be a bool")
        if dmd_enabled and (fake_score_module is None or fake_score_optimizer is None):
            raise ValueError("rCM DMD requires a fake-score module and optimizer")
        if not dmd_enabled and (fake_score_module is not None or fake_score_optimizer is not None):
            raise ValueError("fake-score role cannot be configured while rCM DMD is disabled")
        warmup = non_negative_int(tangent_warmup_steps, field_name="tangent_warmup_steps")
        frequency = non_negative_int(
            student_update_frequency,
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
        student_parameters = trainable_parameters(student_module)
        audit_optimizer_parameters(student_optimizer, student_parameters, role="rCM student")
        fake_parameters: tuple[nn.Parameter, ...] = ()
        if fake_score_module is not None:
            fake_parameters = trainable_parameters(fake_score_module)
            assert fake_score_optimizer is not None
            audit_optimizer_parameters(fake_score_optimizer, fake_parameters, role="rCM fake-score")
            if {id(parameter) for parameter in student_parameters} & {
                id(parameter) for parameter in fake_parameters
            }:
                raise ValueError("rCM student and fake score cannot share trainable parameters")
        validate_stateful_or_none(student_scheduler, field_name="student_scheduler")
        validate_stateful_or_none(fake_score_scheduler, field_name="fake_score_scheduler")
        validate_stateful_or_none(student_ema, field_name="student_ema")
        if not dmd_enabled and fake_score_scheduler is not None:
            raise ValueError("fake_score_scheduler requires DMD")

        self.student_module = student_module
        self.teacher_module = teacher_module
        self.fake_score_module = fake_score_module
        self.loss_adapter = loss_adapter
        self.student_optimizer = student_optimizer
        self.fake_score_optimizer = fake_score_optimizer
        self.student_parameters = student_parameters
        self.fake_score_parameters = fake_parameters
        self.tangent_warmup_steps = warmup
        self.student_update_frequency = frequency
        self.dmd_enabled = dmd_enabled
        self.student_max_grad_norm = positive_float(
            student_max_grad_norm,
            field_name="student_max_grad_norm",
        )
        self.fake_score_max_grad_norm = (
            None
            if not dmd_enabled
            else positive_float(
                fake_score_max_grad_norm,
                field_name="fake_score_max_grad_norm",
            )
        )
        self.gradient_accumulation_steps = accumulation
        self.student_scheduler = student_scheduler
        self.fake_score_scheduler = fake_score_scheduler
        self.student_ema = student_ema
        self.parallel_context = parallel_context or PostTrainingParallelContext.current()
        self.parallel_context.audit_synchronized_module(student_module, role="rCM student")
        if fake_score_module is not None:
            self.parallel_context.audit_synchronized_module(fake_score_module, role="rCM fake-score")
        self.global_step = 0
        self.student_optimizer_steps = 0
        self.fake_score_optimizer_steps = 0
        self._active_phase = "idle"
        self._poisoned = False
        self.teacher_module.eval()

    @property
    def config_digest(self) -> str:
        return str(self.loss_adapter.config_digest)

    def is_student_phase(self, iteration: int) -> bool:
        step = non_negative_int(iteration, field_name="iteration")
        return (
            not self.dmd_enabled
            or step < self.tangent_warmup_steps
            or (step - self.tangent_warmup_steps) % self.student_update_frequency == 0
        )

    def effective_student_iteration(self, iteration: int) -> int:
        step = non_negative_int(iteration, field_name="iteration")
        if not self.dmd_enabled or step < self.tangent_warmup_steps:
            return step
        return self.tangent_warmup_steps + (
            step - self.tangent_warmup_steps
        ) // self.student_update_frequency

    def effective_fake_iteration(self, iteration: int) -> int:
        step = non_negative_int(iteration, field_name="iteration")
        effective = step - self.effective_student_iteration(step) - 1
        if effective < 0:
            raise ValueError("effective fake iteration is only defined during a fake-score phase")
        return effective

    def _normalize_batches(
        self,
        batch: RCMTrainingBatch | Sequence[RCMTrainingBatch],
    ) -> tuple[RCMTrainingBatch, ...]:
        if isinstance(batch, RCMTrainingBatch):
            batches = (batch,)
        elif isinstance(batch, Sequence):
            batches = tuple(batch)
        else:
            raise TypeError("batch must be RCMTrainingBatch or a sequence of microbatches")
        if len(batches) != self.gradient_accumulation_steps:
            raise ValueError(
                "rCM optimizer update requires exactly "
                f"{self.gradient_accumulation_steps} microbatches; got {len(batches)}"
            )
        if not all(isinstance(value, RCMTrainingBatch) for value in batches):
            raise TypeError("every rCM microbatch must be RCMTrainingBatch")
        return batches

    def train_step(
        self,
        batch: RCMTrainingBatch | Sequence[RCMTrainingBatch],
        *,
        generator: torch.Generator | None = None,
    ) -> RCMTrainResult:
        if self._poisoned or self._active_phase != "idle":
            raise RuntimeError("rCM engine has a partially committed update; restore a checkpoint")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be torch.Generator or None")
        batches = self._normalize_batches(batch)
        iteration = self.global_step
        student_phase = self.is_student_phase(iteration)
        phase = "student" if student_phase else "fake-score"
        active_module = self.student_module if student_phase else self.fake_score_module
        parameters = self.student_parameters if student_phase else self.fake_score_parameters
        if active_module is None or not parameters:
            raise RuntimeError(f"rCM {phase} role is unavailable")
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
        self.student_optimizer.zero_grad(set_to_none=True)
        if self.fake_score_optimizer is not None:
            self.fake_score_optimizer.zero_grad(set_to_none=True)
        results: list[RCMLossResult] = []
        optimizer_step_started = False
        try:
            self._active_phase = f"{phase}-backward"
            self.teacher_module.eval()
            if student_phase:
                self.student_module.train()
                if self.fake_score_module is not None:
                    self.fake_score_module.eval()
                frozen = self.fake_score_module
            else:
                self.student_module.eval()
                assert self.fake_score_module is not None
                self.fake_score_module.train()
                frozen = self.student_module
            with _frozen_parameters(frozen):
                for index, (microbatch, weight) in enumerate(zip(batches, weights, strict=True)):
                    final_microbatch = index + 1 == len(batches)
                    with accumulation_context(
                        active_module,
                        final_microbatch=final_microbatch,
                    ):
                        if student_phase:
                            result = self.loss_adapter.student_loss(
                                microbatch,
                                iteration=iteration,
                                effective_student_iteration=self.effective_student_iteration(
                                    iteration
                                ),
                                include_dmd=(
                                    self.dmd_enabled
                                    and iteration >= self.tangent_warmup_steps
                                ),
                                generator=generator,
                            )
                        else:
                            result = self.loss_adapter.fake_score_loss(
                                microbatch,
                                effective_fake_iteration=self.effective_fake_iteration(iteration),
                                generator=generator,
                            )
                        result = _finite_loss(result, role=f"rCM {phase}")
                        check_reported_weight(result, weight, role=f"rCM {phase}")
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
            max_norm = (
                self.student_max_grad_norm
                if student_phase
                else self.fake_score_max_grad_norm
            )
            assert max_norm is not None
            grad_norm = clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
            optimizer_step_started = True
            if student_phase:
                self.student_optimizer.step()
                self.student_optimizer_steps += 1
                self._active_phase = "student-committed"
                if self.student_scheduler is not None:
                    self.student_scheduler.step()
                if self.student_ema is not None:
                    self.student_ema.update(self.student_module)
            else:
                assert self.fake_score_optimizer is not None
                self.fake_score_optimizer.step()
                self.fake_score_optimizer_steps += 1
                self._active_phase = "fake-score-committed"
                if self.fake_score_scheduler is not None:
                    self.fake_score_scheduler.step()
            self.global_step += 1
            self._active_phase = "idle"
        except Exception:
            self.student_optimizer.zero_grad(set_to_none=True)
            if self.fake_score_optimizer is not None:
                self.fake_score_optimizer.zero_grad(set_to_none=True)
            if optimizer_step_started:
                self._poisoned = True
            else:
                self._active_phase = "idle"
            raise

        committed = role_metrics(
            results,
            global_numerator=numerator,
            global_denominator=denominator,
        )
        metrics = {
            **committed,
            "global_step": torch.tensor(self.global_step, device=loss.device),
            "student_optimizer_steps": torch.tensor(
                self.student_optimizer_steps,
                device=loss.device,
            ),
            "fake_score_optimizer_steps": torch.tensor(
                self.fake_score_optimizer_steps,
                device=loss.device,
            ),
            "effective_student_iteration": torch.tensor(
                self.effective_student_iteration(iteration),
                device=loss.device,
            ),
            "accumulated_microbatches": len(batches),
            "grad_norm": grad_norm.detach(),
        }
        if not student_phase:
            metrics["effective_fake_iteration"] = torch.tensor(
                self.effective_fake_iteration(iteration),
                device=loss.device,
            )
        return RCMTrainResult(phase=phase, loss=loss.detach().float(), metrics=metrics)

    def _expected_optimizer_steps(self, global_step: int) -> tuple[int, int]:
        if not self.dmd_enabled:
            return global_step, 0
        warmup_steps = min(global_step, self.tangent_warmup_steps)
        remaining = max(0, global_step - self.tangent_warmup_steps)
        post_warmup_student = (
            0
            if remaining == 0
            else (remaining - 1) // self.student_update_frequency + 1
        )
        student = warmup_steps + post_warmup_student
        return student, global_step - student

    def state_dict(self) -> dict[str, object]:
        if self._poisoned or self._active_phase != "idle":
            raise RuntimeError("cannot checkpoint a partially committed rCM update")
        return {
            "schema": RCM_ENGINE_STATE_SCHEMA,
            "global_step": self.global_step,
            "student_optimizer_steps": self.student_optimizer_steps,
            "fake_score_optimizer_steps": self.fake_score_optimizer_steps,
            "tangent_warmup_steps": self.tangent_warmup_steps,
            "student_update_frequency": self.student_update_frequency,
            "dmd_enabled": self.dmd_enabled,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "config_digest": self.config_digest,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("rCM engine state must be a mapping")
        expected = {
            "schema",
            "global_step",
            "student_optimizer_steps",
            "fake_score_optimizer_steps",
            "tangent_warmup_steps",
            "student_update_frequency",
            "dmd_enabled",
            "gradient_accumulation_steps",
            "config_digest",
        }
        if set(state_dict) != expected:
            raise ValueError("rCM engine state fields differ from the active schema")
        if state_dict["schema"] != RCM_ENGINE_STATE_SCHEMA:
            raise ValueError(f"unsupported rCM engine schema: {state_dict['schema']!r}")
        active = {
            "tangent_warmup_steps": self.tangent_warmup_steps,
            "student_update_frequency": self.student_update_frequency,
            "dmd_enabled": self.dmd_enabled,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "config_digest": self.config_digest,
        }
        for name, value in active.items():
            if state_dict[name] != value:
                raise ValueError(f"saved rCM {name} differs from the active engine")
        global_step = non_negative_int(state_dict["global_step"], field_name="global_step")
        student_steps = non_negative_int(
            state_dict["student_optimizer_steps"],
            field_name="student_optimizer_steps",
        )
        fake_steps = non_negative_int(
            state_dict["fake_score_optimizer_steps"],
            field_name="fake_score_optimizer_steps",
        )
        expected_student, expected_fake = self._expected_optimizer_steps(global_step)
        if (student_steps, fake_steps) != (expected_student, expected_fake):
            raise ValueError("saved rCM optimizer counters violate the active cadence")
        self.global_step = global_step
        self.student_optimizer_steps = student_steps
        self.fake_score_optimizer_steps = fake_steps
        self._active_phase = "idle"
        self._poisoned = False
        self.student_optimizer.zero_grad(set_to_none=True)
        if self.fake_score_optimizer is not None:
            self.fake_score_optimizer.zero_grad(set_to_none=True)


__all__ = ["NativeRCMTrainEngine", "RCM_ENGINE_STATE_SCHEMA", "RCMTrainResult"]
