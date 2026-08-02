"""WorldFoundry-native Distribution Matching Distillation state machine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from worldfoundry.core.gradient import clip_grad_norm_
from worldfoundry.training.optimization import (
    audit_optimizer_parameters,
    trainable_parameters,
)

from ...shared.accumulation import (
    accumulation_context,
    check_reported_weight,
    declared_loss_weight,
    global_denominator,
    global_loss_statistics,
    role_metrics,
)
from ...shared.distributed import PostTrainingParallelContext
from ...shared.validation import (
    non_negative_int,
    positive_float,
    validate_stateful_or_none,
)
from .contracts import DMDLossAdapter, DMDTrainingBatch
from .objective import DMDLossResult

DMD_ENGINE_STATE_SCHEMA = "worldfoundry-dmd-engine"


def _finite_scalar_loss(result: object, *, role: str) -> DMDLossResult:
    if not isinstance(result, DMDLossResult):
        raise TypeError(f"{role} loss adapter must return DMDLossResult")
    if not isinstance(result.loss, torch.Tensor) or result.loss.numel() != 1:
        raise TypeError(f"{role} loss must be one scalar tensor")
    if not bool(torch.isfinite(result.loss.detach()).all()):
        raise FloatingPointError(f"non-finite {role} loss")
    return result


def _microbatches(
    value: DMDTrainingBatch | Sequence[DMDTrainingBatch],
    *,
    expected: int,
    role: str,
) -> tuple[DMDTrainingBatch, ...]:
    if isinstance(value, DMDTrainingBatch):
        batches = (value,)
    elif isinstance(value, Sequence):
        batches = tuple(value)
    else:
        raise TypeError(f"{role} batch must be DMDTrainingBatch or a sequence of them")
    if len(batches) != expected:
        raise ValueError(f"DMD {role} optimizer iteration requires exactly {expected} microbatches; got {len(batches)}")
    if not all(isinstance(batch, DMDTrainingBatch) for batch in batches):
        raise TypeError(f"every DMD {role} microbatch must be DMDTrainingBatch")
    return batches


@dataclass(frozen=True, slots=True)
class DMDTrainResult:
    generator_loss: torch.Tensor
    fake_score_loss: torch.Tensor
    generator_updated: bool
    metrics: Mapping[str, object]


class NativeDMDTrainEngine:
    """Own the three DMD roles, two optimizers, cadence, and commit state."""

    def __init__(
        self,
        *,
        student_module: nn.Module,
        real_score_module: nn.Module,
        fake_score_module: nn.Module,
        loss_adapter: DMDLossAdapter,
        student_optimizer: torch.optim.Optimizer,
        fake_score_optimizer: torch.optim.Optimizer,
        generator_update_interval: int = 5,
        student_max_grad_norm: float = 1.0,
        fake_score_max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        student_scheduler: object | None = None,
        fake_score_scheduler: object | None = None,
        student_scheduler_cadence: str = "iteration",
        student_ema: object | None = None,
        student_ema_start_step: int = 0,
        parallel_context: PostTrainingParallelContext | None = None,
    ) -> None:
        if not all(isinstance(module, nn.Module) for module in (student_module, real_score_module, fake_score_module)):
            raise TypeError("all DMD roles must be nn.Module values")
        if len({id(student_module), id(real_score_module), id(fake_score_module)}) != 3:
            raise ValueError("DMD student, real-score teacher, and fake-score critic must be distinct modules")
        if not isinstance(loss_adapter, DMDLossAdapter):
            raise TypeError("loss_adapter must implement DMDLossAdapter")
        commit_generator_step = getattr(loss_adapter, "commit_generator_step", None)
        if commit_generator_step is not None and not callable(commit_generator_step):
            raise TypeError("loss_adapter.commit_generator_step must be callable")
        interval = non_negative_int(generator_update_interval, field_name="generator_update_interval")
        if interval == 0:
            raise ValueError("generator_update_interval must be positive")
        if student_scheduler_cadence not in {"iteration", "generator-update"}:
            raise ValueError("student_scheduler_cadence must be 'iteration' or 'generator-update'")
        accumulation = non_negative_int(
            gradient_accumulation_steps,
            field_name="gradient_accumulation_steps",
        )
        if accumulation == 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        validate_stateful_or_none(student_scheduler, field_name="student_scheduler")
        validate_stateful_or_none(fake_score_scheduler, field_name="fake_score_scheduler")
        validate_stateful_or_none(student_ema, field_name="student_ema")
        ema_start_step = non_negative_int(
            student_ema_start_step,
            field_name="student_ema_start_step",
        )
        if any(parameter.requires_grad for parameter in real_score_module.parameters()):
            raise ValueError("real-score teacher parameters must be frozen")

        student_parameters = trainable_parameters(student_module)
        fake_score_parameters = trainable_parameters(fake_score_module)
        if {id(parameter) for parameter in student_parameters} & {id(parameter) for parameter in fake_score_parameters}:
            raise ValueError("student and fake-score critic cannot share trainable parameters")
        audit_optimizer_parameters(student_optimizer, student_parameters, role="student")
        audit_optimizer_parameters(fake_score_optimizer, fake_score_parameters, role="fake-score")

        self.student_module = student_module
        self.real_score_module = real_score_module
        self.fake_score_module = fake_score_module
        self.loss_adapter = loss_adapter
        self._commit_generator_step = commit_generator_step
        self.student_optimizer = student_optimizer
        self.fake_score_optimizer = fake_score_optimizer
        self.student_parameters = student_parameters
        self.fake_score_parameters = fake_score_parameters
        self.generator_update_interval = interval
        self.student_max_grad_norm = positive_float(student_max_grad_norm, field_name="student_max_grad_norm")
        self.fake_score_max_grad_norm = positive_float(
            fake_score_max_grad_norm,
            field_name="fake_score_max_grad_norm",
        )
        self.gradient_accumulation_steps = accumulation
        self.student_scheduler = student_scheduler
        self.fake_score_scheduler = fake_score_scheduler
        self.student_scheduler_cadence = student_scheduler_cadence
        self.student_ema = student_ema
        self.student_ema_start_step = ema_start_step
        self.parallel_context = parallel_context or PostTrainingParallelContext.current()
        self.parallel_context.audit_synchronized_module(student_module, role="DMD student")
        self.parallel_context.audit_synchronized_module(fake_score_module, role="DMD fake-score critic")
        self.global_step = 0
        self.student_optimizer_steps = 0
        self.fake_score_optimizer_steps = 0
        self._phase = "idle"
        self._poisoned = False
        self.real_score_module.eval()
        if self.student_ema is not None and self.student_ema_start_step == 0:
            self._start_student_ema()

    def _start_student_ema(self) -> None:
        assert self.student_ema is not None
        start = getattr(self.student_ema, "start", None)
        if start is not None:
            if not callable(start):
                raise TypeError("student_ema.start must be callable")
            start(self.student_module)

    @property
    def schedule_digest(self) -> str:
        return str(self.loss_adapter.schedule_digest)

    def train_step(
        self,
        batch: DMDTrainingBatch | Sequence[DMDTrainingBatch],
        *,
        fake_score_batch: DMDTrainingBatch | Sequence[DMDTrainingBatch] | None = None,
        generator: torch.Generator | None = None,
    ) -> DMDTrainResult:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("DMD engine has a partially committed iteration; restore the last checkpoint")
        batches = _microbatches(
            batch,
            expected=self.gradient_accumulation_steps,
            role="generator",
        )
        fake_batches = (
            batches
            if fake_score_batch is None
            else _microbatches(
                fake_score_batch,
                expected=self.gradient_accumulation_steps,
                role="fake-score",
            )
        )
        generator_due = self.global_step % self.generator_update_interval == 0
        student_results: list[DMDLossResult] = []
        student_weights: list[torch.Tensor] = []
        fake_results: list[DMDLossResult] = []
        fake_weights: list[torch.Tensor] = []
        student_grad_norm = torch.zeros((), device=self.student_parameters[0].device)
        self.student_optimizer.zero_grad(set_to_none=True)
        self.fake_score_optimizer.zero_grad(set_to_none=True)
        optimizer_step_started = False
        try:
            if generator_due:
                self._phase = "student-backward"
                student_weights = [
                    declared_loss_weight(
                        self.loss_adapter,
                        microbatch,
                        role="generator",
                        device=self.student_parameters[0].device,
                    )
                    for microbatch in batches
                ]
                global_student_weight = global_denominator(
                    student_weights,
                    self.parallel_context,
                )
                for index, (microbatch, student_weight) in enumerate(zip(batches, student_weights, strict=True)):
                    final_microbatch = index + 1 == len(batches)
                    with accumulation_context(
                        self.student_module,
                        final_microbatch=final_microbatch,
                    ):
                        student_result = _finite_scalar_loss(
                            self.loss_adapter.generator_loss(microbatch, generator=generator),
                            role="DMD generator",
                        )
                        check_reported_weight(
                            student_result,
                            student_weight,
                            role="DMD generator",
                        )
                        gradient_weight = (
                            student_weight / global_student_weight * float(self.parallel_context.world_size)
                        )
                        (student_result.loss * gradient_weight).backward()
                    student_results.append(student_result)
                student_grad_norm = clip_grad_norm_(
                    self.student_parameters,
                    self.student_max_grad_norm,
                    error_if_nonfinite=True,
                )
                optimizer_step_started = True
                self.student_optimizer.step()
                self.student_optimizer_steps += 1
                self._phase = "student-committed"
                if self.student_ema is not None and self.global_step >= self.student_ema_start_step:
                    self.student_ema.update(self.student_module)

            # Fake-score training observes the committed student update from
            # this iteration rather than the pre-update student.
            self._phase = "fake-score-backward"
            fake_weights = [
                declared_loss_weight(
                    self.loss_adapter,
                    microbatch,
                    role="fake-score",
                    device=self.fake_score_parameters[0].device,
                )
                for microbatch in fake_batches
            ]
            global_fake_weight = global_denominator(
                fake_weights,
                self.parallel_context,
            )
            for index, (microbatch, fake_weight) in enumerate(zip(fake_batches, fake_weights, strict=True)):
                final_microbatch = index + 1 == len(fake_batches)
                with accumulation_context(
                    self.fake_score_module,
                    final_microbatch=final_microbatch,
                ):
                    fake_result = _finite_scalar_loss(
                        self.loss_adapter.fake_score_loss(microbatch, generator=generator),
                        role="DMD fake-score",
                    )
                    check_reported_weight(
                        fake_result,
                        fake_weight,
                        role="DMD fake-score",
                    )
                    gradient_weight = fake_weight / global_fake_weight * float(self.parallel_context.world_size)
                    (fake_result.loss * gradient_weight).backward()
                fake_results.append(fake_result)
            fake_grad_norm = clip_grad_norm_(
                self.fake_score_parameters,
                self.fake_score_max_grad_norm,
                error_if_nonfinite=True,
            )
            optimizer_step_started = True
            self.fake_score_optimizer.step()
            self.fake_score_optimizer_steps += 1
            self._phase = "fake-score-committed"

            if self.fake_score_scheduler is not None:
                self.fake_score_scheduler.step()
            if self.student_scheduler is not None and (self.student_scheduler_cadence == "iteration" or generator_due):
                self.student_scheduler.step()
            if generator_due and self._commit_generator_step is not None:
                # Stateful composite objectives commit auxiliary statistics
                # only after both optimizers have completed.  An exception
                # poisons the iteration, so a checkpoint can never contain a
                # model update without its matching objective state.
                self._commit_generator_step(tuple(student_results))
            next_global_step = self.global_step + 1
            if self.student_ema is not None and self.global_step < self.student_ema_start_step <= next_global_step:
                self._start_student_ema()
            self.global_step = next_global_step
            self._phase = "idle"
        except Exception:
            self.student_optimizer.zero_grad(set_to_none=True)
            self.fake_score_optimizer.zero_grad(set_to_none=True)
            if optimizer_step_started:
                self._poisoned = True
            else:
                self._phase = "idle"
            raise

        assert fake_results
        fake_numerator, fake_denominator, fake_loss = global_loss_statistics(
            fake_results,
            fake_weights,
            self.parallel_context,
        )
        fake_metrics = role_metrics(
            fake_results,
            global_numerator=fake_numerator,
            global_denominator=fake_denominator,
        )
        if not student_results:
            student_loss = torch.zeros((), device=fake_loss.device, dtype=torch.float32)
            student_metrics: Mapping[str, object] = {}
        else:
            student_numerator, student_denominator, student_loss = global_loss_statistics(
                student_results,
                student_weights,
                self.parallel_context,
            )
            student_metrics = role_metrics(
                student_results,
                global_numerator=student_numerator,
                global_denominator=student_denominator,
            )
        metrics: dict[str, object] = {
            "global_step": torch.tensor(self.global_step, device=fake_loss.device, dtype=torch.int64),
            "student_optimizer_steps": torch.tensor(
                self.student_optimizer_steps,
                device=fake_loss.device,
                dtype=torch.int64,
            ),
            "fake_score_optimizer_steps": torch.tensor(
                self.fake_score_optimizer_steps,
                device=fake_loss.device,
                dtype=torch.int64,
            ),
            "accumulated_microbatches": len(batches),
            "generator_microbatches": len(batches) if generator_due else 0,
            "fake_score_microbatches": len(fake_batches),
            "student_grad_norm": student_grad_norm.detach(),
            "fake_score_grad_norm": fake_grad_norm.detach(),
            "student": dict(student_metrics),
            "fake_score": fake_metrics,
        }
        return DMDTrainResult(
            generator_loss=student_loss,
            fake_score_loss=fake_loss,
            generator_updated=generator_due,
            metrics=metrics,
        )

    def state_dict(self) -> dict[str, object]:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("cannot checkpoint a partially committed DMD iteration")
        return {
            "schema": DMD_ENGINE_STATE_SCHEMA,
            "global_step": self.global_step,
            "student_optimizer_steps": self.student_optimizer_steps,
            "fake_score_optimizer_steps": self.fake_score_optimizer_steps,
            "generator_update_interval": self.generator_update_interval,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "student_scheduler_cadence": self.student_scheduler_cadence,
            "student_ema_start_step": self.student_ema_start_step,
            "schedule_digest": self.schedule_digest,
            "data_parallel_size": self.parallel_context.world_size,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("DMD engine state must be a mapping")
        expected = {
            "schema",
            "global_step",
            "student_optimizer_steps",
            "fake_score_optimizer_steps",
            "generator_update_interval",
            "gradient_accumulation_steps",
            "student_scheduler_cadence",
            "student_ema_start_step",
            "schedule_digest",
            "data_parallel_size",
        }
        if set(state_dict) != expected:
            raise ValueError("DMD engine state fields differ from the active schema")
        if state_dict["schema"] != DMD_ENGINE_STATE_SCHEMA:
            raise ValueError(f"unsupported DMD engine schema: {state_dict['schema']!r}")
        if int(state_dict["generator_update_interval"]) != self.generator_update_interval:
            raise ValueError("saved DMD generator cadence differs from the active engine")
        if int(state_dict["gradient_accumulation_steps"]) != self.gradient_accumulation_steps:
            raise ValueError("saved DMD accumulation cadence differs from the active engine")
        if state_dict["student_scheduler_cadence"] != self.student_scheduler_cadence:
            raise ValueError("saved DMD scheduler cadence differs from the active engine")
        if int(state_dict["student_ema_start_step"]) != self.student_ema_start_step:
            raise ValueError("saved DMD EMA start differs from the active engine")
        if state_dict["schedule_digest"] != self.schedule_digest:
            raise ValueError("saved DMD few-step schedule differs from the active engine")
        if int(state_dict["data_parallel_size"]) != self.parallel_context.world_size:
            raise ValueError("saved DMD data-parallel size differs from the active engine")
        global_step = non_negative_int(state_dict["global_step"], field_name="global_step")
        student_steps = non_negative_int(
            state_dict["student_optimizer_steps"],
            field_name="student_optimizer_steps",
        )
        fake_steps = non_negative_int(
            state_dict["fake_score_optimizer_steps"],
            field_name="fake_score_optimizer_steps",
        )
        expected_student_steps = 0 if global_step == 0 else (global_step - 1) // self.generator_update_interval + 1
        if student_steps != expected_student_steps or fake_steps != global_step:
            raise ValueError("saved DMD optimizer counters violate the configured cadence")
        self.global_step = global_step
        self.student_optimizer_steps = student_steps
        self.fake_score_optimizer_steps = fake_steps
        self._phase = "idle"
        self._poisoned = False
        self.student_optimizer.zero_grad(set_to_none=True)
        self.fake_score_optimizer.zero_grad(set_to_none=True)


__all__ = ["DMD_ENGINE_STATE_SCHEMA", "DMDTrainResult", "NativeDMDTrainEngine"]
