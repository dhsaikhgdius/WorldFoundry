"""Atomic generator-then-guidance optimizer engine for native DMD2."""

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
from .contracts import DMD2LossAdapter, DMD2TrainingBatch
from .objective import DMD2LossResult

DMD2_ENGINE_STATE_SCHEMA = "worldfoundry-dmd2-engine"


def _finite_loss(result: object, *, role: str) -> DMD2LossResult:
    if not isinstance(result, DMD2LossResult):
        raise TypeError(f"{role} loss adapter must return DMD2LossResult")
    if not isinstance(result.loss, torch.Tensor) or result.loss.numel() != 1:
        raise TypeError(f"{role} loss must be one scalar tensor")
    if not bool(torch.isfinite(result.loss.detach()).all()):
        raise FloatingPointError(f"non-finite {role} loss")
    return result


@dataclass(frozen=True, slots=True)
class DMD2TrainResult:
    generator_loss: torch.Tensor
    guidance_loss: torch.Tensor
    generator_updated: bool
    metrics: Mapping[str, object]


class NativeDMD2TrainEngine:
    """Commit G first, then train the shared fake-score/discriminator role."""

    def __init__(
        self,
        *,
        student_module: nn.Module,
        teacher_module: nn.Module,
        guidance_module: nn.Module,
        loss_adapter: DMD2LossAdapter,
        student_optimizer: torch.optim.Optimizer,
        guidance_optimizer: torch.optim.Optimizer,
        generator_update_interval: int = 5,
        student_max_grad_norm: float = 1.0,
        guidance_max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        student_scheduler: object | None = None,
        guidance_scheduler: object | None = None,
        student_scheduler_cadence: str = "iteration",
        student_ema: object | None = None,
        parallel_context: PostTrainingParallelContext | None = None,
    ) -> None:
        modules = (student_module, teacher_module, guidance_module)
        if not all(isinstance(module, nn.Module) for module in modules):
            raise TypeError("all DMD2 roles must be nn.Module values")
        if len({id(module) for module in modules}) != 3:
            raise ValueError("DMD2 student, teacher, and guidance modules must be distinct")
        if not isinstance(loss_adapter, DMD2LossAdapter):
            raise TypeError("loss_adapter must implement DMD2LossAdapter")
        if any(parameter.requires_grad for parameter in teacher_module.parameters()):
            raise ValueError("DMD2 teacher parameters must be frozen")
        interval = non_negative_int(generator_update_interval, field_name="generator_update_interval")
        accumulation = non_negative_int(
            gradient_accumulation_steps,
            field_name="gradient_accumulation_steps",
        )
        if interval == 0 or accumulation == 0:
            raise ValueError("DMD2 update interval and accumulation steps must be positive")
        cadence = str(student_scheduler_cadence).strip().lower().replace("_", "-")
        if cadence not in {"iteration", "generator-update"}:
            raise ValueError("student_scheduler_cadence must be 'iteration' or 'generator-update'")
        validate_stateful_or_none(student_scheduler, field_name="student_scheduler")
        validate_stateful_or_none(guidance_scheduler, field_name="guidance_scheduler")
        validate_stateful_or_none(student_ema, field_name="student_ema")

        student_parameters = trainable_parameters(student_module)
        guidance_parameters = trainable_parameters(guidance_module)
        if {id(parameter) for parameter in student_parameters} & {
            id(parameter) for parameter in guidance_parameters
        }:
            raise ValueError("DMD2 student and guidance cannot share trainable parameters")
        audit_optimizer_parameters(student_optimizer, student_parameters, role="DMD2 student")
        audit_optimizer_parameters(guidance_optimizer, guidance_parameters, role="DMD2 guidance")

        self.student_module = student_module
        self.teacher_module = teacher_module
        self.guidance_module = guidance_module
        self.loss_adapter = loss_adapter
        self.student_optimizer = student_optimizer
        self.guidance_optimizer = guidance_optimizer
        self.student_parameters = student_parameters
        self.guidance_parameters = guidance_parameters
        self.generator_update_interval = interval
        self.student_max_grad_norm = positive_float(
            student_max_grad_norm,
            field_name="student_max_grad_norm",
        )
        self.guidance_max_grad_norm = positive_float(
            guidance_max_grad_norm,
            field_name="guidance_max_grad_norm",
        )
        self.gradient_accumulation_steps = accumulation
        self.student_scheduler = student_scheduler
        self.guidance_scheduler = guidance_scheduler
        self.student_scheduler_cadence = cadence
        self.student_ema = student_ema
        self.parallel_context = parallel_context or PostTrainingParallelContext.current()
        self.parallel_context.audit_synchronized_module(student_module, role="DMD2 student")
        self.parallel_context.audit_synchronized_module(guidance_module, role="DMD2 guidance")
        self.global_step = 0
        self.student_optimizer_steps = 0
        self.guidance_optimizer_steps = 0
        self._phase = "idle"
        self._poisoned = False
        self.teacher_module.eval()

    @property
    def config_digest(self) -> str:
        return str(self.loss_adapter.config_digest)

    def _batches(
        self,
        batch: DMD2TrainingBatch | Sequence[DMD2TrainingBatch],
    ) -> tuple[DMD2TrainingBatch, ...]:
        if isinstance(batch, DMD2TrainingBatch):
            batches = (batch,)
        elif isinstance(batch, Sequence):
            batches = tuple(batch)
        else:
            raise TypeError("batch must be DMD2TrainingBatch or a sequence of microbatches")
        if len(batches) != self.gradient_accumulation_steps:
            raise ValueError(
                "DMD2 optimizer iteration requires exactly "
                f"{self.gradient_accumulation_steps} microbatches; got {len(batches)}"
            )
        if not all(isinstance(value, DMD2TrainingBatch) for value in batches):
            raise TypeError("every DMD2 microbatch must be DMD2TrainingBatch")
        return batches

    def train_step(
        self,
        batch: DMD2TrainingBatch | Sequence[DMD2TrainingBatch],
        *,
        generator: torch.Generator | None = None,
    ) -> DMD2TrainResult:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("DMD2 engine has a partially committed iteration; restore the last checkpoint")
        batches = self._batches(batch)
        generator_due = self.global_step % self.generator_update_interval == 0
        generator_results: list[DMD2LossResult] = []
        generator_weights: list[torch.Tensor] = []
        guidance_results: list[DMD2LossResult] = []
        guidance_weights: list[torch.Tensor] = []
        student_grad_norm = torch.zeros((), device=self.student_parameters[0].device)
        self.student_optimizer.zero_grad(set_to_none=True)
        self.guidance_optimizer.zero_grad(set_to_none=True)
        optimizer_mutated = False
        try:
            if generator_due:
                self._phase = "generator-backward"
                self.student_module.train()
                self.guidance_module.eval()
                self.teacher_module.eval()
                generator_weights = [
                    declared_loss_weight(
                        self.loss_adapter,
                        microbatch,
                        role="generator",
                        device=self.student_parameters[0].device,
                    )
                    for microbatch in batches
                ]
                total_generator_weight = global_denominator(
                    generator_weights,
                    self.parallel_context,
                )
                for index, (microbatch, weight) in enumerate(
                    zip(batches, generator_weights, strict=True)
                ):
                    with accumulation_context(
                        self.student_module,
                        final_microbatch=index + 1 == len(batches),
                    ):
                        result = _finite_loss(
                            self.loss_adapter.generator_loss(microbatch, generator=generator),
                            role="DMD2 generator",
                        )
                        check_reported_weight(result, weight, role="DMD2 generator")
                        gradient_weight = (
                            weight / total_generator_weight * float(self.parallel_context.world_size)
                        )
                        (result.loss * gradient_weight).backward()
                    generator_results.append(result)
                if any(parameter.grad is not None for parameter in self.guidance_parameters):
                    raise RuntimeError("DMD2 generator phase produced guidance parameter gradients")
                student_grad_norm = clip_grad_norm_(
                    self.student_parameters,
                    self.student_max_grad_norm,
                    error_if_nonfinite=True,
                )
                optimizer_mutated = True
                self.student_optimizer.step()
                self.student_optimizer_steps += 1
                self._phase = "generator-committed"
                if self.student_ema is not None:
                    self.student_ema.update(self.student_module)

            # This phase intentionally runs after the student commit and uses a
            # fresh student rollout, so the guidance role sees post-G weights.
            self._phase = "guidance-backward"
            self.student_module.eval()
            self.guidance_module.train()
            self.teacher_module.eval()
            self.student_optimizer.zero_grad(set_to_none=True)
            self.guidance_optimizer.zero_grad(set_to_none=True)
            guidance_weights = [
                declared_loss_weight(
                    self.loss_adapter,
                    microbatch,
                    role="guidance",
                    device=self.guidance_parameters[0].device,
                )
                for microbatch in batches
            ]
            total_guidance_weight = global_denominator(guidance_weights, self.parallel_context)
            for index, (microbatch, weight) in enumerate(
                zip(batches, guidance_weights, strict=True)
            ):
                with accumulation_context(
                    self.guidance_module,
                    final_microbatch=index + 1 == len(batches),
                ):
                    result = _finite_loss(
                        self.loss_adapter.guidance_loss(microbatch, generator=generator),
                        role="DMD2 guidance",
                    )
                    check_reported_weight(result, weight, role="DMD2 guidance")
                    gradient_weight = weight / total_guidance_weight * float(self.parallel_context.world_size)
                    (result.loss * gradient_weight).backward()
                guidance_results.append(result)
            if any(parameter.grad is not None for parameter in self.student_parameters):
                raise RuntimeError("DMD2 guidance phase produced student parameter gradients")
            guidance_grad_norm = clip_grad_norm_(
                self.guidance_parameters,
                self.guidance_max_grad_norm,
                error_if_nonfinite=True,
            )
            optimizer_mutated = True
            self.guidance_optimizer.step()
            self.guidance_optimizer_steps += 1
            self._phase = "guidance-committed"
            if self.guidance_scheduler is not None:
                self.guidance_scheduler.step()
            if self.student_scheduler is not None and (
                self.student_scheduler_cadence == "iteration" or generator_due
            ):
                self.student_scheduler.step()
            self.global_step += 1
            self._phase = "idle"
        except Exception:
            self.student_optimizer.zero_grad(set_to_none=True)
            self.guidance_optimizer.zero_grad(set_to_none=True)
            if optimizer_mutated:
                self._poisoned = True
            else:
                self._phase = "idle"
            raise

        guidance_numerator, guidance_denominator, guidance_loss = global_loss_statistics(
            guidance_results,
            guidance_weights,
            self.parallel_context,
        )
        guidance_metrics = role_metrics(
            guidance_results,
            global_numerator=guidance_numerator,
            global_denominator=guidance_denominator,
        )
        if generator_results:
            generator_numerator, generator_denominator, generator_loss = global_loss_statistics(
                generator_results,
                generator_weights,
                self.parallel_context,
            )
            generator_metrics: Mapping[str, object] = role_metrics(
                generator_results,
                global_numerator=generator_numerator,
                global_denominator=generator_denominator,
            )
        else:
            generator_loss = torch.zeros((), device=guidance_loss.device, dtype=torch.float32)
            generator_metrics = {}
        metrics = {
            "global_step": torch.tensor(self.global_step, device=guidance_loss.device, dtype=torch.int64),
            "student_optimizer_steps": torch.tensor(
                self.student_optimizer_steps,
                device=guidance_loss.device,
                dtype=torch.int64,
            ),
            "guidance_optimizer_steps": torch.tensor(
                self.guidance_optimizer_steps,
                device=guidance_loss.device,
                dtype=torch.int64,
            ),
            "accumulated_microbatches": len(batches),
            "student_grad_norm": student_grad_norm.detach(),
            "guidance_grad_norm": guidance_grad_norm.detach(),
            "generator": dict(generator_metrics),
            "guidance": guidance_metrics,
        }
        return DMD2TrainResult(
            generator_loss=generator_loss.detach().float(),
            guidance_loss=guidance_loss.detach().float(),
            generator_updated=generator_due,
            metrics=metrics,
        )

    def state_dict(self) -> dict[str, object]:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("cannot checkpoint a partially committed DMD2 iteration")
        return {
            "schema": DMD2_ENGINE_STATE_SCHEMA,
            "global_step": self.global_step,
            "student_optimizer_steps": self.student_optimizer_steps,
            "guidance_optimizer_steps": self.guidance_optimizer_steps,
            "generator_update_interval": self.generator_update_interval,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "student_scheduler_cadence": self.student_scheduler_cadence,
            "config_digest": self.config_digest,
            "data_parallel_size": self.parallel_context.world_size,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("DMD2 engine state must be a mapping")
        expected = {
            "schema",
            "global_step",
            "student_optimizer_steps",
            "guidance_optimizer_steps",
            "generator_update_interval",
            "gradient_accumulation_steps",
            "student_scheduler_cadence",
            "config_digest",
            "data_parallel_size",
        }
        if set(state_dict) != expected:
            raise ValueError("DMD2 engine state fields differ from the active schema")
        if state_dict["schema"] != DMD2_ENGINE_STATE_SCHEMA:
            raise ValueError(f"unsupported DMD2 engine schema: {state_dict['schema']!r}")
        if str(state_dict["config_digest"]) != self.config_digest:
            raise ValueError("saved DMD2 recipe differs from the active engine")
        for name, active in (
            ("generator_update_interval", self.generator_update_interval),
            ("gradient_accumulation_steps", self.gradient_accumulation_steps),
            ("data_parallel_size", self.parallel_context.world_size),
        ):
            if int(state_dict[name]) != active:
                raise ValueError(f"saved DMD2 {name} differs from the active engine")
        if str(state_dict["student_scheduler_cadence"]) != self.student_scheduler_cadence:
            raise ValueError("saved DMD2 scheduler cadence differs from the active engine")
        global_step = non_negative_int(state_dict["global_step"], field_name="global_step")
        student_steps = non_negative_int(
            state_dict["student_optimizer_steps"],
            field_name="student_optimizer_steps",
        )
        guidance_steps = non_negative_int(
            state_dict["guidance_optimizer_steps"],
            field_name="guidance_optimizer_steps",
        )
        expected_student_steps = 0 if global_step == 0 else (global_step - 1) // self.generator_update_interval + 1
        if student_steps != expected_student_steps or guidance_steps != global_step:
            raise ValueError("saved DMD2 optimizer counters violate the update cadence")
        self.global_step = global_step
        self.student_optimizer_steps = student_steps
        self.guidance_optimizer_steps = guidance_steps
        self._phase = "idle"
        self._poisoned = False


__all__ = ["DMD2_ENGINE_STATE_SCHEMA", "DMD2TrainResult", "NativeDMD2TrainEngine"]
