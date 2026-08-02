"""Atomic fake-score-then-generator engine for native SiD."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
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
from .contracts import SIDLossAdapter, SIDTrainingBatch
from .objective import SIDLossResult

SID_ENGINE_STATE_SCHEMA = "worldfoundry-sid-engine"


def _finite_loss(result: object, *, role: str) -> SIDLossResult:
    if not isinstance(result, SIDLossResult):
        raise TypeError(f"{role} loss adapter must return SIDLossResult")
    if not isinstance(result.loss, torch.Tensor) or result.loss.numel() != 1:
        raise TypeError(f"{role} loss must be one scalar tensor")
    if not bool(torch.isfinite(result.loss.detach()).all()):
        raise FloatingPointError(f"non-finite {role} loss")
    return result


@dataclass(frozen=True, slots=True)
class SIDTrainResult:
    fake_score_loss: torch.Tensor
    generator_loss: torch.Tensor
    target_index: int
    metrics: Mapping[str, object]


class NativeSIDTrainEngine:
    """Commit fake-score first and poison state if the later G commit fails."""

    def __init__(
        self,
        *,
        student_module: nn.Module,
        teacher_module: nn.Module,
        fake_score_module: nn.Module,
        loss_adapter: SIDLossAdapter,
        student_optimizer: torch.optim.Optimizer,
        fake_score_optimizer: torch.optim.Optimizer,
        student_max_grad_norm: float = 1.0,
        fake_score_max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        student_scheduler: object | None = None,
        fake_score_scheduler: object | None = None,
        student_ema: object | None = None,
        parallel_context: PostTrainingParallelContext | None = None,
    ) -> None:
        modules = (student_module, teacher_module, fake_score_module)
        if not all(isinstance(module, nn.Module) for module in modules):
            raise TypeError("all SiD roles must be nn.Module values")
        if len({id(module) for module in modules}) != 3:
            raise ValueError("SiD student, teacher, and fake-score modules must be distinct")
        if not isinstance(loss_adapter, SIDLossAdapter):
            raise TypeError("loss_adapter must implement SIDLossAdapter")
        if any(parameter.requires_grad for parameter in teacher_module.parameters()):
            raise ValueError("SiD teacher parameters must be frozen")
        accumulation = non_negative_int(
            gradient_accumulation_steps,
            field_name="gradient_accumulation_steps",
        )
        if accumulation == 0:
            raise ValueError("SiD gradient accumulation steps must be positive")
        validate_stateful_or_none(student_scheduler, field_name="student_scheduler")
        validate_stateful_or_none(fake_score_scheduler, field_name="fake_score_scheduler")
        validate_stateful_or_none(student_ema, field_name="student_ema")
        student_parameters = trainable_parameters(student_module)
        fake_score_parameters = trainable_parameters(fake_score_module)
        if {id(parameter) for parameter in student_parameters} & {
            id(parameter) for parameter in fake_score_parameters
        }:
            raise ValueError("SiD student and fake-score cannot share trainable parameters")
        audit_optimizer_parameters(student_optimizer, student_parameters, role="SiD student")
        audit_optimizer_parameters(
            fake_score_optimizer,
            fake_score_parameters,
            role="SiD fake-score",
        )

        self.student_module = student_module
        self.teacher_module = teacher_module
        self.fake_score_module = fake_score_module
        self.loss_adapter = loss_adapter
        self.student_optimizer = student_optimizer
        self.fake_score_optimizer = fake_score_optimizer
        self.student_parameters = student_parameters
        self.fake_score_parameters = fake_score_parameters
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
        self.parallel_context.audit_synchronized_module(student_module, role="SiD student")
        self.parallel_context.audit_synchronized_module(fake_score_module, role="SiD fake-score")
        self.global_step = 0
        self.student_optimizer_steps = 0
        self.fake_score_optimizer_steps = 0
        self._phase = "idle"
        self._poisoned = False
        self.teacher_module.eval()

    @property
    def config_digest(self) -> str:
        return str(self.loss_adapter.config_digest)

    def _batches(
        self,
        batch: SIDTrainingBatch | Sequence[SIDTrainingBatch],
        *,
        role: str,
    ) -> tuple[SIDTrainingBatch, ...]:
        if isinstance(batch, SIDTrainingBatch):
            batches = (batch,)
        elif isinstance(batch, Sequence):
            batches = tuple(batch)
        else:
            raise TypeError(f"{role} batch must be SIDTrainingBatch or a sequence of microbatches")
        if len(batches) != self.gradient_accumulation_steps:
            raise ValueError(
                f"SiD {role} optimizer phase requires exactly "
                f"{self.gradient_accumulation_steps} microbatches; got {len(batches)}"
            )
        if not all(isinstance(value, SIDTrainingBatch) for value in batches):
            raise TypeError(f"every SiD {role} microbatch must be SIDTrainingBatch")
        return batches

    def _target_index(self, *, generator: torch.Generator | None) -> int:
        device = self.student_parameters[0].device
        value = torch.randint(
            int(self.loss_adapter.num_student_steps),
            (1,),
            device=device,
            generator=generator,
            dtype=torch.int64,
        )
        if self.parallel_context.world_size > 1:
            if self.parallel_context.rank != 0:
                value.zero_()
            dist.broadcast(value, src=0, group=self.parallel_context.process_group)
        return int(value.item())

    def train_step(
        self,
        fake_score_batch: SIDTrainingBatch | Sequence[SIDTrainingBatch],
        generator_batch: SIDTrainingBatch | Sequence[SIDTrainingBatch],
        *,
        generator: torch.Generator | None = None,
    ) -> SIDTrainResult:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("SiD engine has a partially committed iteration; restore the last checkpoint")
        fake_batches = self._batches(fake_score_batch, role="fake-score")
        generator_batches = self._batches(generator_batch, role="generator")
        target_index = self._target_index(generator=generator)
        fake_results: list[SIDLossResult] = []
        fake_weights: list[torch.Tensor] = []
        generator_results: list[SIDLossResult] = []
        generator_weights: list[torch.Tensor] = []
        self.student_optimizer.zero_grad(set_to_none=True)
        self.fake_score_optimizer.zero_grad(set_to_none=True)
        optimizer_mutated = False
        try:
            self._phase = "fake-score-backward"
            self.student_module.eval()
            self.teacher_module.eval()
            self.fake_score_module.train()
            fake_weights = [
                declared_loss_weight(
                    self.loss_adapter,
                    microbatch,
                    role="fake-score",
                    device=self.fake_score_parameters[0].device,
                )
                for microbatch in fake_batches
            ]
            total_fake_weight = global_denominator(fake_weights, self.parallel_context)
            for index, (microbatch, weight) in enumerate(zip(fake_batches, fake_weights, strict=True)):
                with accumulation_context(
                    self.fake_score_module,
                    final_microbatch=index + 1 == len(fake_batches),
                ):
                    result = _finite_loss(
                        self.loss_adapter.fake_score_loss(
                            microbatch,
                            target_index=target_index,
                            generator=generator,
                        ),
                        role="SiD fake-score",
                    )
                    check_reported_weight(result, weight, role="SiD fake-score")
                    gradient_weight = weight / total_fake_weight * float(self.parallel_context.world_size)
                    (result.loss * gradient_weight).backward()
                fake_results.append(result)
            if any(parameter.grad is not None for parameter in self.student_parameters):
                raise RuntimeError("SiD fake-score phase produced student parameter gradients")
            fake_grad_norm = clip_grad_norm_(
                self.fake_score_parameters,
                self.fake_score_max_grad_norm,
                error_if_nonfinite=True,
            )
            optimizer_mutated = True
            self.fake_score_optimizer.step()
            self.fake_score_optimizer_steps += 1
            if self.fake_score_scheduler is not None:
                self.fake_score_scheduler.step()
            self._phase = "fake-score-committed"

            self._phase = "generator-backward"
            self.student_module.train()
            self.teacher_module.eval()
            self.fake_score_module.eval()
            self.student_optimizer.zero_grad(set_to_none=True)
            self.fake_score_optimizer.zero_grad(set_to_none=True)
            generator_weights = [
                declared_loss_weight(
                    self.loss_adapter,
                    microbatch,
                    role="generator",
                    device=self.student_parameters[0].device,
                )
                for microbatch in generator_batches
            ]
            total_generator_weight = global_denominator(generator_weights, self.parallel_context)
            for index, (microbatch, weight) in enumerate(
                zip(generator_batches, generator_weights, strict=True)
            ):
                with accumulation_context(
                    self.student_module,
                    final_microbatch=index + 1 == len(generator_batches),
                ):
                    result = _finite_loss(
                        self.loss_adapter.generator_loss(
                            microbatch,
                            target_index=target_index,
                            generator=generator,
                        ),
                        role="SiD generator",
                    )
                    check_reported_weight(result, weight, role="SiD generator")
                    gradient_weight = (
                        weight / total_generator_weight * float(self.parallel_context.world_size)
                    )
                    (result.loss * gradient_weight).backward()
                generator_results.append(result)
            if any(parameter.grad is not None for parameter in self.fake_score_parameters):
                raise RuntimeError("SiD generator phase produced fake-score parameter gradients")
            student_grad_norm = clip_grad_norm_(
                self.student_parameters,
                self.student_max_grad_norm,
                error_if_nonfinite=True,
            )
            self.student_optimizer.step()
            self.student_optimizer_steps += 1
            if self.student_scheduler is not None:
                self.student_scheduler.step()
            if self.student_ema is not None:
                self.student_ema.update(self.student_module)
            self._phase = "generator-committed"
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

        fake_numerator, fake_denominator, fake_loss = global_loss_statistics(
            fake_results,
            fake_weights,
            self.parallel_context,
        )
        generator_numerator, generator_denominator, generator_loss = global_loss_statistics(
            generator_results,
            generator_weights,
            self.parallel_context,
        )
        metrics = {
            "global_step": torch.tensor(self.global_step, device=generator_loss.device, dtype=torch.int64),
            "target_index": target_index,
            "student_optimizer_steps": self.student_optimizer_steps,
            "fake_score_optimizer_steps": self.fake_score_optimizer_steps,
            "accumulated_microbatches_per_role": len(fake_batches),
            "student_grad_norm": student_grad_norm.detach(),
            "fake_score_grad_norm": fake_grad_norm.detach(),
            "generator": role_metrics(
                generator_results,
                global_numerator=generator_numerator,
                global_denominator=generator_denominator,
            ),
            "fake_score": role_metrics(
                fake_results,
                global_numerator=fake_numerator,
                global_denominator=fake_denominator,
            ),
        }
        return SIDTrainResult(
            fake_score_loss=fake_loss.detach().float(),
            generator_loss=generator_loss.detach().float(),
            target_index=target_index,
            metrics=metrics,
        )

    def state_dict(self) -> dict[str, object]:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("cannot checkpoint a partially committed SiD iteration")
        return {
            "schema": SID_ENGINE_STATE_SCHEMA,
            "global_step": self.global_step,
            "student_optimizer_steps": self.student_optimizer_steps,
            "fake_score_optimizer_steps": self.fake_score_optimizer_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "config_digest": self.config_digest,
            "data_parallel_size": self.parallel_context.world_size,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("SiD engine state must be a mapping")
        expected = {
            "schema",
            "global_step",
            "student_optimizer_steps",
            "fake_score_optimizer_steps",
            "gradient_accumulation_steps",
            "config_digest",
            "data_parallel_size",
        }
        if set(state_dict) != expected:
            raise ValueError("SiD engine state fields differ from the active schema")
        if state_dict["schema"] != SID_ENGINE_STATE_SCHEMA:
            raise ValueError(f"unsupported SiD engine schema: {state_dict['schema']!r}")
        if str(state_dict["config_digest"]) != self.config_digest:
            raise ValueError("saved SiD recipe differs from the active engine")
        for name, active in (
            ("gradient_accumulation_steps", self.gradient_accumulation_steps),
            ("data_parallel_size", self.parallel_context.world_size),
        ):
            if int(state_dict[name]) != active:
                raise ValueError(f"saved SiD {name} differs from the active engine")
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
            raise ValueError("saved SiD optimizer counters violate the update cadence")
        self.global_step = global_step
        self.student_optimizer_steps = student_steps
        self.fake_score_optimizer_steps = fake_steps
        self._phase = "idle"
        self._poisoned = False


__all__ = ["SID_ENGINE_STATE_SCHEMA", "SIDTrainResult", "NativeSIDTrainEngine"]
