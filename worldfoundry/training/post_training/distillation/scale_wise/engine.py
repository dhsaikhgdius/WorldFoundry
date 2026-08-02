"""Exact fake-updates-then-student state machine for scale-wise distillation."""

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
from .contracts import ScaleWiseLossAdapter, ScaleWiseTrainingBatch
from .objective import ScaleWiseLossResult

SCALE_WISE_ENGINE_STATE_SCHEMA = "worldfoundry-scale-wise-engine"


def _finite_loss(result: object, *, role: str) -> ScaleWiseLossResult:
    if not isinstance(result, ScaleWiseLossResult):
        raise TypeError(f"{role} loss adapter must return ScaleWiseLossResult")
    if not isinstance(result.loss, torch.Tensor) or result.loss.numel() != 1:
        raise TypeError(f"{role} loss must be one scalar tensor")
    if not bool(torch.isfinite(result.loss.detach()).all()):
        raise FloatingPointError(f"non-finite {role} loss")
    return result


def _microbatches(
    value: ScaleWiseTrainingBatch | Sequence[ScaleWiseTrainingBatch],
    *,
    expected: int,
    role: str,
) -> tuple[ScaleWiseTrainingBatch, ...]:
    if isinstance(value, ScaleWiseTrainingBatch):
        batches = (value,)
    elif isinstance(value, Sequence):
        batches = tuple(value)
    else:
        raise TypeError(f"{role} batch must be ScaleWiseTrainingBatch or a sequence")
    if len(batches) != expected:
        raise ValueError(
            f"scale-wise {role} requires exactly {expected} microbatches; "
            f"got {len(batches)}"
        )
    if not all(isinstance(batch, ScaleWiseTrainingBatch) for batch in batches):
        raise TypeError(f"every scale-wise {role} microbatch must be typed")
    return batches


@dataclass(frozen=True, slots=True)
class ScaleWiseTrainResult:
    student_loss: torch.Tensor
    fake_score_loss: torch.Tensor
    interval_index: int
    metrics: Mapping[str, object]


class NativeScaleWiseTrainEngine:
    """Own the SwD roles, update order, accumulation, and commit boundary."""

    def __init__(
        self,
        *,
        student_module: nn.Module,
        teacher_module: nn.Module,
        fake_score_module: nn.Module,
        loss_adapter: ScaleWiseLossAdapter,
        student_optimizer: torch.optim.Optimizer,
        fake_score_optimizer: torch.optim.Optimizer | None,
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
            raise TypeError("all scale-wise roles must be nn.Module values")
        if len({id(module) for module in modules}) != 3:
            raise ValueError("scale-wise student, teacher, and critic must be distinct modules")
        if not isinstance(loss_adapter, ScaleWiseLossAdapter):
            raise TypeError("loss_adapter must implement ScaleWiseLossAdapter")
        if any(parameter.requires_grad for parameter in teacher_module.parameters()):
            raise ValueError("scale-wise teacher parameters must be frozen")
        accumulation = non_negative_int(
            gradient_accumulation_steps,
            field_name="gradient_accumulation_steps",
        )
        if accumulation == 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if loss_adapter.num_intervals <= 0:
            raise ValueError("scale-wise loss adapter must expose schedule intervals")
        fake_updates = non_negative_int(
            loss_adapter.fake_updates_per_iteration,
            field_name="fake_updates_per_iteration",
        )
        if (fake_updates == 0) != (fake_score_optimizer is None):
            raise ValueError(
                "fake_score_optimizer must be present exactly when fake updates are enabled"
            )
        validate_stateful_or_none(student_scheduler, field_name="student_scheduler")
        validate_stateful_or_none(fake_score_scheduler, field_name="fake_score_scheduler")
        validate_stateful_or_none(student_ema, field_name="student_ema")
        if fake_updates == 0 and fake_score_scheduler is not None:
            raise ValueError("fake_score_scheduler requires fake-score updates")

        student_parameters = trainable_parameters(student_module)
        fake_parameters = (
            trainable_parameters(fake_score_module) if fake_updates else ()
        )
        if not fake_updates and any(
            parameter.requires_grad for parameter in fake_score_module.parameters()
        ):
            raise ValueError(
                "MMD-only scale-wise feature extractor must be frozen"
            )
        if {id(value) for value in student_parameters} & {
            id(value) for value in fake_parameters
        }:
            raise ValueError("scale-wise student and critic cannot share trainable parameters")
        audit_optimizer_parameters(
            student_optimizer,
            student_parameters,
            role="scale-wise student",
        )
        if fake_score_optimizer is not None:
            audit_optimizer_parameters(
                fake_score_optimizer,
                fake_parameters,
                role="scale-wise fake score",
            )

        self.student_module = student_module
        self.teacher_module = teacher_module
        self.fake_score_module = fake_score_module
        self.loss_adapter = loss_adapter
        self.student_optimizer = student_optimizer
        self.fake_score_optimizer = fake_score_optimizer
        self.student_parameters = student_parameters
        self.fake_score_parameters = tuple(fake_parameters)
        self.student_max_grad_norm = positive_float(
            student_max_grad_norm,
            field_name="student_max_grad_norm",
        )
        self.fake_score_max_grad_norm = positive_float(
            fake_score_max_grad_norm,
            field_name="fake_score_max_grad_norm",
        )
        self.gradient_accumulation_steps = accumulation
        self.fake_updates_per_iteration = fake_updates
        self.student_scheduler = student_scheduler
        self.fake_score_scheduler = fake_score_scheduler
        self.student_ema = student_ema
        self.parallel_context = parallel_context or PostTrainingParallelContext.current()
        if self.loss_adapter.batch_mmd and self.parallel_context.world_size > 1:
            raise ValueError(
                "batch-coupled MMD requires a global differentiable feature gather"
            )
        self.parallel_context.audit_synchronized_module(
            student_module,
            role="scale-wise student",
        )
        if fake_updates:
            self.parallel_context.audit_synchronized_module(
                fake_score_module,
                role="scale-wise fake score",
            )
        self.global_step = 0
        self.student_optimizer_steps = 0
        self.fake_score_optimizer_steps = 0
        self._phase = "idle"
        self._poisoned = False
        self.teacher_module.eval()

    @property
    def config_digest(self) -> str:
        return str(self.loss_adapter.config_digest)

    @property
    def interval_index(self) -> int:
        return self.global_step % int(self.loss_adapter.num_intervals)

    def _validate_interval(
        self,
        batches: Sequence[ScaleWiseTrainingBatch],
        *,
        role: str,
    ) -> None:
        expected = self.interval_index
        if any(batch.interval_index != expected for batch in batches):
            raise ValueError(
                f"scale-wise {role} batches must use interval {expected} at step "
                f"{self.global_step}"
            )

    def _fake_groups(
        self,
        value: Sequence[ScaleWiseTrainingBatch] | None,
    ) -> tuple[tuple[ScaleWiseTrainingBatch, ...], ...]:
        if not self.fake_updates_per_iteration:
            if value is not None and len(value) != 0:
                raise ValueError("fake batches were supplied while fake updates are disabled")
            return ()
        if value is None or not isinstance(value, Sequence):
            raise TypeError("fresh fake batches must be supplied as one flat sequence")
        flat = tuple(value)
        expected = self.fake_updates_per_iteration * self.gradient_accumulation_steps
        if len(flat) != expected:
            raise ValueError(
                f"scale-wise training requires {expected} fresh fake microbatches; "
                f"got {len(flat)}"
            )
        if not all(isinstance(batch, ScaleWiseTrainingBatch) for batch in flat):
            raise TypeError("every fresh fake microbatch must be ScaleWiseTrainingBatch")
        return tuple(
            flat[offset : offset + self.gradient_accumulation_steps]
            for offset in range(0, expected, self.gradient_accumulation_steps)
        )

    def train_step(
        self,
        student_batch: ScaleWiseTrainingBatch | Sequence[ScaleWiseTrainingBatch],
        *,
        fake_score_batches: Sequence[ScaleWiseTrainingBatch] | None = None,
        generator: torch.Generator | None = None,
    ) -> ScaleWiseTrainResult:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError(
                "scale-wise engine has a partially committed iteration; restore checkpoint"
            )
        student_batches = _microbatches(
            student_batch,
            expected=self.gradient_accumulation_steps,
            role="student",
        )
        fake_groups = self._fake_groups(fake_score_batches)
        self._validate_interval(student_batches, role="student")
        for group in fake_groups:
            self._validate_interval(group, role="fake-score")

        student_results: list[ScaleWiseLossResult] = []
        student_weights: list[torch.Tensor] = []
        fake_results: list[ScaleWiseLossResult] = []
        fake_weights: list[torch.Tensor] = []
        fake_grad_norm = torch.zeros(
            (),
            device=self.student_parameters[0].device,
            dtype=torch.float32,
        )
        optimizer_mutated = False
        self.student_optimizer.zero_grad(set_to_none=True)
        if self.fake_score_optimizer is not None:
            self.fake_score_optimizer.zero_grad(set_to_none=True)
        try:
            self.student_module.eval()
            self.teacher_module.eval()
            self.fake_score_module.train(bool(fake_groups))
            for update_index, group in enumerate(fake_groups):
                assert self.fake_score_optimizer is not None
                self._phase = f"fake-score-{update_index}-backward"
                update_weights = [
                    declared_loss_weight(
                        self.loss_adapter,
                        batch,
                        role="fake-score",
                        device=self.fake_score_parameters[0].device,
                    )
                    for batch in group
                ]
                total_weight = global_denominator(
                    update_weights,
                    self.parallel_context,
                )
                update_results: list[ScaleWiseLossResult] = []
                for index, (batch, weight) in enumerate(
                    zip(group, update_weights, strict=True)
                ):
                    with accumulation_context(
                        self.fake_score_module,
                        final_microbatch=index + 1 == len(group),
                    ):
                        result = _finite_loss(
                            self.loss_adapter.fake_score_loss(
                                batch,
                                generator=generator,
                            ),
                            role="scale-wise fake score",
                        )
                        check_reported_weight(
                            result,
                            weight,
                            role="scale-wise fake score",
                        )
                        gradient_weight = (
                            weight
                            / total_weight
                            * float(self.parallel_context.world_size)
                        )
                        (result.loss * gradient_weight).backward()
                    update_results.append(result)
                if any(parameter.grad is not None for parameter in self.student_parameters):
                    raise RuntimeError(
                        "scale-wise fake-score phase produced student gradients"
                    )
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
                self.fake_score_optimizer.zero_grad(set_to_none=True)
                fake_results.extend(update_results)
                fake_weights.extend(update_weights)
                self._phase = f"fake-score-{update_index}-committed"

            self._phase = "student-backward"
            self.student_module.train()
            self.teacher_module.eval()
            self.fake_score_module.eval()
            for parameter in self.fake_score_parameters:
                parameter.requires_grad_(False)
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
                with accumulation_context(
                    self.student_module,
                    final_microbatch=index + 1 == len(student_batches),
                ):
                    result = _finite_loss(
                        self.loss_adapter.student_loss(
                            batch,
                            generator=generator,
                        ),
                        role="scale-wise student",
                    )
                    check_reported_weight(
                        result,
                        weight,
                        role="scale-wise student",
                    )
                    gradient_weight = (
                        weight
                        / total_student_weight
                        * float(self.parallel_context.world_size)
                    )
                    (result.loss * gradient_weight).backward()
                student_results.append(result)
            if any(parameter.grad is not None for parameter in self.fake_score_parameters):
                raise RuntimeError("scale-wise student phase produced critic gradients")
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
            self.global_step += 1
            self._phase = "idle"
        except Exception:
            self.student_optimizer.zero_grad(set_to_none=True)
            if self.fake_score_optimizer is not None:
                self.fake_score_optimizer.zero_grad(set_to_none=True)
            if optimizer_mutated:
                self._poisoned = True
            else:
                self._phase = "idle"
            raise
        finally:
            for parameter in self.fake_score_parameters:
                parameter.requires_grad_(True)

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
        if fake_results:
            fake_numerator, fake_denominator, fake_loss = global_loss_statistics(
                fake_results,
                fake_weights,
                self.parallel_context,
            )
            fake_metrics: Mapping[str, object] = role_metrics(
                fake_results,
                global_numerator=fake_numerator,
                global_denominator=fake_denominator,
            )
        else:
            fake_loss = torch.zeros_like(student_loss)
            fake_metrics = {}
        completed_interval = (self.global_step - 1) % int(
            self.loss_adapter.num_intervals
        )
        metrics: dict[str, object] = {
            "global_step": self.global_step,
            "interval_index": completed_interval,
            "student_optimizer_steps": self.student_optimizer_steps,
            "fake_score_optimizer_steps": self.fake_score_optimizer_steps,
            "student_grad_norm": student_grad_norm.detach(),
            "fake_score_grad_norm": fake_grad_norm.detach(),
            "student_microbatches": len(student_batches),
            "fake_score_microbatches": sum(len(group) for group in fake_groups),
            "fake_score_updates": len(fake_groups),
            "student": student_metrics,
            "fake_score": fake_metrics,
        }
        return ScaleWiseTrainResult(
            student_loss=student_loss.detach().float(),
            fake_score_loss=fake_loss.detach().float(),
            interval_index=completed_interval,
            metrics=metrics,
        )

    def state_dict(self) -> dict[str, object]:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("cannot checkpoint a partially committed scale-wise iteration")
        return {
            "schema": SCALE_WISE_ENGINE_STATE_SCHEMA,
            "global_step": self.global_step,
            "student_optimizer_steps": self.student_optimizer_steps,
            "fake_score_optimizer_steps": self.fake_score_optimizer_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "fake_updates_per_iteration": self.fake_updates_per_iteration,
            "config_digest": self.config_digest,
            "data_parallel_size": self.parallel_context.world_size,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("scale-wise engine state must be a mapping")
        expected = {
            "schema",
            "global_step",
            "student_optimizer_steps",
            "fake_score_optimizer_steps",
            "gradient_accumulation_steps",
            "fake_updates_per_iteration",
            "config_digest",
            "data_parallel_size",
        }
        if set(state_dict) != expected:
            raise ValueError("scale-wise engine state fields differ from active schema")
        if state_dict["schema"] != SCALE_WISE_ENGINE_STATE_SCHEMA:
            raise ValueError("unsupported scale-wise engine schema")
        if str(state_dict["config_digest"]) != self.config_digest:
            raise ValueError("saved scale-wise configuration differs")
        if int(state_dict["gradient_accumulation_steps"]) != self.gradient_accumulation_steps:
            raise ValueError("saved scale-wise accumulation differs")
        if int(state_dict["fake_updates_per_iteration"]) != self.fake_updates_per_iteration:
            raise ValueError("saved scale-wise fake update cadence differs")
        if int(state_dict["data_parallel_size"]) != self.parallel_context.world_size:
            raise ValueError("saved scale-wise data-parallel size differs")
        global_step = non_negative_int(state_dict["global_step"], field_name="global_step")
        student_steps = non_negative_int(
            state_dict["student_optimizer_steps"],
            field_name="student_optimizer_steps",
        )
        fake_steps = non_negative_int(
            state_dict["fake_score_optimizer_steps"],
            field_name="fake_score_optimizer_steps",
        )
        if student_steps != global_step or fake_steps != (
            global_step * self.fake_updates_per_iteration
        ):
            raise ValueError("saved scale-wise counters violate configured cadence")
        self.global_step = global_step
        self.student_optimizer_steps = student_steps
        self.fake_score_optimizer_steps = fake_steps
        self._phase = "idle"
        self._poisoned = False
        self.student_optimizer.zero_grad(set_to_none=True)
        if self.fake_score_optimizer is not None:
            self.fake_score_optimizer.zero_grad(set_to_none=True)


__all__ = [
    "NativeScaleWiseTrainEngine",
    "SCALE_WISE_ENGINE_STATE_SCHEMA",
    "ScaleWiseTrainResult",
]
