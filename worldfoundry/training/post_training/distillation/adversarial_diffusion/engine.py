"""Atomic, data-parallel ADD generator/discriminator optimizer engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from .contracts import ADDLossAdapter, ADDLossResult, ADDTrainingBatch

ADD_ENGINE_STATE_SCHEMA = "worldfoundry-adversarial-diffusion-engine"


@contextmanager
def _frozen_parameters(module: nn.Module):
    parameters = tuple(module.parameters())
    states = tuple(parameter.requires_grad for parameter in parameters)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, state in zip(parameters, states, strict=True):
            parameter.requires_grad_(state)


def _finite_loss(result: object, *, role: str) -> ADDLossResult:
    if not isinstance(result, ADDLossResult):
        raise TypeError(f"{role} loss adapter must return ADDLossResult")
    if not isinstance(result.loss, torch.Tensor) or result.loss.numel() != 1:
        raise TypeError(f"{role} loss must be one scalar tensor")
    if not bool(torch.isfinite(result.loss.detach()).all()):
        raise FloatingPointError(f"non-finite {role} loss")
    return result


def _microbatches(
    value: ADDTrainingBatch | Sequence[ADDTrainingBatch],
    *,
    expected: int,
    role: str,
) -> tuple[ADDTrainingBatch, ...]:
    if isinstance(value, ADDTrainingBatch):
        batches = (value,)
    elif isinstance(value, Sequence):
        batches = tuple(value)
    else:
        raise TypeError(f"{role} batch must be ADDTrainingBatch or a sequence")
    if len(batches) != expected:
        raise ValueError(f"ADD {role} update requires exactly {expected} microbatches; got {len(batches)}")
    if not all(isinstance(batch, ADDTrainingBatch) for batch in batches):
        raise TypeError(f"every ADD {role} microbatch must be ADDTrainingBatch")
    return batches


@dataclass(frozen=True, slots=True)
class ADDTrainResult:
    generator_loss: torch.Tensor
    discriminator_loss: torch.Tensor
    metrics: Mapping[str, object]


class NativeADDTrainEngine:
    """Commit configured discriminator updates followed by one student update."""

    def __init__(
        self,
        *,
        student_module: nn.Module,
        teacher_module: nn.Module,
        decoder_module: nn.Module,
        discriminator_module: nn.Module,
        discriminator_feature_module: nn.Module,
        loss_adapter: ADDLossAdapter,
        student_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
        discriminator_updates_per_generator: int = 1,
        student_max_grad_norm: float = 1.0,
        discriminator_max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        student_scheduler: object | None = None,
        discriminator_scheduler: object | None = None,
        parallel_context: PostTrainingParallelContext | None = None,
    ) -> None:
        modules = (
            student_module,
            teacher_module,
            decoder_module,
            discriminator_module,
            discriminator_feature_module,
        )
        if not all(isinstance(module, nn.Module) for module in modules):
            raise TypeError("all ADD roles must be nn.Module values")
        if len({id(module) for module in modules}) != len(modules):
            raise ValueError("ADD student, teacher, decoder, discriminator, and feature modules must be distinct")
        if not isinstance(loss_adapter, ADDLossAdapter):
            raise TypeError("loss_adapter must implement ADDLossAdapter")
        for role, module in (
            ("teacher", teacher_module),
            ("decoder", decoder_module),
            ("feature network", discriminator_feature_module),
        ):
            if any(parameter.requires_grad for parameter in module.parameters()):
                raise ValueError(f"ADD {role} parameters must be frozen")
            if any(parameter.grad is not None for parameter in module.parameters()):
                raise ValueError(f"ADD {role} cannot carry parameter gradients")
            module.eval()
        accumulation = non_negative_int(
            gradient_accumulation_steps,
            field_name="gradient_accumulation_steps",
        )
        discriminator_updates = non_negative_int(
            discriminator_updates_per_generator,
            field_name="discriminator_updates_per_generator",
        )
        if accumulation == 0 or discriminator_updates == 0:
            raise ValueError("ADD accumulation and discriminator update counts must be positive")
        validate_stateful_or_none(student_scheduler, field_name="student_scheduler")
        validate_stateful_or_none(discriminator_scheduler, field_name="discriminator_scheduler")
        student_parameters = trainable_parameters(student_module)
        discriminator_parameters = trainable_parameters(discriminator_module)
        role_inventories = {
            "student": {id(parameter) for parameter in student_module.parameters()},
            "teacher": {id(parameter) for parameter in teacher_module.parameters()},
            "decoder": {id(parameter) for parameter in decoder_module.parameters()},
            "feature network": {id(parameter) for parameter in discriminator_feature_module.parameters()},
            "discriminator heads": {id(parameter) for parameter in discriminator_parameters},
        }
        role_names = tuple(role_inventories)
        for index, left in enumerate(role_names):
            for right in role_names[index + 1 :]:
                if role_inventories[left] & role_inventories[right]:
                    raise ValueError(f"ADD roles {left!r} and {right!r} cannot share parameters")
        audit_optimizer_parameters(student_optimizer, student_parameters, role="ADD student")
        audit_optimizer_parameters(
            discriminator_optimizer,
            discriminator_parameters,
            role="ADD discriminator",
        )
        self.student_module = student_module
        self.teacher_module = teacher_module
        self.decoder_module = decoder_module
        self.discriminator_module = discriminator_module
        self.discriminator_feature_module = discriminator_feature_module
        self.loss_adapter = loss_adapter
        self.student_optimizer = student_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.student_parameters = student_parameters
        self.discriminator_parameters = discriminator_parameters
        self.discriminator_updates_per_generator = discriminator_updates
        self.student_max_grad_norm = positive_float(
            student_max_grad_norm,
            field_name="student_max_grad_norm",
        )
        self.discriminator_max_grad_norm = positive_float(
            discriminator_max_grad_norm,
            field_name="discriminator_max_grad_norm",
        )
        self.gradient_accumulation_steps = accumulation
        self.student_scheduler = student_scheduler
        self.discriminator_scheduler = discriminator_scheduler
        self.parallel_context = parallel_context or PostTrainingParallelContext.current()
        self.parallel_context.audit_synchronized_module(student_module, role="ADD student")
        self.parallel_context.audit_synchronized_module(discriminator_module, role="ADD discriminator")
        self.global_step = 0
        self.student_optimizer_steps = 0
        self.discriminator_optimizer_steps = 0
        self._phase = "idle"
        self._poisoned = False

    def _backward_role(
        self,
        batches: tuple[ADDTrainingBatch, ...],
        *,
        role: str,
        module: nn.Module,
        parameters: tuple[nn.Parameter, ...],
        generator: torch.Generator | None,
    ) -> tuple[list[ADDLossResult], list[torch.Tensor]]:
        weights = [
            declared_loss_weight(
                self.loss_adapter,
                batch,
                role=role,
                device=parameters[0].device,
            )
            for batch in batches
        ]
        total_weight = global_denominator(weights, self.parallel_context)
        results: list[ADDLossResult] = []
        loss_method = self.loss_adapter.generator_loss if role == "generator" else self.loss_adapter.discriminator_loss
        for index, (batch, weight) in enumerate(zip(batches, weights, strict=True)):
            with accumulation_context(module, final_microbatch=index + 1 == len(batches)):
                result = _finite_loss(
                    loss_method(batch, generator=generator),
                    role=f"ADD {role}",
                )
                check_reported_weight(result, weight, role=f"ADD {role}")
                gradient_weight = weight / total_weight * float(self.parallel_context.world_size)
                (result.loss * gradient_weight).backward()
            results.append(result)
        return results, weights

    def train_step(
        self,
        batch: ADDTrainingBatch | Sequence[ADDTrainingBatch],
        *,
        discriminator_batch: ADDTrainingBatch | Sequence[ADDTrainingBatch] | None = None,
        generator: torch.Generator | None = None,
    ) -> ADDTrainResult:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("ADD engine has a partially committed iteration; restore the last checkpoint")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator or None")
        generator_batches = _microbatches(
            batch,
            expected=self.gradient_accumulation_steps,
            role="generator",
        )
        discriminator_expected = self.discriminator_updates_per_generator * self.gradient_accumulation_steps
        discriminator_batches = (
            generator_batches * self.discriminator_updates_per_generator
            if discriminator_batch is None
            else _microbatches(
                discriminator_batch,
                expected=discriminator_expected,
                role="discriminator",
            )
        )
        discriminator_groups = tuple(
            discriminator_batches[offset : offset + self.gradient_accumulation_steps]
            for offset in range(0, discriminator_expected, self.gradient_accumulation_steps)
        )
        self.student_optimizer.zero_grad(set_to_none=True)
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        optimizer_mutated = False
        discriminator_losses: list[torch.Tensor] = []
        discriminator_metrics: list[Mapping[str, object]] = []
        try:
            for update_index, discriminator_group in enumerate(discriminator_groups):
                self._phase = f"discriminator-{update_index}-backward"
                self.student_module.eval()
                self.discriminator_module.train()
                self.teacher_module.eval()
                self.decoder_module.eval()
                self.discriminator_feature_module.eval()
                self.student_optimizer.zero_grad(set_to_none=True)
                self.discriminator_optimizer.zero_grad(set_to_none=True)
                results, weights = self._backward_role(
                    discriminator_group,
                    role="discriminator",
                    module=self.discriminator_module,
                    parameters=self.discriminator_parameters,
                    generator=generator,
                )
                if any(parameter.grad is not None for parameter in self.student_parameters):
                    raise RuntimeError("ADD discriminator phase produced student parameter gradients")
                discriminator_grad_norm = clip_grad_norm_(
                    self.discriminator_parameters,
                    self.discriminator_max_grad_norm,
                    error_if_nonfinite=True,
                )
                optimizer_mutated = True
                self.discriminator_optimizer.step()
                self.discriminator_optimizer_steps += 1
                self._phase = f"discriminator-{update_index}-committed"
                if self.discriminator_scheduler is not None:
                    self.discriminator_scheduler.step()
                numerator, denominator, loss = global_loss_statistics(
                    results,
                    weights,
                    self.parallel_context,
                )
                discriminator_losses.append(loss)
                discriminator_metrics.append(
                    role_metrics(
                        results,
                        global_numerator=numerator,
                        global_denominator=denominator,
                    )
                )
            self.discriminator_optimizer.zero_grad(set_to_none=True)

            self._phase = "generator-backward"
            self.student_module.train()
            self.discriminator_module.eval()
            self.teacher_module.eval()
            self.decoder_module.eval()
            self.discriminator_feature_module.eval()
            with _frozen_parameters(self.discriminator_module):
                generator_results, generator_weights = self._backward_role(
                    generator_batches,
                    role="generator",
                    module=self.student_module,
                    parameters=self.student_parameters,
                    generator=generator,
                )
            if any(parameter.grad is not None for parameter in self.discriminator_parameters):
                raise RuntimeError("ADD generator phase produced discriminator parameter gradients")
            for role, module in (
                ("teacher", self.teacher_module),
                ("decoder", self.decoder_module),
                ("feature network", self.discriminator_feature_module),
            ):
                if any(parameter.grad is not None for parameter in module.parameters()):
                    raise RuntimeError(f"ADD frozen {role} received parameter gradients")
            student_grad_norm = clip_grad_norm_(
                self.student_parameters,
                self.student_max_grad_norm,
                error_if_nonfinite=True,
            )
            self.student_optimizer.step()
            self.student_optimizer_steps += 1
            self._phase = "generator-committed"
            if self.student_scheduler is not None:
                self.student_scheduler.step()
            generator_numerator, generator_denominator, generator_loss = global_loss_statistics(
                generator_results,
                generator_weights,
                self.parallel_context,
            )
            generator_metrics = role_metrics(
                generator_results,
                global_numerator=generator_numerator,
                global_denominator=generator_denominator,
            )
            self.student_optimizer.zero_grad(set_to_none=True)
            self.discriminator_optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            self._phase = "idle"
        except Exception:
            self.student_optimizer.zero_grad(set_to_none=True)
            self.discriminator_optimizer.zero_grad(set_to_none=True)
            if optimizer_mutated:
                self._poisoned = True
            else:
                self._phase = "idle"
            raise
        discriminator_loss = torch.stack(discriminator_losses).mean()
        metrics: Mapping[str, object] = {
            "global_step": torch.tensor(
                self.global_step,
                device=generator_loss.device,
                dtype=torch.int64,
            ),
            "student_optimizer_steps": torch.tensor(
                self.student_optimizer_steps,
                device=generator_loss.device,
                dtype=torch.int64,
            ),
            "discriminator_optimizer_steps": torch.tensor(
                self.discriminator_optimizer_steps,
                device=generator_loss.device,
                dtype=torch.int64,
            ),
            "accumulated_microbatches": len(generator_batches),
            "discriminator_updates": len(discriminator_groups),
            "student_grad_norm": student_grad_norm.detach(),
            "discriminator_grad_norm": discriminator_grad_norm.detach(),
            "generator": generator_metrics,
            "discriminator": tuple(discriminator_metrics),
        }
        return ADDTrainResult(
            generator_loss=generator_loss.detach().float(),
            discriminator_loss=discriminator_loss.detach().float(),
            metrics=metrics,
        )

    def state_dict(self) -> dict[str, object]:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("cannot checkpoint a partially committed ADD iteration")
        return {
            "schema": ADD_ENGINE_STATE_SCHEMA,
            "global_step": self.global_step,
            "student_optimizer_steps": self.student_optimizer_steps,
            "discriminator_optimizer_steps": self.discriminator_optimizer_steps,
            "discriminator_updates_per_generator": self.discriminator_updates_per_generator,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "student_max_grad_norm": self.student_max_grad_norm,
            "discriminator_max_grad_norm": self.discriminator_max_grad_norm,
            "student_scheduler_enabled": self.student_scheduler is not None,
            "discriminator_scheduler_enabled": self.discriminator_scheduler is not None,
            "data_parallel_size": self.parallel_context.world_size,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("ADD engine state must be a mapping")
        expected = {
            "schema",
            "global_step",
            "student_optimizer_steps",
            "discriminator_optimizer_steps",
            "discriminator_updates_per_generator",
            "gradient_accumulation_steps",
            "student_max_grad_norm",
            "discriminator_max_grad_norm",
            "student_scheduler_enabled",
            "discriminator_scheduler_enabled",
            "data_parallel_size",
        }
        if set(state_dict) != expected:
            raise ValueError("ADD engine state fields differ from the active schema")
        if state_dict["schema"] != ADD_ENGINE_STATE_SCHEMA:
            raise ValueError(f"unsupported ADD engine schema: {state_dict['schema']!r}")
        for name, active in (
            ("discriminator_updates_per_generator", self.discriminator_updates_per_generator),
            ("gradient_accumulation_steps", self.gradient_accumulation_steps),
            ("data_parallel_size", self.parallel_context.world_size),
        ):
            if int(state_dict[name]) != active:
                raise ValueError(f"saved ADD {name} differs from the active engine")
        for name, active in (
            ("student_max_grad_norm", self.student_max_grad_norm),
            ("discriminator_max_grad_norm", self.discriminator_max_grad_norm),
        ):
            if float(state_dict[name]) != active:
                raise ValueError(f"saved ADD {name} differs from the active engine")
        for name, active in (
            ("student_scheduler_enabled", self.student_scheduler is not None),
            ("discriminator_scheduler_enabled", self.discriminator_scheduler is not None),
        ):
            if not isinstance(state_dict[name], bool) or state_dict[name] is not active:
                raise ValueError(f"saved ADD {name} differs from the active engine")
        global_step = non_negative_int(state_dict["global_step"], field_name="global_step")
        student_steps = non_negative_int(
            state_dict["student_optimizer_steps"],
            field_name="student_optimizer_steps",
        )
        discriminator_steps = non_negative_int(
            state_dict["discriminator_optimizer_steps"],
            field_name="discriminator_optimizer_steps",
        )
        if student_steps != global_step:
            raise ValueError("saved ADD student counter violates the update cadence")
        if discriminator_steps != global_step * self.discriminator_updates_per_generator:
            raise ValueError("saved ADD discriminator counter violates the update cadence")
        self.global_step = global_step
        self.student_optimizer_steps = student_steps
        self.discriminator_optimizer_steps = discriminator_steps
        self._phase = "idle"
        self._poisoned = False
        self.student_optimizer.zero_grad(set_to_none=True)
        self.discriminator_optimizer.zero_grad(set_to_none=True)


__all__ = ["ADD_ENGINE_STATE_SCHEMA", "ADDTrainResult", "NativeADDTrainEngine"]
