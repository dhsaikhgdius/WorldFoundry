"""Three-optimizer SenseFlow state machine with immediate post-generator IDA."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

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
from .contracts import (
    SenseFlowGeneratorPhase,
    SenseFlowLossAdapter,
    SenseFlowLossResult,
    SenseFlowPreparedBatch,
    SenseFlowTrainingBatch,
)
from .math import audit_ida_alignment, implicit_distribution_alignment_

SENSEFLOW_ENGINE_STATE_SCHEMA = "worldfoundry-senseflow-engine"


def _finite_loss(result: object, *, role: str) -> SenseFlowLossResult:
    if not isinstance(result, SenseFlowLossResult):
        raise TypeError(f"{role} loss adapter must return SenseFlowLossResult")
    if not isinstance(result.loss, Tensor) or result.loss.numel() != 1:
        raise TypeError(f"{role} loss must be one scalar tensor")
    if not bool(torch.isfinite(result.loss.detach()).all()):
        raise FloatingPointError(f"non-finite {role} loss")
    return result


def _microbatches(
    value: SenseFlowTrainingBatch | Sequence[SenseFlowTrainingBatch],
    *,
    expected: int,
) -> tuple[SenseFlowTrainingBatch, ...]:
    if isinstance(value, SenseFlowTrainingBatch):
        batches = (value,)
    elif isinstance(value, Sequence):
        batches = tuple(value)
    else:
        raise TypeError("SenseFlow batch must be a batch or sequence of microbatches")
    if len(batches) != expected:
        raise ValueError(
            "SenseFlow optimizer iteration requires exactly "
            f"{expected} microbatches; got {len(batches)}"
        )
    if not all(isinstance(batch, SenseFlowTrainingBatch) for batch in batches):
        raise TypeError("every SenseFlow microbatch must be SenseFlowTrainingBatch")
    return batches


def _module_owned_by_adapter(module: nn.Module, adapter: object, *, role: str) -> None:
    if getattr(adapter, "module", None) is not module:
        raise ValueError(f"SenseFlow engine and loss adapter must share the {role} module")


@dataclass(frozen=True, slots=True)
class SenseFlowTrainResult:
    generator_loss: Tensor
    fake_score_loss: Tensor
    discriminator_loss: Tensor
    generator_updated: bool
    metrics: Mapping[str, object]


class NativeSenseFlowTrainEngine:
    """Commit G, apply IDA, then update fake score and discriminator every iteration."""

    def __init__(
        self,
        *,
        student_module: nn.Module,
        teacher_module: nn.Module,
        fake_score_module: nn.Module,
        discriminator_module: nn.Module,
        loss_adapter: SenseFlowLossAdapter,
        student_optimizer: torch.optim.Optimizer,
        fake_score_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
        student_max_grad_norm: float = 1.0,
        fake_score_max_grad_norm: float = 1.0,
        discriminator_max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        seed: int = 71801,
        student_scheduler: object | None = None,
        fake_score_scheduler: object | None = None,
        discriminator_scheduler: object | None = None,
        student_scheduler_cadence: str = "iteration",
        discriminator_frozen_modules: Sequence[nn.Module] = (),
        parallel_context: PostTrainingParallelContext | None = None,
    ) -> None:
        modules = (
            student_module,
            teacher_module,
            fake_score_module,
            discriminator_module,
        )
        if not all(isinstance(module, nn.Module) for module in modules):
            raise TypeError("all SenseFlow roles must be nn.Module values")
        if len({id(module) for module in modules}) != len(modules):
            raise ValueError("SenseFlow roles must be distinct modules")
        if not isinstance(loss_adapter, SenseFlowLossAdapter):
            raise TypeError("loss_adapter must implement SenseFlowLossAdapter")
        _module_owned_by_adapter(student_module, loss_adapter.student, role="student")
        _module_owned_by_adapter(teacher_module, loss_adapter.teacher, role="teacher")
        _module_owned_by_adapter(fake_score_module, loss_adapter.fake_score, role="fake score")
        _module_owned_by_adapter(
            discriminator_module,
            loss_adapter.discriminator,
            role="discriminator",
        )
        if any(parameter.requires_grad for parameter in teacher_module.parameters()):
            raise ValueError("SenseFlow teacher parameters must be frozen")
        accumulation = non_negative_int(
            gradient_accumulation_steps,
            field_name="gradient_accumulation_steps",
        )
        if accumulation == 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if (
            isinstance(loss_adapter.generator_update_interval, bool)
            or int(loss_adapter.generator_update_interval) <= 0
        ):
            raise ValueError("SenseFlow generator update interval must be positive")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("SenseFlow seed must be a non-negative integer")
        cadence = str(student_scheduler_cadence).strip().lower().replace("_", "-")
        if cadence not in {"iteration", "generator-update"}:
            raise ValueError("student_scheduler_cadence must be 'iteration' or 'generator-update'")
        for name, scheduler in (
            ("student_scheduler", student_scheduler),
            ("fake_score_scheduler", fake_score_scheduler),
            ("discriminator_scheduler", discriminator_scheduler),
        ):
            validate_stateful_or_none(scheduler, field_name=name)
        frozen_discriminator_modules = tuple(discriminator_frozen_modules)
        discriminator_descendants = {id(module) for module in discriminator_module.modules()}
        if len({id(module) for module in frozen_discriminator_modules}) != len(
            frozen_discriminator_modules
        ):
            raise ValueError("SenseFlow discriminator frozen modules cannot contain duplicates")
        for module in frozen_discriminator_modules:
            if not isinstance(module, nn.Module) or id(module) not in discriminator_descendants:
                raise ValueError(
                    "SenseFlow frozen feature modules must belong to the discriminator"
                )
            if any(parameter.requires_grad for parameter in module.parameters()):
                raise ValueError("SenseFlow discriminator feature modules must be frozen")

        student_parameters = trainable_parameters(student_module)
        fake_parameters = trainable_parameters(fake_score_module)
        discriminator_parameters = trainable_parameters(discriminator_module)
        inventories = tuple(
            {id(parameter) for parameter in parameters}
            for parameters in (student_parameters, fake_parameters, discriminator_parameters)
        )
        if any(
            inventories[left] & inventories[right]
            for left in range(len(inventories))
            for right in range(left + 1, len(inventories))
        ):
            raise ValueError("SenseFlow trainable roles cannot share parameters")
        audit_optimizer_parameters(student_optimizer, student_parameters, role="SenseFlow student")
        audit_optimizer_parameters(
            fake_score_optimizer,
            fake_parameters,
            role="SenseFlow fake score",
        )
        audit_optimizer_parameters(
            discriminator_optimizer,
            discriminator_parameters,
            role="SenseFlow discriminator",
        )
        if loss_adapter.ida_enabled:
            audit_ida_alignment(student_module, fake_score_module)

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
        self.generator_update_interval = int(loss_adapter.generator_update_interval)
        self.ida_decay = float(loss_adapter.ida_decay)
        self.ida_enabled = bool(loss_adapter.ida_enabled)
        self.student_max_grad_norm = positive_float(
            student_max_grad_norm,
            field_name="student_max_grad_norm",
        )
        self.fake_score_max_grad_norm = positive_float(
            fake_score_max_grad_norm,
            field_name="fake_score_max_grad_norm",
        )
        self.discriminator_max_grad_norm = positive_float(
            discriminator_max_grad_norm,
            field_name="discriminator_max_grad_norm",
        )
        self.gradient_accumulation_steps = accumulation
        self.student_scheduler = student_scheduler
        self.fake_score_scheduler = fake_score_scheduler
        self.discriminator_scheduler = discriminator_scheduler
        self.student_scheduler_cadence = cadence
        self.discriminator_frozen_modules = frozen_discriminator_modules
        self.parallel_context = parallel_context or PostTrainingParallelContext.current()
        self.parallel_context.audit_synchronized_module(student_module, role="SenseFlow student")
        self.parallel_context.audit_synchronized_module(fake_score_module, role="SenseFlow fake score")
        self.parallel_context.audit_synchronized_module(
            discriminator_module,
            role="SenseFlow discriminator",
        )
        device = student_parameters[0].device
        if any(parameter.device != device for parameter in fake_parameters + discriminator_parameters):
            raise ValueError("SenseFlow trainable roles must use one local accelerator device")
        rank_seed = seed + self.parallel_context.rank
        if rank_seed >= 2**63:
            raise ValueError("SenseFlow seed plus data-parallel rank exceeds torch RNG range")
        self.generator = torch.Generator(device=device).manual_seed(rank_seed)
        self.global_step = 0
        self.student_optimizer_steps = 0
        self.fake_score_optimizer_steps = 0
        self.discriminator_optimizer_steps = 0
        self.ida_updates = 0
        self._phase = "idle"
        self._poisoned = False
        self.teacher_module.eval()
        for module in self.discriminator_frozen_modules:
            module.eval()

    def _zero_grad(self) -> None:
        self.student_optimizer.zero_grad(set_to_none=True)
        self.fake_score_optimizer.zero_grad(set_to_none=True)
        self.discriminator_optimizer.zero_grad(set_to_none=True)

    def _assert_no_gradients(self, parameters: tuple[nn.Parameter, ...], *, role: str) -> None:
        if any(parameter.grad is not None for parameter in parameters):
            raise RuntimeError(f"SenseFlow phase produced unexpected {role} gradients")

    def train_step(
        self,
        batch: SenseFlowTrainingBatch | Sequence[SenseFlowTrainingBatch],
    ) -> SenseFlowTrainResult:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError(
                "SenseFlow engine has a partially committed iteration; restore the last checkpoint"
            )
        batches = _microbatches(batch, expected=self.gradient_accumulation_steps)
        generator_due = (self.global_step + 1) % self.generator_update_interval == 0
        rng_before = self.generator.get_state().clone()
        prepared_batches: list[SenseFlowPreparedBatch] = []
        generator_results: list[SenseFlowLossResult] = []
        generator_weights: list[Tensor] = []
        fake_results: list[SenseFlowLossResult] = []
        fake_weights: list[Tensor] = []
        discriminator_results: list[SenseFlowLossResult] = []
        discriminator_weights: list[Tensor] = []
        zero = torch.zeros((), device=self.student_parameters[0].device, dtype=torch.float32)
        student_grad_norm = zero
        ida_shift = zero
        state_mutated = False
        self._zero_grad()
        try:
            self._phase = "generator-forward"
            self.student_module.train()
            self.teacher_module.eval()
            self.fake_score_module.eval()
            self.discriminator_module.eval()
            if generator_due:
                generator_weights = [
                    declared_loss_weight(
                        self.loss_adapter,
                        microbatch,
                        role="generator",
                        device=self.student_parameters[0].device,
                    )
                    for microbatch in batches
                ]
                generator_denominator = global_denominator(
                    generator_weights,
                    self.parallel_context,
                )
            else:
                generator_denominator = None
            for index, microbatch in enumerate(batches):
                context = (
                    accumulation_context(
                        self.student_module,
                        final_microbatch=index + 1 == len(batches),
                    )
                    if generator_due
                    else torch.no_grad()
                )
                with context:
                    phase = self.loss_adapter.generator_phase(
                        microbatch,
                        update=generator_due,
                        generator=self.generator,
                    )
                    if not isinstance(phase, SenseFlowGeneratorPhase):
                        raise TypeError("SenseFlow generator phase returned an incompatible value")
                    prepared_batches.append(phase.prepared)
                    if generator_due:
                        assert generator_denominator is not None
                        result = _finite_loss(phase.loss_result, role="SenseFlow generator")
                        weight = generator_weights[index]
                        check_reported_weight(result, weight, role="SenseFlow generator")
                        gradient_weight = (
                            weight
                            / generator_denominator
                            * float(self.parallel_context.world_size)
                        )
                        (result.loss * gradient_weight).backward()
                        generator_results.append(result)
                    elif phase.loss_result is not None:
                        raise RuntimeError("SenseFlow skipped generator phase unexpectedly returned a loss")
            self._assert_no_gradients(self.fake_score_parameters, role="fake-score")
            self._assert_no_gradients(self.discriminator_parameters, role="discriminator")
            if generator_due:
                self._phase = "generator-backward"
                student_grad_norm = clip_grad_norm_(
                    self.student_parameters,
                    self.student_max_grad_norm,
                    error_if_nonfinite=True,
                )
                state_mutated = True
                self.student_optimizer.step()
                self.student_optimizer_steps += 1
                self._phase = "generator-committed"
                if self.ida_enabled:
                    update = implicit_distribution_alignment_(
                        self.student_module,
                        self.fake_score_module,
                        decay=self.ida_decay,
                    )
                    ida_shift = update.mean_absolute_shift.detach()
                    self.ida_updates += 1
                self._phase = "ida-committed"
            if self.student_scheduler is not None and (
                self.student_scheduler_cadence == "iteration" or generator_due
            ):
                state_mutated = True
                self.student_scheduler.step()

            self._phase = "fake-score-backward"
            self._zero_grad()
            self.student_module.eval()
            self.fake_score_module.train()
            self.discriminator_module.eval()
            fake_weights = [
                declared_loss_weight(
                    self.loss_adapter,
                    microbatch,
                    role="fake-score",
                    device=self.fake_score_parameters[0].device,
                )
                for microbatch in batches
            ]
            fake_denominator = global_denominator(fake_weights, self.parallel_context)
            for index, (prepared, weight) in enumerate(
                zip(prepared_batches, fake_weights, strict=True)
            ):
                with accumulation_context(
                    self.fake_score_module,
                    final_microbatch=index + 1 == len(prepared_batches),
                ):
                    result = _finite_loss(
                        self.loss_adapter.fake_score_loss(
                            prepared,
                            generator=self.generator,
                        ),
                        role="SenseFlow fake score",
                    )
                    check_reported_weight(result, weight, role="SenseFlow fake score")
                    gradient_weight = (
                        weight / fake_denominator * float(self.parallel_context.world_size)
                    )
                    (result.loss * gradient_weight).backward()
                fake_results.append(result)
            self._assert_no_gradients(self.student_parameters, role="student")
            self._assert_no_gradients(self.discriminator_parameters, role="discriminator")
            fake_grad_norm = clip_grad_norm_(
                self.fake_score_parameters,
                self.fake_score_max_grad_norm,
                error_if_nonfinite=True,
            )
            state_mutated = True
            self.fake_score_optimizer.step()
            self.fake_score_optimizer_steps += 1
            if self.fake_score_scheduler is not None:
                self.fake_score_scheduler.step()
            self._phase = "fake-score-committed"

            self._phase = "discriminator-backward"
            self._zero_grad()
            self.fake_score_module.eval()
            self.discriminator_module.train()
            for module in self.discriminator_frozen_modules:
                module.eval()
            discriminator_weights = [
                declared_loss_weight(
                    self.loss_adapter,
                    microbatch,
                    role="discriminator",
                    device=self.discriminator_parameters[0].device,
                )
                for microbatch in batches
            ]
            discriminator_denominator = global_denominator(
                discriminator_weights,
                self.parallel_context,
            )
            for index, (prepared, weight) in enumerate(
                zip(prepared_batches, discriminator_weights, strict=True)
            ):
                with accumulation_context(
                    self.discriminator_module,
                    final_microbatch=index + 1 == len(prepared_batches),
                ):
                    result = _finite_loss(
                        self.loss_adapter.discriminator_loss(prepared),
                        role="SenseFlow discriminator",
                    )
                    check_reported_weight(result, weight, role="SenseFlow discriminator")
                    gradient_weight = (
                        weight
                        / discriminator_denominator
                        * float(self.parallel_context.world_size)
                    )
                    (result.loss * gradient_weight).backward()
                discriminator_results.append(result)
            self._assert_no_gradients(self.student_parameters, role="student")
            self._assert_no_gradients(self.fake_score_parameters, role="fake-score")
            discriminator_grad_norm = clip_grad_norm_(
                self.discriminator_parameters,
                self.discriminator_max_grad_norm,
                error_if_nonfinite=True,
            )
            state_mutated = True
            self.discriminator_optimizer.step()
            self.discriminator_optimizer_steps += 1
            if self.discriminator_scheduler is not None:
                self.discriminator_scheduler.step()
            self._phase = "discriminator-committed"
            self.global_step += 1
            self._phase = "idle"
        except Exception:
            self._zero_grad()
            if state_mutated:
                self._poisoned = True
            else:
                self.generator.set_state(rng_before)
                self._phase = "idle"
            raise

        fake_numerator, fake_denominator, fake_loss = global_loss_statistics(
            fake_results,
            fake_weights,
            self.parallel_context,
        )
        discriminator_numerator, discriminator_denominator, discriminator_loss = (
            global_loss_statistics(
                discriminator_results,
                discriminator_weights,
                self.parallel_context,
            )
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
            generator_loss = torch.zeros_like(fake_loss)
            generator_metrics = {}
        metrics = {
            "global_step": torch.tensor(
                self.global_step,
                device=fake_loss.device,
                dtype=torch.int64,
            ),
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
            "discriminator_optimizer_steps": torch.tensor(
                self.discriminator_optimizer_steps,
                device=fake_loss.device,
                dtype=torch.int64,
            ),
            "ida_updates": torch.tensor(
                self.ida_updates,
                device=fake_loss.device,
                dtype=torch.int64,
            ),
            "ida_mean_absolute_shift": ida_shift,
            "accumulated_microbatches": len(batches),
            "student_grad_norm": student_grad_norm.detach(),
            "fake_score_grad_norm": fake_grad_norm.detach(),
            "discriminator_grad_norm": discriminator_grad_norm.detach(),
            "generator": dict(generator_metrics),
            "fake_score": role_metrics(
                fake_results,
                global_numerator=fake_numerator,
                global_denominator=fake_denominator,
            ),
            "discriminator": role_metrics(
                discriminator_results,
                global_numerator=discriminator_numerator,
                global_denominator=discriminator_denominator,
            ),
        }
        return SenseFlowTrainResult(
            generator_loss=generator_loss.detach().float(),
            fake_score_loss=fake_loss.detach().float(),
            discriminator_loss=discriminator_loss.detach().float(),
            generator_updated=generator_due,
            metrics=metrics,
        )

    def state_dict(self) -> dict[str, object]:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("cannot checkpoint a partially committed SenseFlow iteration")
        return {
            "schema": SENSEFLOW_ENGINE_STATE_SCHEMA,
            "global_step": self.global_step,
            "student_optimizer_steps": self.student_optimizer_steps,
            "fake_score_optimizer_steps": self.fake_score_optimizer_steps,
            "discriminator_optimizer_steps": self.discriminator_optimizer_steps,
            "ida_updates": self.ida_updates,
            "generator_update_interval": self.generator_update_interval,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "student_scheduler_cadence": self.student_scheduler_cadence,
            "data_parallel_size": self.parallel_context.world_size,
            "rng_state": self.generator.get_state().clone(),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("SenseFlow engine state must be a mapping")
        expected = {
            "schema",
            "global_step",
            "student_optimizer_steps",
            "fake_score_optimizer_steps",
            "discriminator_optimizer_steps",
            "ida_updates",
            "generator_update_interval",
            "gradient_accumulation_steps",
            "student_scheduler_cadence",
            "data_parallel_size",
            "rng_state",
        }
        if set(state_dict) != expected:
            raise ValueError("SenseFlow engine state fields differ from the active schema")
        if state_dict["schema"] != SENSEFLOW_ENGINE_STATE_SCHEMA:
            raise ValueError(f"unsupported SenseFlow engine schema: {state_dict['schema']!r}")
        for name, active in (
            ("generator_update_interval", self.generator_update_interval),
            ("gradient_accumulation_steps", self.gradient_accumulation_steps),
            ("data_parallel_size", self.parallel_context.world_size),
        ):
            if int(state_dict[name]) != active:
                raise ValueError(f"saved SenseFlow {name} differs from the active engine")
        if str(state_dict["student_scheduler_cadence"]) != self.student_scheduler_cadence:
            raise ValueError("saved SenseFlow scheduler cadence differs from the active engine")
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
        ida_updates = non_negative_int(state_dict["ida_updates"], field_name="ida_updates")
        if student_steps != global_step // self.generator_update_interval:
            raise ValueError("saved SenseFlow student counter violates TTUR cadence")
        if fake_steps != global_step or discriminator_steps != global_step:
            raise ValueError("saved SenseFlow auxiliary counters violate update cadence")
        expected_ida = student_steps if self.ida_enabled else 0
        if ida_updates != expected_ida:
            raise ValueError("saved SenseFlow IDA counter violates generator cadence")
        rng_state = state_dict["rng_state"]
        if not isinstance(rng_state, Tensor) or rng_state.dtype != torch.uint8 or rng_state.ndim != 1:
            raise TypeError("saved SenseFlow rng_state must be a rank-one uint8 tensor")
        previous_rng = self.generator.get_state().clone()
        try:
            self.generator.set_state(rng_state.detach().cpu())
        except Exception:
            self.generator.set_state(previous_rng)
            raise
        self.global_step = global_step
        self.student_optimizer_steps = student_steps
        self.fake_score_optimizer_steps = fake_steps
        self.discriminator_optimizer_steps = discriminator_steps
        self.ida_updates = ida_updates
        self._phase = "idle"
        self._poisoned = False


__all__ = [
    "SENSEFLOW_ENGINE_STATE_SCHEMA",
    "NativeSenseFlowTrainEngine",
    "SenseFlowTrainResult",
]
