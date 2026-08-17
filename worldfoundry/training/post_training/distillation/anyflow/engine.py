"""Scalable optimizer engines for native AnyFlow training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

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
from .contracts import (
    AnyFlowLossResult,
    AnyFlowOnPolicyLossAdapter,
    AnyFlowPretrainLossAdapter,
    AnyFlowTrainingBatch,
)
from .synchronization import ANYFLOW_DECISION_RNG_SCHEMA, AnyFlowDecisionRNG

ANYFLOW_PRETRAIN_ENGINE_STATE_SCHEMA = "worldfoundry-anyflow-pretrain-engine"
ANYFLOW_ON_POLICY_ENGINE_STATE_SCHEMA = "worldfoundry-anyflow-on-policy-engine"


def _finite_result(value: object, *, role: str) -> AnyFlowLossResult:
    if not isinstance(value, AnyFlowLossResult):
        raise TypeError(f"{role} objective must return AnyFlowLossResult")
    if not isinstance(value.loss, Tensor) or value.loss.numel() != 1:
        raise TypeError(f"{role} objective loss must be a scalar tensor")
    if not bool(torch.isfinite(value.loss.detach())):
        raise FloatingPointError(f"non-finite {role} loss")
    return value


def _batches(
    value: AnyFlowTrainingBatch | Sequence[AnyFlowTrainingBatch],
    *,
    accumulation_steps: int,
) -> tuple[AnyFlowTrainingBatch, ...]:
    if isinstance(value, AnyFlowTrainingBatch):
        batches = (value,)
    elif isinstance(value, Sequence):
        batches = tuple(value)
    else:
        raise TypeError("AnyFlow batch must be one batch or a microbatch sequence")
    if len(batches) != accumulation_steps:
        raise ValueError(
            f"AnyFlow optimizer update requires exactly {accumulation_steps} microbatches; got {len(batches)}"
        )
    if not all(isinstance(batch, AnyFlowTrainingBatch) for batch in batches):
        raise TypeError("every AnyFlow microbatch must be AnyFlowTrainingBatch")
    return batches


def _decision_state(
    value: object,
    *,
    seed: int,
    expected_draw_count: int,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("saved AnyFlow decision state must be a mapping")
    if value.get("schema") != ANYFLOW_DECISION_RNG_SCHEMA:
        raise ValueError("saved AnyFlow decision state has an unsupported schema")
    if int(value.get("seed", -1)) != seed:
        raise ValueError("saved AnyFlow decision seed differs from the active engine")
    if int(value.get("draw_count", -1)) != expected_draw_count:
        raise ValueError("saved AnyFlow decision count violates the optimizer cadence")
    return value


@dataclass(frozen=True, slots=True)
class AnyFlowPretrainResult:
    loss: Tensor
    metrics: Mapping[str, object]


class NativeAnyFlowPretrainEngine:
    """Apply one weighted FAR FlowMap optimizer update atomically."""

    def __init__(
        self,
        *,
        student_module: nn.Module,
        loss_adapter: AnyFlowPretrainLossAdapter,
        optimizer: torch.optim.Optimizer,
        decisions: AnyFlowDecisionRNG,
        max_grad_norm: float,
        gradient_accumulation_steps: int = 1,
        scheduler: object | None = None,
        student_ema: object | None = None,
        parallel_context: PostTrainingParallelContext | None = None,
    ) -> None:
        if not isinstance(student_module, nn.Module):
            raise TypeError("AnyFlow student_module must be nn.Module")
        if not isinstance(loss_adapter, AnyFlowPretrainLossAdapter):
            raise TypeError("loss_adapter must implement AnyFlowPretrainLossAdapter")
        if not isinstance(decisions, AnyFlowDecisionRNG):
            raise TypeError("decisions must be AnyFlowDecisionRNG")
        if loss_adapter.decisions is not decisions:
            raise ValueError("AnyFlow engine and objective must share one decision RNG")
        if loss_adapter.student.module is not student_module:
            raise ValueError("AnyFlow engine and objective must share the student module")
        draws_per_loss = non_negative_int(
            loss_adapter.decision_draws_per_student_loss,
            field_name="decision_draws_per_student_loss",
        )
        if draws_per_loss == 0:
            raise ValueError("AnyFlow pretraining must declare positive decision cadence")
        accumulation = non_negative_int(
            gradient_accumulation_steps,
            field_name="gradient_accumulation_steps",
        )
        if accumulation == 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        validate_stateful_or_none(scheduler, field_name="scheduler")
        validate_stateful_or_none(student_ema, field_name="student_ema")
        if student_ema is not None and not callable(getattr(student_ema, "update", None)):
            raise TypeError("student_ema must expose update(module)")
        parameters = trainable_parameters(student_module)
        audit_optimizer_parameters(optimizer, parameters, role="AnyFlow student")
        context = parallel_context or PostTrainingParallelContext.current()
        if context.world_size > 1 and decisions.synchronizer is None:
            raise ValueError("distributed AnyFlow training requires synchronized randomized decisions")
        synchronized_world_size = getattr(decisions.synchronizer, "world_size", None)
        if synchronized_world_size is not None and int(synchronized_world_size) != (context.world_size):
            raise ValueError("AnyFlow decision synchronizer and data-parallel group differ")
        context.audit_synchronized_module(student_module, role="AnyFlow student")
        self.student_module = student_module
        self.loss_adapter = loss_adapter
        self.optimizer = optimizer
        self.decisions = decisions
        self.decision_draws_per_student_loss = draws_per_loss
        self.parameters = parameters
        self.max_grad_norm = positive_float(max_grad_norm, field_name="max_grad_norm")
        self.gradient_accumulation_steps = accumulation
        self.scheduler = scheduler
        self.student_ema = student_ema
        self.parallel_context = context
        self.global_step = 0
        self.optimizer_steps = 0
        self._phase = "idle"
        self._poisoned = False

    @property
    def config_state(self) -> dict[str, object]:
        return dict(self.loss_adapter.config_state)

    def train_step(
        self,
        batch: AnyFlowTrainingBatch | Sequence[AnyFlowTrainingBatch],
        *,
        generator: torch.Generator | None = None,
    ) -> AnyFlowPretrainResult:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("AnyFlow pretrain engine has a partial update; restore a checkpoint")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be torch.Generator or None")
        batches = _batches(
            batch,
            accumulation_steps=self.gradient_accumulation_steps,
        )
        weights = [
            declared_loss_weight(
                self.loss_adapter,
                microbatch,
                role="student",
                device=self.parameters[0].device,
            )
            for microbatch in batches
        ]
        total_weight = global_denominator(weights, self.parallel_context)
        results: list[AnyFlowLossResult] = []
        self.optimizer.zero_grad(set_to_none=True)
        optimizer_started = False
        try:
            self._phase = "student-backward"
            self.student_module.train()
            for index, (microbatch, weight) in enumerate(zip(batches, weights, strict=True)):
                with accumulation_context(
                    self.student_module,
                    final_microbatch=index + 1 == len(batches),
                ):
                    result = _finite_result(
                        self.loss_adapter.student_loss(
                            microbatch,
                            generator=generator,
                        ),
                        role="AnyFlow pretrain",
                    )
                    check_reported_weight(result, weight, role="AnyFlow pretrain")
                    gradient_weight = weight / total_weight * float(self.parallel_context.world_size)
                    (result.loss * gradient_weight).backward()
                results.append(result)
            grad_norm = clip_grad_norm_(
                self.parameters,
                self.max_grad_norm,
                error_if_nonfinite=True,
            )
            optimizer_started = True
            self.optimizer.step()
            self.optimizer_steps += 1
            self._phase = "student-committed"
            if self.scheduler is not None:
                self.scheduler.step()
            if self.student_ema is not None:
                self.student_ema.update(self.student_module)
            self.global_step += 1
            self._phase = "idle"
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            if optimizer_started:
                self._poisoned = True
            else:
                self._phase = "idle"
            raise
        numerator, denominator, loss = global_loss_statistics(
            results,
            weights,
            self.parallel_context,
        )
        metrics = role_metrics(
            results,
            global_numerator=numerator,
            global_denominator=denominator,
        )
        metrics.update(
            {
                "global_step": self.global_step,
                "optimizer_steps": self.optimizer_steps,
                "grad_norm": grad_norm.detach(),
            }
        )
        return AnyFlowPretrainResult(loss=loss.detach().float(), metrics=metrics)

    def state_dict(self) -> dict[str, object]:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("cannot checkpoint a partially committed AnyFlow update")
        return {
            "schema": ANYFLOW_PRETRAIN_ENGINE_STATE_SCHEMA,
            "global_step": self.global_step,
            "optimizer_steps": self.optimizer_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "configuration": self.config_state,
            "data_parallel_size": self.parallel_context.world_size,
            "decisions": self.decisions.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        expected = {
            "schema",
            "global_step",
            "optimizer_steps",
            "gradient_accumulation_steps",
            "configuration",
            "data_parallel_size",
            "decisions",
        }
        if not isinstance(state_dict, Mapping) or set(state_dict) != expected:
            raise ValueError("AnyFlow pretrain engine state fields differ from the schema")
        if state_dict["schema"] != ANYFLOW_PRETRAIN_ENGINE_STATE_SCHEMA:
            raise ValueError("unsupported AnyFlow pretrain engine schema")
        if state_dict["configuration"] != self.config_state:
            raise ValueError("saved AnyFlow pretrain configuration differs")
        for name, active in (
            ("gradient_accumulation_steps", self.gradient_accumulation_steps),
            ("data_parallel_size", self.parallel_context.world_size),
        ):
            if int(state_dict[name]) != active:
                raise ValueError(f"saved AnyFlow pretrain {name} differs")
        step = non_negative_int(state_dict["global_step"], field_name="global_step")
        optimizer_steps = non_negative_int(
            state_dict["optimizer_steps"],
            field_name="optimizer_steps",
        )
        if optimizer_steps != step:
            raise ValueError("saved AnyFlow pretrain optimizer counter is inconsistent")
        decisions = _decision_state(
            state_dict["decisions"],
            seed=self.decisions.seed,
            expected_draw_count=(self.decision_draws_per_student_loss * self.gradient_accumulation_steps * step),
        )
        self.decisions.load_state_dict(decisions)
        self.global_step = step
        self.optimizer_steps = optimizer_steps
        self._phase = "idle"
        self._poisoned = False
        self.optimizer.zero_grad(set_to_none=True)


@dataclass(frozen=True, slots=True)
class AnyFlowOnPolicyResult:
    generator_loss: Tensor
    fake_score_loss: Tensor
    metrics: Mapping[str, object]


class NativeAnyFlowOnPolicyEngine:
    """Commit generator/FAR first, then fresh fake-score updates."""

    def __init__(
        self,
        *,
        student_module: nn.Module,
        real_score_module: nn.Module,
        fake_score_module: nn.Module,
        loss_adapter: AnyFlowOnPolicyLossAdapter,
        student_optimizer: torch.optim.Optimizer,
        fake_score_optimizer: torch.optim.Optimizer,
        decisions: AnyFlowDecisionRNG,
        discriminator_update_ratio: int,
        student_max_grad_norm: float,
        fake_score_max_grad_norm: float,
        gradient_accumulation_steps: int = 1,
        student_scheduler: object | None = None,
        fake_score_scheduler: object | None = None,
        student_ema: object | None = None,
        parallel_context: PostTrainingParallelContext | None = None,
    ) -> None:
        modules = (student_module, real_score_module, fake_score_module)
        if not all(isinstance(module, nn.Module) for module in modules):
            raise TypeError("all AnyFlow on-policy roles must be nn.Module")
        if len({id(module) for module in modules}) != 3:
            raise ValueError("AnyFlow on-policy role modules must be distinct")
        if any(parameter.requires_grad for parameter in real_score_module.parameters()):
            raise ValueError("AnyFlow real score parameters must be frozen")
        if not isinstance(loss_adapter, AnyFlowOnPolicyLossAdapter):
            raise TypeError("loss_adapter must implement AnyFlowOnPolicyLossAdapter")
        if not isinstance(decisions, AnyFlowDecisionRNG):
            raise TypeError("decisions must be AnyFlowDecisionRNG")
        if loss_adapter.decisions is not decisions:
            raise ValueError("AnyFlow engine and objective must share one decision RNG")
        if loss_adapter.student.module is not student_module:
            raise ValueError("AnyFlow objective student differs from the engine module")
        if loss_adapter.real_score.module is not real_score_module:
            raise ValueError("AnyFlow objective real score differs from the engine module")
        if loss_adapter.fake_score.module is not fake_score_module:
            raise ValueError("AnyFlow objective fake score differs from the engine module")
        ratio = non_negative_int(
            discriminator_update_ratio,
            field_name="discriminator_update_ratio",
        )
        accumulation = non_negative_int(
            gradient_accumulation_steps,
            field_name="gradient_accumulation_steps",
        )
        if ratio == 0 or accumulation == 0:
            raise ValueError("AnyFlow update ratio and accumulation must be positive")
        if int(loss_adapter.discriminator_update_ratio) != ratio:
            raise ValueError("AnyFlow objective and engine update ratios differ")
        generator_draws = non_negative_int(
            loss_adapter.generator_decision_draws,
            field_name="generator_decision_draws",
        )
        fake_draws = non_negative_int(
            loss_adapter.fake_score_decision_draws,
            field_name="fake_score_decision_draws",
        )
        if generator_draws == 0 or fake_draws == 0:
            raise ValueError("AnyFlow on-policy objectives must declare decision cadence")
        validate_stateful_or_none(student_scheduler, field_name="student_scheduler")
        validate_stateful_or_none(fake_score_scheduler, field_name="fake_score_scheduler")
        validate_stateful_or_none(student_ema, field_name="student_ema")
        if student_ema is not None and not callable(getattr(student_ema, "update", None)):
            raise TypeError("student_ema must expose update(module)")
        student_parameters = trainable_parameters(student_module)
        fake_parameters = trainable_parameters(fake_score_module)
        if {id(value) for value in student_parameters} & {id(value) for value in fake_parameters}:
            raise ValueError("AnyFlow student and fake score cannot share parameters")
        audit_optimizer_parameters(
            student_optimizer,
            student_parameters,
            role="AnyFlow student",
        )
        audit_optimizer_parameters(
            fake_score_optimizer,
            fake_parameters,
            role="AnyFlow fake score",
        )
        context = parallel_context or PostTrainingParallelContext.current()
        if context.world_size > 1 and decisions.synchronizer is None:
            raise ValueError("distributed AnyFlow training requires synchronized randomized decisions")
        synchronized_world_size = getattr(decisions.synchronizer, "world_size", None)
        if synchronized_world_size is not None and int(synchronized_world_size) != (context.world_size):
            raise ValueError("AnyFlow decision synchronizer and data-parallel group differ")
        context.audit_synchronized_module(student_module, role="AnyFlow student")
        context.audit_synchronized_module(fake_score_module, role="AnyFlow fake score")
        self.student_module = student_module
        self.real_score_module = real_score_module
        self.fake_score_module = fake_score_module
        self.loss_adapter = loss_adapter
        self.student_optimizer = student_optimizer
        self.fake_score_optimizer = fake_score_optimizer
        self.decisions = decisions
        self.student_parameters = student_parameters
        self.fake_score_parameters = fake_parameters
        self.discriminator_update_ratio = ratio
        self.generator_decision_draws = generator_draws
        self.fake_score_decision_draws = fake_draws
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
        self.parallel_context = context
        self.global_step = 0
        self.student_optimizer_steps = 0
        self.fake_score_optimizer_steps = 0
        self._phase = "idle"
        self._poisoned = False
        self.real_score_module.eval()

    @property
    def config_state(self) -> dict[str, object]:
        return dict(self.loss_adapter.config_state)

    def _role_backward(
        self,
        batches: tuple[AnyFlowTrainingBatch, ...],
        *,
        role: str,
        module: nn.Module,
        parameters: tuple[nn.Parameter, ...],
        generator: torch.Generator | None,
    ) -> tuple[list[AnyFlowLossResult], list[Tensor], Tensor]:
        weights = [
            declared_loss_weight(
                self.loss_adapter,
                microbatch,
                role=role,
                device=parameters[0].device,
            )
            for microbatch in batches
        ]
        total_weight = global_denominator(weights, self.parallel_context)
        results: list[AnyFlowLossResult] = []
        method = self.loss_adapter.generator_loss if role == "generator" else self.loss_adapter.fake_score_loss
        for index, (microbatch, weight) in enumerate(zip(batches, weights, strict=True)):
            with accumulation_context(
                module,
                final_microbatch=index + 1 == len(batches),
            ):
                result = _finite_result(
                    method(microbatch, generator=generator),
                    role=f"AnyFlow {role}",
                )
                check_reported_weight(result, weight, role=f"AnyFlow {role}")
                gradient_weight = weight / total_weight * float(self.parallel_context.world_size)
                (result.loss * gradient_weight).backward()
            results.append(result)
        return results, weights, total_weight

    def _fake_score_groups(
        self,
        value: Sequence[AnyFlowTrainingBatch] | None,
    ) -> tuple[tuple[AnyFlowTrainingBatch, ...], ...]:
        if value is None or not isinstance(value, Sequence):
            raise TypeError("fresh AnyFlow fake-score batches must be supplied as one flat sequence")
        flat = tuple(value)
        expected = self.discriminator_update_ratio * self.gradient_accumulation_steps
        if len(flat) != expected:
            raise ValueError(
                f"AnyFlow on-policy training requires {expected} fresh fake-score microbatches; got {len(flat)}"
            )
        if not all(isinstance(batch, AnyFlowTrainingBatch) for batch in flat):
            raise TypeError("every fresh AnyFlow fake-score microbatch must be AnyFlowTrainingBatch")
        return tuple(
            flat[offset : offset + self.gradient_accumulation_steps]
            for offset in range(0, expected, self.gradient_accumulation_steps)
        )

    def train_step(
        self,
        batch: AnyFlowTrainingBatch | Sequence[AnyFlowTrainingBatch],
        *,
        fake_score_batches: Sequence[AnyFlowTrainingBatch] | None = None,
        generator: torch.Generator | None = None,
    ) -> AnyFlowOnPolicyResult:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("AnyFlow on-policy engine has a partial iteration; restore a checkpoint")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be torch.Generator or None")
        batches = _batches(
            batch,
            accumulation_steps=self.gradient_accumulation_steps,
        )
        fake_score_groups = self._fake_score_groups(fake_score_batches)
        generator_results: list[AnyFlowLossResult] = []
        generator_weights: list[Tensor] = []
        fake_results: list[AnyFlowLossResult] = []
        fake_weights: list[Tensor] = []
        self.student_optimizer.zero_grad(set_to_none=True)
        self.fake_score_optimizer.zero_grad(set_to_none=True)
        optimizer_mutated = False
        try:
            self._phase = "generator-backward"
            self.student_module.train()
            self.fake_score_module.eval()
            self.real_score_module.eval()
            generator_results, generator_weights, _ = self._role_backward(
                batches,
                role="generator",
                module=self.student_module,
                parameters=self.student_parameters,
                generator=generator,
            )
            if any(parameter.grad is not None for parameter in self.fake_score_parameters):
                raise RuntimeError("AnyFlow generator phase produced fake-score gradients")
            student_grad_norm = clip_grad_norm_(
                self.student_parameters,
                self.student_max_grad_norm,
                error_if_nonfinite=True,
            )
            optimizer_mutated = True
            self.student_optimizer.step()
            self.student_optimizer_steps += 1
            self._phase = "generator-committed"
            if self.student_scheduler is not None:
                self.student_scheduler.step()
            if self.student_ema is not None:
                self.student_ema.update(self.student_module)

            fake_grad_norm = torch.zeros(
                (),
                device=self.fake_score_parameters[0].device,
                dtype=torch.float32,
            )
            for update_index, fake_batches in enumerate(fake_score_groups):
                self._phase = f"fake-score-{update_index}-backward"
                self.student_module.eval()
                self.fake_score_module.train()
                self.real_score_module.eval()
                self.student_optimizer.zero_grad(set_to_none=True)
                self.fake_score_optimizer.zero_grad(set_to_none=True)
                results, weights, _ = self._role_backward(
                    fake_batches,
                    role="fake-score",
                    module=self.fake_score_module,
                    parameters=self.fake_score_parameters,
                    generator=generator,
                )
                fake_results.extend(results)
                fake_weights.extend(weights)
                if any(parameter.grad is not None for parameter in self.student_parameters):
                    raise RuntimeError("AnyFlow fake-score phase produced student gradients")
                fake_grad_norm = clip_grad_norm_(
                    self.fake_score_parameters,
                    self.fake_score_max_grad_norm,
                    error_if_nonfinite=True,
                )
                self.fake_score_optimizer.step()
                self.fake_score_optimizer_steps += 1
                self._phase = f"fake-score-{update_index}-committed"
                if self.fake_score_scheduler is not None:
                    self.fake_score_scheduler.step()
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

        generator_numerator, generator_denominator, generator_loss = global_loss_statistics(
            generator_results,
            generator_weights,
            self.parallel_context,
        )
        fake_numerator, fake_denominator, fake_loss = global_loss_statistics(
            fake_results,
            fake_weights,
            self.parallel_context,
        )
        metrics = {
            "global_step": self.global_step,
            "student_optimizer_steps": self.student_optimizer_steps,
            "fake_score_optimizer_steps": self.fake_score_optimizer_steps,
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
        return AnyFlowOnPolicyResult(
            generator_loss=generator_loss.detach().float(),
            fake_score_loss=fake_loss.detach().float(),
            metrics=metrics,
        )

    def state_dict(self) -> dict[str, object]:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("cannot checkpoint a partial AnyFlow on-policy iteration")
        return {
            "schema": ANYFLOW_ON_POLICY_ENGINE_STATE_SCHEMA,
            "global_step": self.global_step,
            "student_optimizer_steps": self.student_optimizer_steps,
            "fake_score_optimizer_steps": self.fake_score_optimizer_steps,
            "discriminator_update_ratio": self.discriminator_update_ratio,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "configuration": self.config_state,
            "data_parallel_size": self.parallel_context.world_size,
            "decisions": self.decisions.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        expected = {
            "schema",
            "global_step",
            "student_optimizer_steps",
            "fake_score_optimizer_steps",
            "discriminator_update_ratio",
            "gradient_accumulation_steps",
            "configuration",
            "data_parallel_size",
            "decisions",
        }
        if not isinstance(state_dict, Mapping) or set(state_dict) != expected:
            raise ValueError("AnyFlow on-policy engine state fields differ from schema")
        if state_dict["schema"] != ANYFLOW_ON_POLICY_ENGINE_STATE_SCHEMA:
            raise ValueError("unsupported AnyFlow on-policy engine schema")
        if state_dict["configuration"] != self.config_state:
            raise ValueError("saved AnyFlow on-policy configuration differs")
        for name, active in (
            ("discriminator_update_ratio", self.discriminator_update_ratio),
            ("gradient_accumulation_steps", self.gradient_accumulation_steps),
            ("data_parallel_size", self.parallel_context.world_size),
        ):
            if int(state_dict[name]) != active:
                raise ValueError(f"saved AnyFlow on-policy {name} differs")
        step = non_negative_int(state_dict["global_step"], field_name="global_step")
        student_steps = non_negative_int(
            state_dict["student_optimizer_steps"],
            field_name="student_optimizer_steps",
        )
        fake_steps = non_negative_int(
            state_dict["fake_score_optimizer_steps"],
            field_name="fake_score_optimizer_steps",
        )
        if student_steps != step or fake_steps != step * self.discriminator_update_ratio:
            raise ValueError("saved AnyFlow optimizer counters violate the update ratio")
        expected_draws = (
            step
            * self.gradient_accumulation_steps
            * (self.generator_decision_draws + self.fake_score_decision_draws * self.discriminator_update_ratio)
        )
        decisions = _decision_state(
            state_dict["decisions"],
            seed=self.decisions.seed,
            expected_draw_count=expected_draws,
        )
        self.decisions.load_state_dict(decisions)
        self.global_step = step
        self.student_optimizer_steps = student_steps
        self.fake_score_optimizer_steps = fake_steps
        self._phase = "idle"
        self._poisoned = False
        self.student_optimizer.zero_grad(set_to_none=True)
        self.fake_score_optimizer.zero_grad(set_to_none=True)


__all__ = [
    "ANYFLOW_ON_POLICY_ENGINE_STATE_SCHEMA",
    "ANYFLOW_PRETRAIN_ENGINE_STATE_SCHEMA",
    "AnyFlowOnPolicyResult",
    "AnyFlowPretrainResult",
    "NativeAnyFlowOnPolicyEngine",
    "NativeAnyFlowPretrainEngine",
]
