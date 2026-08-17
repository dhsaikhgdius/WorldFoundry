"""Atomic progressive-distillation optimizer and stage state machine."""

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
    global_denominator,
    global_loss_statistics,
    role_metrics,
)
from ...shared.distributed import PostTrainingParallelContext
from ...shared.validation import positive_float
from ..causal_consistency.ema import FrozenModuleEMA
from .contracts import ProgressiveDistillationBatch, ProgressiveRandomInputs
from .objective import (
    ProgressiveDistillationLossResult,
    ProgressiveDistillationObjective,
)

PROGRESSIVE_DISTILLATION_ENGINE_STATE_SCHEMA = (
    "worldfoundry-progressive-distillation-engine"
)


def _execution_module(module: nn.Module) -> nn.Module:
    from torch.nn.parallel import DistributedDataParallel

    return module.module if isinstance(module, DistributedDataParallel) else module


def _microbatches(
    value: ProgressiveDistillationBatch | Sequence[ProgressiveDistillationBatch],
    *,
    expected: int,
) -> tuple[ProgressiveDistillationBatch, ...]:
    if isinstance(value, ProgressiveDistillationBatch):
        batches = (value,)
    elif isinstance(value, Sequence):
        batches = tuple(value)
    else:
        raise TypeError("progressive batch must be one batch or a sequence")
    if len(batches) != expected:
        raise ValueError(
            "progressive optimizer iteration requires "
            f"{expected} microbatches; got {len(batches)}"
        )
    if not all(isinstance(batch, ProgressiveDistillationBatch) for batch in batches):
        raise TypeError("every progressive microbatch must use the native contract")
    return batches


def _audit_module_state(source: nn.Module, target: nn.Module, *, role: str) -> None:
    source_module = _execution_module(source)
    target_module = _execution_module(target)
    source_state = source_module.state_dict()
    target_state = target_module.state_dict()
    if tuple(source_state) != tuple(target_state):
        raise ValueError(f"progressive {role} state inventories differ")
    for name in source_state:
        if (
            source_state[name].shape != target_state[name].shape
            or source_state[name].dtype != target_state[name].dtype
        ):
            raise ValueError(
                f"progressive {role} tensor {name!r} shape or dtype differs"
            )


def _copy_module_state(source: nn.Module, target: nn.Module, *, role: str) -> None:
    _audit_module_state(source, target, role=role)
    source_module = _execution_module(source)
    target_module = _execution_module(target)
    incompatible = target_module.load_state_dict(source_module.state_dict(), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"progressive {role} strict state copy failed")


@torch.no_grad()
def _copy_module_buffers(source: nn.Module, target: nn.Module) -> None:
    """Keep mutable buffers aligned with the online student after an EMA step."""

    source_buffers = dict(_execution_module(source).named_buffers())
    target_buffers = dict(_execution_module(target).named_buffers())
    if tuple(source_buffers) != tuple(target_buffers):
        raise RuntimeError("progressive student and EMA buffer inventories differ")
    for name, source_buffer in source_buffers.items():
        target_buffer = target_buffers[name]
        if (
            source_buffer.shape != target_buffer.shape
            or source_buffer.dtype != target_buffer.dtype
            or source_buffer.device != target_buffer.device
        ):
            raise RuntimeError(
                f"progressive EMA buffer {name!r} shape, dtype, or device differs"
            )
        target_buffer.copy_(source_buffer.detach())


@dataclass(frozen=True, slots=True)
class ProgressiveDistillationTrainResult:
    loss: Tensor
    metrics: Mapping[str, object]


class NativeProgressiveDistillationTrainEngine:
    """Train each halving stage and atomically promote its EMA at the boundary."""

    def __init__(
        self,
        *,
        student_module: nn.Module,
        teacher_module: nn.Module,
        ema_target_module: nn.Module,
        objective: ProgressiveDistillationObjective,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        parallel_context: PostTrainingParallelContext | None = None,
        seed: int = 0,
        initialize_student_from_teacher: bool = True,
        initialize_ema_target: bool = True,
    ) -> None:
        modules = (student_module, teacher_module, ema_target_module)
        if not all(isinstance(module, nn.Module) for module in modules):
            raise TypeError("all progressive roles must be nn.Module values")
        if len({id(module) for module in modules}) != 3:
            raise ValueError("progressive roles must be distinct modules")
        parameter_ids = tuple(
            {id(parameter) for parameter in module.parameters()} for module in modules
        )
        if any(
            parameter_ids[left] & parameter_ids[right]
            for left in range(len(parameter_ids))
            for right in range(left + 1, len(parameter_ids))
        ):
            raise ValueError("progressive roles cannot share parameters")
        if not isinstance(objective, ProgressiveDistillationObjective):
            raise TypeError("objective must be ProgressiveDistillationObjective")
        if (
            objective.student_module is not student_module
            or objective.teacher_module is not teacher_module
        ):
            raise ValueError("progressive objective and engine role modules differ")
        if (
            isinstance(gradient_accumulation_steps, bool)
            or not isinstance(gradient_accumulation_steps, int)
            or gradient_accumulation_steps <= 0
        ):
            raise ValueError("gradient_accumulation_steps must be positive")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(initialize_student_from_teacher, bool):
            raise TypeError("initialize_student_from_teacher must be bool")
        if not isinstance(initialize_ema_target, bool):
            raise TypeError("initialize_ema_target must be bool")

        teacher_module.requires_grad_(False)
        ema_target_module.requires_grad_(False)
        teacher_module.eval()
        ema_target_module.eval()
        parameters = trainable_parameters(student_module)
        audit_optimizer_parameters(
            optimizer,
            parameters,
            role="progressive student",
        )
        source_module = _execution_module(student_module)
        target_module = _execution_module(ema_target_module)
        if initialize_student_from_teacher:
            _copy_module_state(
                _execution_module(teacher_module),
                source_module,
                role="teacher-to-student initialization",
            )
        if initialize_ema_target:
            _copy_module_state(
                source_module,
                target_module,
                role="student-to-EMA initialization",
            )
        ema_updater = FrozenModuleEMA(
            source_module,
            target_module,
            decay=objective.config.ema_decay,
            initialize_target=False,
        )
        _audit_module_state(
            target_module,
            _execution_module(teacher_module),
            role="EMA-to-teacher compatibility audit",
        )
        context = parallel_context or PostTrainingParallelContext.current()
        context.audit_synchronized_module(
            student_module,
            role="progressive student",
        )
        rng_device = parameters[0].device
        rng = torch.Generator(device=rng_device)
        rng.manual_seed((seed + context.rank) % (2**63 - 1))

        self.student_module = student_module
        self.teacher_module = teacher_module
        self.ema_target_module = ema_target_module
        self.objective = objective
        self.optimizer = optimizer
        self.student_parameters = parameters
        self.max_grad_norm = positive_float(
            max_grad_norm,
            field_name="max_grad_norm",
        )
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.parallel_context = context
        self.ema_updater = ema_updater
        self._base_learning_rates = tuple(
            float(group["lr"]) for group in optimizer.param_groups
        )
        if not self._base_learning_rates or any(
            value <= 0.0 for value in self._base_learning_rates
        ):
            raise ValueError(
                "progressive optimizer parameter groups require positive learning rates"
            )
        self._rng = rng
        self._rng_device = str(rng_device)
        self.global_step = 0
        self.optimizer_steps = 0
        self.stage_index = 0
        self.stage_step = 0
        self._complete = False
        self._phase = "idle"
        self._poisoned = False

    @property
    def student_num_steps(self) -> int:
        return self.objective.config.student_steps[self.stage_index]

    @property
    def teacher_num_steps(self) -> int:
        return self.objective.config.teacher_steps[self.stage_index]

    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def remaining_optimizer_steps(self) -> int:
        if self._complete:
            return 0
        remaining_stages = self.objective.config.stage_count - self.stage_index - 1
        return (
            self.objective.config.optimizer_steps_per_stage
            - self.stage_step
            + remaining_stages * self.objective.config.optimizer_steps_per_stage
        )

    def _learning_rate_multiplier(self) -> float:
        if self.objective.config.learning_rate_anneal == "constant":
            return 1.0
        return 1.0 - self.stage_step / float(
            self.objective.config.optimizer_steps_per_stage
        )

    def _apply_stage_learning_rate(self) -> tuple[float, ...]:
        multiplier = self._learning_rate_multiplier()
        values = tuple(
            base * multiplier for base in self._base_learning_rates
        )
        for group, value in zip(
            self.optimizer.param_groups,
            values,
            strict=True,
        ):
            group["lr"] = value
        return values

    def _restore_base_learning_rates(self) -> None:
        for group, value in zip(
            self.optimizer.param_groups,
            self._base_learning_rates,
            strict=True,
        ):
            group["lr"] = value

    def sample_random_inputs(
        self,
        batch: ProgressiveDistillationBatch,
    ) -> ProgressiveRandomInputs:
        if not isinstance(batch, ProgressiveDistillationBatch):
            raise TypeError("batch must be ProgressiveDistillationBatch")
        clean = batch.clean_latents
        if not isinstance(clean, Tensor) or str(clean.device) != self._rng_device:
            raise ValueError(
                "progressive batch device differs from the checkpointed RNG device"
            )
        noise = torch.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.dtype,
            generator=self._rng,
        )
        indices = torch.randint(
            0,
            self.student_num_steps,
            (batch.batch_size,),
            device=clean.device,
            dtype=torch.int64,
            generator=self._rng,
        )
        return ProgressiveRandomInputs(noise=noise, timestep_indices=indices)

    def _finish_stage_if_due(self) -> bool:
        config = self.objective.config
        if self.stage_step != config.optimizer_steps_per_stage:
            return False
        if self.student_num_steps == config.end_num_steps:
            _copy_module_state(
                self.ema_target_module,
                self.student_module,
                role="final EMA-to-student export",
            )
            self.optimizer.state.clear()
            self.optimizer.zero_grad(set_to_none=True)
            self._complete = True
            return True
        _copy_module_state(
            self.ema_target_module,
            self.teacher_module,
            role="EMA-to-next-teacher promotion",
        )
        _copy_module_state(
            self.ema_target_module,
            self.student_module,
            role="EMA-to-next-student initialization",
        )
        self.optimizer.state.clear()
        self.optimizer.zero_grad(set_to_none=True)
        self._restore_base_learning_rates()
        self.stage_index += 1
        self.stage_step = 0
        self.teacher_module.requires_grad_(False)
        self.ema_target_module.requires_grad_(False)
        self.teacher_module.eval()
        self.ema_target_module.eval()
        return True

    def train_step(
        self,
        batch: ProgressiveDistillationBatch
        | Sequence[ProgressiveDistillationBatch],
    ) -> ProgressiveDistillationTrainResult:
        if self._complete:
            raise RuntimeError("progressive distillation has reached end_num_steps")
        if self._poisoned or self._phase != "idle":
            raise RuntimeError(
                "progressive engine has a partially committed update; restore a checkpoint"
            )
        batches = _microbatches(
            batch,
            expected=self.gradient_accumulation_steps,
        )
        weights = [
            torch.tensor(
                self.objective.loss_denominator(microbatch),
                device=self.student_parameters[0].device,
                dtype=torch.float32,
            )
            for microbatch in batches
        ]
        total_weight = global_denominator(weights, self.parallel_context)
        active_student_steps = self.student_num_steps
        active_teacher_steps = self.teacher_num_steps
        active_stage = self.stage_index
        results: list[ProgressiveDistillationLossResult] = []
        self.optimizer.zero_grad(set_to_none=True)
        optimizer_mutated = False
        try:
            self._phase = "student-backward"
            self.student_module.train()
            self.teacher_module.eval()
            self.ema_target_module.eval()
            for index, (microbatch, weight) in enumerate(
                zip(batches, weights, strict=True)
            ):
                random_inputs = self.sample_random_inputs(microbatch)
                with accumulation_context(
                    self.student_module,
                    final_microbatch=index + 1 == len(batches),
                ):
                    result = self.objective.loss(
                        microbatch,
                        random_inputs=random_inputs,
                        student_num_steps=active_student_steps,
                    )
                    reported = torch.as_tensor(
                        result.metrics.get("loss_denominator"),
                        device=weight.device,
                        dtype=torch.float32,
                    )
                    if reported.numel() != 1 or not torch.equal(
                        reported.detach().reshape(()),
                        weight,
                    ):
                        raise RuntimeError(
                            "progressive declared and realized loss denominators differ"
                        )
                    gradient_weight = (
                        weight
                        / total_weight
                        * float(self.parallel_context.world_size)
                    )
                    (result.loss * gradient_weight).backward()
                results.append(result)
            if any(
                parameter.grad is not None
                for parameter in self.teacher_module.parameters()
            ):
                raise RuntimeError("progressive teacher received gradients")
            if any(
                parameter.grad is not None
                for parameter in self.ema_target_module.parameters()
            ):
                raise RuntimeError("progressive EMA target received gradients")
            grad_norm = clip_grad_norm_(
                self.student_parameters,
                self.max_grad_norm,
                error_if_nonfinite=True,
            )
            applied_learning_rates = self._apply_stage_learning_rate()
            optimizer_mutated = True
            self.optimizer.step()
            self.optimizer_steps += 1
            self.ema_updater.update()
            _copy_module_buffers(
                self.student_module,
                self.ema_target_module,
            )
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            self.stage_step += 1
            stage_finished = self._finish_stage_if_due()
            self._phase = "idle"
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True)
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
            **role_metrics(
                results,
                global_numerator=numerator,
                global_denominator=denominator,
            ),
            "global_step": self.global_step,
            "optimizer_steps": self.optimizer_steps,
            "grad_norm": grad_norm.detach(),
            "learning_rates": applied_learning_rates,
            "accumulated_microbatches": len(batches),
            "trained_stage_index": active_stage,
            "trained_student_num_steps": active_student_steps,
            "trained_teacher_num_steps": active_teacher_steps,
            "stage_finished": stage_finished,
            "training_complete": self._complete,
            "active_stage_index": self.stage_index,
            "active_stage_step": self.stage_step,
            "active_student_num_steps": self.student_num_steps,
            "active_teacher_num_steps": self.teacher_num_steps,
        }
        return ProgressiveDistillationTrainResult(
            loss=loss.detach().float(),
            metrics=metrics,
        )

    def state_dict(self) -> dict[str, object]:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError(
                "cannot checkpoint a partially committed progressive update"
            )
        return {
            "schema": PROGRESSIVE_DISTILLATION_ENGINE_STATE_SCHEMA,
            "global_step": self.global_step,
            "optimizer_steps": self.optimizer_steps,
            "stage_index": self.stage_index,
            "stage_step": self.stage_step,
            "student_num_steps": self.student_num_steps,
            "teacher_num_steps": self.teacher_num_steps,
            "complete": self._complete,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "data_parallel_size": self.parallel_context.world_size,
            "rng_device": self._rng_device,
            "rng_state": self._rng.get_state().clone(),
            "base_learning_rates": self._base_learning_rates,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("progressive engine state must be a mapping")
        expected = {
            "schema",
            "global_step",
            "optimizer_steps",
            "stage_index",
            "stage_step",
            "student_num_steps",
            "teacher_num_steps",
            "complete",
            "gradient_accumulation_steps",
            "data_parallel_size",
            "rng_device",
            "rng_state",
            "base_learning_rates",
        }
        if set(state_dict) != expected:
            raise ValueError(
                "progressive engine state fields differ from the active schema"
            )
        if state_dict["schema"] != PROGRESSIVE_DISTILLATION_ENGINE_STATE_SCHEMA:
            raise ValueError("unsupported progressive engine schema")
        active = {
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "data_parallel_size": self.parallel_context.world_size,
            "rng_device": self._rng_device,
        }
        for name, value in active.items():
            if state_dict[name] != value:
                raise ValueError(
                    f"saved progressive {name} differs from the active engine"
                )
        saved_base_learning_rates = state_dict["base_learning_rates"]
        if (
            not isinstance(saved_base_learning_rates, (tuple, list))
            or tuple(float(value) for value in saved_base_learning_rates)
            != self._base_learning_rates
        ):
            raise ValueError(
                "saved progressive base learning rates differ from the active optimizer"
            )

        def non_negative_int(name: str) -> int:
            value = state_dict[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"progressive {name} must be a non-negative integer")
            return value

        global_step = non_negative_int("global_step")
        optimizer_steps = non_negative_int("optimizer_steps")
        stage_index = non_negative_int("stage_index")
        stage_step = non_negative_int("stage_step")
        complete = state_dict["complete"]
        if not isinstance(complete, bool):
            raise TypeError("progressive complete flag must be bool")
        config = self.objective.config
        if stage_index >= config.stage_count:
            raise ValueError("progressive saved stage_index is out of range")
        if complete:
            if (
                stage_index != config.stage_count - 1
                or stage_step != config.optimizer_steps_per_stage
            ):
                raise ValueError("progressive completed stage state is inconsistent")
        elif stage_step >= config.optimizer_steps_per_stage:
            raise ValueError("progressive active stage_step is out of range")
        expected_global_step = (
            stage_index * config.optimizer_steps_per_stage + stage_step
        )
        if global_step != expected_global_step or optimizer_steps != global_step:
            raise ValueError("progressive global and stage counters are inconsistent")
        if state_dict["student_num_steps"] != config.student_steps[stage_index]:
            raise ValueError("progressive saved student_num_steps is inconsistent")
        if state_dict["teacher_num_steps"] != config.teacher_steps[stage_index]:
            raise ValueError("progressive saved teacher_num_steps is inconsistent")
        rng_state = state_dict["rng_state"]
        if not isinstance(rng_state, Tensor) or rng_state.dtype != torch.uint8:
            raise TypeError("progressive rng_state must be a uint8 tensor")

        self.optimizer.zero_grad(set_to_none=True)
        self._rng.set_state(rng_state.detach().cpu())
        self.global_step = global_step
        self.optimizer_steps = optimizer_steps
        self.stage_index = stage_index
        self.stage_step = stage_step
        self._complete = complete
        if not complete and stage_step == 0:
            self._restore_base_learning_rates()
        self._phase = "idle"
        self._poisoned = False


__all__ = [
    "NativeProgressiveDistillationTrainEngine",
    "PROGRESSIVE_DISTILLATION_ENGINE_STATE_SCHEMA",
    "ProgressiveDistillationTrainResult",
]
