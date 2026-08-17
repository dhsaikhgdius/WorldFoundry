"""Optimizer state machine for native DiffusionOPD."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from worldfoundry.core.gradient import clip_grad_norm_, get_total_norm
from worldfoundry.training.optimization import (
    audit_optimizer_parameters,
    trainable_parameters,
)

from ...shared.accumulation import accumulation_context, global_denominator
from ...shared.distributed import PostTrainingParallelContext
from .contracts import DiffusionOPDTrajectory
from .objective import DiffusionOPDLoss, diffusion_opd_loss
from .trajectory import NativeDiffusionOPDTrajectoryReplay

DIFFUSION_OPD_ENGINE_STATE_SCHEMA = "worldfoundry-diffusion-opd-engine"


@dataclass(frozen=True, slots=True)
class DiffusionOPDTrainResult:
    """One accumulated student update and per-domain diagnostics."""

    loss: torch.Tensor
    domain_losses: Mapping[str, torch.Tensor]
    gradient_norm: torch.Tensor
    sample_count: int
    transition_count: int
    microbatches: int


class NativeDiffusionOPDEngine:
    """Replay frozen teachers and update only the current student."""

    def __init__(
        self,
        *,
        student_module: nn.Module,
        student_replay: NativeDiffusionOPDTrajectoryReplay,
        teacher_replays: Mapping[str, NativeDiffusionOPDTrajectoryReplay],
        optimizer: torch.optim.Optimizer,
        add_kl_coefficient: bool,
        gradient_accumulation_steps: int,
        max_grad_norm: float | None,
        parallel_context: PostTrainingParallelContext | None = None,
    ) -> None:
        if not isinstance(student_module, nn.Module):
            raise TypeError("DiffusionOPD student_module must be nn.Module")
        if not isinstance(student_replay, NativeDiffusionOPDTrajectoryReplay):
            raise TypeError("student_replay must be NativeDiffusionOPDTrajectoryReplay")
        teachers = dict(teacher_replays)
        if not teachers or not all(
            isinstance(value, NativeDiffusionOPDTrajectoryReplay) for value in teachers.values()
        ):
            raise ValueError("DiffusionOPD teacher replay registry cannot be empty")
        if isinstance(gradient_accumulation_steps, bool) or int(gradient_accumulation_steps) <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        parameters = trainable_parameters(student_module)
        audit_optimizer_parameters(optimizer, parameters, role="DiffusionOPD student")
        context = parallel_context or PostTrainingParallelContext.current()
        context.audit_synchronized_module(student_module, role="DiffusionOPD student")
        self.student_module = student_module
        self.student_replay = student_replay
        self.teacher_replays = teachers
        self.optimizer = optimizer
        self.parameters = parameters
        self.add_kl_coefficient = bool(add_kl_coefficient)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        self.parallel_context = context
        self.global_step = 0

    def train_step(
        self,
        trajectories: Sequence[DiffusionOPDTrajectory],
    ) -> DiffusionOPDTrainResult:
        values = tuple(trajectories)
        if len(values) != self.gradient_accumulation_steps or not all(
            isinstance(value, DiffusionOPDTrajectory) for value in values
        ):
            raise ValueError(
                f"DiffusionOPD update requires exactly {self.gradient_accumulation_steps} typed trajectories"
            )
        domain_counts = {domain: 0 for domain in self.teacher_replays}
        for trajectory in values:
            if trajectory.domain not in domain_counts:
                raise ValueError(f"DiffusionOPD trajectory domain {trajectory.domain!r} has no teacher")
            domain_counts[trajectory.domain] += 1
        if len(set(domain_counts.values())) != 1 or 0 in domain_counts.values():
            raise ValueError("DiffusionOPD optimizer updates require complete balanced teacher-domain cycles")
        weights = [
            torch.tensor(
                trajectory.batch_size * trajectory.selected_steps,
                device=self.parameters[0].device,
                dtype=torch.float32,
            )
            for trajectory in values
        ]
        denominator = global_denominator(weights, self.parallel_context)
        results: list[DiffusionOPDLoss] = []
        domain_results: dict[str, list[DiffusionOPDLoss]] = {}
        self.optimizer.zero_grad(set_to_none=True)
        self.student_module.train()
        try:
            for index, (trajectory, weight) in enumerate(zip(values, weights, strict=True)):
                teacher = self.teacher_replays.get(trajectory.domain)
                assert teacher is not None
                with torch.no_grad():
                    teacher_result = teacher.replay(trajectory, training=False)
                with accumulation_context(
                    self.student_module,
                    final_microbatch=index + 1 == len(values),
                ):
                    student_result = self.student_replay.replay(
                        trajectory,
                        training=True,
                    )
                    result = diffusion_opd_loss(
                        student_result.transition_means,
                        teacher_result.transition_means,
                        trajectory.transition_scales,
                        add_kl_coefficient=self.add_kl_coefficient,
                    )
                    gradient_weight = weight / denominator * float(self.parallel_context.world_size)
                    (result.loss * gradient_weight).backward()
                results.append(result)
                domain_results.setdefault(trajectory.domain, []).append(result)
            if self.max_grad_norm is None:
                gradient_norm = get_total_norm(
                    tuple(parameter.grad for parameter in self.parameters if parameter.grad is not None),
                    error_if_nonfinite=True,
                )
            else:
                gradient_norm = clip_grad_norm_(
                    self.parameters,
                    self.max_grad_norm,
                    error_if_nonfinite=True,
                )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True)
            raise

        numerator = sum(result.loss.detach().float() * weight for result, weight in zip(results, weights, strict=True))
        statistics = torch.stack((numerator, sum(weights)))
        if self.parallel_context.world_size > 1:
            torch.distributed.all_reduce(
                statistics,
                group=self.parallel_context.process_group,
            )
        global_numerator, global_weight = statistics.unbind()
        domain_losses = {
            domain: torch.stack([result.loss.detach().float() for result in domain_values]).mean()
            for domain, domain_values in domain_results.items()
        }
        return DiffusionOPDTrainResult(
            loss=global_numerator / global_weight,
            domain_losses=domain_losses,
            gradient_norm=gradient_norm.detach(),
            sample_count=sum(value.batch_size for value in values),
            transition_count=sum(value.batch_size * value.selected_steps for value in values),
            microbatches=len(values),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": DIFFUSION_OPD_ENGINE_STATE_SCHEMA,
            "global_step": self.global_step,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping) or set(state_dict) != {
            "schema",
            "global_step",
        }:
            raise ValueError("DiffusionOPD engine state fields differ")
        if state_dict["schema"] != DIFFUSION_OPD_ENGINE_STATE_SCHEMA:
            raise ValueError("unsupported DiffusionOPD engine state schema")
        step = int(state_dict["global_step"])
        if step < 0:
            raise ValueError("DiffusionOPD global_step cannot be negative")
        self.global_step = step


__all__ = [
    "DIFFUSION_OPD_ENGINE_STATE_SCHEMA",
    "DiffusionOPDTrainResult",
    "NativeDiffusionOPDEngine",
]
