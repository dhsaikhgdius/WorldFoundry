"""Native Reward-Forcing optimizer engine."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from ...shared.distributed import PostTrainingParallelContext
from ..dmd.contracts import DMDTrainingBatch
from ..dmd.engine import DMDTrainResult, NativeDMDTrainEngine
from .objective import NativeRewardForcingLossAdapter


class NativeRewardForcingTrainEngine(NativeDMDTrainEngine):
    """DMD engine with the released Reward-Forcing evaluation semantics."""

    def __init__(
        self,
        *,
        student_module: nn.Module,
        real_score_module: nn.Module,
        fake_score_module: nn.Module,
        loss_adapter: NativeRewardForcingLossAdapter,
        student_optimizer: torch.optim.Optimizer,
        fake_score_optimizer: torch.optim.Optimizer,
        generator_update_interval: int,
        student_max_grad_norm: float,
        fake_score_max_grad_norm: float,
        gradient_accumulation_steps: int,
        student_scheduler: object | None = None,
        fake_score_scheduler: object | None = None,
        student_scheduler_cadence: str = "generator-update",
        student_ema: object | None = None,
        student_ema_start_step: int = 0,
        parallel_context: PostTrainingParallelContext | None = None,
    ) -> None:
        if not isinstance(loss_adapter, NativeRewardForcingLossAdapter):
            raise TypeError("Reward-Forcing engine requires NativeRewardForcingLossAdapter")
        super().__init__(
            student_module=student_module,
            real_score_module=real_score_module,
            fake_score_module=fake_score_module,
            loss_adapter=loss_adapter,
            student_optimizer=student_optimizer,
            fake_score_optimizer=fake_score_optimizer,
            generator_update_interval=generator_update_interval,
            student_max_grad_norm=student_max_grad_norm,
            fake_score_max_grad_norm=fake_score_max_grad_norm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            student_scheduler=student_scheduler,
            fake_score_scheduler=fake_score_scheduler,
            student_scheduler_cadence=student_scheduler_cadence,
            student_ema=student_ema,
            student_ema_start_step=student_ema_start_step,
            parallel_context=parallel_context,
        )

    def generator_update_due(self) -> bool:
        """Keep the released Reward-Forcing cadence, starting at iteration zero."""

        return self.global_step % self.generator_update_interval == 0

    @property
    def generator_update_phase(self) -> str:
        return "start-of-interval"

    def _expected_student_optimizer_steps(self, completed_iterations: int) -> int:
        if completed_iterations == 0:
            return 0
        return (completed_iterations - 1) // self.generator_update_interval + 1

    def _set_released_eval_modes(self) -> None:
        self.student_module.eval()
        self.real_score_module.eval()
        self.fake_score_module.eval()
        decoder = self.loss_adapter.reward_decoder.module
        if not isinstance(decoder, nn.Module):
            raise TypeError("Reward-Forcing reward decoder lost its nn.Module")
        decoder.eval()
        motion_module = self.loss_adapter.motion_reward.owned_module
        if motion_module is not None:
            if not isinstance(motion_module, nn.Module):
                raise TypeError("Reward-Forcing motion reward lost its nn.Module ownership")
            motion_module.eval()

    def train_step(
        self,
        batch: DMDTrainingBatch | Sequence[DMDTrainingBatch],
        *,
        fake_score_batch: DMDTrainingBatch | Sequence[DMDTrainingBatch] | None = None,
        generator: torch.Generator | None = None,
    ) -> DMDTrainResult:
        """Keep released eval-mode semantics across every optimizer boundary."""

        self._set_released_eval_modes()
        try:
            return super().train_step(
                batch,
                fake_score_batch=fake_score_batch,
                generator=generator,
            )
        finally:
            self._set_released_eval_modes()

__all__ = ["NativeRewardForcingTrainEngine"]
