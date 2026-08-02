"""Exact-resume engine identity for native Reward-Forcing."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import torch
from torch import nn

from ...shared.distributed import PostTrainingParallelContext
from ..dmd.contracts import DMDTrainingBatch
from ..dmd.engine import DMDTrainResult, NativeDMDTrainEngine
from .objective import NativeRewardForcingLossAdapter

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class NativeRewardForcingTrainEngine(NativeDMDTrainEngine):
    """DMD engine whose resume gate covers all Reward-Forcing behavior.

    The shared DMD engine records ``schedule_digest`` at every checkpoint.  A
    Reward-Forcing run also depends on its two optimizer configurations, so
    this specialization exposes the builder's composite execution digest at
    that gate rather than accepting a checkpoint with silently different
    learning rates, clipping, or accumulation.
    """

    def __init__(
        self,
        *,
        execution_digest: str,
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
        digest = str(execution_digest).strip().lower()
        if _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError("Reward-Forcing execution_digest must be 64 lowercase hex")
        if not isinstance(loss_adapter, NativeRewardForcingLossAdapter):
            raise TypeError("Reward-Forcing engine requires NativeRewardForcingLossAdapter")
        self.execution_digest = digest
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

    @property
    def schedule_digest(self) -> str:
        return self.execution_digest

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

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Name a composite-behavior mismatch before the shared DMD checks."""

        if (
            isinstance(state_dict, Mapping)
            and "schedule_digest" in state_dict
            and state_dict["schedule_digest"] != self.execution_digest
        ):
            raise ValueError("saved Reward-Forcing execution behavior differs from the active engine")
        super().load_state_dict(state_dict)


__all__ = ["NativeRewardForcingTrainEngine"]
