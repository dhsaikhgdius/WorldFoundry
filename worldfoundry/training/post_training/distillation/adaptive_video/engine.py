"""Atomic adaptive-video engine built on the shared DMD state machine."""

from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from ...shared.distributed import PostTrainingParallelContext
from ..dmd.engine import NativeDMDTrainEngine
from .contracts import AdaptiveVideoLossAdapter

ADAPTIVE_VIDEO_ENGINE_STATE_SCHEMA = "worldfoundry-adaptive-video-engine"


class NativeAdaptiveVideoTrainEngine(NativeDMDTrainEngine):
    """Checkpoint DMD counters and adaptive regression EMA as one commit."""

    def __init__(
        self,
        *,
        student_module: nn.Module,
        real_score_module: nn.Module,
        fake_score_module: nn.Module,
        loss_adapter: AdaptiveVideoLossAdapter,
        student_optimizer: object,
        fake_score_optimizer: object,
        generator_update_interval: int = 5,
        student_max_grad_norm: float = 10.0,
        fake_score_max_grad_norm: float = 10.0,
        gradient_accumulation_steps: int = 1,
        student_scheduler: object | None = None,
        fake_score_scheduler: object | None = None,
        student_scheduler_cadence: str = "iteration",
        student_ema: object | None = None,
        student_ema_start_step: int = 0,
        parallel_context: PostTrainingParallelContext | None = None,
    ) -> None:
        if not isinstance(loss_adapter, AdaptiveVideoLossAdapter):
            raise TypeError(
                "loss_adapter must implement AdaptiveVideoLossAdapter"
            )
        self.adaptive_loss_adapter = loss_adapter
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

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": ADAPTIVE_VIDEO_ENGINE_STATE_SCHEMA,
            "global_step": self.global_step,
            "config_digest": self.adaptive_loss_adapter.config_digest,
            "dmd_engine": super().state_dict(),
            "objective": dict(self.adaptive_loss_adapter.adaptive_state_dict()),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("adaptive video engine state must be a mapping")
        expected = {
            "schema",
            "global_step",
            "config_digest",
            "dmd_engine",
            "objective",
        }
        if set(state_dict) != expected:
            raise ValueError(
                "adaptive video engine state fields differ from the active schema"
            )
        if state_dict["schema"] != ADAPTIVE_VIDEO_ENGINE_STATE_SCHEMA:
            raise ValueError("unsupported adaptive video engine state schema")
        if state_dict["config_digest"] != self.adaptive_loss_adapter.config_digest:
            raise ValueError("saved adaptive video configuration differs")
        dmd_state = state_dict["dmd_engine"]
        objective_state = state_dict["objective"]
        if not isinstance(dmd_state, Mapping) or not isinstance(
            objective_state,
            Mapping,
        ):
            raise TypeError("adaptive video nested engine states must be mappings")
        if int(state_dict["global_step"]) != int(dmd_state.get("global_step", -1)):
            raise ValueError("adaptive video outer and DMD global steps differ")
        previous_objective = dict(self.adaptive_loss_adapter.adaptive_state_dict())
        self.adaptive_loss_adapter.load_adaptive_state_dict(objective_state)
        try:
            super().load_state_dict(dmd_state)
        except Exception:
            self.adaptive_loss_adapter.load_adaptive_state_dict(
                previous_objective
            )
            raise


__all__ = [
    "ADAPTIVE_VIDEO_ENGINE_STATE_SCHEMA",
    "NativeAdaptiveVideoTrainEngine",
]
