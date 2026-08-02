"""Resumable DMD execution wrapper for Self-Gradient-Forcing randomness."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from ..dmd.contracts import DMDTrainingBatch
from ..dmd.engine import DMDTrainResult, NativeDMDTrainEngine
from .rollout import SelfGradientForcingSampler

SELF_GRADIENT_FORCING_ENGINE_STATE_SCHEMA = "worldfoundry-self-gradient-forcing-engine"


class NativeSelfGradientForcingTrainEngine:
    """Compose the native DMD engine with checkpointed replay randomness."""

    def __init__(
        self,
        dmd_engine: NativeDMDTrainEngine,
        sampler: SelfGradientForcingSampler,
    ) -> None:
        if not isinstance(dmd_engine, NativeDMDTrainEngine):
            raise TypeError("dmd_engine must be NativeDMDTrainEngine")
        if not isinstance(sampler, SelfGradientForcingSampler):
            raise TypeError("sampler must be SelfGradientForcingSampler")
        if getattr(dmd_engine.loss_adapter, "student_sampler", None) is not sampler:
            raise ValueError("DMD loss adapter must use the supplied Self-Gradient-Forcing sampler")
        if dmd_engine.student_module is not sampler.adapter.module:
            raise ValueError("DMD engine and Self-Gradient-Forcing sampler must share the student module")
        self.dmd_engine = dmd_engine
        self.sampler = sampler

    @property
    def global_step(self) -> int:
        return self.dmd_engine.global_step

    @property
    def student_optimizer_steps(self) -> int:
        return self.dmd_engine.student_optimizer_steps

    @property
    def fake_score_optimizer_steps(self) -> int:
        return self.dmd_engine.fake_score_optimizer_steps

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.dmd_engine.gradient_accumulation_steps

    @property
    def generator_update_interval(self) -> int:
        return self.dmd_engine.generator_update_interval

    @property
    def schedule_digest(self) -> str:
        return self.dmd_engine.schedule_digest

    def train_step(
        self,
        batch: DMDTrainingBatch | Sequence[DMDTrainingBatch],
        *,
        fake_score_batch: DMDTrainingBatch | Sequence[DMDTrainingBatch] | None = None,
        generator: torch.Generator | None = None,
    ) -> DMDTrainResult:
        if generator is not None and generator is not self.sampler.generator:
            raise ValueError("Self-Gradient-Forcing engine owns its checkpointed generator")
        sampler_state = self.sampler.state_dict()
        try:
            return self.dmd_engine.train_step(
                batch,
                fake_score_batch=fake_score_batch,
                generator=self.sampler.generator,
            )
        except Exception:
            self.sampler.load_state_dict(sampler_state)
            raise

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": SELF_GRADIENT_FORCING_ENGINE_STATE_SCHEMA,
            "global_step": self.global_step,
            "dmd_engine": self.dmd_engine.state_dict(),
            "sampler": self.sampler.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("Self-Gradient-Forcing engine state must be a mapping")
        if set(state_dict) != {"schema", "global_step", "dmd_engine", "sampler"}:
            raise ValueError("Self-Gradient-Forcing engine state fields differ from the active schema")
        if state_dict["schema"] != SELF_GRADIENT_FORCING_ENGINE_STATE_SCHEMA:
            raise ValueError(
                f"unsupported Self-Gradient-Forcing engine schema: {state_dict['schema']!r}"
            )
        saved_dmd = state_dict["dmd_engine"]
        saved_sampler = state_dict["sampler"]
        if not isinstance(saved_dmd, Mapping) or not isinstance(saved_sampler, Mapping):
            raise TypeError("Self-Gradient-Forcing nested engine states must be mappings")
        saved_global_step = state_dict["global_step"]
        if (
            isinstance(saved_global_step, bool)
            or not isinstance(saved_global_step, int)
            or saved_global_step < 0
        ):
            raise ValueError("saved Self-Gradient-Forcing global_step is invalid")
        if int(saved_dmd.get("global_step", -1)) != saved_global_step:
            raise ValueError("saved Self-Gradient-Forcing wrapper and DMD steps differ")
        previous_dmd = self.dmd_engine.state_dict()
        previous_sampler = self.sampler.state_dict()
        try:
            self.dmd_engine.load_state_dict(saved_dmd)
            self.sampler.load_state_dict(saved_sampler)
        except Exception:
            self.dmd_engine.load_state_dict(previous_dmd)
            self.sampler.load_state_dict(previous_sampler)
            raise


__all__ = [
    "SELF_GRADIENT_FORCING_ENGINE_STATE_SCHEMA",
    "NativeSelfGradientForcingTrainEngine",
]
