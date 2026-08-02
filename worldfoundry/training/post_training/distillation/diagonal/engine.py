"""Atomic DMD execution wrapper for diagonal sampler and motion EMA state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from ..dmd.contracts import DMDTrainingBatch
from ..dmd.engine import DMDTrainResult, NativeDMDTrainEngine
from .objective import DiagonalDMDLossAdapter
from .rollout import DiagonalRolloutSampler

DIAGONAL_ENGINE_STATE_SCHEMA = "worldfoundry-diagonal-engine"


class NativeDiagonalTrainEngine:
    """Checkpoint DMD counters, rollout RNG, and motion target as one unit."""

    def __init__(
        self,
        dmd_engine: NativeDMDTrainEngine,
        sampler: DiagonalRolloutSampler,
        objective: DiagonalDMDLossAdapter,
    ) -> None:
        if not isinstance(dmd_engine, NativeDMDTrainEngine):
            raise TypeError("dmd_engine must be NativeDMDTrainEngine")
        if not isinstance(sampler, DiagonalRolloutSampler):
            raise TypeError("sampler must be DiagonalRolloutSampler")
        if not isinstance(objective, DiagonalDMDLossAdapter):
            raise TypeError("objective must be DiagonalDMDLossAdapter")
        if dmd_engine.loss_adapter is not objective:
            raise ValueError("DMD engine must use the supplied diagonal objective")
        if objective.student_sampler is not sampler:
            raise ValueError("diagonal objective must use the supplied rollout sampler")
        if dmd_engine.student_module is not sampler.adapter.module:
            raise ValueError("DMD engine and diagonal sampler must share the student module")
        engine_parallel = dmd_engine.parallel_context
        sampler_parallel = sampler.parallel_context
        if (
            engine_parallel.rank != sampler_parallel.rank
            or engine_parallel.world_size != sampler_parallel.world_size
            or engine_parallel.process_group is not sampler_parallel.process_group
        ):
            raise ValueError("DMD engine and diagonal sampler must share a data-parallel group")
        self.dmd_engine = dmd_engine
        self.sampler = sampler
        self.objective = objective

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
            raise ValueError("diagonal engine owns its checkpointed generator")
        sampler_state = self.sampler.state_dict()
        objective_state = self.objective.state_dict()
        try:
            return self.dmd_engine.train_step(
                batch,
                fake_score_batch=fake_score_batch,
                generator=self.sampler.generator,
            )
        except Exception:
            self.sampler.load_state_dict(sampler_state)
            self.objective.load_state_dict(objective_state)
            raise

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": DIAGONAL_ENGINE_STATE_SCHEMA,
            "global_step": self.global_step,
            "dmd_engine": self.dmd_engine.state_dict(),
            "sampler": self.sampler.state_dict(),
            "objective": self.objective.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("diagonal engine state must be a mapping")
        if set(state_dict) != {
            "schema",
            "global_step",
            "dmd_engine",
            "sampler",
            "objective",
        }:
            raise ValueError("diagonal engine state fields differ from the active schema")
        if state_dict["schema"] != DIAGONAL_ENGINE_STATE_SCHEMA:
            raise ValueError(f"unsupported diagonal engine schema: {state_dict['schema']!r}")
        saved_global_step = state_dict["global_step"]
        if (
            isinstance(saved_global_step, bool)
            or not isinstance(saved_global_step, int)
            or saved_global_step < 0
        ):
            raise ValueError("saved diagonal global_step is invalid")
        saved_dmd = state_dict["dmd_engine"]
        saved_sampler = state_dict["sampler"]
        saved_objective = state_dict["objective"]
        if not all(isinstance(value, Mapping) for value in (saved_dmd, saved_sampler, saved_objective)):
            raise TypeError("diagonal nested engine states must be mappings")
        if int(saved_dmd.get("global_step", -1)) != saved_global_step:
            raise ValueError("saved diagonal wrapper and DMD steps differ")
        previous_dmd = self.dmd_engine.state_dict()
        previous_sampler = self.sampler.state_dict()
        previous_objective = self.objective.state_dict()
        try:
            self.objective.load_state_dict(saved_objective)
            self.sampler.load_state_dict(saved_sampler)
            self.dmd_engine.load_state_dict(saved_dmd)
        except Exception:
            self.objective.load_state_dict(previous_objective)
            self.sampler.load_state_dict(previous_sampler)
            self.dmd_engine.load_state_dict(previous_dmd)
            raise


__all__ = ["DIAGONAL_ENGINE_STATE_SCHEMA", "NativeDiagonalTrainEngine"]
