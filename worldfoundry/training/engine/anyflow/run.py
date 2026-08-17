"""Lifecycle and full-model export for native AnyFlow runs."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.training.checkpoint.artifacts import TrainingCheckpointArtifact
from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingState
from worldfoundry.training.distributed.parallel import DistributedTrainingContext
from worldfoundry.training.post_training.distillation.anyflow.ema import AnyFlowEMA
from worldfoundry.training.post_training.distillation.anyflow.session import (
    AnyFlowOnPolicyRunSummary,
    NativeAnyFlowOnPolicyTrainingSession,
    NativeAnyFlowPretrainingSession,
)
from worldfoundry.training.post_training.shared.session import (
    SingleOptimizerRunSummary,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.tuning.full_model import (
    DEFAULT_MAX_SHARD_SIZE_BYTES,
    FullModelArtifact,
    inspect_full_model,
)

from ..artifacts import export_full_model
from .roles import AnyFlowRoleBundle

AnyFlowSession = (
    NativeAnyFlowPretrainingSession | NativeAnyFlowOnPolicyTrainingSession
)
AnyFlowRunSummary = SingleOptimizerRunSummary | AnyFlowOnPolicyRunSummary


class AnyFlowTrainingRun:
    """Own an executable AnyFlow session, exact resume state, and export."""

    def __init__(
        self,
        *,
        recipe: PostTrainingRecipe,
        session: AnyFlowSession,
        checkpoint_state: TrainingState,
        checkpointer: TrainingCheckpointer,
        roles: AnyFlowRoleBundle,
        student_ema: AnyFlowEMA,
        output_dir: Path,
        resume_artifact: TrainingCheckpointArtifact | None,
        distributed_context: DistributedTrainingContext | None,
    ) -> None:
        self.recipe = recipe
        self.session = session
        self.checkpoint_state = checkpoint_state
        self.checkpointer = checkpointer
        self.roles = roles
        self.student_ema = student_ema
        self.output_dir = output_dir
        self.resume_artifact = resume_artifact
        self.distributed_context = distributed_context
        self.rank = 0 if distributed_context is None else distributed_context.rank
        self.world_size = (
            1 if distributed_context is None else distributed_context.world_size
        )
        self._summary: AnyFlowRunSummary | None = None

    @property
    def is_coordinator(self) -> bool:
        return self.rank == 0

    def run(self, *, max_steps: int) -> AnyFlowRunSummary:
        if isinstance(self.session, NativeAnyFlowPretrainingSession):
            summary: AnyFlowRunSummary = self.session.run(max_steps=max_steps)
        else:
            summary = self.session.run(
                max_steps=max_steps,
                generator=self.checkpoint_state.objective_generator,
            )
        self._summary = summary
        cadence = self.recipe.checkpoint.export_every_steps
        if cadence and summary.final_step % cadence == 0:
            self._export_student(require_complete=False)
        return summary

    def _export_student(
        self,
        output_dir: str | Path | None = None,
        *,
        require_complete: bool,
    ) -> FullModelArtifact | TrainingCheckpointArtifact:
        if require_complete and self._summary is None:
            raise RuntimeError("AnyFlow training must complete before export")
        step = self.session.engine.global_step
        if self.recipe.export.format == "distributed-checkpoint":
            self.session.wait_for_checkpoints()
            destination = self.checkpointer.root / f"step-{step:08d}"
            if destination.exists():
                return self.checkpointer.inspect(destination)
            artifact = self.checkpointer.save(
                self.checkpoint_state,
                asynchronous=False,
            )
            if not isinstance(artifact, TrainingCheckpointArtifact):
                raise RuntimeError("synchronous AnyFlow checkpoint export did not commit")
            return artifact
        destination = Path(
            output_dir
            or self.output_dir / "exports" / f"step-{step:08d}" / "student"
        ).expanduser().resolve()
        if destination.exists():
            return inspect_full_model(destination)
        with self.student_ema.apply_to(self.roles.student.module):
            return export_full_model(
                self.roles.student.module,
                destination,
                distributed_context=self.distributed_context,
                role="AnyFlow EMA student",
                max_shard_size_bytes=int(
                    self.recipe.export.options.get(
                        "max_shard_size_bytes",
                        DEFAULT_MAX_SHARD_SIZE_BYTES,
                    )
                ),
            )

    def export_student(
        self,
        output_dir: str | Path | None = None,
    ) -> FullModelArtifact | TrainingCheckpointArtifact:
        return self._export_student(output_dir, require_complete=True)

    def close(self) -> None:
        try:
            self.session.wait_for_checkpoints()
        finally:
            if self.distributed_context is not None:
                self.distributed_context.close()


__all__ = ["AnyFlowRunSummary", "AnyFlowTrainingRun"]
