"""SANA SCM-LADD exact-resume run and student export lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from worldfoundry.core.io.integrity import replace_json_atomic
from worldfoundry.training.checkpoint.artifacts import TrainingCheckpointArtifact
from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingState
from worldfoundry.training.distributed.parallel import DistributedTrainingContext
from worldfoundry.training.post_training.distillation.scm_ladd.session import (
    NativeSCMLADDTrainingSession,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.tuning.full_model import (
    DEFAULT_MAX_SHARD_SIZE_BYTES,
    FullModelArtifact,
    inspect_full_model,
)
from worldfoundry.training.tuning.peft import PeftAdapterArtifact, inspect_peft_adapter

from ..artifacts import export_full_model, export_peft_application
from .scm_ladd_roles import SanaSCMLADDRoleBundle

SANA_SCM_LADD_RUN_SCHEMA = "worldfoundry-sana-scm-ladd-run"


class SanaSCMLADDTrainingRun:
    def __init__(
        self,
        *,
        recipe: PostTrainingRecipe,
        session: NativeSCMLADDTrainingSession,
        checkpoint_state: TrainingState,
        checkpointer: TrainingCheckpointer,
        roles: SanaSCMLADDRoleBundle,
        output_dir: Path,
        data_identity: Mapping[str, object],
        resume_artifact: TrainingCheckpointArtifact | None,
        distributed_context: DistributedTrainingContext | None,
    ) -> None:
        self.recipe = recipe
        self.session = session
        self.checkpoint_state = checkpoint_state
        self.checkpointer = checkpointer
        self.roles = roles
        self.output_dir = output_dir
        self.data_identity = dict(data_identity)
        self.resume_artifact = resume_artifact
        self.distributed_context = distributed_context
        self.rank = 0 if distributed_context is None else distributed_context.rank
        self.world_size = 1 if distributed_context is None else distributed_context.world_size
        self.manifest_path = output_dir / "run.json"
        self._summary = None

    @property
    def is_coordinator(self) -> bool:
        return self.rank == 0

    def _write_status(self, status: str, *, max_steps: int, error: Exception | None = None) -> None:
        if not self.is_coordinator:
            return
        replace_json_atomic(
            self.manifest_path,
            {
                "schema": SANA_SCM_LADD_RUN_SCHEMA,
                "status": status,
                "run_id": self.recipe.run.id,
                "recipe_digest": self.recipe.digest,
                "rank_count": self.world_size,
                "max_steps": int(max_steps),
                "progress": self.session.progress.state_dict(),
                "error": (
                    None
                    if error is None
                    else {"type": type(error).__name__, "message": str(error)}
                ),
            },
            root=self.output_dir,
        )

    def run(self, *, max_steps: int) -> object:
        self._write_status("running", max_steps=max_steps)
        try:
            self._summary = self.session.run(
                max_steps=max_steps,
                generator=self.checkpoint_state.objective_generator,
                boundary_every_steps=self.recipe.checkpoint.export_every_steps,
                boundary_sink=(
                    self._export_student_if_due
                    if self.recipe.checkpoint.export_every_steps
                    else None
                ),
            )
        except Exception as error:
            self._write_status("failed", max_steps=max_steps, error=error)
            raise
        self._write_status("complete", max_steps=max_steps)
        return self._summary

    def _export_metadata(self) -> dict[str, object]:
        return {
            "run_id": self.recipe.run.id,
            "recipe_digest": self.recipe.digest,
            "global_step": self.session.engine.global_step,
            "role": "student",
        }

    def _export_student_artifact(
        self,
        output_dir: str | Path | None = None,
        *,
        require_complete: bool,
    ) -> PeftAdapterArtifact | FullModelArtifact | TrainingCheckpointArtifact:
        if require_complete and self._summary is None:
            raise RuntimeError("SANA SCM-LADD training must complete before export")
        step = self.session.engine.global_step
        export_format = self.recipe.export.format
        if export_format == "distributed-checkpoint":
            if output_dir is not None:
                raise ValueError("distributed-checkpoint export path is owned by the run")
            destination = self.checkpointer.root / f"step-{step:08d}"
            artifact = (
                self.checkpointer.inspect(destination)
                if destination.exists()
                else self.checkpointer.save(self.checkpoint_state, asynchronous=False)
            )
            if not isinstance(artifact, TrainingCheckpointArtifact):
                raise RuntimeError("synchronous DCP export did not commit")
            return artifact

        destination = Path(
            output_dir or self.output_dir / "exports" / f"step-{step:08d}" / "student"
        ).expanduser().resolve()
        metadata = self._export_metadata()
        if export_format == "peft":
            application = self.roles.student_peft
            if application is None:
                raise RuntimeError("PEFT export requires a LoRA SCM student")
            if destination.exists():
                artifact = inspect_peft_adapter(destination)
                if dict(artifact.metadata) != metadata:
                    raise ValueError("existing SCM student PEFT artifact differs")
                return artifact
            return export_peft_application(
                application,
                destination,
                metadata=metadata,
                distributed_context=self.distributed_context,
                role="SANA SCM-LADD student",
            )
        if export_format != "safetensors":
            raise ValueError(f"unsupported SANA SCM-LADD export format: {export_format!r}")
        if self.roles.student_peft is not None:
            raise RuntimeError("full export cannot serialize an unmerged LoRA SCM student")
        if destination.exists():
            artifact = inspect_full_model(destination)
            if dict(artifact.metadata) != metadata:
                raise ValueError("existing SCM student full-model artifact differs")
            return artifact
        return export_full_model(
            self.roles.student.module,
            destination,
            metadata=metadata,
            distributed_context=self.distributed_context,
            role="SANA SCM-LADD student",
            max_shard_size_bytes=int(
                self.recipe.export.options.get(
                    "max_shard_size_bytes",
                    DEFAULT_MAX_SHARD_SIZE_BYTES,
                )
            ),
        )

    def _export_student_if_due(self, _previous_step: int, _current_step: int) -> None:
        self._export_student_artifact(require_complete=False)

    def export_student(
        self,
        output_dir: str | Path | None = None,
    ) -> PeftAdapterArtifact | FullModelArtifact | TrainingCheckpointArtifact:
        return self._export_student_artifact(output_dir, require_complete=True)

    def close(self) -> None:
        if self.distributed_context is not None:
            self.distributed_context.close()


__all__ = ["SANA_SCM_LADD_RUN_SCHEMA", "SanaSCMLADDTrainingRun"]
