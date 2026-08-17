"""Shared lifecycle for native student-distillation training runs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from worldfoundry.core.io.integrity import append_jsonl_durable, replace_json_atomic
from worldfoundry.core.time import utc_now_iso
from worldfoundry.training.checkpoint.artifacts import TrainingCheckpointArtifact
from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingState
from worldfoundry.training.distributed.parallel import DistributedTrainingContext
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.tuning.full_model import (
    DEFAULT_MAX_SHARD_SIZE_BYTES,
    FullModelArtifact,
    inspect_full_model,
)
from worldfoundry.training.tuning.peft import (
    PeftAdapterArtifact,
    PeftLoraApplication,
    inspect_peft_adapter,
)

from .artifacts import export_full_model, export_peft_application


class DistillationStudent(Protocol):
    trainable_module: object


class DistillationRoles(Protocol):
    student: DistillationStudent
    student_peft: PeftLoraApplication | None

    def runtime_identity(self) -> dict[str, object]: ...


class DistillationEngine(Protocol):
    global_step: int


class DistillationSession(Protocol):
    engine: DistillationEngine
    progress: object

    def run(self, **kwargs: object) -> object: ...

    def wait_for_checkpoints(self) -> None: ...


class StudentDistillationTrainingRun:
    """Own output, exact-resume state, metrics, and student export."""

    run_schema = "worldfoundry-student-distillation-run"
    algorithm_label = "student distillation"
    export_role_label = "distilled student"

    def __init__(
        self,
        *,
        recipe: PostTrainingRecipe,
        session: DistillationSession,
        checkpoint_state: TrainingState,
        checkpointer: TrainingCheckpointer,
        roles: DistillationRoles,
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
        self.data_identity = MappingProxyType(dict(data_identity))
        self.resume_artifact = resume_artifact
        self.distributed_context = distributed_context
        self.rank = 0 if distributed_context is None else distributed_context.rank
        self.world_size = 1 if distributed_context is None else distributed_context.world_size
        self.manifest_path = output_dir / "run.json"
        self.metrics_path = output_dir / "metrics.jsonl"
        self._summary = None
        self._exported_steps: set[int] = set()

    @property
    def is_coordinator(self) -> bool:
        return self.rank == 0

    def _manifest(
        self,
        *,
        status: str,
        max_steps: int,
        error: object | None = None,
    ) -> dict[str, object]:
        return {
            "schema": self.run_schema,
            "status": status,
            "run_id": self.recipe.run.id,
            "recipe": self.recipe.to_dict(),
            "rank_count": self.world_size,
            "data": dict(self.data_identity),
            "roles": self.roles.runtime_identity(),
            "max_steps": int(max_steps),
            "progress": self.session.progress.state_dict(),
            "summary": (
                None
                if self._summary is None
                else {name: getattr(self._summary, name) for name in self._summary.__dataclass_fields__}
            ),
            "resumed_from": (
                None
                if self.resume_artifact is None
                else {
                    "path": str(self.resume_artifact.path),
                    "global_step": self.resume_artifact.global_step,
                    "identity": dict(self.resume_artifact.identity),
                }
            ),
            "error": error,
            "updated_at": utc_now_iso(),
        }

    def run(self, *, max_steps: int) -> object:
        if self.is_coordinator:
            replace_json_atomic(
                self.manifest_path,
                self._manifest(status="running", max_steps=max_steps),
                root=self.output_dir,
            )
        try:
            self._summary = self.session.run(
                max_steps=max_steps,
                generator=self.checkpoint_state.objective_generator,
                boundary_every_steps=self.recipe.checkpoint.export_every_steps,
                boundary_sink=(self._export_student_if_due if self.recipe.checkpoint.export_every_steps else None),
            )
        except Exception as error:
            if self.is_coordinator:
                replace_json_atomic(
                    self.manifest_path,
                    self._manifest(
                        status="failed",
                        max_steps=max_steps,
                        error={"type": type(error).__name__, "message": str(error)},
                    ),
                    root=self.output_dir,
                )
            raise
        if self.is_coordinator:
            replace_json_atomic(
                self.manifest_path,
                self._manifest(status="complete", max_steps=max_steps),
                root=self.output_dir,
            )
        return self._summary

    def _record_export(
        self,
        artifact: PeftAdapterArtifact | FullModelArtifact | TrainingCheckpointArtifact,
    ) -> None:
        step = self.session.engine.global_step
        if step in self._exported_steps:
            return
        self._exported_steps.add(step)
        if not self.is_coordinator:
            return
        append_jsonl_durable(
            self.metrics_path,
            {
                "schema": "worldfoundry-trained-artifact-event",
                "global_step": step,
                "role": "student",
                "format": self.recipe.export.format,
                "path": str(artifact.path),
                "file_size_bytes": dict(artifact.file_size_bytes),
                "run_id": self.recipe.run.id,
                "recorded_at": utc_now_iso(),
            },
            root=self.output_dir,
        )

    def _export_student_artifact(
        self,
        output_dir: str | Path | None = None,
        *,
        require_complete: bool,
    ) -> PeftAdapterArtifact | FullModelArtifact | TrainingCheckpointArtifact:
        if require_complete and self._summary is None:
            raise RuntimeError(f"{self.algorithm_label} training must complete before export")
        step = self.session.engine.global_step
        export_format = self.recipe.export.format
        if export_format == "distributed-checkpoint":
            if output_dir is not None:
                raise ValueError("distributed-checkpoint export path is owned by the run checkpointer")
            # A checkpoint cadence may already be staging this exact step.
            # Join it before inspecting/saving so scheduled export cannot race
            # a second writer for the same immutable DCP destination.
            self.session.wait_for_checkpoints()
            destination = self.checkpointer.root / f"step-{step:08d}"
            artifact = (
                self.checkpointer.inspect(destination)
                if destination.exists()
                else self.checkpointer.save(self.checkpoint_state, asynchronous=False)
            )
            if not isinstance(artifact, TrainingCheckpointArtifact):
                raise RuntimeError("synchronous DCP export did not return a committed artifact")
        else:
            destination = (
                Path(output_dir or (self.output_dir / "exports" / f"step-{step:08d}" / "student"))
                .expanduser()
                .resolve()
            )
            if export_format == "peft":
                application = self.roles.student_peft
                if application is None:
                    raise RuntimeError(f"{self.algorithm_label} PEFT export requires a LoRA student")
                if destination.exists():
                    artifact = inspect_peft_adapter(destination)
                else:
                    artifact = export_peft_application(
                        application,
                        destination,
                        distributed_context=self.distributed_context,
                        role=self.export_role_label,
                    )
            elif export_format == "safetensors":
                if self.roles.student_peft is not None:
                    raise RuntimeError(f"full {self.algorithm_label} export cannot serialize an unmerged PEFT student")
                if destination.exists():
                    artifact = inspect_full_model(destination)
                else:
                    artifact = export_full_model(
                        self.roles.student.trainable_module,
                        destination,
                        distributed_context=self.distributed_context,
                        role=self.export_role_label,
                        max_shard_size_bytes=int(
                            self.recipe.export.options.get(
                                "max_shard_size_bytes",
                                DEFAULT_MAX_SHARD_SIZE_BYTES,
                            )
                        ),
                    )
            else:
                raise ValueError(f"unsupported {self.algorithm_label} export format: {export_format!r}")
        self._record_export(artifact)
        return artifact

    def _export_student_if_due(self, previous_step: int, current_step: int) -> None:
        cadence = self.recipe.checkpoint.export_every_steps
        if not cadence or current_step // cadence <= previous_step // cadence:
            return
        self._export_student_artifact(require_complete=False)

    def export_student(
        self,
        output_dir: str | Path | None = None,
    ) -> PeftAdapterArtifact | FullModelArtifact | TrainingCheckpointArtifact:
        """Export the configured student artifact."""

        return self._export_student_artifact(output_dir, require_complete=True)

    def export_student_peft(
        self,
        output_dir: str | Path | None = None,
    ) -> PeftAdapterArtifact:
        """Explicit PEFT-only compatibility surface."""

        artifact = self._export_student_artifact(output_dir, require_complete=True)
        if not isinstance(artifact, PeftAdapterArtifact):
            raise RuntimeError("configured distillation export is not PEFT")
        return artifact

    def close(self) -> None:
        if self.distributed_context is not None:
            self.distributed_context.close()


__all__ = [
    "DistillationRoles",
    "DistillationSession",
    "StudentDistillationTrainingRun",
]
