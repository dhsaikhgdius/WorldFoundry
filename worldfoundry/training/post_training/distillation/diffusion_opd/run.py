"""Checkpoint, resume, and export lifecycle for native DiffusionOPD."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from worldfoundry.training.checkpoint.artifacts import TrainingCheckpointArtifact
from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingProgress, TrainingState
from worldfoundry.training.distributed.parallel import DistributedTrainingContext
from worldfoundry.training.engine.artifacts import (
    create_run_directory,
    export_full_model,
    export_peft_application,
)
from worldfoundry.training.recipes.post_training.algorithms.diffusion_opd import (
    DiffusionOPDAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.tuning.full_model import (
    DEFAULT_MAX_SHARD_SIZE_BYTES,
    FullModelArtifact,
)
from worldfoundry.training.tuning.peft import PeftAdapterArtifact, PeftLoraApplication

from .batching import NativeDiffusionOPDDataLoader
from .builder import NativeDiffusionOPDTrainingStack
from .session import DiffusionOPDIterationResult, NativeDiffusionOPDTrainingSession


@dataclass(frozen=True, slots=True)
class DiffusionOPDRunSummary:
    """Outcome of one bounded on-policy distillation invocation."""

    initial_optimizer_step: int
    final_optimizer_step: int
    iterations: int
    final_loss: float

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_optimizer_step": self.initial_optimizer_step,
            "final_optimizer_step": self.final_optimizer_step,
            "iterations": self.iterations,
            "final_loss": self.final_loss,
        }


class NativeDiffusionOPDTrainingRun:
    """Execute student rollouts, teacher replay, exact resume, and export."""

    def __init__(
        self,
        *,
        recipe: PostTrainingRecipe,
        session: NativeDiffusionOPDTrainingSession,
        checkpoint_state: TrainingState,
        checkpointer: TrainingCheckpointer,
        student_module: torch.nn.Module,
        student_tuning: PeftLoraApplication | None,
        output_dir: str | Path,
        distributed_context: DistributedTrainingContext | None,
        resume_artifact: TrainingCheckpointArtifact | None,
    ) -> None:
        self.recipe = recipe
        self.session = session
        self.checkpoint_state = checkpoint_state
        self.checkpointer = checkpointer
        self.student_module = student_module
        self.student_tuning = student_tuning
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.distributed_context = distributed_context
        self.resume_artifact = resume_artifact
        self._summary: DiffusionOPDRunSummary | None = None

    @property
    def engine(self):
        return self.session.engine

    def run(self, *, max_iterations: int) -> DiffusionOPDRunSummary:
        if isinstance(max_iterations, bool) or int(max_iterations) <= 0:
            raise ValueError("max_iterations must be positive")
        initial_step = self.engine.global_step
        final_result: DiffusionOPDIterationResult | None = None
        try:
            for _ in range(int(max_iterations)):
                final_result = self.session.train_iteration(
                    generator=self.checkpoint_state.objective_generator,
                )
        finally:
            self.session.wait_for_checkpoints()
        assert final_result is not None
        self._summary = DiffusionOPDRunSummary(
            initial_optimizer_step=initial_step,
            final_optimizer_step=self.engine.global_step,
            iterations=int(max_iterations),
            final_loss=float(final_result.update.loss.detach().float().item()),
        )
        return self._summary

    def export_student(
        self,
        output_dir: str | Path | None = None,
    ) -> PeftAdapterArtifact | FullModelArtifact | TrainingCheckpointArtifact:
        if self._summary is None:
            raise RuntimeError("DiffusionOPD training must complete before export")
        self.session.wait_for_checkpoints()
        step = self.engine.global_step
        if self.recipe.export.format == "distributed-checkpoint":
            if output_dir is not None:
                raise ValueError("distributed-checkpoint export path is owned by the checkpointer")
            destination = self.checkpointer.root / f"step-{step:08d}"
            if destination.exists():
                return self.checkpointer.inspect(destination)
            artifact = self.checkpointer.save(self.checkpoint_state, asynchronous=False)
            if not isinstance(artifact, TrainingCheckpointArtifact):
                raise RuntimeError("synchronous DiffusionOPD checkpoint export did not commit")
            return artifact

        destination = (
            Path(output_dir or self.output_dir / "exports" / f"step-{step:08d}" / "student").expanduser().resolve()
        )
        if self.recipe.export.format == "peft":
            if self.student_tuning is None:
                raise RuntimeError("PEFT export requires a LoRA DiffusionOPD student")
            return export_peft_application(
                self.student_tuning,
                destination,
                distributed_context=self.distributed_context,
                role="DiffusionOPD student",
            )
        if self.recipe.export.format == "safetensors":
            if self.student_tuning is not None:
                raise RuntimeError("full-model export cannot serialize an unmerged LoRA student")
            return export_full_model(
                self.student_module,
                destination,
                distributed_context=self.distributed_context,
                role="DiffusionOPD student",
                max_shard_size_bytes=int(
                    self.recipe.export.options.get(
                        "max_shard_size_bytes",
                        DEFAULT_MAX_SHARD_SIZE_BYTES,
                    )
                ),
            )
        raise ValueError(f"unsupported DiffusionOPD export format: {self.recipe.export.format!r}")

    def close(self) -> None:
        try:
            self.session.wait_for_checkpoints()
        finally:
            if self.distributed_context is not None:
                self.distributed_context.close()


def build_native_diffusion_opd_training_run(
    recipe: PostTrainingRecipe,
    *,
    stack: NativeDiffusionOPDTrainingStack,
    dataloader: NativeDiffusionOPDDataLoader,
    student_module: torch.nn.Module,
    student_tuning: PeftLoraApplication | None,
    objective_generator: torch.Generator,
    output_dir: str | Path,
    resume_checkpoint: str | Path | None = None,
    distributed_context: DistributedTrainingContext | None = None,
) -> NativeDiffusionOPDTrainingRun:
    """Bind materialized model roles to the complete DiffusionOPD lifecycle."""

    if not isinstance(recipe.algorithm, DiffusionOPDAlgorithmSpec):
        raise TypeError("DiffusionOPD run requires DiffusionOPDAlgorithmSpec")
    if stack.recipe != recipe:
        raise ValueError("DiffusionOPD stack and run recipes differ")
    if student_module is not stack.engine.student_module:
        raise ValueError("DiffusionOPD run student differs from the built stack")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        if resume_checkpoint is None or not destination.is_dir():
            raise FileExistsError(f"post-training run output already exists: {destination}")
    else:
        create_run_directory(destination, distributed_context)

    progress = TrainingProgress(optimizer_steps=stack.engine.global_step)
    checkpoint_state = TrainingState(
        model=student_module,
        optimizer=stack.optimizer,
        engine=stack.engine,
        dataloader=dataloader,
        objective_generator=objective_generator,
        progress=progress,
        identity={"recipe": recipe.to_dict()},
        ignore_frozen_parameters=recipe.tuning.mode == "lora",
        **stack.checkpoint_state_kwargs(),
    )
    checkpointer = TrainingCheckpointer(destination / "checkpoints")
    resume_artifact = None
    if resume_checkpoint is not None:
        resume_artifact = checkpointer.load(checkpoint_state, resume_checkpoint)
    session = NativeDiffusionOPDTrainingSession(
        sampler=stack.sampler,
        engine=stack.engine,
        dataloader=dataloader,
        progress=progress,
        checkpoint_state=checkpoint_state,
        checkpointer=checkpointer,
        save_every_steps=recipe.checkpoint.save_every_steps,
        asynchronous_checkpoints=recipe.checkpoint.async_save,
    )
    return NativeDiffusionOPDTrainingRun(
        recipe=recipe,
        session=session,
        checkpoint_state=checkpoint_state,
        checkpointer=checkpointer,
        student_module=student_module,
        student_tuning=student_tuning,
        output_dir=destination,
        distributed_context=distributed_context,
        resume_artifact=resume_artifact,
    )


__all__ = [
    "DiffusionOPDRunSummary",
    "NativeDiffusionOPDTrainingRun",
    "build_native_diffusion_opd_training_run",
]
