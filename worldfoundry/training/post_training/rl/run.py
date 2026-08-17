"""Model-neutral flow-policy training run lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from worldfoundry.training.checkpoint.artifacts import TrainingCheckpointArtifact
from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingProgress, TrainingState
from worldfoundry.training.distributed.parallel import DistributedTrainingContext
from worldfoundry.training.engine.artifacts import export_full_model, export_peft_application
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

from .algorithms.flow_policy.builder import NativeFlowPolicyTrainingStack
from .algorithms.flow_policy.session import (
    FlowPolicyIterationResult,
    NativeFlowPolicyTrainingSession,
)
from .batching import NativeFlowPolicyDataLoader
from .contracts import TrajectoryRewardAdapter


@dataclass(frozen=True, slots=True)
class FlowPolicyRunSummary:
    """Optimizer and reward outcome of one bounded run invocation."""

    initial_optimizer_step: int
    final_optimizer_step: int
    rollout_iterations: int
    policy_optimizer_steps: int
    final_policy_loss: float
    final_scalar_reward_mean: float

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_optimizer_step": self.initial_optimizer_step,
            "final_optimizer_step": self.final_optimizer_step,
            "rollout_iterations": self.rollout_iterations,
            "policy_optimizer_steps": self.policy_optimizer_steps,
            "final_policy_loss": self.final_policy_loss,
            "final_scalar_reward_mean": self.final_scalar_reward_mean,
        }


class NativeFlowPolicyTrainingRun:
    """Execute rollout, reward, exact replay, checkpoint, resume, and export."""

    def __init__(
        self,
        *,
        recipe: PostTrainingRecipe,
        session: NativeFlowPolicyTrainingSession,
        dataloader: NativeFlowPolicyDataLoader,
        checkpoint_state: TrainingState,
        checkpointer: TrainingCheckpointer,
        policy_module: torch.nn.Module,
        policy_tuning: PeftLoraApplication | None,
        output_dir: str | Path,
        distributed_context: DistributedTrainingContext | None = None,
        closeables: tuple[object, ...] = (),
    ) -> None:
        self.recipe = recipe
        self.session = session
        self.dataloader = dataloader
        self.checkpoint_state = checkpoint_state
        self.checkpointer = checkpointer
        self.policy_module = policy_module
        self.policy_tuning = policy_tuning
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.distributed_context = distributed_context
        self.closeables = tuple(closeables)
        self.rank = 0 if distributed_context is None else distributed_context.rank
        self.world_size = 1 if distributed_context is None else distributed_context.world_size
        self._summary: FlowPolicyRunSummary | None = None

    @property
    def is_coordinator(self) -> bool:
        return self.rank == 0

    def run(self, *, max_iterations: int) -> FlowPolicyRunSummary:
        if isinstance(max_iterations, bool) or int(max_iterations) <= 0:
            raise ValueError("max_iterations must be positive")
        iterations = int(max_iterations)
        initial_step = self.session.engine.global_step
        iterator = iter(self.dataloader)
        final_result: FlowPolicyIterationResult | None = None
        try:
            cadence = self.recipe.checkpoint.export_every_steps
            if cadence and initial_step > 0 and initial_step % cadence == 0:
                self._export_policy_artifact(require_complete=False)
            for _ in range(iterations):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(self.dataloader)
                    try:
                        batch = next(iterator)
                    except StopIteration as error:
                        raise RuntimeError("flow-policy dataloader is empty") from error
                previous_step = self.session.engine.global_step
                result = self.session.train_iteration(
                    batch,
                    generator=self.checkpoint_state.objective_generator,
                )
                if not isinstance(result, FlowPolicyIterationResult) or not result.updates:
                    raise RuntimeError("flow-policy iteration did not produce an optimizer update")
                final_result = result
                self._export_policy_if_due(
                    previous_step,
                    self.session.engine.global_step,
                )
        finally:
            self.session.wait_for_checkpoints()

        assert final_result is not None
        scalar_mean = final_result.rewards.scalar_rewards.detach().float().mean()
        policy_loss = final_result.updates[-1].policy_loss.detach().float()
        self._summary = FlowPolicyRunSummary(
            initial_optimizer_step=initial_step,
            final_optimizer_step=self.session.engine.global_step,
            rollout_iterations=iterations,
            policy_optimizer_steps=self.session.engine.global_step - initial_step,
            final_policy_loss=float(policy_loss.item()),
            final_scalar_reward_mean=float(scalar_mean.item()),
        )
        return self._summary

    def _export_policy_artifact(
        self,
        output_dir: str | Path | None = None,
        *,
        require_complete: bool,
    ) -> PeftAdapterArtifact | FullModelArtifact | TrainingCheckpointArtifact:
        if require_complete and self._summary is None:
            raise RuntimeError("flow-policy training must complete before export")
        self.session.wait_for_checkpoints()
        step = self.session.engine.global_step
        export_format = self.recipe.export.format
        if export_format == "distributed-checkpoint":
            if output_dir is not None:
                raise ValueError("distributed-checkpoint export path is owned by the checkpointer")
            destination = self.checkpointer.root / f"step-{step:08d}"
            if destination.exists():
                return self.checkpointer.inspect(destination)
            artifact = self.checkpointer.save(self.checkpoint_state, asynchronous=False)
            if not isinstance(artifact, TrainingCheckpointArtifact):
                raise RuntimeError("synchronous checkpoint export did not commit")
            return artifact

        destination = (
            Path(output_dir or self.output_dir / "exports" / f"step-{step:08d}" / "policy").expanduser().resolve()
        )
        if export_format == "peft":
            if self.policy_tuning is None:
                raise RuntimeError("PEFT export requires a LoRA policy")
            if destination.exists():
                return inspect_peft_adapter(destination)
            return export_peft_application(
                self.policy_tuning,
                destination,
                distributed_context=self.distributed_context,
                role="flow policy",
            )
        if export_format == "safetensors":
            if self.policy_tuning is not None:
                raise RuntimeError("full-model export cannot serialize an unmerged LoRA policy")
            if destination.exists():
                return inspect_full_model(destination)
            return export_full_model(
                self.policy_module,
                destination,
                distributed_context=self.distributed_context,
                role="flow policy",
                max_shard_size_bytes=int(
                    self.recipe.export.options.get(
                        "max_shard_size_bytes",
                        DEFAULT_MAX_SHARD_SIZE_BYTES,
                    )
                ),
            )
        raise ValueError(f"unsupported flow-policy export format: {export_format!r}")

    def _export_policy_if_due(
        self,
        previous_step: int,
        current_step: int,
    ) -> None:
        cadence = self.recipe.checkpoint.export_every_steps
        if cadence and current_step // cadence > previous_step // cadence:
            self._export_policy_artifact(require_complete=False)

    def export_policy(
        self,
        output_dir: str | Path | None = None,
    ) -> PeftAdapterArtifact | FullModelArtifact | TrainingCheckpointArtifact:
        return self._export_policy_artifact(
            output_dir,
            require_complete=True,
        )

    def close(self) -> None:
        try:
            self.session.wait_for_checkpoints()
            for resource in reversed(self.closeables):
                close = getattr(resource, "close", None)
                if callable(close):
                    close()
        finally:
            if self.distributed_context is not None:
                self.distributed_context.close()


def build_native_flow_policy_training_run(
    recipe: PostTrainingRecipe,
    *,
    stack: NativeFlowPolicyTrainingStack,
    dataloader: NativeFlowPolicyDataLoader,
    reward_adapter: TrajectoryRewardAdapter,
    policy_module: torch.nn.Module,
    policy_tuning: PeftLoraApplication | None,
    objective_generator: torch.Generator,
    output_dir: str | Path,
    resume_identity: dict[str, object],
    resume_checkpoint: str | Path | None = None,
    distributed_context: DistributedTrainingContext | None = None,
    closeables: tuple[object, ...] = (),
) -> NativeFlowPolicyTrainingRun:
    """Bind a materialized model stack to exact-resume run state."""

    progress = TrainingProgress(optimizer_steps=stack.engine.global_step)
    checkpoint_state = TrainingState(
        model=policy_module,
        optimizer=stack.optimizer,
        engine=stack.engine,
        dataloader=dataloader,
        objective_generator=objective_generator,
        progress=progress,
        identity=resume_identity,
        ignore_frozen_parameters=recipe.tuning.mode == "lora",
        **stack.checkpoint_state_kwargs(),
    )
    destination = Path(output_dir).expanduser().resolve()
    checkpointer = TrainingCheckpointer(destination / "checkpoints")
    if resume_checkpoint is not None:
        checkpointer.load(checkpoint_state, resume_checkpoint)
    session = stack.session_type(
        sampler=stack.sampler,
        reward_adapter=reward_adapter,
        scalarizer=stack.scalarizer,
        engine=stack.engine,
        progress=progress,
        sde_index_schedule=stack.sde_index_schedule,
        old_log_prob_source=stack.old_log_prob_source,
        advantage_epsilon=stack.advantage_epsilon,
        advantage_normalization=stack.advantage_normalization,
        advantage_clip_max=stack.advantage_clip_max,
        checkpoint_state=checkpoint_state,
        checkpointer=checkpointer,
        save_every_steps=recipe.checkpoint.save_every_steps,
        asynchronous_checkpoints=recipe.checkpoint.async_save,
        **stack.session_kwargs,
    )
    if not isinstance(session, NativeFlowPolicyTrainingSession):
        raise TypeError("flow-policy session factory returned an incompatible session")
    return NativeFlowPolicyTrainingRun(
        recipe=recipe,
        session=session,
        dataloader=dataloader,
        checkpoint_state=checkpoint_state,
        checkpointer=checkpointer,
        policy_module=policy_module,
        policy_tuning=policy_tuning,
        output_dir=destination,
        distributed_context=distributed_context,
        closeables=closeables,
    )


__all__ = [
    "FlowPolicyRunSummary",
    "NativeFlowPolicyTrainingRun",
    "build_native_flow_policy_training_run",
]
