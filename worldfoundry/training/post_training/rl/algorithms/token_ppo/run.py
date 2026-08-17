"""Executable checkpoint, resume, and export lifecycle for native token PPO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import torch
from torch import nn

from worldfoundry.training.checkpoint.artifacts import TrainingCheckpointArtifact
from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingProgress, TrainingState
from worldfoundry.training.engine.artifacts import create_run_directory, export_full_model
from worldfoundry.training.recipes.post_training.algorithms.token_ppo import (
    TokenPPOAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.tuning.full_model import (
    DEFAULT_MAX_SHARD_SIZE_BYTES,
    FullModelArtifact,
    inspect_full_model,
)

from .batching import NativeTokenPPODataLoader, TokenPPOSample
from .builder import build_native_token_ppo_training_stack
from .contracts import (
    TokenPPOReplayAdapter,
    TokenPPORolloutAdapter,
    TokenPPOTerminalRewardAdapter,
)
from .session import NativeTokenPPOTrainingSession, TokenPPOIterationResult


@dataclass(frozen=True, slots=True)
class TokenPPORunSummary:
    """Outcome of one bounded native PPO invocation."""

    initial_optimizer_step: int
    final_optimizer_step: int
    rollout_iterations: int
    optimizer_steps: int
    final_loss: float
    final_policy_loss: float
    final_value_loss: float
    final_scalar_reward_mean: float

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_optimizer_step": self.initial_optimizer_step,
            "final_optimizer_step": self.final_optimizer_step,
            "rollout_iterations": self.rollout_iterations,
            "optimizer_steps": self.optimizer_steps,
            "final_loss": self.final_loss,
            "final_policy_loss": self.final_policy_loss,
            "final_value_loss": self.final_value_loss,
            "final_scalar_reward_mean": self.final_scalar_reward_mean,
        }


class NativeTokenPPOTrainingRun:
    """Execute actor-critic rollout, PPO epochs, exact resume, and export."""

    def __init__(
        self,
        *,
        recipe: PostTrainingRecipe,
        session: NativeTokenPPOTrainingSession,
        dataloader: NativeTokenPPODataLoader,
        checkpoint_state: TrainingState,
        checkpointer: TrainingCheckpointer,
        model: nn.Module,
        output_dir: str | Path,
        resume_artifact: TrainingCheckpointArtifact | None,
    ) -> None:
        self.recipe = recipe
        self.session = session
        self.dataloader = dataloader
        self.checkpoint_state = checkpoint_state
        self.checkpointer = checkpointer
        self.model = model
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.resume_artifact = resume_artifact
        self._summary: TokenPPORunSummary | None = None

    @property
    def engine(self):
        return self.session.engine

    def run(self, *, max_iterations: int) -> TokenPPORunSummary:
        if isinstance(max_iterations, bool) or int(max_iterations) <= 0:
            raise ValueError("max_iterations must be positive")
        initial_step = self.engine.global_step
        final_result: TokenPPOIterationResult | None = None
        try:
            for _ in range(int(max_iterations)):
                final_result = self.session.train_iteration(
                    next(self.dataloader),
                    generator=self.checkpoint_state.objective_generator,
                )
        finally:
            self.session.wait_for_checkpoints()
        assert final_result is not None
        update = final_result.updates[-1]
        scalar_reward = final_result.rewards.scalar_rewards.detach().float().mean()
        self._summary = TokenPPORunSummary(
            initial_optimizer_step=initial_step,
            final_optimizer_step=self.engine.global_step,
            rollout_iterations=int(max_iterations),
            optimizer_steps=self.engine.global_step - initial_step,
            final_loss=float(update.loss.detach().float().item()),
            final_policy_loss=float(update.policy_loss.detach().float().item()),
            final_value_loss=float(update.value_loss.detach().float().item()),
            final_scalar_reward_mean=float(scalar_reward.item()),
        )
        return self._summary

    def export_actor_critic(
        self,
        output_dir: str | Path | None = None,
    ) -> FullModelArtifact | TrainingCheckpointArtifact:
        if self._summary is None:
            raise RuntimeError("token PPO training must complete before export")
        step = self.engine.global_step
        if self.recipe.export.format == "distributed-checkpoint":
            if output_dir is not None:
                raise ValueError("distributed-checkpoint export path is owned by the checkpointer")
            # A checkpoint cadence may already be staging this exact step.
            # Join it before inspecting/saving so export cannot race a second
            # writer for the same immutable DCP destination.
            self.session.wait_for_checkpoints()
            destination = self.checkpointer.root / f"step-{step:08d}"
            if destination.exists():
                return self.checkpointer.inspect(destination)
            artifact = self.checkpointer.save(self.checkpoint_state, asynchronous=False)
            if not isinstance(artifact, TrainingCheckpointArtifact):
                raise RuntimeError("synchronous PPO checkpoint export did not commit")
            return artifact

        destination = (
            Path(output_dir or self.output_dir / "exports" / f"step-{step:08d}" / "actor-critic").expanduser().resolve()
        )
        if destination.exists():
            return inspect_full_model(destination)
        return export_full_model(
            self.model,
            destination,
            distributed_context=None,
            role="token PPO actor-critic",
            max_shard_size_bytes=int(
                self.recipe.export.options.get(
                    "max_shard_size_bytes",
                    DEFAULT_MAX_SHARD_SIZE_BYTES,
                )
            ),
        )


def materialize_token_ppo_training_run(
    recipe: PostTrainingRecipe,
    *,
    rollout_adapter: TokenPPORolloutAdapter,
    replay_adapter: TokenPPOReplayAdapter,
    reward_adapter: TokenPPOTerminalRewardAdapter,
    samples: Sequence[TokenPPOSample],
    base_dir: str | Path = ".",
    output_dir: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    device: str | torch.device | None = None,
    initialization_seed: int | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeTokenPPOTrainingRun:
    """Materialize a complete single-process actor-critic PPO run."""

    if not isinstance(recipe.algorithm, TokenPPOAlgorithmSpec):
        raise TypeError("token PPO materialization requires TokenPPOAlgorithmSpec")
    if recipe.tuning.mode != "full":
        raise ValueError("token PPO materialization currently supports full tuning")
    if recipe.distributed.backend != "single":
        raise ValueError("token PPO materialization currently supports single-process runs")
    model = replay_adapter.module
    if not isinstance(model, nn.Module):
        raise TypeError("replay_adapter.module must be an nn.Module")
    options = dict(recipe.data.options)
    unknown_options = set(options) - {"batch_size"}
    if unknown_options:
        raise ValueError(f"unknown token PPO data options: {sorted(unknown_options)}")
    batch_size = int(options.get("batch_size", 1))
    if batch_size % recipe.algorithm.update_partitions:
        raise ValueError("token PPO data.options.batch_size must be divisible by algorithm.update_partitions")

    root = Path(base_dir).expanduser().resolve()
    destination = Path(output_dir or recipe.run.output_dir)
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    if destination.exists():
        if resume_checkpoint is None or not destination.is_dir():
            raise FileExistsError(f"post-training run output already exists: {destination}")
    else:
        create_run_directory(destination, None)

    if device is not None:
        model.to(torch.device(device))
    parameter = next(model.parameters(), None)
    model_device = torch.device("cpu") if parameter is None else parameter.device
    seed = recipe.data.shuffle_seed if initialization_seed is None else int(initialization_seed)
    objective_generator = torch.Generator(device=model_device).manual_seed(seed)
    stack = build_native_token_ppo_training_stack(
        recipe,
        rollout_adapter=rollout_adapter,
        replay_adapter=replay_adapter,
        reward_adapter=reward_adapter,
        initial_policy_revision=recipe.model.checkpoint,
        fused_adamw=fused_adamw,
    )
    dataloader = NativeTokenPPODataLoader(
        tuple(samples),
        batch_size=batch_size,
        policy_revision=lambda: stack.engine.current_policy_revision,
        sampling_temperature=stack.sampling_temperature,
        shuffle=recipe.data.shuffle,
        shuffle_seed=recipe.data.shuffle_seed,
        tail_policy=recipe.data.tail_policy,
    )
    progress = TrainingProgress(optimizer_steps=stack.engine.global_step)
    checkpoint_state = TrainingState(
        model=model,
        optimizer=stack.optimizer,
        engine=stack.engine,
        dataloader=dataloader,
        objective_generator=objective_generator,
        progress=progress,
        identity={
            "recipe": recipe.to_dict(),
            "token_ppo": {
                "sample_ids": [sample.sample_id for sample in samples],
                "batch_size": batch_size,
                "initialization_seed": seed,
            },
        },
        **stack.checkpoint_state_kwargs(),
    )
    checkpointer = TrainingCheckpointer(destination / "checkpoints")
    resume_artifact = None
    if resume_checkpoint is not None:
        resume_artifact = checkpointer.load(checkpoint_state, resume_checkpoint)
    session = stack.build_session(
        progress,
        checkpoint_state=checkpoint_state,
        checkpointer=checkpointer,
        save_every_steps=recipe.checkpoint.save_every_steps,
        asynchronous_checkpoints=recipe.checkpoint.async_save,
    )
    return NativeTokenPPOTrainingRun(
        recipe=recipe,
        session=session,
        dataloader=dataloader,
        checkpoint_state=checkpoint_state,
        checkpointer=checkpointer,
        model=model,
        output_dir=destination,
        resume_artifact=resume_artifact,
    )


__all__ = [
    "NativeTokenPPOTrainingRun",
    "TokenPPORunSummary",
    "materialize_token_ppo_training_run",
]
