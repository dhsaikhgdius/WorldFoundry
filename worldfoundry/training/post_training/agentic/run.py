"""Executable lifecycle for native agentic token-policy training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from worldfoundry.training.checkpoint.artifacts import TrainingCheckpointArtifact
from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingProgress, TrainingState
from worldfoundry.training.engine.artifacts import create_run_directory, export_full_model
from worldfoundry.training.post_training.rl.algorithms.token_policy.builder import (
    build_native_token_policy_training_stack,
)
from worldfoundry.training.post_training.rl.algorithms.token_policy.contracts import (
    TokenTrajectoryRewardAdapter,
)
from worldfoundry.training.post_training.shared.building import resolve_tensor_dtype
from worldfoundry.training.recipes.post_training.algorithms.token_policy import (
    TokenPolicyAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.tuning.full_model import (
    DEFAULT_MAX_SHARD_SIZE_BYTES,
    FullModelArtifact,
    inspect_full_model,
)

from .batching import AgenticPrompt, NativeAgenticPromptLoader, load_agentic_prompts
from .causal_lm import (
    AgenticChatCodec,
    CausalLMAgenticPolicyAdapter,
    CausalLMGenerationConfig,
)
from .rewards import AgenticRewardComponent, AgenticTrajectoryRewardAdapter
from .rollout import AgenticRolloutAdapter, NativeAgenticRolloutAdapter
from .session import AgenticIterationResult, NativeAgenticTrainingSession
from .tools import AgentToolExecutor


@dataclass(frozen=True, slots=True)
class AgenticRunSummary:
    """Result of one bounded invocation of an agentic training run."""

    initial_optimizer_step: int
    final_optimizer_step: int
    rollout_iterations: int
    policy_optimizer_steps: int
    completed_rollouts: int
    final_policy_loss: float
    final_scalar_reward_mean: float

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_optimizer_step": self.initial_optimizer_step,
            "final_optimizer_step": self.final_optimizer_step,
            "rollout_iterations": self.rollout_iterations,
            "policy_optimizer_steps": self.policy_optimizer_steps,
            "completed_rollouts": self.completed_rollouts,
            "final_policy_loss": self.final_policy_loss,
            "final_scalar_reward_mean": self.final_scalar_reward_mean,
        }


class NativeAgenticTrainingRun:
    """Run prompt loading, multi-turn rollout, token learning, and artifacts."""

    def __init__(
        self,
        *,
        recipe: PostTrainingRecipe,
        session: NativeAgenticTrainingSession,
        dataloader: NativeAgenticPromptLoader,
        checkpoint_state: TrainingState,
        checkpointer: TrainingCheckpointer,
        policy_module: nn.Module,
        output_dir: str | Path,
        resume_artifact: TrainingCheckpointArtifact | None,
        closeables: Sequence[object] = (),
    ) -> None:
        self.recipe = recipe
        self.session = session
        self.dataloader = dataloader
        self.checkpoint_state = checkpoint_state
        self.checkpointer = checkpointer
        self.policy_module = policy_module
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.resume_artifact = resume_artifact
        self.closeables = tuple(closeables)
        self._summary: AgenticRunSummary | None = None

    @property
    def engine(self):
        return self.session.token_policy_session.engine

    @property
    def rollout_adapter(self) -> AgenticRolloutAdapter:
        return self.session.rollout_adapter

    def close(self) -> None:
        """Release caller-selected remote runtimes or HTTP clients."""

        for resource in reversed(self.closeables):
            close = getattr(resource, "close", None)
            if callable(close):
                close()
                continue
            shutdown = getattr(resource, "shutdown", None)
            if callable(shutdown):
                shutdown()

    def run(self, *, max_iterations: int) -> AgenticRunSummary:
        if isinstance(max_iterations, bool) or int(max_iterations) <= 0:
            raise ValueError("max_iterations must be positive")
        iterations = int(max_iterations)
        initial_step = self.engine.global_step
        final_result: AgenticIterationResult | None = None
        try:
            for _ in range(iterations):
                final_result = self.session.train_iteration(
                    next(self.dataloader),
                    generator=self.checkpoint_state.objective_generator,
                )
        finally:
            self.session.wait_for_checkpoints()
        assert final_result is not None
        final_update = final_result.token_policy.updates[-1]
        scalar_reward = final_result.token_policy.rewards.scalar_rewards.detach().float().mean()
        self._summary = AgenticRunSummary(
            initial_optimizer_step=initial_step,
            final_optimizer_step=self.engine.global_step,
            rollout_iterations=iterations,
            policy_optimizer_steps=self.engine.global_step - initial_step,
            completed_rollouts=self.rollout_adapter.completed_rollouts,
            final_policy_loss=float(final_update.loss.detach().float().item()),
            final_scalar_reward_mean=float(scalar_reward.item()),
        )
        return self._summary

    def export_policy(
        self,
        output_dir: str | Path | None = None,
    ) -> FullModelArtifact | TrainingCheckpointArtifact:
        if self._summary is None:
            raise RuntimeError("agentic training must complete before export")
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
                raise RuntimeError("synchronous agentic checkpoint export did not commit")
            return artifact

        destination = (
            Path(output_dir or self.output_dir / "exports" / f"step-{step:08d}" / "policy").expanduser().resolve()
        )
        if destination.exists():
            return inspect_full_model(destination)
        return export_full_model(
            self.policy_module,
            destination,
            distributed_context=None,
            role="agentic policy",
            max_shard_size_bytes=int(
                self.recipe.export.options.get(
                    "max_shard_size_bytes",
                    DEFAULT_MAX_SHARD_SIZE_BYTES,
                )
            ),
        )


def _resolved_prompts(
    recipe: PostTrainingRecipe,
    *,
    prompts: Sequence[AgenticPrompt] | None,
    base_dir: Path,
) -> tuple[AgenticPrompt, ...]:
    if prompts is not None:
        selected = tuple(
            prompt for prompt in prompts if isinstance(prompt, AgenticPrompt) and prompt.split == recipe.data.split
        )
        if len(selected) != len(prompts):
            raise ValueError("provided agentic prompts must all match the recipe split")
        return selected
    manifest = Path(recipe.data.manifest)
    if not manifest.is_absolute():
        manifest = base_dir / manifest
    return load_agentic_prompts(manifest, split=recipe.data.split)


def materialize_agentic_training_run(
    recipe: PostTrainingRecipe,
    *,
    policy_module: nn.Module,
    codec: AgenticChatCodec,
    tool_executor: AgentToolExecutor | None = None,
    reward_components: Sequence[AgenticRewardComponent] = (),
    rollout_adapter: AgenticRolloutAdapter | None = None,
    reward_adapter: TokenTrajectoryRewardAdapter | None = None,
    closeables: Sequence[object] = (),
    prompts: Sequence[AgenticPrompt] | None = None,
    generation: CausalLMGenerationConfig | None = None,
    base_dir: str | Path = ".",
    output_dir: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    device: str | torch.device | None = None,
    initialization_seed: int | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> NativeAgenticTrainingRun:
    """Materialize a complete single-process causal-LM agentic training run."""

    if not isinstance(recipe.algorithm, TokenPolicyAlgorithmSpec):
        raise TypeError("agentic training requires a token-policy algorithm")
    if recipe.tuning.mode != "full":
        raise ValueError("agentic materialization currently supports full tuning")
    if recipe.distributed.backend != "single":
        raise ValueError("agentic materialization currently supports single-process training")
    if not isinstance(policy_module, nn.Module):
        raise TypeError("policy_module must be an nn.Module")

    options = dict(recipe.data.options)
    unknown_options = set(options) - {
        "groups_per_batch",
        "max_new_tokens",
        "max_turns",
    }
    if unknown_options:
        raise ValueError(f"unknown agentic data options: {sorted(unknown_options)}")
    groups_per_batch = int(options.get("groups_per_batch", 1))
    max_turns = int(options.get("max_turns", 8))
    resolved_generation = generation or CausalLMGenerationConfig(max_new_tokens=int(options.get("max_new_tokens", 512)))

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
        policy_module.to(torch.device(device))
    parameter = next(policy_module.parameters(), None)
    policy_device = torch.device("cpu") if parameter is None else parameter.device
    resolved_seed = recipe.data.shuffle_seed if initialization_seed is None else int(initialization_seed)
    objective_generator = torch.Generator(device=policy_device).manual_seed(resolved_seed)
    selected_prompts = _resolved_prompts(recipe, prompts=prompts, base_dir=root)

    policy_adapter = CausalLMAgenticPolicyAdapter(
        policy_module,
        codec,
        generation=resolved_generation,
        compute_dtype=resolve_tensor_dtype(recipe.runtime.param_dtype),
    )
    resolved_rollout = rollout_adapter
    if resolved_rollout is None:
        if tool_executor is None:
            raise ValueError("local Agentic rollout requires tool_executor")
        resolved_rollout = NativeAgenticRolloutAdapter(policy_adapter, tool_executor)
    elif tool_executor is not None:
        raise ValueError("tool_executor cannot be combined with an injected rollout_adapter")

    resolved_reward = reward_adapter
    if resolved_reward is None:
        resolved_reward = AgenticTrajectoryRewardAdapter(tuple(reward_components))
    elif reward_components:
        raise ValueError("reward_components cannot be combined with an injected reward_adapter")
    stack = build_native_token_policy_training_stack(
        recipe,
        rollout_adapter=resolved_rollout,
        replay_adapter=policy_adapter,
        reward_adapter=resolved_reward,
        initial_policy_revision=recipe.model.checkpoint,
        fused_adamw=fused_adamw,
    )
    dataloader = NativeAgenticPromptLoader(
        selected_prompts,
        group_size=stack.group_size,
        groups_per_batch=groups_per_batch,
        policy_revision=lambda: stack.engine.current_policy_revision,
        sampling_temperature=stack.sampling_temperature,
        max_turns=max_turns,
        shuffle=recipe.data.shuffle,
        shuffle_seed=recipe.data.shuffle_seed,
        tail_policy=recipe.data.tail_policy,
    )
    progress = TrainingProgress(optimizer_steps=stack.engine.global_step)
    checkpoint_state = TrainingState(
        model=policy_module,
        optimizer=stack.optimizer,
        engine=stack.engine,
        dataloader=dataloader,
        objective_generator=objective_generator,
        progress=progress,
        identity={
            "recipe": recipe.to_dict(),
            "agentic": {
                "prompt_ids": [prompt.prompt_id for prompt in selected_prompts],
                "groups_per_batch": groups_per_batch,
                "max_turns": max_turns,
                "max_new_tokens": resolved_generation.max_new_tokens,
                "initialization_seed": resolved_seed,
            },
        },
        **stack.checkpoint_state_kwargs(),
    )
    checkpointer = TrainingCheckpointer(destination / "checkpoints")
    resume_artifact = None
    if resume_checkpoint is not None:
        resume_artifact = checkpointer.load(checkpoint_state, resume_checkpoint)
    token_session = stack.build_session(
        progress,
        checkpoint_state=checkpoint_state,
        checkpointer=checkpointer,
        save_every_steps=recipe.checkpoint.save_every_steps,
        asynchronous_checkpoints=recipe.checkpoint.async_save,
    )
    return NativeAgenticTrainingRun(
        recipe=recipe,
        session=NativeAgenticTrainingSession(resolved_rollout, token_session),
        dataloader=dataloader,
        checkpoint_state=checkpoint_state,
        checkpointer=checkpointer,
        policy_module=policy_module,
        output_dir=destination,
        resume_artifact=resume_artifact,
        closeables=closeables,
    )


__all__ = [
    "AgenticRunSummary",
    "NativeAgenticTrainingRun",
    "materialize_agentic_training_run",
]
