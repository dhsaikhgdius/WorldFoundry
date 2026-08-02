"""Wan flow-policy run lifecycle, global metrics, checkpoints, and export."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import torch
import torch.distributed as dist

from worldfoundry.core.io.integrity import append_jsonl_durable, replace_json_atomic
from worldfoundry.core.time import utc_now_iso
from worldfoundry.training.checkpoint.artifacts import TrainingCheckpointArtifact
from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingState
from worldfoundry.training.distributed.fsdp import FSDP2Application
from worldfoundry.training.distributed.parallel import DistributedTrainingContext
from worldfoundry.training.models.wan import WanTrainAdapter
from worldfoundry.training.post_training.rl.algorithms.flow_policy.session import (
    FlowPolicyIterationResult,
    NativeFlowPolicyTrainingSession,
)
from worldfoundry.training.post_training.rl.batching import NativeFlowPolicyDataLoader
from worldfoundry.training.post_training.rl.trajectory_rewards import (
    DecodedTerminalRewardAdapter,
)
from worldfoundry.training.post_training.shared.role_checkpoints import (
    ResolvedRoleCheckpoint,
)
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

from ..artifacts import export_full_model, export_peft_application
from .roles import fsdp_identity, peft_identity

WAN_FLOW_POLICY_RUN_SCHEMA = "worldfoundry-wan-flow-policy-run"


@dataclass(frozen=True, slots=True)
class WanFlowPolicyRoleBundle:
    policy: WanTrainAdapter
    reference: WanTrainAdapter | None
    policy_checkpoint: ResolvedRoleCheckpoint
    reference_checkpoint: ResolvedRoleCheckpoint | None
    policy_peft: PeftLoraApplication | None
    policy_fsdp: FSDP2Application | None
    reference_fsdp: FSDP2Application | None

    def runtime_identity(self) -> dict[str, object]:
        return {
            "policy": {
                "checkpoint": {
                    **self.policy_checkpoint.to_dict(),
                    "digest": self.policy_checkpoint.digest,
                },
                "peft": peft_identity(self.policy_peft),
                "fsdp2": fsdp_identity(self.policy_fsdp),
            },
            "reference": (
                None
                if self.reference_checkpoint is None
                else {
                    "checkpoint": {
                        **self.reference_checkpoint.to_dict(),
                        "digest": self.reference_checkpoint.digest,
                    },
                    "peft": None,
                    "fsdp2": fsdp_identity(self.reference_fsdp),
                }
            ),
        }


@dataclass(frozen=True, slots=True)
class WanFlowPolicyRunSummary:
    initial_optimizer_step: int
    final_optimizer_step: int
    rollout_iterations: int
    policy_optimizer_steps: int
    final_policy_loss: float
    final_scalar_reward_mean: float
    final_raw_reward_means: Mapping[str, float]
    final_normalized_reward_means: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "final_raw_reward_means",
            MappingProxyType(dict(self.final_raw_reward_means)),
        )
        object.__setattr__(
            self,
            "final_normalized_reward_means",
            MappingProxyType(dict(self.final_normalized_reward_means)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_optimizer_step": self.initial_optimizer_step,
            "final_optimizer_step": self.final_optimizer_step,
            "rollout_iterations": self.rollout_iterations,
            "policy_optimizer_steps": self.policy_optimizer_steps,
            "final_policy_loss": self.final_policy_loss,
            "final_scalar_reward_mean": self.final_scalar_reward_mean,
            "final_raw_reward_means": dict(self.final_raw_reward_means),
            "final_normalized_reward_means": dict(self.final_normalized_reward_means),
        }


class WanFlowPolicyTrainingRun:
    """Own a scalable native rollout/reward/update/checkpoint lifecycle."""

    run_schema = WAN_FLOW_POLICY_RUN_SCHEMA

    def __init__(
        self,
        *,
        recipe: PostTrainingRecipe,
        session: NativeFlowPolicyTrainingSession,
        dataloader: NativeFlowPolicyDataLoader,
        checkpoint_state: TrainingState,
        checkpointer: TrainingCheckpointer,
        roles: WanFlowPolicyRoleBundle,
        reward_adapter: DecodedTerminalRewardAdapter,
        output_dir: Path,
        data_identity: Mapping[str, object],
        reward_identity: Mapping[str, object],
        resume_artifact: TrainingCheckpointArtifact | None,
        distributed_context: DistributedTrainingContext | None,
    ) -> None:
        self.recipe = recipe
        self.session = session
        self.dataloader = dataloader
        self.checkpoint_state = checkpoint_state
        self.checkpointer = checkpointer
        self.roles = roles
        self.reward_adapter = reward_adapter
        self.output_dir = output_dir
        self.data_identity = MappingProxyType(dict(data_identity))
        self.reward_identity = MappingProxyType(dict(reward_identity))
        self.resume_artifact = resume_artifact
        self.distributed_context = distributed_context
        self.rank = 0 if distributed_context is None else distributed_context.rank
        self.world_size = 1 if distributed_context is None else distributed_context.world_size
        self.manifest_path = output_dir / "run.json"
        self.metrics_path = output_dir / "metrics.jsonl"
        self._summary: WanFlowPolicyRunSummary | None = None
        self._exported_steps: set[int] = set()

    @property
    def is_coordinator(self) -> bool:
        return self.rank == 0

    def _manifest(
        self,
        *,
        status: str,
        max_iterations: int,
        error: object | None = None,
    ) -> dict[str, object]:
        return {
            "schema": self.run_schema,
            "status": status,
            "run_id": self.recipe.run.id,
            "recipe_digest": self.recipe.digest,
            "recipe": self.recipe.to_dict(),
            "rank_count": self.world_size,
            "data": dict(self.data_identity),
            "reward": dict(self.reward_identity),
            "roles": self.roles.runtime_identity(),
            "max_iterations": int(max_iterations),
            "progress": self.session.progress.state_dict(),
            "summary": None if self._summary is None else self._summary.to_dict(),
            "resumed_from": (
                None
                if self.resume_artifact is None
                else {
                    "path": str(self.resume_artifact.path),
                    "global_step": self.resume_artifact.global_step,
                    "manifest_sha256": self.resume_artifact.manifest_sha256,
                    "identity_digest": self.resume_artifact.identity_digest,
                }
            ),
            "error": error,
            "updated_at": utc_now_iso(),
        }

    def _global_iteration_metrics(self, result: object) -> dict[str, object]:

        if not isinstance(result, FlowPolicyIterationResult) or not result.updates:
            raise TypeError("flow-policy iteration result is incomplete")
        reward_ids = self.reward_adapter.reward_ids
        raw_results = self.reward_adapter.last_results
        if len(raw_results) != result.trajectory.batch_size:
            raise RuntimeError("flow-policy reward results differ from the trajectory batch")
        raw_sums = [sum(float(item.values[reward_id]) for item in raw_results) for reward_id in reward_ids]
        normalized_sums = [
            float(result.rewards.normalized_components[reward_id].sum().item()) for reward_id in reward_ids
        ]
        values = torch.tensor(
            [
                *raw_sums,
                *normalized_sums,
                float(result.rewards.scalar_rewards.sum().item()),
                float(result.updates[-1].policy_loss.item()) * result.trajectory.batch_size,
                float(result.trajectory.batch_size),
            ],
            device=result.trajectory.latents.device,
            dtype=torch.float64,
        )
        if self.world_size > 1:
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        count = float(values[-1].item())
        if count <= 0:
            raise RuntimeError("flow-policy global metric weight must be positive")
        reward_count = len(reward_ids)
        raw_means = {reward_id: float(values[index].item() / count) for index, reward_id in enumerate(reward_ids)}
        normalized_means = {
            reward_id: float(values[reward_count + index].item() / count) for index, reward_id in enumerate(reward_ids)
        }
        return {
            "raw_reward_means": raw_means,
            "normalized_reward_means": normalized_means,
            "scalar_reward_mean": float(values[2 * reward_count].item() / count),
            "policy_loss": float(values[2 * reward_count + 1].item() / count),
            "sample_count": int(count),
        }

    def run(self, *, max_iterations: int) -> WanFlowPolicyRunSummary:
        if isinstance(max_iterations, bool) or int(max_iterations) <= 0:
            raise ValueError("max_iterations must be positive")
        iterations = int(max_iterations)
        if self.is_coordinator:
            replace_json_atomic(
                self.manifest_path,
                self._manifest(status="running", max_iterations=iterations),
                root=self.output_dir,
            )
        initial_step = self.session.engine.global_step
        iterator = iter(self.dataloader)
        final_metrics: dict[str, object] | None = None
        try:
            try:
                for iteration in range(iterations):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(self.dataloader)
                        try:
                            batch = next(iterator)
                        except StopIteration as error:
                            raise RuntimeError("flow-policy dataloader is empty") from error
                    iteration_initial_step = self.session.engine.global_step
                    result = self.session.train_iteration(
                        batch,
                        generator=self.checkpoint_state.objective_generator,
                    )
                    final_metrics = self._global_iteration_metrics(result)
                    if self.is_coordinator:
                        append_jsonl_durable(
                            self.metrics_path,
                            {
                                "schema": f"worldfoundry-{self.recipe.algorithm.type}-iteration-event",
                                "rollout_iteration": iteration + 1,
                                "global_step": self.session.engine.global_step,
                                "policy_revision": self.session.engine.current_policy_revision,
                                **final_metrics,
                                "run_id": self.recipe.run.id,
                                "recipe_digest": self.recipe.digest,
                                "recorded_at": utc_now_iso(),
                            },
                            root=self.output_dir,
                        )
                    self._export_policy_if_due(
                        initial_optimizer_step=iteration_initial_step,
                        final_optimizer_step=self.session.engine.global_step,
                    )
                assert final_metrics is not None
                self._summary = WanFlowPolicyRunSummary(
                    initial_optimizer_step=initial_step,
                    final_optimizer_step=self.session.engine.global_step,
                    rollout_iterations=iterations,
                    policy_optimizer_steps=self.session.engine.global_step - initial_step,
                    final_policy_loss=float(final_metrics["policy_loss"]),
                    final_scalar_reward_mean=float(final_metrics["scalar_reward_mean"]),
                    final_raw_reward_means=final_metrics["raw_reward_means"],  # type: ignore[arg-type]
                    final_normalized_reward_means=final_metrics["normalized_reward_means"],  # type: ignore[arg-type]
                )
            finally:
                self.session.wait_for_checkpoints()
        except Exception as error:
            if self.is_coordinator:
                replace_json_atomic(
                    self.manifest_path,
                    self._manifest(
                        status="failed",
                        max_iterations=iterations,
                        error={"type": type(error).__name__, "message": str(error)},
                    ),
                    root=self.output_dir,
                )
            raise
        if self.is_coordinator:
            replace_json_atomic(
                self.manifest_path,
                self._manifest(status="complete", max_iterations=iterations),
                root=self.output_dir,
            )
        return self._summary

    def _policy_export_metadata(self) -> dict[str, object]:
        return {
            "run_id": self.recipe.run.id,
            "recipe_digest": self.recipe.digest,
            "global_step": self.session.engine.global_step,
            "role": "policy",
            "roles": self.roles.runtime_identity(),
            "data": dict(self.data_identity),
            "reward": dict(self.reward_identity),
        }

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
        file_digests = (
            dict(artifact.file_sha256)
            if isinstance(artifact, TrainingCheckpointArtifact)
            else dict(artifact.file_digests)
        )
        append_jsonl_durable(
            self.metrics_path,
            {
                "schema": "worldfoundry-trained-artifact-event",
                "global_step": step,
                "role": "policy",
                "format": self.recipe.export.format,
                "path": str(artifact.path),
                "manifest_sha256": artifact.manifest_sha256,
                "file_sha256": file_digests,
                "run_id": self.recipe.run.id,
                "recipe_digest": self.recipe.digest,
                "recorded_at": utc_now_iso(),
            },
            root=self.output_dir,
        )

    def _export_policy_artifact(
        self,
        output_dir: str | Path | None = None,
        *,
        require_complete: bool,
    ) -> PeftAdapterArtifact | FullModelArtifact | TrainingCheckpointArtifact:
        if require_complete and self._summary is None:
            raise RuntimeError("Wan policy training must complete before export")
        self.session.wait_for_checkpoints()
        step = self.session.engine.global_step
        metadata = self._policy_export_metadata()
        export_format = self.recipe.export.format
        if export_format == "distributed-checkpoint":
            if output_dir is not None:
                raise ValueError("distributed-checkpoint export path is owned by the run checkpointer")
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
                Path(output_dir or (self.output_dir / "exports" / f"step-{step:08d}" / "policy")).expanduser().resolve()
            )
            if export_format == "peft":
                application = self.roles.policy_peft
                if application is None:
                    raise RuntimeError("Wan flow-policy PEFT export requires a LoRA policy")
                if destination.exists():
                    artifact = inspect_peft_adapter(destination)
                    if dict(artifact.metadata) != metadata:
                        raise ValueError("existing flow-policy policy artifact metadata differs")
                else:
                    artifact = export_peft_application(
                        application,
                        destination,
                        metadata=metadata,
                        distributed_context=self.distributed_context,
                        role="Wan policy",
                    )
            elif export_format == "safetensors":
                if self.roles.policy_peft is not None:
                    raise RuntimeError("full flow-policy export cannot serialize an unmerged PEFT policy")
                if destination.exists():
                    artifact = inspect_full_model(destination)
                    if dict(artifact.metadata) != metadata:
                        raise ValueError("existing flow-policy policy artifact metadata differs")
                else:
                    artifact = export_full_model(
                        self.roles.policy.trainable_module,
                        destination,
                        metadata=metadata,
                        distributed_context=self.distributed_context,
                        role="Wan policy",
                        max_shard_size_bytes=int(
                            self.recipe.export.options.get(
                                "max_shard_size_bytes",
                                DEFAULT_MAX_SHARD_SIZE_BYTES,
                            )
                        ),
                    )
            else:
                raise ValueError(f"unsupported flow-policy export format: {export_format!r}")
        self._record_export(artifact)
        return artifact

    def _export_policy_if_due(
        self,
        *,
        initial_optimizer_step: int,
        final_optimizer_step: int,
    ) -> None:
        cadence = self.recipe.checkpoint.export_every_steps
        if not cadence or final_optimizer_step // cadence <= initial_optimizer_step // cadence:
            return
        self._export_policy_artifact(require_complete=False)

    def export_policy(
        self,
        output_dir: str | Path | None = None,
    ) -> PeftAdapterArtifact | FullModelArtifact | TrainingCheckpointArtifact:
        """Export the configured, digest-audited policy artifact."""

        return self._export_policy_artifact(output_dir, require_complete=True)

    def export_policy_peft(
        self,
        output_dir: str | Path | None = None,
    ) -> PeftAdapterArtifact:
        """Explicit PEFT-only compatibility surface."""

        artifact = self._export_policy_artifact(output_dir, require_complete=True)
        if not isinstance(artifact, PeftAdapterArtifact):
            raise RuntimeError("configured flow-policy export is not PEFT")
        return artifact

    def close(self) -> None:
        try:
            self.session.wait_for_checkpoints()
        finally:
            if self.distributed_context is not None:
                self.distributed_context.close()


__all__ = [
    "WAN_FLOW_POLICY_RUN_SCHEMA",
    "WanFlowPolicyRoleBundle",
    "WanFlowPolicyRunSummary",
    "WanFlowPolicyTrainingRun",
]
