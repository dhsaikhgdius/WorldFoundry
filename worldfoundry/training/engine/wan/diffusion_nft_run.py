"""Wan DiffusionNFT run lifecycle, metrics, checkpoints, and export."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from worldfoundry.core.io.integrity import append_jsonl_durable, replace_json_atomic
from worldfoundry.core.time import utc_now_iso
from worldfoundry.training.models.wan import WanTrainAdapter
from worldfoundry.training.post_training.rl.algorithms.diffusion_nft.session import (
    DiffusionNFTIterationResult,
)
from worldfoundry.training.post_training.shared.role_checkpoints import (
    ResolvedRoleCheckpoint,
)
from worldfoundry.training.tuning.peft import PeftLoraApplication

from .flow_policy_run import WanFlowPolicyTrainingRun
from .roles import peft_identity

WAN_DIFFUSION_NFT_RUN_SCHEMA = "worldfoundry-wan-diffusion-nft-run"


@dataclass(frozen=True, slots=True)
class WanDiffusionNFTRoleBundle:
    """Independent policy, behavior-policy, and optional reference roles."""

    policy: WanTrainAdapter
    old_policy: WanTrainAdapter
    reference: WanTrainAdapter | None
    policy_checkpoint: ResolvedRoleCheckpoint
    old_policy_checkpoint: ResolvedRoleCheckpoint
    reference_checkpoint: ResolvedRoleCheckpoint | None
    policy_peft: PeftLoraApplication | None
    old_policy_peft: PeftLoraApplication | None

    def runtime_identity(self) -> dict[str, object]:
        def checkpoint(role: ResolvedRoleCheckpoint) -> dict[str, object]:
            return role.to_dict()

        return {
            "policy": {
                "checkpoint": checkpoint(self.policy_checkpoint),
                "peft": peft_identity(self.policy_peft),
                "fsdp2": None,
            },
            "old_policy": {
                "checkpoint": checkpoint(self.old_policy_checkpoint),
                "peft": peft_identity(self.old_policy_peft),
                "fsdp2": None,
            },
            "reference": (
                None
                if self.reference_checkpoint is None
                else {
                    "checkpoint": checkpoint(self.reference_checkpoint),
                    "peft": None,
                    "fsdp2": None,
                }
            ),
        }


@dataclass(frozen=True, slots=True)
class WanDiffusionNFTRunSummary:
    initial_optimizer_step: int
    final_optimizer_step: int
    collection_iterations: int
    final_loss: float
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
            "collection_iterations": self.collection_iterations,
            "final_loss": self.final_loss,
            "final_policy_loss": self.final_policy_loss,
            "final_scalar_reward_mean": self.final_scalar_reward_mean,
            "final_raw_reward_means": dict(self.final_raw_reward_means),
            "final_normalized_reward_means": dict(self.final_normalized_reward_means),
        }


class WanDiffusionNFTTrainingRun(WanFlowPolicyTrainingRun):
    """Own the single-device Wan terminal-collection training lifecycle."""

    run_schema = WAN_DIFFUSION_NFT_RUN_SCHEMA

    def _iteration_metrics(
        self,
        result: DiffusionNFTIterationResult,
    ) -> dict[str, object]:
        if result.scalarization is None or result.reward_components is None:
            raise TypeError("Wan DiffusionNFT requires terminal reward scalarization")
        reward_ids = self.reward_adapter.reward_ids
        raw_results = self.reward_adapter.last_results
        if len(raw_results) != result.rollout.batch_size:
            raise RuntimeError("DiffusionNFT reward results differ from the rollout batch")
        raw_means = {
            reward_id: sum(float(item.values[reward_id]) for item in raw_results) / result.rollout.batch_size
            for reward_id in reward_ids
        }
        normalized_means: dict[str, float] = {}
        for reward_id in reward_ids:
            value = result.scalarization.normalized_components[reward_id]
            if not isinstance(value, torch.Tensor):
                raise TypeError("normalized DiffusionNFT rewards must be tensors")
            normalized_means[reward_id] = float(value.float().mean().item())
        scalar_rewards = result.scalarization.scalar_rewards
        if not isinstance(scalar_rewards, torch.Tensor):
            raise TypeError("scalar DiffusionNFT rewards must be a tensor")
        return {
            "raw_reward_means": raw_means,
            "normalized_reward_means": normalized_means,
            "scalar_reward_mean": float(scalar_rewards.float().mean().item()),
            "loss": float(result.update.loss.item()),
            "policy_loss": float(result.update.policy_loss.item()),
            "sample_count": result.rollout.batch_size,
            "collection_policy_revision": result.rollout.policy_revision,
            "next_collection_policy_revision": (self.session.engine.current_collection_policy_revision),
        }

    def run(self, *, max_iterations: int) -> WanDiffusionNFTRunSummary:
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
                for iteration_index in range(iterations):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(self.dataloader)
                        try:
                            batch = next(iterator)
                        except StopIteration as error:
                            raise RuntimeError("DiffusionNFT dataloader is empty") from error
                    iteration_initial_step = self.session.engine.global_step
                    result = self.session.train_iteration(
                        batch,
                        generator=self.checkpoint_state.objective_generator,
                    )
                    final_metrics = self._iteration_metrics(result)
                    if self.is_coordinator:
                        append_jsonl_durable(
                            self.metrics_path,
                            {
                                "schema": "worldfoundry-diffusion-nft-iteration-event",
                                "collection_iteration": iteration_index + 1,
                                "global_step": self.session.engine.global_step,
                                **final_metrics,
                                "run_id": self.recipe.run.id,
                                "recorded_at": utc_now_iso(),
                            },
                            root=self.output_dir,
                        )
                    self._export_policy_if_due(
                        initial_optimizer_step=iteration_initial_step,
                        final_optimizer_step=self.session.engine.global_step,
                    )
                assert final_metrics is not None
                self._summary = WanDiffusionNFTRunSummary(
                    initial_optimizer_step=initial_step,
                    final_optimizer_step=self.session.engine.global_step,
                    collection_iterations=iterations,
                    final_loss=float(final_metrics["loss"]),
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


__all__ = [
    "WAN_DIFFUSION_NFT_RUN_SCHEMA",
    "WanDiffusionNFTRoleBundle",
    "WanDiffusionNFTRunSummary",
    "WanDiffusionNFTTrainingRun",
]
