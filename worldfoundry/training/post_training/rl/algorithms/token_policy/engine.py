"""Algorithm-neutral optimizer state machine for packed token policies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from uuid import uuid4

import torch
import torch.distributed as dist
from torch import nn

from worldfoundry.core.gradient import clip_grad_norm_
from worldfoundry.core.io.integrity import canonical_sha256
from worldfoundry.training.optimization import (
    audit_optimizer_parameters,
    trainable_parameters,
)

from ....shared.distributed import PostTrainingParallelContext
from ....shared.partitioning import balanced_contiguous_partitions
from ....shared.validation import non_negative_int, positive_float
from ...objectives.group_advantages import normalize_data_parallel_grouped_advantages
from .contracts import (
    PackedTokenReplayBatch,
    PackedTokenTrajectory,
    TokenPolicyReplayAdapter,
    TokenReplayResult,
)
from .packing import slice_packed_token_trajectory
from .stages import TokenPolicyStage, TokenPolicyStageLoss

TOKEN_POLICY_ENGINE_STATE_SCHEMA = "worldfoundry-token-policy-engine"


@dataclass(frozen=True, slots=True)
class TokenPolicyStepResult:
    """One optimizer partition from a frozen packed-token anchor."""

    loss: torch.Tensor
    metrics: Mapping[str, object]
    trajectory_complete: bool
    optimizer_committed: bool
    replay_microbatches: int
    sample_count: int
    token_count: int


@dataclass(frozen=True, slots=True)
class _TokenPolicyAnchor:
    old_log_probs: torch.Tensor
    advantages: torch.Tensor


class NativeTokenPolicyEngine:
    """Own old-policy anchors, optimizer partitions, accumulation, and resume.

    ``updates_per_trajectory`` splits the rank-local rollout shard into that
    many balanced contiguous optimizer partitions.  Every sample participates
    in one step only.  ``replay_microbatch_size`` chunks the current partition
    for gradient accumulation and never repeats the complete trajectory.
    """

    def __init__(
        self,
        replay_adapter: TokenPolicyReplayAdapter,
        optimizer: torch.optim.Optimizer,
        *,
        algorithm: TokenPolicyStage,
        initial_policy_revision: str,
        old_log_prob_source: str = "rollout",
        max_grad_norm: float = 1.0,
        updates_per_trajectory: int = 1,
        first_update_log_ratio_tolerance: float = 1.0e-5,
        parallel_context: PostTrainingParallelContext | None = None,
        replay_microbatch_size: int | None = None,
    ) -> None:
        if not isinstance(replay_adapter, TokenPolicyReplayAdapter):
            raise TypeError("replay_adapter must implement TokenPolicyReplayAdapter")
        if not isinstance(replay_adapter.module, nn.Module):
            raise TypeError("replay_adapter.module must be an nn.Module")
        if not isinstance(algorithm, TokenPolicyStage):
            raise TypeError("algorithm must implement TokenPolicyStage")
        if not isinstance(initial_policy_revision, str) or not initial_policy_revision.strip():
            raise ValueError("initial_policy_revision must be a non-empty string")
        source = str(old_log_prob_source).strip().lower()
        if source not in {"rollout", "replay"}:
            raise ValueError("old_log_prob_source must be 'rollout' or 'replay'")
        updates = non_negative_int(
            updates_per_trajectory,
            field_name="updates_per_trajectory",
        )
        if updates == 0:
            raise ValueError("updates_per_trajectory must be positive")
        if updates > 1 and not algorithm.supports_multi_update:
            raise ValueError(f"{algorithm.name} does not support multiple updates per trajectory")
        if replay_microbatch_size is not None and (
            isinstance(replay_microbatch_size, bool) or int(replay_microbatch_size) <= 0
        ):
            raise ValueError("replay_microbatch_size must be a positive integer")
        anchor_tolerance = float(first_update_log_ratio_tolerance)
        if not isfinite(anchor_tolerance) or anchor_tolerance < 0:
            raise ValueError("first_update_log_ratio_tolerance must be finite and non-negative")

        parameters = trainable_parameters(replay_adapter.module)
        audit_optimizer_parameters(
            optimizer,
            parameters,
            role=f"{algorithm.name} policy",
        )
        self.replay_adapter = replay_adapter
        self.optimizer = optimizer
        self.algorithm = algorithm
        self.parameters = parameters
        self.initial_policy_revision = initial_policy_revision
        self.old_log_prob_source = source
        self.max_grad_norm = positive_float(
            max_grad_norm,
            field_name="max_grad_norm",
        )
        self.updates_per_trajectory = updates
        self.first_update_log_ratio_tolerance = anchor_tolerance
        self.parallel_context = parallel_context or PostTrainingParallelContext.current()
        self.replay_microbatch_size = None if replay_microbatch_size is None else int(replay_microbatch_size)
        self.parallel_context.audit_synchronized_module(
            replay_adapter.module,
            role=f"{algorithm.name} policy",
        )
        self.global_step = 0
        self._active_trajectory: PackedTokenTrajectory | None = None
        self._active_anchor: _TokenPolicyAnchor | None = None
        self._active_id: str | None = None
        self._active_updates = 0
        self._poisoned = False

    def _policy_revision_for_step(self, optimizer_steps: int) -> str:
        if optimizer_steps == 0:
            return self.initial_policy_revision
        return canonical_sha256(
            {
                "schema": "worldfoundry-logical-token-policy-revision",
                "initial_policy_revision": self.initial_policy_revision,
                "optimizer_steps": optimizer_steps,
            }
        )

    def _ensure_healthy(self) -> None:
        if self._poisoned:
            raise RuntimeError(
                "token-policy engine is poisoned after optimizer.step began; "
                "restore the complete training state into a fresh engine"
            )

    @property
    def current_policy_revision(self) -> str:
        return self._policy_revision_for_step(self.global_step)

    @property
    def is_poisoned(self) -> bool:
        return self._poisoned

    @property
    def has_active_trajectory(self) -> bool:
        return self._active_trajectory is not None

    def _optimizer_partitions(self, batch_size: int) -> tuple[tuple[int, int], ...]:
        if self.updates_per_trajectory > batch_size:
            raise ValueError("updates_per_trajectory cannot exceed trajectory batch_size")
        return balanced_contiguous_partitions(
            batch_size,
            self.updates_per_trajectory,
        )

    def _microbatch_ranges(self, start: int, end: int):
        size = self.replay_microbatch_size or (end - start)
        for chunk_start in range(start, end, size):
            yield chunk_start, min(end, chunk_start + size)

    def _replay_chunk(
        self,
        chunk: PackedTokenReplayBatch,
        *,
        training: bool,
    ) -> TokenReplayResult:
        replay = self.replay_adapter.replay(chunk, training=training)
        if not isinstance(replay, TokenReplayResult):
            raise TypeError("token replay must return TokenReplayResult")
        if int(replay.log_probs.shape[0]) != chunk.token_count:
            raise ValueError("replayed log_probs do not match the packed token span")
        if replay.sampling_temperature != chunk.sampling_temperature:
            raise ValueError("token replay used a different sampling temperature than rollout")
        return replay

    def _capture_replay_anchor(
        self,
        trajectory: PackedTokenTrajectory,
    ) -> torch.Tensor:
        captured: list[torch.Tensor] = []
        with torch.no_grad():
            for partition_start, partition_end in self._optimizer_partitions(trajectory.batch_size):
                for start, end in self._microbatch_ranges(
                    partition_start,
                    partition_end,
                ):
                    chunk = slice_packed_token_trajectory(trajectory, start, end)
                    if chunk.token_count == 0:
                        continue
                    captured.append(self._replay_chunk(chunk, training=False).log_probs.detach().float())
        if not captured:
            return trajectory.old_log_probs.detach().float().clone()
        return torch.cat(captured, dim=0)

    def prepare_trajectory(
        self,
        trajectory: PackedTokenTrajectory,
        rewards: torch.Tensor,
        *,
        advantage_epsilon: float = 1.0e-8,
        advantage_normalization: str = "group-population-variance",
        advantage_clip_max: float | None = None,
    ) -> str:
        """Freeze old log-probabilities and grouped advantages before updates."""

        self._ensure_healthy()
        if self.has_active_trajectory:
            raise RuntimeError("finish the active trajectory before preparing another")
        if not isinstance(trajectory, PackedTokenTrajectory):
            raise TypeError("trajectory must be a PackedTokenTrajectory")
        if trajectory.policy_revision != self.current_policy_revision:
            raise ValueError("trajectory policy revision differs from the active pre-update policy")
        self.parallel_context.audit_local_group_ownership(trajectory.group_ids)
        if (
            not isinstance(rewards, torch.Tensor)
            or tuple(rewards.shape) != (trajectory.batch_size,)
            or not rewards.is_floating_point()
        ):
            raise ValueError("rewards must be a floating tensor with shape [B]")
        partitions = self._optimizer_partitions(trajectory.batch_size)
        if trajectory.token_count == 0:
            if self.updates_per_trajectory != 1:
                raise ValueError("an all-empty token trajectory requires updates_per_trajectory=1")
        else:
            for start, end in partitions:
                if int(trajectory.lengths[start:end].sum().item()) == 0:
                    raise ValueError("every token-policy optimizer partition must contain at least one response token")

        old_log_probs = (
            trajectory.old_log_probs.detach().float().clone()
            if self.old_log_prob_source == "rollout"
            else self._capture_replay_anchor(trajectory)
        )
        device = self.parameters[0].device
        advantages = normalize_data_parallel_grouped_advantages(
            rewards.to(device=device),
            trajectory.group_ids,
            parallel_context=self.parallel_context,
            epsilon=advantage_epsilon,
            clip_max=advantage_clip_max,
            normalization=advantage_normalization,
        ).advantages.detach()
        self._active_trajectory = trajectory
        self._active_anchor = _TokenPolicyAnchor(
            old_log_probs=old_log_probs,
            advantages=advantages,
        )
        self._active_id = uuid4().hex
        self._active_updates = 0
        return self._active_id

    def _audit_distributed_backward_calls(self, local_count: int) -> None:
        if self.parallel_context.world_size == 1:
            return
        local = torch.tensor(
            local_count,
            device=self.parameters[0].device,
            dtype=torch.int64,
        )
        gathered = [torch.zeros_like(local) for _ in range(self.parallel_context.world_size)]
        dist.all_gather(
            gathered,
            local,
            group=self.parallel_context.process_group,
        )
        if len({int(value.item()) for value in gathered}) != 1:
            raise ValueError(
                "distributed token replay requires the same backward-call count "
                "for the current optimizer partition on every rank"
            )

    def _empty_step(self, *, anchor_id: str) -> TokenPolicyStepResult:
        zero = torch.zeros((), device=self.parameters[0].device)
        assert self._active_trajectory is not None
        sample_count = self._active_trajectory.batch_size
        metrics: dict[str, object] = {
            "global_step": torch.tensor(
                self.global_step,
                device=zero.device,
                dtype=torch.int64,
            ),
            "sample_count": torch.tensor(
                sample_count,
                device=zero.device,
                dtype=torch.int64,
            ),
            "token_count": torch.zeros((), device=zero.device, dtype=torch.int64),
            "replay_microbatches": torch.zeros(
                (),
                device=zero.device,
                dtype=torch.int64,
            ),
        }
        self.finish_trajectory(anchor_id=anchor_id)
        return TokenPolicyStepResult(
            loss=zero,
            metrics=metrics,
            trajectory_complete=True,
            optimizer_committed=False,
            replay_microbatches=0,
            sample_count=sample_count,
            token_count=0,
        )

    def train_step(self, *, anchor_id: str) -> TokenPolicyStepResult:
        self._ensure_healthy()
        if not self.has_active_trajectory:
            raise RuntimeError("no token-policy trajectory is prepared")
        if anchor_id != self._active_id:
            raise ValueError("token-policy anchor does not belong to this engine")
        if self._active_updates >= self.updates_per_trajectory:
            raise RuntimeError("active trajectory exhausted its configured updates")
        assert self._active_trajectory is not None
        assert self._active_anchor is not None
        trajectory = self._active_trajectory
        anchor = self._active_anchor
        partition_start, partition_end = self._optimizer_partitions(trajectory.batch_size)[self._active_updates]
        microbatch_ranges = tuple(self._microbatch_ranges(partition_start, partition_end))
        replay_ranges = tuple(
            (start, end)
            for start, end in microbatch_ranges
            if slice_packed_token_trajectory(trajectory, start, end).token_count > 0
        )
        self._audit_distributed_backward_calls(len(replay_ranges))
        if trajectory.token_count == 0:
            return self._empty_step(anchor_id=anchor_id)

        partition_lengths = trajectory.lengths[partition_start:partition_end]
        total_weight = self.algorithm.loss_weight(partition_lengths)
        if total_weight <= 0:
            raise ValueError("token-policy reduction selected no trainable units")
        distributed_scale = self.parallel_context.scale_local_mean(
            torch.ones((), device=self.parameters[0].device),
            total_weight,
        ).detach()

        self.optimizer.zero_grad(set_to_none=True)
        numerator_sum = torch.zeros((), device=self.parameters[0].device)
        metric_sums: dict[str, torch.Tensor] = {}
        metric_weights: dict[str, int] = {}
        ratio_values: list[torch.Tensor] = []
        replay_microbatches = 0
        optimizer_step_started = False
        try:
            for start, end in replay_ranges:
                chunk = slice_packed_token_trajectory(trajectory, start, end)
                replay = self._replay_chunk(chunk, training=True)
                old_log_probs = anchor.old_log_probs[chunk.token_start : chunk.token_end].to(
                    device=replay.log_probs.device,
                    dtype=replay.log_probs.dtype,
                )
                if self._active_updates == 0 and self.old_log_prob_source == "replay":
                    actual_anchor_error = (replay.log_probs.detach().float() - old_log_probs.detach().float()).abs()
                    if not bool((actual_anchor_error <= self.first_update_log_ratio_tolerance).all()):
                        raise ValueError(
                            "first differentiable token-policy replay must match its frozen old-log-probability anchor"
                        )
                advantages = anchor.advantages[chunk.sequence_start : chunk.sequence_end]
                stage_loss = self.algorithm.loss(
                    replay.log_probs,
                    old_log_probs,
                    advantages,
                    chunk.lengths,
                    optimizer_step=self.global_step,
                )
                if not isinstance(stage_loss, TokenPolicyStageLoss):
                    raise TypeError("TokenPolicyStage.loss must return TokenPolicyStageLoss")
                expected_weight = self.algorithm.loss_weight(chunk.lengths)
                if stage_loss.denominator != expected_weight:
                    raise ValueError("token stage returned an inconsistent denominator")
                if not bool(torch.isfinite(stage_loss.numerator.detach())):
                    raise FloatingPointError("non-finite token-policy loss")
                (stage_loss.numerator / float(total_weight) * distributed_scale).backward()
                numerator_sum += stage_loss.numerator.detach()
                ratio = stage_loss.ratio.detach()
                ratio_values.append(ratio)
                metric_weight = int(ratio.numel())
                for name, value in stage_loss.metrics.items():
                    if not isinstance(name, str) or not name:
                        raise ValueError("token stage metric names must be non-empty")
                    if not torch.is_tensor(value) or value.numel() != 1:
                        raise ValueError(f"token stage metric {name!r} must be a scalar tensor")
                    if not bool(torch.isfinite(value.detach())):
                        raise FloatingPointError(f"non-finite token stage metric: {name}")
                    metric_sums[name] = (
                        metric_sums.get(
                            name,
                            torch.zeros_like(numerator_sum),
                        )
                        + value.detach() * metric_weight
                    )
                    metric_weights[name] = metric_weights.get(name, 0) + metric_weight
                replay_microbatches += 1
            if replay_microbatches == 0:
                raise RuntimeError("non-empty trajectory produced no replay microbatches")
            grad_norm = clip_grad_norm_(
                self.parameters,
                self.max_grad_norm,
                error_if_nonfinite=True,
            )
            optimizer_step_started = True
            self.optimizer.step()
            self.global_step += 1
            self._active_updates += 1

            loss = numerator_sum / float(total_weight)
            ratios = torch.cat(ratio_values)
            ratio_std = (
                ratios.std() if ratios.numel() > 1 else torch.zeros((), device=ratios.device, dtype=ratios.dtype)
            )
            complete = self._active_updates == self.updates_per_trajectory
            sample_count = partition_end - partition_start
            token_count = int(partition_lengths.sum().item())
            metrics: dict[str, object] = {
                "global_step": torch.tensor(
                    self.global_step,
                    device=loss.device,
                    dtype=torch.int64,
                ),
                "trajectory_update": torch.tensor(
                    self._active_updates,
                    device=loss.device,
                    dtype=torch.int64,
                ),
                "ratio_mean": ratios.mean(),
                "ratio_std": ratio_std,
                "ratio_min": ratios.min(),
                "ratio_max": ratios.max(),
                "sample_count": torch.tensor(
                    sample_count,
                    device=loss.device,
                    dtype=torch.int64,
                ),
                "token_count": torch.tensor(
                    token_count,
                    device=loss.device,
                    dtype=torch.int64,
                ),
                "replay_microbatches": torch.tensor(
                    replay_microbatches,
                    device=loss.device,
                    dtype=torch.int64,
                ),
                **{name: value / metric_weights[name] for name, value in metric_sums.items()},
                "grad_norm": grad_norm.detach(),
            }
            result = TokenPolicyStepResult(
                loss=loss,
                metrics=metrics,
                trajectory_complete=complete,
                optimizer_committed=True,
                replay_microbatches=replay_microbatches,
                sample_count=sample_count,
                token_count=token_count,
            )
            if complete:
                self.finish_trajectory(anchor_id=anchor_id)
            return result
        except Exception:
            if optimizer_step_started:
                self._poisoned = True
            try:
                self.optimizer.zero_grad(set_to_none=True)
            except Exception:
                self._poisoned = True
            raise

    def finish_trajectory(self, *, anchor_id: str) -> None:
        self._ensure_healthy()
        if not self.has_active_trajectory or anchor_id != self._active_id:
            raise ValueError("active token-policy anchor does not match")
        self._active_trajectory = None
        self._active_anchor = None
        self._active_id = None
        self._active_updates = 0

    def state_dict(self) -> dict[str, object]:
        self._ensure_healthy()
        if self.has_active_trajectory:
            raise RuntimeError("checkpoint token policy only at a completed trajectory boundary")
        return {
            "schema": TOKEN_POLICY_ENGINE_STATE_SCHEMA,
            "algorithm": self.algorithm.name,
            "algorithm_state": dict(self.algorithm.state_fields),
            "global_step": self.global_step,
            "initial_policy_revision": self.initial_policy_revision,
            "current_policy_revision": self.current_policy_revision,
            "old_log_prob_source": self.old_log_prob_source,
            "updates_per_trajectory": self.updates_per_trajectory,
            "first_update_log_ratio_tolerance": self.first_update_log_ratio_tolerance,
            "replay_microbatch_size": self.replay_microbatch_size,
            "max_grad_norm": self.max_grad_norm,
            "data_parallel_size": self.parallel_context.world_size,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        self._ensure_healthy()
        if not isinstance(state_dict, Mapping):
            raise TypeError("token-policy engine state must be a mapping")
        expected = self.state_dict()
        if set(state_dict) != set(expected):
            raise ValueError("token-policy engine state fields differ from the active schema")
        for name in (
            "schema",
            "algorithm",
            "algorithm_state",
            "initial_policy_revision",
            "old_log_prob_source",
            "updates_per_trajectory",
            "first_update_log_ratio_tolerance",
            "replay_microbatch_size",
            "max_grad_norm",
            "data_parallel_size",
        ):
            if state_dict[name] != expected[name]:
                raise ValueError(f"saved token-policy {name} differs from the active engine")
        global_step = non_negative_int(
            state_dict["global_step"],
            field_name="global_step",
        )
        candidate_revision = self._policy_revision_for_step(global_step)
        if state_dict["current_policy_revision"] != candidate_revision:
            raise ValueError("saved logical token-policy revision is invalid")
        if global_step % self.updates_per_trajectory:
            raise ValueError("saved optimizer step is outside a completed trajectory boundary")
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step = global_step


__all__ = [
    "TOKEN_POLICY_ENGINE_STATE_SCHEMA",
    "NativeTokenPolicyEngine",
    "TokenPolicyStepResult",
]
