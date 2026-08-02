"""Algorithm-neutral optimizer state machine for stochastic flow policies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, prod
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
from ...contracts import FlowTrajectory, FlowTrajectoryReplayAdapter
from ...objectives.group_advantages import normalize_data_parallel_grouped_advantages
from ...trajectory import slice_flow_trajectory
from ..stage import AnchorField, StageAlgorithm, StageAnchor, StageLoss
from .reference_kl import shared_variance_gaussian_kl


@dataclass(frozen=True, slots=True)
class FlowPolicyStepResult:
    """One committed optimizer partition from a prepared trajectory anchor."""

    loss: torch.Tensor
    policy_loss: torch.Tensor
    reference_kl: torch.Tensor | None
    metrics: Mapping[str, object]
    trajectory_complete: bool
    sample_count: int
    token_count: int
    replay_microbatches: int


class NativeFlowPolicyEngine:
    """Own replay anchors, optimizer partitions, accumulation, and resume state.

    ``updates_per_trajectory`` is the number of balanced contiguous partitions
    in the rank-local rollout shard.  Each sample belongs to exactly one
    optimizer step.  ``replay_microbatch_size`` only chunks the current
    partition for gradient accumulation; it never creates another data epoch.
    """

    def __init__(
        self,
        replay_adapter: FlowTrajectoryReplayAdapter,
        optimizer: torch.optim.Optimizer,
        *,
        algorithm: StageAlgorithm,
        initial_policy_revision: str,
        state_schema: str,
        display_name: str,
        anchor_schema: str,
        max_grad_norm: float = 1.0,
        updates_per_trajectory: int = 1,
        reference_replay_adapter: FlowTrajectoryReplayAdapter | None = None,
        reference_kl_weight: float = 0.0,
        parallel_context: PostTrainingParallelContext | None = None,
        replay_microbatch_size: int | None = None,
    ) -> None:
        if not isinstance(replay_adapter, FlowTrajectoryReplayAdapter):
            raise TypeError("replay_adapter must implement FlowTrajectoryReplayAdapter")
        if not isinstance(replay_adapter.module, nn.Module):
            raise TypeError("replay_adapter.module must be an nn.Module")
        if not isinstance(algorithm, StageAlgorithm):
            raise TypeError("algorithm must implement StageAlgorithm")
        if not algorithm.anchor_fields or AnchorField.OLD_LOG_PROBS not in algorithm.anchor_fields:
            raise ValueError("flow-policy algorithms must anchor old_log_probs")
        if not algorithm.anchor_fields <= frozenset(AnchorField):
            raise ValueError("algorithm declares an unsupported anchor field")
        if algorithm.requires_reference_replay and reference_replay_adapter is None:
            raise ValueError(f"{algorithm.name} requires a frozen reference replay adapter")
        if not isinstance(initial_policy_revision, str) or not initial_policy_revision.strip():
            raise ValueError("initial_policy_revision must be a non-empty string")
        for field_name, value in (
            ("state_schema", state_schema),
            ("display_name", display_name),
            ("anchor_schema", anchor_schema),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        updates = non_negative_int(updates_per_trajectory, field_name="updates_per_trajectory")
        if updates == 0:
            raise ValueError("updates_per_trajectory must be positive")
        if updates > 1 and not algorithm.supports_multi_update:
            raise ValueError(f"{algorithm.name} does not support multiple updates per trajectory")
        kl_weight = float(reference_kl_weight)
        if not isfinite(kl_weight) or kl_weight < 0:
            raise ValueError("reference_kl_weight must be finite and non-negative")
        if kl_weight > 0 and reference_replay_adapter is None:
            raise ValueError("positive reference_kl_weight requires reference_replay_adapter")
        if reference_replay_adapter is not None:
            if not isinstance(reference_replay_adapter, FlowTrajectoryReplayAdapter):
                raise TypeError("reference_replay_adapter must implement FlowTrajectoryReplayAdapter")
            if any(parameter.requires_grad for parameter in reference_replay_adapter.module.parameters()):
                raise ValueError("reference policy parameters must be frozen")
        if replay_microbatch_size is not None and (
            isinstance(replay_microbatch_size, bool) or int(replay_microbatch_size) <= 0
        ):
            raise ValueError("replay_microbatch_size must be a positive integer")

        parameters = trainable_parameters(replay_adapter.module)
        audit_optimizer_parameters(optimizer, parameters, role=f"{display_name} policy")
        self.replay_adapter = replay_adapter
        self.reference_replay_adapter = reference_replay_adapter
        self.optimizer = optimizer
        self.algorithm = algorithm
        self.parameters = parameters
        self.initial_policy_revision = initial_policy_revision
        self.state_schema = state_schema
        self.display_name = display_name
        self.anchor_schema = anchor_schema
        self.max_grad_norm = positive_float(max_grad_norm, field_name="max_grad_norm")
        self.updates_per_trajectory = updates
        self.reference_kl_weight = kl_weight
        self.parallel_context = parallel_context or PostTrainingParallelContext.current()
        self.replay_microbatch_size = None if replay_microbatch_size is None else int(replay_microbatch_size)
        self.parallel_context.audit_synchronized_module(replay_adapter.module, role=f"{display_name} policy")
        self.global_step = 0
        self._active_trajectory: FlowTrajectory | None = None
        self._active_anchor: StageAnchor | None = None
        self._active_id: str | None = None
        self._active_old_log_prob_source: str | None = None
        self._active_updates = 0
        self._owner_token = uuid4().hex
        self._poisoned = False

    def _ensure_healthy(self) -> None:
        if self._poisoned:
            raise RuntimeError(
                f"{self.display_name} engine is poisoned after optimizer.step began; "
                "restore the complete training state into a fresh engine"
            )

    @property
    def is_poisoned(self) -> bool:
        return self._poisoned

    @property
    def current_policy_revision(self) -> str:
        return self._policy_revision_for_step(self.global_step)

    def _policy_revision_for_step(self, global_step: int) -> str:
        if global_step == 0:
            return self.initial_policy_revision
        return canonical_sha256(
            {
                "schema": "worldfoundry-logical-policy-revision",
                "initial_policy_revision": self.initial_policy_revision,
                "optimizer_steps": global_step,
            }
        )

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
        microbatch_size = self.replay_microbatch_size or (end - start)
        for chunk_start in range(start, end, microbatch_size):
            yield chunk_start, min(end, chunk_start + microbatch_size)

    def _audit_distributed_backward_calls(self, local_count: int) -> None:
        if self.parallel_context.world_size == 1:
            return
        local_chunks = torch.tensor(
            local_count,
            device=self.parameters[0].device,
            dtype=torch.int64,
        )
        gathered = [torch.zeros_like(local_chunks) for _ in range(self.parallel_context.world_size)]
        dist.all_gather(gathered, local_chunks, group=self.parallel_context.process_group)
        if len({int(value.item()) for value in gathered}) != 1:
            raise ValueError(
                "distributed flow replay requires the same backward-call count "
                "for the current optimizer partition on every rank"
            )

    def _capture_replay_anchor(
        self,
        trajectory: FlowTrajectory,
        *,
        capture_log_probs: bool,
        capture_transition_means: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        log_probs: list[torch.Tensor] = []
        transition_means: list[torch.Tensor] = []
        with torch.no_grad():
            for partition_start, partition_end in self._optimizer_partitions(trajectory.batch_size):
                for start, end in self._microbatch_ranges(
                    partition_start,
                    partition_end,
                ):
                    chunk = slice_flow_trajectory(trajectory, start, end)
                    replay = self.replay_adapter.replay(chunk, training=False)
                    if capture_log_probs:
                        log_probs.append(replay.log_probs.detach().float())
                    if capture_transition_means:
                        transition_means.append(replay.transition_means.detach().float())
        resolved_log_probs = torch.cat(log_probs, dim=0) if log_probs else None
        resolved_means = torch.cat(transition_means, dim=0) if transition_means else None
        return resolved_log_probs, resolved_means

    def _validate_trajectory_start(self, trajectory: FlowTrajectory) -> torch.Tensor | None:
        self._ensure_healthy()
        if self.has_active_trajectory:
            raise RuntimeError("finish the active trajectory before preparing another")
        if not isinstance(trajectory, FlowTrajectory):
            raise TypeError("trajectory must be FlowTrajectory")
        if trajectory.policy_revision != self.current_policy_revision:
            raise ValueError("trajectory policy revision differs from the active pre-update policy")
        self.parallel_context.audit_local_group_ownership(trajectory.group_ids)
        self._optimizer_partitions(trajectory.batch_size)
        mask = trajectory.update_step_mask
        if mask is None:
            return None
        if not self.algorithm.supports_update_step_mask:
            raise ValueError(f"{self.algorithm.name} does not support an update_step_mask")
        if self.reference_kl_weight > 0 or self.algorithm.requires_reference_replay:
            raise ValueError("masked flow-policy updates cannot be combined with reference replay")
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise TypeError("update_step_mask must be a bool torch.Tensor")
        if tuple(mask.shape) != (trajectory.batch_size, len(trajectory.step_indices)):
            raise ValueError("update_step_mask must have shape [B,K]")
        if not bool(mask.any(dim=1).all()):
            raise ValueError("update_step_mask must select at least one transition per sample")
        return mask.detach().to(device=self.parameters[0].device)

    def _activate_trajectory(
        self,
        trajectory: FlowTrajectory,
        advantages: torch.Tensor,
        *,
        old_log_prob_source: str,
        update_step_mask: torch.Tensor | None,
    ) -> str:
        if not isinstance(advantages, torch.Tensor) or tuple(advantages.shape) != (trajectory.batch_size,):
            raise ValueError("advantages must have shape [B]")
        if not bool(torch.isfinite(advantages).all()):
            raise ValueError("advantages must be finite")
        source = str(old_log_prob_source).strip().lower()
        if source not in {"rollout", "replay"}:
            raise ValueError("old_log_prob_source must be 'rollout' or 'replay'")
        needs_old_means = AnchorField.OLD_TRANSITION_MEANS in self.algorithm.anchor_fields
        device = self.parameters[0].device
        replay_log_probs: torch.Tensor | None = None
        old_means: torch.Tensor | None = None
        if source == "replay" or needs_old_means:
            replay_log_probs, old_means = self._capture_replay_anchor(
                trajectory,
                capture_log_probs=source == "replay",
                capture_transition_means=needs_old_means,
            )
        if source == "replay":
            old_log_probs = replay_log_probs
        else:
            old_log_probs = trajectory.old_log_probs.detach().to(
                device=device,
                dtype=torch.float32,
            )
        assert old_log_probs is not None
        if needs_old_means and old_means is None:
            raise RuntimeError(f"{self.display_name} failed to capture old transition means")
        frozen_advantages = advantages.detach().to(device=device, dtype=torch.float32)
        active_id = canonical_sha256(
            {
                "schema": self.anchor_schema,
                "owner_token": self._owner_token,
                "sample_ids": trajectory.sample_ids,
                "schedule_digest": trajectory.schedule_digest,
                "policy_revision": trajectory.policy_revision,
                "optimizer_step": self.global_step,
                "update_step_mask": (
                    None
                    if update_step_mask is None
                    else update_step_mask.to(device="cpu", dtype=torch.uint8).tolist()
                ),
            }
        )
        self._active_trajectory = trajectory
        self._active_anchor = StageAnchor(
            old_log_probs=old_log_probs.clone(),
            advantages=frozen_advantages.clone(),
            old_transition_means=None if old_means is None else old_means.clone(),
            update_step_mask=None if update_step_mask is None else update_step_mask.clone(),
        )
        self._active_id = active_id
        self._active_old_log_prob_source = source
        self._active_updates = 0
        return active_id

    def prepare_trajectory(
        self,
        trajectory: FlowTrajectory,
        rewards: torch.Tensor,
        *,
        old_log_prob_source: str = "rollout",
        advantage_epsilon: float = 1.0e-8,
        advantage_normalization: str = "group-population-variance",
        advantage_clip_max: float | None = None,
    ) -> str:
        update_step_mask = self._validate_trajectory_start(trajectory)
        if not isinstance(rewards, torch.Tensor) or tuple(rewards.shape) != (trajectory.batch_size,):
            raise ValueError("rewards must have shape [B]")
        device = self.parameters[0].device
        advantages = normalize_data_parallel_grouped_advantages(
            rewards.to(device=device),
            trajectory.group_ids,
            parallel_context=self.parallel_context,
            epsilon=advantage_epsilon,
            clip_max=advantage_clip_max,
            normalization=advantage_normalization,
        ).advantages.detach()
        return self._activate_trajectory(
            trajectory,
            advantages,
            old_log_prob_source=old_log_prob_source,
            update_step_mask=update_step_mask,
        )

    def prepare_trajectory_from_advantages(
        self,
        trajectory: FlowTrajectory,
        advantages: torch.Tensor,
        *,
        old_log_prob_source: str = "rollout",
    ) -> str:
        """Freeze caller-computed advantages without applying another normalization."""

        update_step_mask = self._validate_trajectory_start(trajectory)
        return self._activate_trajectory(
            trajectory,
            advantages,
            old_log_prob_source=old_log_prob_source,
            update_step_mask=update_step_mask,
        )

    def train_step(self, *, anchor_id: str) -> FlowPolicyStepResult:
        self._ensure_healthy()
        if not self.has_active_trajectory:
            raise RuntimeError(f"no {self.display_name} trajectory is prepared")
        if anchor_id != self._active_id:
            raise ValueError(f"{self.display_name} anchor does not belong to this active engine state")
        if self._active_updates >= self.updates_per_trajectory:
            raise RuntimeError("active trajectory exhausted its configured policy updates")
        assert self._active_trajectory is not None
        assert self._active_anchor is not None
        self.optimizer.zero_grad(set_to_none=True)
        optimizer_step_started = False
        try:
            partition_start, partition_end = self._optimizer_partitions(self._active_trajectory.batch_size)[
                self._active_updates
            ]
            sample_count = partition_end - partition_start
            step_count = len(self._active_trajectory.step_indices)
            active_mask = self._active_anchor.update_step_mask
            transition_weight = (
                sample_count * step_count
                if active_mask is None
                else int(active_mask[partition_start:partition_end].sum().item())
            )
            if transition_weight <= 0:
                raise RuntimeError("optimizer partition selects no policy transitions")
            microbatch_ranges = tuple(self._microbatch_ranges(partition_start, partition_end))
            replay_microbatches = len(microbatch_ranges)
            self._audit_distributed_backward_calls(replay_microbatches)
            scale = self.parallel_context.scale_local_mean(
                torch.ones((), device=self.parameters[0].device),
                transition_weight,
            ).detach()
            policy_sum = torch.zeros((), device=self.parameters[0].device)
            reference_sum = torch.zeros_like(policy_sum)
            total_sum = torch.zeros_like(policy_sum)
            metric_sums: dict[str, torch.Tensor] = {}
            ratio_values: list[torch.Tensor] = []
            for start, end in microbatch_ranges:
                chunk = slice_flow_trajectory(self._active_trajectory, start, end)
                replay = self.replay_adapter.replay(chunk, training=True)
                old_means = self._active_anchor.old_transition_means
                stage_anchor = StageAnchor(
                    old_log_probs=self._active_anchor.old_log_probs[start:end],
                    advantages=self._active_anchor.advantages[start:end],
                    old_transition_means=None if old_means is None else old_means[start:end],
                    update_step_mask=(
                        None if active_mask is None else active_mask[start:end]
                    ),
                )
                if self._active_updates == 0 and self._active_old_log_prob_source == "replay":
                    replay_log_probs = replay.log_probs.detach().float()
                    anchor_log_probs = stage_anchor.old_log_probs.detach().to(
                        device=replay_log_probs.device,
                        dtype=torch.float32,
                    )
                    if replay_log_probs.shape != anchor_log_probs.shape or not torch.equal(
                        replay_log_probs,
                        anchor_log_probs,
                    ):
                        raise ValueError(
                            f"{self.display_name} first replay does not exactly match its old log-probability anchor"
                        )
                if self._active_updates == 0 and stage_anchor.old_transition_means is not None:
                    replay_means = replay.transition_means.detach().float()
                    anchor_means = stage_anchor.old_transition_means.detach().to(
                        device=replay_means.device,
                        dtype=torch.float32,
                    )
                    if replay_means.shape != anchor_means.shape or not torch.equal(
                        replay_means,
                        anchor_means,
                    ):
                        raise ValueError(
                            f"{self.display_name} first replay transition means "
                            "do not exactly match their old-policy anchor"
                        )
                reference = None
                if self.reference_kl_weight > 0 or self.algorithm.requires_reference_replay:
                    assert self.reference_replay_adapter is not None
                    with torch.no_grad():
                        reference = self.reference_replay_adapter.replay(
                            chunk,
                            training=False,
                        )
                objective = self.algorithm.loss(
                    replay,
                    stage_anchor,
                    reference,
                    optimizer_step=self.global_step,
                )
                if not isinstance(objective, StageLoss):
                    raise TypeError("StageAlgorithm.loss must return StageLoss")
                if (
                    self._active_updates == 0
                    and self._active_old_log_prob_source == "replay"
                    and not torch.equal(
                        objective.ratio.detach(),
                        torch.ones_like(objective.ratio.detach()),
                    )
                ):
                    raise ValueError(f"{self.display_name} first update requires an exact ratio-one anchor")
                chunk_weight = (
                    (end - start) * step_count
                    if stage_anchor.update_step_mask is None
                    else int(stage_anchor.update_step_mask.sum().item())
                )
                chunk_loss = objective.loss
                chunk_reference: torch.Tensor | None = None
                if self.reference_kl_weight > 0:
                    assert reference is not None
                    chunk_reference = shared_variance_gaussian_kl(
                        replay.transition_means,
                        reference.transition_means,
                        replay.transition_scales,
                    )
                    chunk_loss = chunk_loss + self.reference_kl_weight * chunk_reference
                if not bool(torch.isfinite(chunk_loss.detach())):
                    raise FloatingPointError(f"non-finite {self.display_name} loss")
                (chunk_loss * (float(chunk_weight) / float(transition_weight)) * scale).backward()
                policy_sum += objective.loss.detach() * chunk_weight
                total_sum += chunk_loss.detach() * chunk_weight
                if chunk_reference is not None:
                    reference_sum += chunk_reference.detach() * chunk_weight
                ratio = objective.ratio.detach()
                if stage_anchor.update_step_mask is not None:
                    ratio = ratio[stage_anchor.update_step_mask]
                ratio_values.append(ratio.reshape(-1))
                for name, value in objective.metrics.items():
                    if not isinstance(name, str) or not name:
                        raise ValueError("StageAlgorithm metric names must be non-empty strings")
                    if not torch.is_tensor(value) or value.numel() != 1:
                        raise ValueError(f"StageAlgorithm metric {name!r} must be a scalar tensor")
                    metric_sums[name] = (
                        metric_sums.get(name, torch.zeros_like(policy_sum)) + value.detach() * chunk_weight
                    )
            policy_loss = policy_sum / transition_weight
            total_loss = total_sum / transition_weight
            reference_kl = reference_sum / transition_weight if self.reference_kl_weight > 0 else None
            ratios = torch.cat(ratio_values, dim=0)
            grad_norm = clip_grad_norm_(self.parameters, self.max_grad_norm, error_if_nonfinite=True)
            optimizer_step_started = True
            self.optimizer.step()
            self.global_step += 1
            self._active_updates += 1

            complete = self._active_updates == self.updates_per_trajectory
            token_count = transition_weight * prod(
                int(size) for size in self._active_trajectory.latents.shape[3:]
            )
            metrics: dict[str, object] = {
                "global_step": torch.tensor(self.global_step, device=total_loss.device, dtype=torch.int64),
                "trajectory_update": torch.tensor(self._active_updates, device=total_loss.device, dtype=torch.int64),
                "ratio_mean": ratios.mean(),
                "ratio_std": ratios.std(correction=0),
                "ratio_min": ratios.min(),
                "ratio_max": ratios.max(),
                "sample_count": torch.tensor(sample_count, device=total_loss.device, dtype=torch.int64),
                "token_count": torch.tensor(token_count, device=total_loss.device, dtype=torch.int64),
                "replay_microbatches": torch.tensor(
                    replay_microbatches,
                    device=total_loss.device,
                    dtype=torch.int64,
                ),
                **{name: value / transition_weight for name, value in metric_sums.items()},
                "grad_norm": grad_norm.detach(),
            }
            if reference_kl is not None:
                metrics["reference_kl"] = reference_kl.detach()
            result = FlowPolicyStepResult(
                loss=total_loss.detach(),
                policy_loss=policy_loss.detach(),
                reference_kl=None if reference_kl is None else reference_kl.detach(),
                metrics=metrics,
                trajectory_complete=complete,
                sample_count=sample_count,
                token_count=token_count,
                replay_microbatches=replay_microbatches,
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
            raise ValueError(f"active {self.display_name} anchor does not match")
        self._active_trajectory = None
        self._active_anchor = None
        self._active_id = None
        self._active_old_log_prob_source = None
        self._active_updates = 0

    def state_dict(self) -> dict[str, object]:
        self._ensure_healthy()
        if self.has_active_trajectory:
            raise RuntimeError(f"checkpoint {self.display_name} only at a completed trajectory boundary")
        algorithm_fields = dict(self.algorithm.state_fields)
        shared_fields: dict[str, object] = {
            "schema": self.state_schema,
            "global_step": self.global_step,
            "initial_policy_revision": self.initial_policy_revision,
            "current_policy_revision": self.current_policy_revision,
            "updates_per_trajectory": self.updates_per_trajectory,
            "reference_kl_weight": self.reference_kl_weight,
            "replay_microbatch_size": self.replay_microbatch_size,
            "data_parallel_size": self.parallel_context.world_size,
        }
        overlap = set(algorithm_fields) & set(shared_fields)
        if overlap:
            raise ValueError(f"algorithm state fields collide with learner state: {sorted(overlap)}")
        return {**shared_fields, **algorithm_fields}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        self._ensure_healthy()
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"{self.display_name} engine state must be a mapping")
        expected_state = self.state_dict()
        if set(state_dict) != set(expected_state):
            raise ValueError(f"{self.display_name} engine state fields differ from the active schema")
        if state_dict["schema"] != self.state_schema:
            raise ValueError(f"unsupported {self.display_name} engine schema: {state_dict['schema']!r}")
        if state_dict["initial_policy_revision"] != self.initial_policy_revision:
            raise ValueError("saved initial policy revision differs from the active engine")
        for name, value in self.algorithm.state_fields.items():
            if state_dict[name] != value:
                raise ValueError(f"saved {self.display_name} {name} differs from the active engine")
        if int(state_dict["updates_per_trajectory"]) != self.updates_per_trajectory:
            raise ValueError("saved update count differs from the active engine")
        if float(state_dict["reference_kl_weight"]) != self.reference_kl_weight:
            raise ValueError(f"saved {self.display_name} reference KL weight differs from the active engine")
        saved_microbatch = state_dict["replay_microbatch_size"]
        if saved_microbatch is not None:
            saved_microbatch = int(saved_microbatch)
        if saved_microbatch != self.replay_microbatch_size:
            raise ValueError(f"saved {self.display_name} replay microbatch size differs")
        if int(state_dict["data_parallel_size"]) != self.parallel_context.world_size:
            raise ValueError(f"saved {self.display_name} data-parallel size differs from the active engine")
        global_step = non_negative_int(state_dict["global_step"], field_name="global_step")
        if state_dict["current_policy_revision"] != self._policy_revision_for_step(global_step):
            raise ValueError("saved logical policy revision is invalid")
        if global_step % self.updates_per_trajectory:
            raise ValueError("saved optimizer step is outside a completed trajectory boundary")
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step = global_step


__all__ = ["FlowPolicyStepResult", "NativeFlowPolicyEngine"]
