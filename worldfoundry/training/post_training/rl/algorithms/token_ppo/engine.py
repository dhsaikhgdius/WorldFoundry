"""Native optimizer state machine for classic packed-token PPO."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

import torch
from torch import nn

from worldfoundry.core.gradient import clip_grad_norm_
from worldfoundry.training.optimization import (
    audit_optimizer_parameters,
    trainable_parameters,
)
from worldfoundry.training.recipes.post_training.common import (
    scheduled_clip_range,
    validate_clip_schedule,
)

from ....shared.distributed import PostTrainingParallelContext
from .contracts import (
    PackedTokenPPOReplayBatch,
    PackedTokenPPOTrajectory,
    TokenPPOReplayAdapter,
    TokenPPOReplayResult,
    slice_token_ppo_trajectory,
)
from .math import packed_gae, scatter_terminal_rewards
from .objective import (
    TOKEN_PPO_REDUCTIONS,
    token_ppo_loss,
    token_ppo_reduction_weight,
)

TOKEN_PPO_ENGINE_STATE_SCHEMA = "worldfoundry-token-ppo-engine"


@dataclass(frozen=True, slots=True)
class TokenPPOAnchor:
    """Frozen behavior policy, critic, GAE, and return tensors."""

    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    token_rewards: torch.Tensor


@dataclass(frozen=True, slots=True)
class TokenPPOStepResult:
    """One optimizer partition from a frozen PPO trajectory anchor."""

    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    metrics: Mapping[str, torch.Tensor]
    trajectory_complete: bool
    replay_microbatches: int
    sample_count: int
    token_count: int


class NativeTokenPPOEngine:
    """Freeze one PPO anchor, then update disjoint contiguous partitions."""

    def __init__(
        self,
        replay_adapter: TokenPPOReplayAdapter,
        optimizer: torch.optim.Optimizer,
        *,
        initial_policy_revision: str,
        update_epochs: int = 1,
        update_partitions: int = 1,
        replay_microbatch_size: int | None = None,
        clip_range: float = 0.2,
        clip_range_high: float | None = None,
        clip_schedule: str = "constant",
        clip_schedule_steps: int | None = None,
        value_clip_range: float = 0.2,
        vf_coef: float = 0.5,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        reduction: str = "token-mean",
        horizon: int = 8192,
        max_grad_norm: float = 1.0,
        parallel_context: PostTrainingParallelContext | None = None,
    ) -> None:
        if not isinstance(replay_adapter, TokenPPOReplayAdapter):
            raise TypeError("replay_adapter must implement TokenPPOReplayAdapter")
        if not isinstance(replay_adapter.module, nn.Module):
            raise TypeError("replay_adapter.module must be an nn.Module")
        if not isinstance(initial_policy_revision, str) or not initial_policy_revision.strip():
            raise ValueError("initial_policy_revision must be a non-empty string")
        if isinstance(update_epochs, bool) or int(update_epochs) <= 0:
            raise ValueError("update_epochs must be positive")
        if isinstance(update_partitions, bool) or int(update_partitions) <= 0:
            raise ValueError("update_partitions must be positive")
        if replay_microbatch_size is not None and (
            isinstance(replay_microbatch_size, bool) or int(replay_microbatch_size) <= 0
        ):
            raise ValueError("replay_microbatch_size must be positive")
        values = {
            "clip_range": float(clip_range),
            "value_clip_range": float(value_clip_range),
            "vf_coef": float(vf_coef),
            "gamma": float(gamma),
            "gae_lambda": float(gae_lambda),
            "max_grad_norm": float(max_grad_norm),
        }
        if any(not isfinite(value) for value in values.values()):
            raise ValueError("PPO numeric settings must be finite")
        resolved_clip_high = None if clip_range_high is None else float(clip_range_high)
        if resolved_clip_high is not None and (not isfinite(resolved_clip_high) or resolved_clip_high < 0):
            raise ValueError("clip_range_high must be finite and non-negative")
        if values["clip_range"] < 0 or values["value_clip_range"] < 0:
            raise ValueError("policy and value clip ranges must be non-negative")
        if values["vf_coef"] < 0 or values["max_grad_norm"] <= 0:
            raise ValueError("vf_coef must be non-negative and max_grad_norm must be positive")
        if not 0 <= values["gamma"] <= 1 or not 0 <= values["gae_lambda"] <= 1:
            raise ValueError("gamma and gae_lambda must be in [0,1]")
        resolved_reduction = str(reduction).strip().lower().replace("_", "-")
        if resolved_reduction not in TOKEN_PPO_REDUCTIONS:
            raise ValueError(f"reduction must be one of {sorted(TOKEN_PPO_REDUCTIONS)}")
        if isinstance(horizon, bool) or int(horizon) <= 0:
            raise ValueError("horizon must be positive")
        resolved_schedule, resolved_schedule_steps = validate_clip_schedule(
            clip_schedule,
            clip_schedule_steps,
        )

        parameters = trainable_parameters(replay_adapter.module)
        audit_optimizer_parameters(optimizer, parameters, role="token PPO actor-critic")
        self.replay_adapter = replay_adapter
        self.optimizer = optimizer
        self.parameters = parameters
        self.initial_policy_revision = initial_policy_revision
        self.update_epochs = int(update_epochs)
        self.update_partitions = int(update_partitions)
        self.replay_microbatch_size = None if replay_microbatch_size is None else int(replay_microbatch_size)
        self.clip_range = values["clip_range"]
        self.clip_range_high = resolved_clip_high
        self.clip_schedule = resolved_schedule
        self.clip_schedule_steps = resolved_schedule_steps
        self.value_clip_range = values["value_clip_range"]
        self.vf_coef = values["vf_coef"]
        self.gamma = values["gamma"]
        self.gae_lambda = values["gae_lambda"]
        self.reduction = resolved_reduction
        self.horizon = int(horizon)
        self.max_grad_norm = values["max_grad_norm"]
        self.parallel_context = parallel_context or PostTrainingParallelContext.current()
        self.parallel_context.audit_synchronized_module(
            replay_adapter.module,
            role="token PPO actor-critic",
        )
        self.global_step = 0
        self._active_trajectory: PackedTokenPPOTrajectory | None = None
        self._active_anchor: TokenPPOAnchor | None = None
        self._active_epoch = 0
        self._active_partition = 0

    @property
    def current_policy_revision(self) -> str:
        if self.global_step == 0:
            return self.initial_policy_revision
        return f"{self.initial_policy_revision}:step-{self.global_step}"

    @property
    def has_active_trajectory(self) -> bool:
        return self._active_trajectory is not None

    @property
    def active_anchor(self) -> TokenPPOAnchor | None:
        return self._active_anchor

    @property
    def updates_per_trajectory(self) -> int:
        return self.update_epochs * self.update_partitions

    def _optimizer_partitions(self, batch_size: int) -> tuple[tuple[int, int], ...]:
        if batch_size % self.update_partitions:
            raise ValueError("local PPO trajectory batch_size must be divisible by update_partitions")
        partition_size = batch_size // self.update_partitions
        return tuple((index * partition_size, (index + 1) * partition_size) for index in range(self.update_partitions))

    def _microbatch_ranges(self, start: int, end: int):
        size = self.replay_microbatch_size or (end - start)
        for chunk_start in range(start, end, size):
            yield chunk_start, min(end, chunk_start + size)

    def _replay(
        self,
        chunk: PackedTokenPPOReplayBatch,
        *,
        training: bool,
    ) -> TokenPPOReplayResult:
        result = self.replay_adapter.replay(chunk, training=training)
        if not isinstance(result, TokenPPOReplayResult):
            raise TypeError("token PPO replay must return TokenPPOReplayResult")
        if tuple(result.log_probs.shape) != (chunk.token_count,):
            raise ValueError("replayed actor-critic tensors do not match the packed token span")
        if result.sampling_temperature != chunk.sampling_temperature:
            raise ValueError("PPO replay changed the rollout sampling temperature")
        return result

    def _capture_old_values(self, trajectory: PackedTokenPPOTrajectory) -> torch.Tensor:
        values: list[torch.Tensor] = []
        with torch.no_grad():
            for partition_start, partition_end in self._optimizer_partitions(trajectory.batch_size):
                for start, end in self._microbatch_ranges(partition_start, partition_end):
                    chunk = slice_token_ppo_trajectory(trajectory, start, end)
                    values.append(self._replay(chunk, training=False).values.detach().float())
        return torch.cat(values)

    def prepare_trajectory(
        self,
        trajectory: PackedTokenPPOTrajectory,
        terminal_rewards: torch.Tensor,
    ) -> TokenPPOAnchor:
        """Capture critic values and derive packed token rewards, GAE, and returns."""

        if self.has_active_trajectory:
            raise RuntimeError("finish the active PPO trajectory before preparing another")
        if not isinstance(trajectory, PackedTokenPPOTrajectory):
            raise TypeError("trajectory must be PackedTokenPPOTrajectory")
        if trajectory.policy_revision != self.current_policy_revision:
            raise ValueError("trajectory policy revision differs from the active policy")
        if (
            not isinstance(terminal_rewards, torch.Tensor)
            or tuple(terminal_rewards.shape) != (trajectory.batch_size,)
            or not terminal_rewards.is_floating_point()
            or not bool(torch.isfinite(terminal_rewards).all())
        ):
            raise ValueError("terminal_rewards must be finite floating values with shape [B]")

        old_values = self._capture_old_values(trajectory)
        token_rewards = scatter_terminal_rewards(
            terminal_rewards.to(device=old_values.device, dtype=torch.float32),
            trajectory.lengths,
        )
        advantages, returns = packed_gae(
            token_rewards,
            old_values,
            trajectory.lengths,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        anchor = TokenPPOAnchor(
            old_log_probs=trajectory.old_log_probs.detach().float().clone(),
            old_values=old_values.detach().clone(),
            advantages=advantages.detach(),
            returns=returns.detach(),
            token_rewards=token_rewards.detach(),
        )
        self._active_trajectory = trajectory
        self._active_anchor = anchor
        self._active_epoch = 0
        self._active_partition = 0
        return anchor

    def train_step(self) -> TokenPPOStepResult:
        """Update one contiguous trajectory partition using the frozen anchor."""

        if not self.has_active_trajectory or self._active_anchor is None:
            raise RuntimeError("no PPO trajectory is prepared")
        trajectory = self._active_trajectory
        anchor = self._active_anchor
        assert trajectory is not None
        partition_start, partition_end = self._optimizer_partitions(trajectory.batch_size)[self._active_partition]
        partition = slice_token_ppo_trajectory(
            trajectory,
            partition_start,
            partition_end,
        )
        total_weight = token_ppo_reduction_weight(
            partition.lengths,
            partition.loss_mask,
            reduction=self.reduction,
        )
        distributed_scale = self.parallel_context.scale_local_mean(
            torch.ones((), device=self.parameters[0].device),
            total_weight,
        ).detach()
        active_clip_range = scheduled_clip_range(
            self.clip_range,
            schedule=self.clip_schedule,
            schedule_steps=self.clip_schedule_steps,
            optimizer_step=self.global_step,
        )
        active_clip_range_high = (
            None
            if self.clip_range_high is None
            else scheduled_clip_range(
                self.clip_range_high,
                schedule=self.clip_schedule,
                schedule_steps=self.clip_schedule_steps,
                optimizer_step=self.global_step,
            )
        )

        self.optimizer.zero_grad(set_to_none=True)
        policy_numerator = torch.zeros((), device=self.parameters[0].device)
        value_numerator = torch.zeros((), device=self.parameters[0].device)
        ratios: list[torch.Tensor] = []
        policy_clipped: list[torch.Tensor] = []
        value_clipped: list[torch.Tensor] = []
        active_values: list[torch.Tensor] = []
        active_returns: list[torch.Tensor] = []
        replay_microbatches = 0
        for start, end in self._microbatch_ranges(partition_start, partition_end):
            chunk = slice_token_ppo_trajectory(trajectory, start, end)
            replay = self._replay(chunk, training=True)
            span = slice(chunk.token_start, chunk.token_end)
            terms = token_ppo_loss(
                replay.log_probs,
                replay.values,
                anchor.old_log_probs[span].to(replay.log_probs),
                anchor.old_values[span].to(device=replay.values.device),
                anchor.advantages[span].to(replay.log_probs),
                anchor.returns[span].to(device=replay.values.device),
                chunk.lengths,
                loss_mask=chunk.loss_mask,
                clip_range=active_clip_range,
                clip_range_high=active_clip_range_high,
                value_clip_range=self.value_clip_range,
                reduction=self.reduction,
                horizon=self.horizon,
            )
            expected_weight = token_ppo_reduction_weight(
                chunk.lengths,
                chunk.loss_mask,
                reduction=self.reduction,
            )
            if terms.denominator != expected_weight:
                raise ValueError("PPO microbatch returned an inconsistent reduction weight")
            combined = terms.policy_numerator + self.vf_coef * terms.value_numerator
            (combined / float(total_weight) * distributed_scale).backward()
            policy_numerator += terms.policy_numerator.detach()
            value_numerator += terms.value_numerator.detach()
            ratios.append(terms.ratio.detach())
            policy_clipped.append(terms.policy_clipped.detach())
            value_clipped.append(terms.value_clipped.detach())
            active = (
                torch.ones_like(replay.values, dtype=torch.bool)
                if chunk.loss_mask is None
                else chunk.loss_mask.to(device=replay.values.device)
            )
            active_values.append(replay.values.detach().float()[active])
            active_returns.append(
                anchor.returns[span].to(device=replay.values.device, dtype=torch.float32).detach()[active]
            )
            replay_microbatches += 1

        grad_norm = clip_grad_norm_(
            self.parameters,
            self.max_grad_norm,
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.global_step += 1
        completed_epoch = self._active_epoch + 1
        completed_partition = self._active_partition + 1
        self._active_partition += 1
        if self._active_partition == self.update_partitions:
            self._active_partition = 0
            self._active_epoch += 1

        policy_loss = policy_numerator / float(total_weight)
        value_loss = value_numerator / float(total_weight)
        loss = policy_loss + self.vf_coef * value_loss
        ratio = torch.cat(ratios)
        values = torch.cat(active_values)
        returns = torch.cat(active_returns)
        return_variance = returns.var(unbiased=False)
        explained_variance = (
            1.0 - (returns - values).var(unbiased=False) / return_variance
            if bool(return_variance > 0)
            else return_variance.new_zeros(())
        )
        complete = self._active_epoch == self.update_epochs
        metrics = {
            "global_step": loss.new_tensor(self.global_step, dtype=torch.int64),
            "update_epoch": loss.new_tensor(completed_epoch, dtype=torch.int64),
            "update_partition": loss.new_tensor(completed_partition, dtype=torch.int64),
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "clip_range": loss.new_tensor(active_clip_range),
            "clip_range_high": loss.new_tensor(
                active_clip_range if active_clip_range_high is None else active_clip_range_high
            ),
            "ratio_mean": ratio.mean(),
            "ratio_std": ratio.std(unbiased=False),
            "policy_clip_fraction": torch.cat(policy_clipped).float().mean(),
            "value_clip_fraction": torch.cat(value_clipped).float().mean(),
            "value_mean": values.mean(),
            "return_mean": returns.mean(),
            "explained_variance": explained_variance,
            "grad_norm": grad_norm.detach(),
        }
        result = TokenPPOStepResult(
            loss=loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            metrics=metrics,
            trajectory_complete=complete,
            replay_microbatches=replay_microbatches,
            sample_count=partition.batch_size,
            token_count=partition.token_count,
        )
        if complete:
            self._active_trajectory = None
            self._active_anchor = None
            self._active_epoch = 0
            self._active_partition = 0
        return result

    def state_dict(self) -> dict[str, object]:
        if self.has_active_trajectory:
            raise RuntimeError("checkpoint PPO only after every partition and epoch for a trajectory completes")
        return {
            "schema": TOKEN_PPO_ENGINE_STATE_SCHEMA,
            "global_step": self.global_step,
            "initial_policy_revision": self.initial_policy_revision,
            "update_epochs": self.update_epochs,
            "update_partitions": self.update_partitions,
            "replay_microbatch_size": self.replay_microbatch_size,
            "clip_range": self.clip_range,
            "clip_range_high": self.clip_range_high,
            "clip_schedule": self.clip_schedule,
            "clip_schedule_steps": self.clip_schedule_steps,
            "value_clip_range": self.value_clip_range,
            "vf_coef": self.vf_coef,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "reduction": self.reduction,
            "horizon": self.horizon,
            "max_grad_norm": self.max_grad_norm,
            "data_parallel_size": self.parallel_context.world_size,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if self.has_active_trajectory:
            raise RuntimeError("cannot load PPO state while a trajectory is active")
        expected = self.state_dict()
        if not isinstance(state_dict, Mapping) or set(state_dict) != set(expected):
            raise ValueError("token PPO engine state fields differ from the active schema")
        for name, value in expected.items():
            if name != "global_step" and state_dict[name] != value:
                raise ValueError(f"saved token PPO {name} differs from the active engine")
        step = int(state_dict["global_step"])
        if step < 0 or step % self.updates_per_trajectory:
            raise ValueError("saved token PPO step is outside a completed trajectory boundary")
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step = step


__all__ = [
    "TOKEN_PPO_ENGINE_STATE_SCHEMA",
    "NativeTokenPPOEngine",
    "TokenPPOAnchor",
    "TokenPPOStepResult",
]
