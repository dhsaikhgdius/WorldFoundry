"""Gradient-accumulating optimizer engine for DDRL."""

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
    advantage_normalization_mode,
)

from ....shared.distributed import PostTrainingParallelContext
from ....shared.validation import non_negative_int, positive_float
from .contracts import (
    DDRLDataRegularizerAdapter,
    DDRLReplayAdapter,
    DDRLTrajectory,
)
from .objective import (
    DDRL_ADVANTAGE_EPSILON,
    DDRLLoss,
    ddrl_group_advantages,
    ddrl_loss,
)

DDRL_ENGINE_STATE_SCHEMA = "worldfoundry-ddrl-engine"
_ENGINE_STATE_FIELDS = frozenset(
    {
        "schema",
        "global_step",
        "optimizer_steps",
        "last_trajectory_id",
        "clip_range",
        "loss_scale",
        "advantage_epsilon",
        "advantage_normalization",
        "advantage_clip_min",
        "advantage_clip_max",
        "exponential_advantage",
        "kl_beta",
        "data_beta",
        "data_on_first_step_only",
        "max_grad_norm",
        "data_parallel_size",
    }
)


@dataclass(frozen=True, slots=True)
class DDRLStepResult:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    reference_kl: torch.Tensor | None
    data_loss: torch.Tensor | None
    advantages: torch.Tensor
    ratios: torch.Tensor
    gradient_norm: torch.Tensor
    train_on: tuple[int, ...]
    metrics: Mapping[str, object]


def _non_negative_float(value: float, *, field_name: str) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return resolved


def _optional_finite_float(value: float | None, *, field_name: str) -> float | None:
    if value is None:
        return None
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{field_name} must be finite or None")
    return resolved


class NativeDDRLEngine:
    """Accumulate every selected transition before one optimizer commit."""

    def __init__(
        self,
        replay_adapter: DDRLReplayAdapter,
        optimizer: torch.optim.Optimizer,
        *,
        clip_range: float,
        loss_scale: float = 10.0,
        advantage_epsilon: float = DDRL_ADVANTAGE_EPSILON,
        advantage_normalization: str = "group-sample-std",
        advantage_clip_min: float | None = None,
        advantage_clip_max: float | None = None,
        exponential_advantage: bool = False,
        kl_beta: float = 0.0,
        data_beta: float = 0.0,
        data_regularizer: DDRLDataRegularizerAdapter | None = None,
        data_on_first_step_only: bool = False,
        max_grad_norm: float = 1.0,
        parallel_context: PostTrainingParallelContext | None = None,
    ) -> None:
        if not isinstance(replay_adapter, DDRLReplayAdapter):
            raise TypeError("replay_adapter must implement DDRLReplayAdapter")
        policy_module = replay_adapter.module
        if not isinstance(policy_module, nn.Module):
            raise TypeError("replay_adapter.module must be an nn.Module")
        parameters = trainable_parameters(policy_module)
        audit_optimizer_parameters(optimizer, parameters, role="DDRL policy")
        resolved_clip = float(clip_range)
        if not isfinite(resolved_clip) or not 0 < resolved_clip < 1:
            raise ValueError("clip_range must be finite and in (0,1)")
        lower = _optional_finite_float(
            advantage_clip_min,
            field_name="advantage_clip_min",
        )
        upper = _optional_finite_float(
            advantage_clip_max,
            field_name="advantage_clip_max",
        )
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError("advantage_clip_min must be smaller than advantage_clip_max")
        if not isinstance(exponential_advantage, bool):
            raise TypeError("exponential_advantage must be a bool")
        if not isinstance(data_on_first_step_only, bool):
            raise TypeError("data_on_first_step_only must be a bool")
        resolved_data_beta = _non_negative_float(data_beta, field_name="data_beta")
        if resolved_data_beta > 0:
            if not isinstance(data_regularizer, DDRLDataRegularizerAdapter):
                raise TypeError("positive data_beta requires DDRLDataRegularizerAdapter")
            if data_regularizer.module is not policy_module:
                raise ValueError("data regularizer and replay adapter must own the same policy module")
        elif data_regularizer is not None:
            raise ValueError("data_regularizer is unused when data_beta is zero")

        self.replay_adapter = replay_adapter
        self.data_regularizer = data_regularizer
        self.policy_module = policy_module
        self.optimizer = optimizer
        self.parameters = parameters
        self.clip_range = resolved_clip
        self.loss_scale = positive_float(loss_scale, field_name="loss_scale")
        self.advantage_epsilon = positive_float(
            advantage_epsilon,
            field_name="advantage_epsilon",
        )
        self.advantage_normalization = advantage_normalization_mode(
            advantage_normalization,
            field_name="advantage_normalization",
        )
        self.advantage_clip_min = lower
        self.advantage_clip_max = upper
        self.exponential_advantage = exponential_advantage
        self.kl_beta = _non_negative_float(kl_beta, field_name="kl_beta")
        self.data_beta = resolved_data_beta
        self.data_on_first_step_only = data_on_first_step_only
        self.max_grad_norm = positive_float(max_grad_norm, field_name="max_grad_norm")
        self.parallel_context = parallel_context or PostTrainingParallelContext.current()
        self.parallel_context.audit_synchronized_module(policy_module, role="DDRL policy")
        self.global_step = 0
        self.optimizer_steps = 0
        self.last_trajectory_id: str | None = None
        self._phase = "idle"
        self._poisoned = False

    def _replay_mean(
        self,
        trajectory: DDRLTrajectory,
        position: int,
    ) -> torch.Tensor:
        current = self.replay_adapter.replay_mean(
            trajectory,
            position,
            training=True,
        )
        if not isinstance(current, torch.Tensor):
            raise TypeError("DDRL replay mean must be a torch.Tensor")
        return current

    def _data_loss(
        self,
        trajectory: DDRLTrajectory,
        position: int,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor | None:
        if self.data_regularizer is None:
            return None
        if self.data_on_first_step_only and position != 0:
            return None
        value = self.data_regularizer.loss(
            trajectory,
            position,
            generator=generator,
            training=True,
        )
        if not isinstance(value, torch.Tensor):
            raise TypeError("DDRL data regularizer must return a torch.Tensor")
        if value.numel() != 1 and (value.ndim == 0 or int(value.shape[0]) != trajectory.batch_size):
            raise ValueError("DDRL data regularizer must return a scalar or a batch-leading loss tensor")
        return value

    def train_trajectory(
        self,
        trajectory: DDRLTrajectory,
        rewards: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> DDRLStepResult:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("DDRL engine has a partially committed update; restore the last checkpoint")
        if not isinstance(trajectory, DDRLTrajectory):
            raise TypeError("trajectory must be DDRLTrajectory")
        if trajectory.trajectory_id == self.last_trajectory_id:
            raise ValueError("a DDRL trajectory can be optimized only once")
        if (
            not isinstance(rewards, torch.Tensor)
            or not rewards.is_floating_point()
            or tuple(rewards.shape) != (trajectory.batch_size,)
        ):
            raise ValueError("rewards must be a floating tensor with shape [B]")
        if self.kl_beta > 0 and trajectory.reference_means is None:
            raise ValueError("positive kl_beta requires trajectory reference_means")
        self.parallel_context.audit_local_group_ownership(trajectory.group_ids)
        advantage_result = ddrl_group_advantages(
            rewards.to(device=self.parameters[0].device),
            trajectory.group_ids,
            epsilon=self.advantage_epsilon,
            normalization=self.advantage_normalization,
            clip_min=self.advantage_clip_min,
            clip_max=self.advantage_clip_max,
            exponential=self.exponential_advantage,
            parallel_context=self.parallel_context,
        )
        advantages = advantage_result.advantages.detach()
        transition_weight = trajectory.batch_size * trajectory.step_count
        scale = self.parallel_context.scale_local_mean(
            torch.ones((), device=self.parameters[0].device),
            transition_weight,
        ).detach()
        self.policy_module.train()
        self.optimizer.zero_grad(set_to_none=True)
        optimizer_started = False
        policy_sum = torch.zeros((), device=self.parameters[0].device)
        total_sum = torch.zeros_like(policy_sum)
        reference_sum = torch.zeros_like(policy_sum)
        data_sum = torch.zeros_like(policy_sum)
        ratios: list[torch.Tensor] = []
        log_ratios: list[torch.Tensor] = []
        clip_fractions: list[torch.Tensor] = []
        try:
            for position in range(trajectory.step_count):
                self._phase = "forward"
                next_latents = trajectory.next_latents[:, position].detach().clone()
                old_means = trajectory.old_means[:, position].detach().clone()
                reference_means = (
                    None
                    if trajectory.reference_means is None
                    else trajectory.reference_means[:, position].detach().clone()
                )
                current_means = self._replay_mean(trajectory, position)
                data_loss = self._data_loss(
                    trajectory,
                    position,
                    generator=generator,
                )
                objective: DDRLLoss = ddrl_loss(
                    next_latents=next_latents,
                    current_means=current_means,
                    old_means=old_means,
                    advantages=advantages,
                    clip_range=self.clip_range,
                    reference_means=reference_means,
                    data_loss=data_loss,
                    kl_beta=self.kl_beta,
                    data_beta=self.data_beta if data_loss is not None else 0.0,
                )
                self._phase = "backward"
                (objective.loss * (scale * self.loss_scale / trajectory.step_count)).backward()
                policy_sum += objective.policy_loss.detach()
                total_sum += objective.loss.detach()
                if objective.reference_kl is not None:
                    reference_sum += objective.reference_kl.detach()
                if objective.data_loss is not None:
                    data_sum += objective.data_loss.detach()
                ratios.append(objective.ratio.detach())
                log_ratios.append(objective.log_ratio.detach())
                clip_fractions.append(objective.clip_fraction.detach())
            if not any(parameter.grad is not None for parameter in self.parameters):
                raise RuntimeError("DDRL objective is disconnected from the policy parameters")
            gradient_norm = clip_grad_norm_(
                self.parameters,
                self.max_grad_norm,
                error_if_nonfinite=True,
            )
            self._phase = "optimizer"
            optimizer_started = True
            self.optimizer.step()
            self.optimizer_steps += 1
            self.global_step += 1
            self.last_trajectory_id = trajectory.trajectory_id
            self._phase = "idle"
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            if optimizer_started:
                self._poisoned = True
            else:
                self._phase = "idle"
            raise

        step_count = float(trajectory.step_count)
        policy_loss = policy_sum / step_count
        total_loss = total_sum / step_count
        reference_kl = reference_sum / step_count if self.kl_beta > 0 else None
        data_component = data_sum / step_count if self.data_beta > 0 else None
        ratio_tensor = torch.stack(ratios, dim=1)
        log_ratio_tensor = torch.stack(log_ratios, dim=1)
        metrics: dict[str, object] = {
            "global_step": torch.tensor(
                self.global_step,
                device=total_loss.device,
                dtype=torch.int64,
            ),
            "train_on_steps": torch.tensor(
                trajectory.step_count,
                device=total_loss.device,
                dtype=torch.int64,
            ),
            "ratio_mean": ratio_tensor.mean(),
            "ratio_std": ratio_tensor.std(correction=0),
            "ratio_min": ratio_tensor.min(),
            "ratio_max": ratio_tensor.max(),
            "log_ratio_mean": log_ratio_tensor.mean(),
            "clip_fraction": torch.stack(clip_fractions).mean(),
            "gradient_norm": gradient_norm.detach().float(),
        }
        if reference_kl is not None:
            metrics["reference_kl"] = reference_kl
        if data_component is not None:
            metrics["data_loss"] = data_component
        return DDRLStepResult(
            loss=total_loss.float(),
            policy_loss=policy_loss.float(),
            reference_kl=None if reference_kl is None else reference_kl.float(),
            data_loss=None if data_component is None else data_component.float(),
            advantages=advantages.float(),
            ratios=ratio_tensor.float(),
            gradient_norm=gradient_norm.detach().float(),
            train_on=trajectory.train_on,
            metrics=metrics,
        )

    def state_dict(self) -> dict[str, object]:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("cannot checkpoint a partially committed DDRL update")
        return {
            "schema": DDRL_ENGINE_STATE_SCHEMA,
            "global_step": self.global_step,
            "optimizer_steps": self.optimizer_steps,
            "last_trajectory_id": self.last_trajectory_id,
            "clip_range": self.clip_range,
            "loss_scale": self.loss_scale,
            "advantage_epsilon": self.advantage_epsilon,
            "advantage_normalization": self.advantage_normalization,
            "advantage_clip_min": self.advantage_clip_min,
            "advantage_clip_max": self.advantage_clip_max,
            "exponential_advantage": self.exponential_advantage,
            "kl_beta": self.kl_beta,
            "data_beta": self.data_beta,
            "data_on_first_step_only": self.data_on_first_step_only,
            "max_grad_norm": self.max_grad_norm,
            "data_parallel_size": self.parallel_context.world_size,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("DDRL engine state must be a mapping")
        if set(state_dict) != _ENGINE_STATE_FIELDS:
            raise ValueError("DDRL engine state fields differ from the active schema")
        if state_dict["schema"] != DDRL_ENGINE_STATE_SCHEMA:
            raise ValueError(f"unsupported DDRL engine schema: {state_dict['schema']!r}")
        configured_values = {
            "clip_range": self.clip_range,
            "loss_scale": self.loss_scale,
            "advantage_epsilon": self.advantage_epsilon,
            "advantage_normalization": self.advantage_normalization,
            "advantage_clip_min": self.advantage_clip_min,
            "advantage_clip_max": self.advantage_clip_max,
            "exponential_advantage": self.exponential_advantage,
            "kl_beta": self.kl_beta,
            "data_beta": self.data_beta,
            "data_on_first_step_only": self.data_on_first_step_only,
            "max_grad_norm": self.max_grad_norm,
            "data_parallel_size": self.parallel_context.world_size,
        }
        if any(state_dict[name] != value for name, value in configured_values.items()):
            raise ValueError("saved DDRL configuration differs from the active engine")
        global_step = non_negative_int(state_dict["global_step"], field_name="global_step")
        optimizer_steps = non_negative_int(
            state_dict["optimizer_steps"],
            field_name="optimizer_steps",
        )
        if global_step != optimizer_steps:
            raise ValueError("saved DDRL optimizer/global counters differ")
        last_trajectory_id = state_dict["last_trajectory_id"]
        if last_trajectory_id is not None and (
            not isinstance(last_trajectory_id, str) or not last_trajectory_id.strip()
        ):
            raise ValueError("saved DDRL trajectory identity is invalid")
        if (global_step == 0) != (last_trajectory_id is None):
            raise ValueError("saved DDRL trajectory identity is inconsistent with its step")
        self.global_step = global_step
        self.optimizer_steps = optimizer_steps
        self.last_trajectory_id = last_trajectory_id
        self._phase = "idle"
        self._poisoned = False
        self.optimizer.zero_grad(set_to_none=True)


__all__ = ["DDRL_ENGINE_STATE_SCHEMA", "DDRLStepResult", "NativeDDRLEngine"]
