"""Stateful WorldFoundry-native DiffusionNFT optimizer engine."""

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

from ....shared.contracts import FlowPredictionAdapter
from ....shared.distributed import PostTrainingParallelContext
from ....shared.validation import non_negative_int, positive_float
from .contracts import DiffusionNFTRollout, OldPolicyRefresh, validate_mix_beta
from .objective import (
    DiffusionNFTLoss,
    diffusion_nft_forward_process,
    diffusion_nft_loss,
    diffusion_nft_reward_weights,
)

DIFFUSION_NFT_ENGINE_STATE_SCHEMA = "worldfoundry-diffusion-nft-engine"
_DIFFUSION_NFT_ENGINE_STATE_FIELDS = frozenset(
    {
        "schema",
        "global_step",
        "optimizer_steps",
        "old_policy_refreshes",
        "last_old_policy_refresh_step",
        "last_collection_id",
        "initial_old_policy_revision",
        "collection_policy_revision",
        "beta",
        "advantage_clip_max",
        "advantage_epsilon",
        "advantage_normalization",
        "advantage_mode",
        "reference_mse_weight",
        "reconstruction_mae_floor",
        "max_grad_norm",
        "old_policy_refresh",
        "updates_per_rollout",
        "data_parallel_size",
    }
)


@dataclass(frozen=True, slots=True)
class DiffusionNFTStepResult:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    reference_mse: torch.Tensor | None
    advantages: torch.Tensor
    reward_probabilities: torch.Tensor
    times: torch.Tensor
    gradient_norm: torch.Tensor
    old_policy_refreshed: bool
    old_policy_retention: float | None
    metrics: Mapping[str, object]


def _adapter_module(adapter: FlowPredictionAdapter, *, role: str) -> nn.Module:
    if not isinstance(adapter, FlowPredictionAdapter):
        raise TypeError(f"{role} must implement FlowPredictionAdapter")
    module = adapter.module
    if not isinstance(module, nn.Module):
        raise TypeError(f"{role}.module must be an nn.Module")
    return module


def _parameter_inventory(module: nn.Module) -> dict[str, nn.Parameter]:
    return dict(module.named_parameters())


def _buffer_inventory(module: nn.Module) -> dict[str, torch.Tensor]:
    return dict(module.named_buffers())


def _audit_matching_modules(policy: nn.Module, old_policy: nn.Module) -> None:
    policy_parameters = _parameter_inventory(policy)
    old_parameters = _parameter_inventory(old_policy)
    if set(policy_parameters) != set(old_parameters):
        raise ValueError("policy and old policy parameter names must match exactly")
    for name, parameter in policy_parameters.items():
        old_parameter = old_parameters[name]
        if parameter.shape != old_parameter.shape or parameter.dtype != old_parameter.dtype:
            raise ValueError(f"policy and old policy parameter {name!r} differ in shape or dtype")
    policy_buffers = _buffer_inventory(policy)
    old_buffers = _buffer_inventory(old_policy)
    if set(policy_buffers) != set(old_buffers):
        raise ValueError("policy and old policy buffer names must match exactly")
    for name, buffer in policy_buffers.items():
        old_buffer = old_buffers[name]
        if buffer.shape != old_buffer.shape or buffer.dtype != old_buffer.dtype:
            raise ValueError(f"policy and old policy buffer {name!r} differ in shape or dtype")


def _audit_no_shared_parameters(left: nn.Module, right: nn.Module, *, roles: str) -> None:
    left_ids = {id(parameter) for parameter in left.parameters()}
    right_ids = {id(parameter) for parameter in right.parameters()}
    if left_ids & right_ids:
        raise ValueError(f"{roles} cannot share parameter objects")


class NativeDiffusionNFTEngine:
    """Own one forward-process update and the independent old-policy anchor."""

    def __init__(
        self,
        policy: FlowPredictionAdapter,
        old_policy: FlowPredictionAdapter,
        optimizer: torch.optim.Optimizer,
        *,
        initial_old_policy_revision: str,
        beta: float,
        advantage_clip_max: float,
        advantage_epsilon: float = 1.0e-4,
        advantage_normalization: str = "group-population-std",
        advantage_mode: str = "all",
        reference_policy: FlowPredictionAdapter | None = None,
        reference_mse_weight: float = 0.0,
        reconstruction_mae_floor: float = 1.0e-5,
        max_grad_norm: float = 1.0,
        old_policy_refresh: OldPolicyRefresh | None = None,
        updates_per_rollout: int = 1,
        parallel_context: PostTrainingParallelContext | None = None,
    ) -> None:
        policy_module = _adapter_module(policy, role="policy")
        old_policy_module = _adapter_module(old_policy, role="old_policy")
        if policy_module is old_policy_module:
            raise ValueError("policy and old policy must be distinct modules")
        _audit_no_shared_parameters(
            policy_module,
            old_policy_module,
            roles="policy and old policy",
        )
        _audit_matching_modules(policy_module, old_policy_module)
        if isinstance(updates_per_rollout, bool) or int(updates_per_rollout) != 1:
            raise ValueError(
                "DiffusionNFT supports exactly one optimizer update per collected rollout; "
                "multi-update replay would reuse a stale old-policy anchor"
            )
        resolved_advantage_normalization = advantage_normalization_mode(
            advantage_normalization,
            field_name="advantage_normalization",
        )
        resolved_advantage_mode = str(advantage_mode).strip().lower().replace("-", "_")
        if resolved_advantage_mode not in {
            "all",
            "positive_only",
            "negative_only",
            "one_only",
            "binary",
        }:
            raise ValueError("advantage_mode must be all, positive_only, negative_only, one_only, or binary")
        resolved_reference_weight = float(reference_mse_weight)
        if not isfinite(resolved_reference_weight) or resolved_reference_weight < 0:
            raise ValueError("reference_mse_weight must be finite and non-negative")
        reference_module: nn.Module | None = None
        if reference_policy is not None:
            reference_module = _adapter_module(reference_policy, role="reference_policy")
            if reference_module is policy_module or reference_module is old_policy_module:
                raise ValueError("reference policy must be a distinct module")
            _audit_no_shared_parameters(
                policy_module,
                reference_module,
                roles="policy and reference policy",
            )
            _audit_no_shared_parameters(
                old_policy_module,
                reference_module,
                roles="old policy and reference policy",
            )
        elif resolved_reference_weight != 0:
            raise ValueError("positive reference_mse_weight requires reference_policy")

        parameters = trainable_parameters(policy_module)
        audit_optimizer_parameters(optimizer, parameters, role="DiffusionNFT policy")
        old_policy_module.requires_grad_(False)
        old_policy_module.eval()
        if reference_module is not None:
            reference_module.requires_grad_(False)
            reference_module.eval()

        self.policy = policy
        self.old_policy = old_policy
        self.reference_policy = reference_policy
        self.policy_module = policy_module
        self.old_policy_module = old_policy_module
        self.reference_module = reference_module
        self.optimizer = optimizer
        self.parameters = parameters
        resolved_initial_revision = str(initial_old_policy_revision).strip()
        if not resolved_initial_revision:
            raise ValueError("initial_old_policy_revision must be a non-empty string")
        self.initial_old_policy_revision = resolved_initial_revision
        self.beta = validate_mix_beta(beta)
        self.advantage_clip_max = positive_float(
            advantage_clip_max,
            field_name="advantage_clip_max",
        )
        self.advantage_epsilon = positive_float(
            advantage_epsilon,
            field_name="advantage_epsilon",
        )
        self.advantage_normalization = resolved_advantage_normalization
        self.advantage_mode = resolved_advantage_mode
        self.reference_mse_weight = resolved_reference_weight
        self.reconstruction_mae_floor = positive_float(
            reconstruction_mae_floor,
            field_name="reconstruction_mae_floor",
        )
        self.max_grad_norm = positive_float(max_grad_norm, field_name="max_grad_norm")
        self.old_policy_refresh = old_policy_refresh or OldPolicyRefresh()
        if not isinstance(self.old_policy_refresh, OldPolicyRefresh):
            raise TypeError("old_policy_refresh must be OldPolicyRefresh")
        self.parallel_context = parallel_context or PostTrainingParallelContext.current()
        self.parallel_context.audit_synchronized_module(
            policy_module,
            role="DiffusionNFT policy",
        )
        self.global_step = 0
        self.optimizer_steps = 0
        self.old_policy_refreshes = 0
        self.last_old_policy_refresh_step = -1
        self.last_collection_id: str | None = None
        self._phase = "idle"
        self._poisoned = False
        self._refresh_old_policy_parameters(retention=0.0)

    @property
    def current_collection_policy_revision(self) -> str:
        """Identity of the exact old-policy state used for the next collection."""

        if self.old_policy_refreshes == 0:
            return self.initial_old_policy_revision
        return (
            f"{self.initial_old_policy_revision}:refresh-{self.old_policy_refreshes}"
            f":step-{self.last_old_policy_refresh_step}"
        )

    @torch.no_grad()
    def _refresh_old_policy_parameters(self, *, retention: float) -> None:
        if not isfinite(retention) or not 0 <= retention <= 1:
            raise ValueError("old-policy retention must be finite and in [0,1]")
        policy_parameters = _parameter_inventory(self.policy_module)
        old_parameters = _parameter_inventory(self.old_policy_module)
        for name, old_parameter in old_parameters.items():
            policy_parameter = policy_parameters[name].detach().to(device=old_parameter.device)
            old_parameter.mul_(retention).add_(policy_parameter, alpha=1 - retention)
        policy_buffers = _buffer_inventory(self.policy_module)
        old_buffers = _buffer_inventory(self.old_policy_module)
        for name, old_buffer in old_buffers.items():
            old_buffer.copy_(policy_buffers[name].detach().to(device=old_buffer.device))
        self.old_policy_module.requires_grad_(False)
        self.old_policy_module.eval()

    def _predict(
        self,
        adapter: FlowPredictionAdapter,
        noisy_latents: torch.Tensor,
        times: torch.Tensor,
        batch: DiffusionNFTRollout,
        *,
        training: bool,
    ) -> torch.Tensor:
        prediction = adapter.predict_velocity(
            noisy_latents,
            times,
            sample_ids=batch.sample_ids,
            conditioning=batch.conditioning,
            training=training,
        )
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("DiffusionNFT flow prediction must be a torch.Tensor")
        return prediction

    def train_step(
        self,
        batch: DiffusionNFTRollout,
        *,
        generator: torch.Generator | None = None,
    ) -> DiffusionNFTStepResult:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("DiffusionNFT engine has a partially committed update; restore the last checkpoint")
        if not isinstance(batch, DiffusionNFTRollout):
            raise TypeError("batch must be DiffusionNFTRollout")
        if batch.collection_id == self.last_collection_id:
            raise ValueError("a collected DiffusionNFT rollout can be optimized only once")
        if batch.policy_revision != self.current_collection_policy_revision:
            raise ValueError(
                "DiffusionNFT rollout was collected by a stale behavior policy: "
                f"expected {self.current_collection_policy_revision!r}, "
                f"got {batch.policy_revision!r}"
            )
        self.parallel_context.audit_local_group_ownership(batch.group_ids)
        clean = batch.clean_latents
        rewards = batch.rewards.to(device=clean.device)
        reward_weights = diffusion_nft_reward_weights(
            rewards,
            batch.group_ids,
            advantage_clip_max=self.advantage_clip_max,
            epsilon=self.advantage_epsilon,
            normalization=self.advantage_normalization,
            advantage_mode=self.advantage_mode,
            parallel_context=self.parallel_context,
        )
        times = torch.rand(
            (batch.batch_size,),
            device=clean.device,
            dtype=torch.float32,
            generator=generator,
        )
        noise = torch.randn(
            tuple(clean.shape),
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
        forward = diffusion_nft_forward_process(clean, times, noise)
        self.policy_module.train()
        self.old_policy_module.eval()
        if self.reference_module is not None:
            self.reference_module.eval()

        optimizer_step_started = False
        self.optimizer.zero_grad(set_to_none=True)
        try:
            self._phase = "forward"
            with torch.no_grad():
                old_prediction = self._predict(
                    self.old_policy,
                    forward.noisy_latents,
                    forward.times,
                    batch,
                    training=False,
                ).detach()
                reference_prediction = (
                    None
                    if self.reference_policy is None
                    else self._predict(
                        self.reference_policy,
                        forward.noisy_latents,
                        forward.times,
                        batch,
                        training=False,
                    ).detach()
                )
            policy_prediction = self._predict(
                self.policy,
                forward.noisy_latents,
                forward.times,
                batch,
                training=True,
            )
            result: DiffusionNFTLoss = diffusion_nft_loss(
                clean_latents=clean,
                noisy_latents=forward.noisy_latents,
                times=forward.times,
                target_velocity=forward.target_velocity,
                policy_prediction=policy_prediction,
                old_policy_prediction=old_prediction,
                reward_probabilities=reward_weights.reward_probabilities,
                beta=self.beta,
                advantage_clip_max=self.advantage_clip_max,
                reference_prediction=reference_prediction,
                reference_mse_weight=self.reference_mse_weight,
                reconstruction_mae_floor=self.reconstruction_mae_floor,
            )
            self._phase = "backward"
            self.parallel_context.scale_local_mean(result.loss, batch.batch_size).backward()
            gradient_norm = clip_grad_norm_(
                self.parameters,
                self.max_grad_norm,
                error_if_nonfinite=True,
            )
            optimizer_step_started = True
            self.optimizer.step()
            self.optimizer_steps += 1
            self.global_step += 1
            self._phase = "policy-committed"
            refreshed = self.old_policy_refresh.should_refresh(self.optimizer_steps)
            retention: float | None = None
            if refreshed:
                retention = self.old_policy_refresh.retention(self.optimizer_steps)
                self._refresh_old_policy_parameters(retention=retention)
                self.old_policy_refreshes += 1
                self.last_old_policy_refresh_step = self.optimizer_steps
            self.last_collection_id = batch.collection_id
            self._phase = "idle"
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            if optimizer_step_started:
                self._poisoned = True
            else:
                self._phase = "idle"
            raise

        reference_mse = None if result.reference_mse is None else result.reference_mse.detach().float()
        metrics: dict[str, object] = {
            "global_step": torch.tensor(self.global_step, device=clean.device, dtype=torch.int64),
            "policy_loss": result.policy_loss.detach().float(),
            "flow_matching_mse": result.flow_matching_mse.detach().float(),
            "old_policy_mse": result.old_policy_mse.detach().float(),
            "positive_reconstruction": result.positive_reconstruction.detach().float().mean(),
            "negative_reconstruction": result.negative_reconstruction.detach().float().mean(),
            "reward_probability_mean": reward_weights.reward_probabilities.detach().float().mean(),
            "gradient_norm": gradient_norm.detach().float(),
            "old_policy_refreshes": torch.tensor(
                self.old_policy_refreshes,
                device=clean.device,
                dtype=torch.int64,
            ),
        }
        if reference_mse is not None:
            metrics["reference_mse"] = reference_mse
        return DiffusionNFTStepResult(
            loss=result.loss.detach().float(),
            policy_loss=result.policy_loss.detach().float(),
            reference_mse=reference_mse,
            advantages=reward_weights.advantages.detach().float(),
            reward_probabilities=reward_weights.reward_probabilities.detach().float(),
            times=forward.times.detach().float(),
            gradient_norm=gradient_norm.detach().float(),
            old_policy_refreshed=refreshed,
            old_policy_retention=retention,
            metrics=metrics,
        )

    def state_dict(self) -> dict[str, object]:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("cannot checkpoint a partially committed DiffusionNFT update")
        return {
            "schema": DIFFUSION_NFT_ENGINE_STATE_SCHEMA,
            "global_step": self.global_step,
            "optimizer_steps": self.optimizer_steps,
            "old_policy_refreshes": self.old_policy_refreshes,
            "last_old_policy_refresh_step": self.last_old_policy_refresh_step,
            "last_collection_id": self.last_collection_id,
            "initial_old_policy_revision": self.initial_old_policy_revision,
            "collection_policy_revision": self.current_collection_policy_revision,
            "beta": self.beta,
            "advantage_clip_max": self.advantage_clip_max,
            "advantage_epsilon": self.advantage_epsilon,
            "advantage_normalization": self.advantage_normalization,
            "advantage_mode": self.advantage_mode,
            "reference_mse_weight": self.reference_mse_weight,
            "reconstruction_mae_floor": self.reconstruction_mae_floor,
            "max_grad_norm": self.max_grad_norm,
            "old_policy_refresh": self.old_policy_refresh.state_dict(),
            "updates_per_rollout": 1,
            "data_parallel_size": self.parallel_context.world_size,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("DiffusionNFT engine state must be a mapping")
        if set(state_dict) != _DIFFUSION_NFT_ENGINE_STATE_FIELDS:
            raise ValueError("DiffusionNFT engine state fields differ from the active schema")
        if state_dict["schema"] != DIFFUSION_NFT_ENGINE_STATE_SCHEMA:
            raise ValueError(f"unsupported DiffusionNFT engine schema: {state_dict['schema']!r}")
        saved_refresh = OldPolicyRefresh.from_state_dict(state_dict["old_policy_refresh"])
        configured_values = {
            "beta": self.beta,
            "advantage_clip_max": self.advantage_clip_max,
            "advantage_epsilon": self.advantage_epsilon,
            "advantage_normalization": self.advantage_normalization,
            "advantage_mode": self.advantage_mode,
            "reference_mse_weight": self.reference_mse_weight,
            "reconstruction_mae_floor": self.reconstruction_mae_floor,
            "max_grad_norm": self.max_grad_norm,
            "updates_per_rollout": 1,
            "data_parallel_size": self.parallel_context.world_size,
            "initial_old_policy_revision": self.initial_old_policy_revision,
        }
        if any(state_dict[name] != value for name, value in configured_values.items()):
            raise ValueError("saved DiffusionNFT configuration differs from the active engine")
        if saved_refresh != self.old_policy_refresh:
            raise ValueError("saved old-policy refresh differs from the active engine")
        global_step = non_negative_int(state_dict["global_step"], field_name="global_step")
        optimizer_steps = non_negative_int(
            state_dict["optimizer_steps"],
            field_name="optimizer_steps",
        )
        refreshes = non_negative_int(
            state_dict["old_policy_refreshes"],
            field_name="old_policy_refreshes",
        )
        if optimizer_steps != global_step:
            raise ValueError("saved DiffusionNFT optimizer/global counters differ")
        expected_refreshes = optimizer_steps // self.old_policy_refresh.update_interval
        if refreshes != expected_refreshes:
            raise ValueError("saved DiffusionNFT old-policy refresh count violates its cadence")
        last_refresh = int(state_dict["last_old_policy_refresh_step"])
        expected_last_refresh = (
            -1 if expected_refreshes == 0 else expected_refreshes * self.old_policy_refresh.update_interval
        )
        if last_refresh != expected_last_refresh:
            raise ValueError("saved DiffusionNFT last old-policy refresh step is inconsistent")
        last_collection_id = state_dict["last_collection_id"]
        if last_collection_id is not None and (
            not isinstance(last_collection_id, str) or not last_collection_id.strip()
        ):
            raise ValueError("saved DiffusionNFT collection identity is invalid")
        expected_collection_revision = (
            self.initial_old_policy_revision
            if refreshes == 0
            else (
                f"{self.initial_old_policy_revision}:refresh-{refreshes}"
                f":step-{last_refresh}"
            )
        )
        if state_dict["collection_policy_revision"] != expected_collection_revision:
            raise ValueError("saved DiffusionNFT collection-policy revision is inconsistent")
        self.optimizer.zero_grad(set_to_none=True)
        self.old_policy_module.requires_grad_(False)
        self.old_policy_module.eval()
        if self.reference_module is not None:
            self.reference_module.requires_grad_(False)
            self.reference_module.eval()
        self.global_step = global_step
        self.optimizer_steps = optimizer_steps
        self.old_policy_refreshes = refreshes
        self.last_old_policy_refresh_step = last_refresh
        self.last_collection_id = last_collection_id
        self._phase = "idle"
        self._poisoned = False


__all__ = [
    "DIFFUSION_NFT_ENGINE_STATE_SCHEMA",
    "DiffusionNFTStepResult",
    "NativeDiffusionNFTEngine",
]
