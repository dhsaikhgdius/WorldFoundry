"""Optimizer engine for paired Diffusion-DPO updates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from worldfoundry.core.gradient import clip_grad_norm_
from worldfoundry.training.optimization import (
    audit_optimizer_parameters,
    trainable_parameters,
)

from ....shared.building import prediction_module
from ....shared.contracts import FlowPredictionAdapter
from ....shared.distributed import PostTrainingParallelContext
from ....shared.validation import non_negative_int, positive_float
from .contracts import DiffusionDPOBatch
from .objective import (
    DiffusionDPOLoss,
    diffusion_dpo_loss,
    sample_diffusion_dpo_forward_process,
)

DIFFUSION_DPO_ENGINE_STATE_SCHEMA = "worldfoundry-diffusion-dpo-engine"
_ENGINE_STATE_FIELDS = frozenset(
    {
        "schema",
        "global_step",
        "optimizer_steps",
        "last_batch_id",
        "beta",
        "max_grad_norm",
        "data_parallel_size",
    }
)


@dataclass(frozen=True, slots=True)
class DiffusionDPOStepResult:
    loss: torch.Tensor
    logits: torch.Tensor
    current_mse: torch.Tensor
    reference_mse: torch.Tensor
    preference_accuracy: torch.Tensor
    times: torch.Tensor
    gradient_norm: torch.Tensor
    metrics: Mapping[str, object]


def _audit_independent_modules(policy: nn.Module, reference: nn.Module) -> None:
    if policy is reference:
        raise ValueError("policy and reference policy must be distinct modules")
    policy_parameters = {id(parameter) for parameter in policy.parameters()}
    reference_parameters = {id(parameter) for parameter in reference.parameters()}
    if policy_parameters & reference_parameters:
        raise ValueError("policy and reference policy cannot share parameter objects")


class NativeDiffusionDPOEngine:
    """Own one frozen-reference preference update over clean latent pairs."""

    def __init__(
        self,
        policy: FlowPredictionAdapter,
        reference_policy: FlowPredictionAdapter,
        optimizer: torch.optim.Optimizer,
        *,
        beta: float,
        max_grad_norm: float = 1.0,
        parallel_context: PostTrainingParallelContext | None = None,
    ) -> None:
        policy_module = prediction_module(policy, role="Diffusion-DPO policy")
        reference_module = prediction_module(
            reference_policy,
            role="Diffusion-DPO reference policy",
        )
        _audit_independent_modules(policy_module, reference_module)
        parameters = trainable_parameters(policy_module)
        audit_optimizer_parameters(optimizer, parameters, role="Diffusion-DPO policy")
        reference_module.requires_grad_(False)
        reference_module.zero_grad(set_to_none=True)
        reference_module.eval()

        self.policy = policy
        self.reference_policy = reference_policy
        self.policy_module = policy_module
        self.reference_module = reference_module
        self.optimizer = optimizer
        self.parameters = parameters
        self.beta = positive_float(beta, field_name="beta")
        self.max_grad_norm = positive_float(max_grad_norm, field_name="max_grad_norm")
        self.parallel_context = parallel_context or PostTrainingParallelContext.current()
        self.parallel_context.audit_synchronized_module(
            policy_module,
            role="Diffusion-DPO policy",
        )
        self.global_step = 0
        self.optimizer_steps = 0
        self.last_batch_id: str | None = None
        self._phase = "idle"
        self._poisoned = False

    def _predict(
        self,
        adapter: FlowPredictionAdapter,
        noisy_latents: torch.Tensor,
        times: torch.Tensor,
        batch: DiffusionDPOBatch,
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
            raise TypeError("Diffusion-DPO flow prediction must be a torch.Tensor")
        return prediction

    def train_step(
        self,
        batch: DiffusionDPOBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> DiffusionDPOStepResult:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("Diffusion-DPO engine has a partially committed update; restore the last checkpoint")
        if not isinstance(batch, DiffusionDPOBatch):
            raise TypeError("batch must be DiffusionDPOBatch")

        forward = sample_diffusion_dpo_forward_process(
            batch.clean_latents,
            generator=generator,
        )
        self.policy_module.train()
        self.reference_module.eval()
        optimizer_started = False
        self.optimizer.zero_grad(set_to_none=True)
        try:
            self._phase = "forward"
            with torch.no_grad():
                reference_prediction = self._predict(
                    self.reference_policy,
                    forward.noisy_latents,
                    forward.times,
                    batch,
                    training=False,
                ).detach()
            policy_prediction = self._predict(
                self.policy,
                forward.noisy_latents,
                forward.times,
                batch,
                training=True,
            )
            result: DiffusionDPOLoss = diffusion_dpo_loss(
                target_velocity=forward.target_velocity,
                policy_prediction=policy_prediction,
                reference_prediction=reference_prediction,
                beta=self.beta,
            )
            self._phase = "backward"
            self.parallel_context.scale_local_mean(result.loss, batch.pair_count).backward()
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
            self.last_batch_id = batch.batch_id
            self._phase = "idle"
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            if optimizer_started:
                self._poisoned = True
            else:
                self._phase = "idle"
            raise

        metrics: dict[str, object] = {
            "global_step": torch.tensor(
                self.global_step,
                device=batch.clean_latents.device,
                dtype=torch.int64,
            ),
            "pair_count": torch.tensor(
                batch.pair_count,
                device=batch.clean_latents.device,
                dtype=torch.int64,
            ),
            "logit_mean": result.logits.detach().float().mean(),
            "preference_accuracy": result.preference_accuracy.detach().float(),
            "current_chosen_mse": result.current_mse[0::2].detach().float().mean(),
            "current_rejected_mse": result.current_mse[1::2].detach().float().mean(),
            "reference_chosen_mse": result.reference_mse[0::2].detach().float().mean(),
            "reference_rejected_mse": result.reference_mse[1::2].detach().float().mean(),
            "gradient_norm": gradient_norm.detach().float(),
        }
        return DiffusionDPOStepResult(
            loss=result.loss.detach().float(),
            logits=result.logits.detach().float(),
            current_mse=result.current_mse.detach().float(),
            reference_mse=result.reference_mse.detach().float(),
            preference_accuracy=result.preference_accuracy.detach().float(),
            times=forward.times.detach().float(),
            gradient_norm=gradient_norm.detach().float(),
            metrics=metrics,
        )

    def state_dict(self) -> dict[str, object]:
        if self._poisoned or self._phase != "idle":
            raise RuntimeError("cannot checkpoint a partially committed Diffusion-DPO update")
        return {
            "schema": DIFFUSION_DPO_ENGINE_STATE_SCHEMA,
            "global_step": self.global_step,
            "optimizer_steps": self.optimizer_steps,
            "last_batch_id": self.last_batch_id,
            "beta": self.beta,
            "max_grad_norm": self.max_grad_norm,
            "data_parallel_size": self.parallel_context.world_size,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("Diffusion-DPO engine state must be a mapping")
        if set(state_dict) != _ENGINE_STATE_FIELDS:
            raise ValueError("Diffusion-DPO engine state fields differ from the active schema")
        if state_dict["schema"] != DIFFUSION_DPO_ENGINE_STATE_SCHEMA:
            raise ValueError(f"unsupported Diffusion-DPO engine schema: {state_dict['schema']!r}")
        configured_values = {
            "beta": self.beta,
            "max_grad_norm": self.max_grad_norm,
            "data_parallel_size": self.parallel_context.world_size,
        }
        if any(state_dict[name] != value for name, value in configured_values.items()):
            raise ValueError("saved Diffusion-DPO configuration differs from the active engine")
        global_step = non_negative_int(state_dict["global_step"], field_name="global_step")
        optimizer_steps = non_negative_int(
            state_dict["optimizer_steps"],
            field_name="optimizer_steps",
        )
        if optimizer_steps != global_step:
            raise ValueError("saved Diffusion-DPO optimizer/global counters differ")
        last_batch_id = state_dict["last_batch_id"]
        if last_batch_id is not None and (not isinstance(last_batch_id, str) or not last_batch_id.strip()):
            raise ValueError("saved Diffusion-DPO batch identity is invalid")
        if (global_step == 0) != (last_batch_id is None):
            raise ValueError("saved Diffusion-DPO batch identity is inconsistent with its step")

        self.global_step = global_step
        self.optimizer_steps = optimizer_steps
        self.last_batch_id = last_batch_id
        self._phase = "idle"
        self._poisoned = False
        self.optimizer.zero_grad(set_to_none=True)
        self.reference_module.requires_grad_(False)
        self.reference_module.zero_grad(set_to_none=True)
        self.reference_module.eval()


__all__ = [
    "DIFFUSION_DPO_ENGINE_STATE_SCHEMA",
    "DiffusionDPOStepResult",
    "NativeDiffusionDPOEngine",
]
