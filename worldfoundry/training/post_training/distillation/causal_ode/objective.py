"""Official-semantics Causal ODE trajectory regression objective."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ..causal.contracts import CausalCleanPredictionAdapter
from .config import CausalODEConfig
from .contracts import CausalODETrainingBatch


def _frame_axis(frame_dim: int, ndim: int) -> int:
    axis = frame_dim if frame_dim >= 0 else ndim + frame_dim
    if axis <= 0 or axis >= ndim:
        raise ValueError(f"frame_dim {frame_dim} is invalid for a {ndim}-dimensional latent")
    return axis


@dataclass(frozen=True, slots=True)
class PreparedCausalODEBatch:
    batch: CausalODETrainingBatch
    noisy_latents: Tensor
    clean_context: Tensor
    target_latents: Tensor
    trajectory_indices: Tensor
    timesteps: Tensor
    valid_frame_mask: Tensor
    loss_denominator: int


@dataclass(frozen=True, slots=True)
class CausalODELossResult:
    loss: Tensor
    metrics: Mapping[str, object]


class CausalODEObjective:
    """Select one trajectory state per sample and regress the shared clean target."""

    def __init__(self, student: CausalCleanPredictionAdapter, config: CausalODEConfig) -> None:
        if not isinstance(student, CausalCleanPredictionAdapter):
            raise TypeError("student must implement CausalCleanPredictionAdapter")
        if not isinstance(config, CausalODEConfig):
            raise TypeError("config must be CausalODEConfig")
        self.student = student
        self.config = config

    @property
    def num_trajectory_steps(self) -> int:
        return len(self.config.trajectory_timesteps)

    def sample_trajectory_indices(
        self,
        batch: CausalODETrainingBatch,
        *,
        generator: torch.Generator,
    ) -> Tensor:
        trajectories = batch.ode_trajectories
        if not isinstance(trajectories, Tensor):
            raise TypeError("native Causal ODE requires torch.Tensor trajectories")
        return torch.randint(
            self.num_trajectory_steps,
            (batch.batch_size,),
            device=trajectories.device,
            generator=generator,
            dtype=torch.int64,
        )

    def prepare(
        self,
        batch: CausalODETrainingBatch,
        trajectory_indices: Tensor,
    ) -> PreparedCausalODEBatch:
        if not isinstance(batch, CausalODETrainingBatch):
            raise TypeError("batch must be CausalODETrainingBatch")
        trajectories = batch.ode_trajectories
        if not isinstance(trajectories, Tensor) or not trajectories.is_floating_point():
            raise TypeError("ode_trajectories must be a floating torch.Tensor")
        expected_states = self.num_trajectory_steps + 1
        if trajectories.shape[1] != expected_states:
            raise ValueError(
                f"ODE trajectory has {trajectories.shape[1]} states; expected {expected_states}"
            )
        if (
            not isinstance(trajectory_indices, Tensor)
            or trajectory_indices.dtype != torch.int64
            or tuple(trajectory_indices.shape) != (batch.batch_size,)
        ):
            raise TypeError("trajectory_indices must be an int64 tensor with shape [B]")
        indices = trajectory_indices.to(device=trajectories.device)
        if bool(((indices < 0) | (indices >= self.num_trajectory_steps)).any()):
            raise ValueError("trajectory_indices are outside the configured trajectory")

        batch_indices = torch.arange(batch.batch_size, device=trajectories.device)
        noisy = trajectories[batch_indices, indices]
        clean = trajectories[:, -1]
        target = trajectories[:, -2]
        if noisy.shape != clean.shape or noisy.shape != target.shape:
            raise RuntimeError("ODE trajectory states do not share one latent shape")
        axis = _frame_axis(self.config.frame_dim, noisy.ndim)
        frame_count = int(noisy.shape[axis])
        selected = torch.as_tensor(
            self.config.trajectory_timesteps,
            device=noisy.device,
            dtype=torch.float32,
        )[indices]
        timesteps = selected[:, None].expand(batch.batch_size, frame_count)
        valid_frame_mask = timesteps != 0
        elements_per_frame = noisy.numel() // (batch.batch_size * frame_count)
        denominator = int(valid_frame_mask.sum().item()) * elements_per_frame
        if denominator <= 0:
            raise ValueError("Causal ODE sampled an empty t!=0 loss mask")
        return PreparedCausalODEBatch(
            batch=batch,
            noisy_latents=noisy,
            clean_context=clean,
            target_latents=target,
            trajectory_indices=indices,
            timesteps=timesteps,
            valid_frame_mask=valid_frame_mask,
            loss_denominator=denominator,
        )

    def loss(self, prepared: PreparedCausalODEBatch) -> CausalODELossResult:
        if not isinstance(prepared, PreparedCausalODEBatch):
            raise TypeError("prepared must be PreparedCausalODEBatch")
        prediction = self.student.predict_clean(
            prepared.noisy_latents,
            prepared.timesteps,
            clean_context=prepared.clean_context,
            sample_ids=prepared.batch.sample_ids,
            conditioning=prepared.batch.conditioning,
            training=True,
        )
        if not isinstance(prediction, Tensor) or prediction.shape != prepared.target_latents.shape:
            raise ValueError("Causal ODE student prediction must match the target latent shape")
        axis = _frame_axis(self.config.frame_dim, prediction.ndim)
        prediction_by_frame = prediction.movedim(axis, 1)
        target_by_frame = prepared.target_latents.movedim(axis, 1)
        selected_prediction = prediction_by_frame[prepared.valid_frame_mask]
        selected_target = target_by_frame[prepared.valid_frame_mask]
        if selected_prediction.numel() != prepared.loss_denominator:
            raise RuntimeError("Causal ODE realized loss mask differs from its declared denominator")
        loss = F.mse_loss(selected_prediction, selected_target, reduction="mean")
        if not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError("non-finite Causal ODE loss")
        per_sample = (prediction - prepared.target_latents).float().square().flatten(1).mean(dim=1)
        return CausalODELossResult(
            loss=loss,
            metrics={
                "loss_denominator": prepared.loss_denominator,
                "trajectory_indices": prepared.trajectory_indices.detach(),
                "timesteps": prepared.timesteps[:, 0].detach(),
                "per_sample_mse": per_sample.detach(),
            },
        )


__all__ = ["CausalODELossResult", "CausalODEObjective", "PreparedCausalODEBatch"]
