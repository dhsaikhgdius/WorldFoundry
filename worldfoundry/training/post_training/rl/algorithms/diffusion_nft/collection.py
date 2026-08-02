"""Terminal-only flow collection for native DiffusionNFT."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

import torch

from ....shared.contracts import FlowPredictionAdapter
from ...contracts import FlowRolloutBatch
from ...transitions.flow_sde import flow_ode_step
from .contracts import DiffusionNFTTerminalLatents


def _slice_conditioning(
    conditioning: Mapping[str, object],
    *,
    start: int,
    end: int,
    batch_size: int,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in conditioning.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == batch_size:
            result[str(key)] = value[start:end]
        else:
            result[str(key)] = value
    return result


class NativeDiffusionNFTTerminalCollector:
    """Run no-grad flow ODE collection and retain only its terminal state."""

    def __init__(
        self,
        policy: FlowPredictionAdapter,
        *,
        sigmas: tuple[float, ...],
        group_size: int,
        latent_dtype: torch.dtype,
        forward_batch_size: int | None = None,
    ) -> None:
        if not isinstance(policy, FlowPredictionAdapter):
            raise TypeError("DiffusionNFT collection policy must implement FlowPredictionAdapter")
        if latent_dtype not in {torch.bfloat16, torch.float16, torch.float32}:
            raise ValueError("DiffusionNFT collection latent_dtype must be bfloat16, float16, or float32")
        schedule = tuple(float(value) for value in sigmas)
        if (
            len(schedule) < 2
            or schedule[0] != 1.0
            or schedule[-1] != 0.0
            or any(left <= right for left, right in zip(schedule, schedule[1:]))
        ):
            raise ValueError("DiffusionNFT collection requires a descending sigma schedule from 1 to 0")
        if isinstance(group_size, bool) or int(group_size) < 2:
            raise ValueError("DiffusionNFT collection group_size must be at least two")
        if forward_batch_size is not None and (isinstance(forward_batch_size, bool) or int(forward_batch_size) <= 0):
            raise ValueError("DiffusionNFT collection forward_batch_size must be positive")
        self.policy = policy
        self.module = policy.module
        self.sigmas = schedule
        self.group_size = int(group_size)
        self.latent_dtype = latent_dtype
        self.forward_batch_size = None if forward_batch_size is None else int(forward_batch_size)

    def _predict_velocity(
        self,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        batch: FlowRolloutBatch,
    ) -> torch.Tensor:
        batch_size = int(latents.shape[0])
        chunk_size = self.forward_batch_size or batch_size
        predictions: list[torch.Tensor] = []
        for start in range(0, batch_size, chunk_size):
            end = min(batch_size, start + chunk_size)
            prediction = self.policy.predict_velocity(
                latents[start:end],
                sigma[start:end],
                sample_ids=batch.sample_ids[start:end],
                conditioning=_slice_conditioning(
                    batch.conditioning,
                    start=start,
                    end=end,
                    batch_size=batch_size,
                ),
                training=False,
            )
            if not isinstance(prediction, torch.Tensor) or prediction.shape != latents[start:end].shape:
                raise ValueError("DiffusionNFT collection prediction must match its latent chunk")
            predictions.append(prediction)
        return torch.cat(predictions, dim=0)

    def _validate_batch(self, batch: FlowRolloutBatch) -> torch.Tensor:
        if not isinstance(batch, FlowRolloutBatch):
            raise TypeError("DiffusionNFT collector requires FlowRolloutBatch")
        counts = Counter(batch.group_ids)
        invalid = sorted(group for group, count in counts.items() if count != self.group_size)
        if invalid:
            raise ValueError(f"DiffusionNFT rollout group sizes differ from the collection recipe: {invalid}")
        schedule = batch.sigmas
        if not isinstance(schedule, torch.Tensor):
            raise TypeError("DiffusionNFT rollout sigmas must be a torch.Tensor")
        schedule = schedule.detach().to(device=batch.initial_latents.device, dtype=torch.float32)
        expected = torch.tensor(self.sigmas, device=schedule.device, dtype=torch.float32)
        if schedule.ndim == 1:
            matches = tuple(schedule.shape) == tuple(expected.shape) and torch.equal(
                schedule,
                expected,
            )
        elif schedule.ndim == 2 and int(schedule.shape[0]) == batch.batch_size:
            matches = tuple(schedule.shape[1:]) == tuple(expected.shape) and torch.equal(
                schedule,
                expected.unsqueeze(0).expand_as(schedule),
            )
        else:
            matches = False
        if not matches:
            raise ValueError("DiffusionNFT rollout sigmas differ from the collection recipe")
        return schedule

    def collect(
        self,
        batch: FlowRolloutBatch,
        *,
        collection_id: str,
    ) -> DiffusionNFTTerminalLatents:
        schedule = self._validate_batch(batch)
        current = batch.initial_latents.detach().to(dtype=self.latent_dtype)
        batch_size = batch.batch_size
        with torch.no_grad():
            for index in range(len(self.sigmas) - 1):
                if schedule.ndim == 1:
                    sigma = schedule[index].expand(batch_size)
                    sigma_next = schedule[index + 1].expand(batch_size)
                else:
                    sigma = schedule[:, index]
                    sigma_next = schedule[:, index + 1]
                velocity = self._predict_velocity(current, sigma, batch)
                current = flow_ode_step(
                    velocity,
                    current,
                    sigma,
                    sigma_next,
                ).to(dtype=self.latent_dtype)
        return DiffusionNFTTerminalLatents(
            collection_id=collection_id,
            policy_revision=batch.policy_revision,
            sample_ids=batch.sample_ids,
            group_ids=batch.group_ids,
            clean_latents=current.detach(),
            transition_count=len(self.sigmas) - 1,
            conditioning=batch.conditioning,
            metadata=batch.metadata,
        )


__all__ = ["NativeDiffusionNFTTerminalCollector"]
