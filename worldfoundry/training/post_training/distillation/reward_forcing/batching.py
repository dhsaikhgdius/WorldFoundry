"""Prompt-preserving data loader for Reward-Forcing."""

from __future__ import annotations

from collections.abc import Iterator

from worldfoundry.training.data.rollout_cache import RolloutConditionedPrompt

from ..self_forcing.batching import NativeSelfForcingDataLoader
from .contracts import RewardForcingTrainingBatch


class NativeRewardForcingDataLoader(NativeSelfForcingDataLoader):
    """Reuse prompt-only conditioning batches while retaining reward text."""

    def __iter__(self) -> Iterator[RewardForcingTrainingBatch]:
        for values in self.source:
            base = self._batch(values)
            if not isinstance(values, tuple) or not all(
                isinstance(value, RolloutConditionedPrompt) for value in values
            ):
                raise TypeError("Reward-Forcing prompt source must emit conditioned prompt tuples")
            yield RewardForcingTrainingBatch(
                sample_ids=base.sample_ids,
                clean_latents=base.clean_latents,
                conditioning=base.conditioning,
                unconditional_conditioning=base.unconditional_conditioning,
                loss_mask=base.loss_mask,
                sample_weights=base.sample_weights,
                prompts=tuple(value.record.prompt for value in values),
            )


__all__ = ["NativeRewardForcingDataLoader"]
