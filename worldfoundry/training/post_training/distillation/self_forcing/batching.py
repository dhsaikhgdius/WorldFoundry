"""Prompt-only batches for data-free causal Self-Forcing distillation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence

import torch

from worldfoundry.training.data.rollout_cache import RolloutConditionedPrompt

from ...shared.batching import batch_shared_conditioning
from ..dmd.contracts import DMDTrainingBatch

SELF_FORCING_DATA_LOADER_STATE_SCHEMA = "worldfoundry-self-forcing-data-loader"


def _batched_conditioning(
    values: Sequence[RolloutConditionedPrompt],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    key_sets = {tuple(sorted(value.conditioning)) for value in values}
    if len(key_sets) != 1 or not key_sets or not next(iter(key_sets)):
        raise ValueError("Self-Forcing prompts must expose identical non-empty condition keys")
    result: dict[str, torch.Tensor] = {}
    for key in next(iter(key_sets)):
        tensors: list[torch.Tensor] = []
        expected_shape: tuple[int, ...] | None = None
        for value in values:
            tensor = value.conditioning[key].detach()
            if tensor.ndim >= 2 and int(tensor.shape[0]) == 1:
                tensor = tensor[0]
            shape = tuple(int(size) for size in tensor.shape)
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                raise ValueError(f"Self-Forcing condition {key!r} shapes differ across prompts")
            target_dtype = dtype if tensor.is_floating_point() else tensor.dtype
            tensor = tensor.to(device=device, dtype=target_dtype)
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Self-Forcing condition {key!r} contains NaN or infinity")
            tensors.append(tensor)
        result[key] = torch.stack(tensors)
    return result


class NativeSelfForcingDataLoader(Iterable[DMDTrainingBatch]):
    """Turn checkpointable conditioned prompts into latent-shape templates."""

    def __init__(
        self,
        source: Iterable[tuple[RolloutConditionedPrompt, ...]],
        *,
        latent_shape: Sequence[int],
        device: str | torch.device,
        dtype: torch.dtype,
        shared_unconditional_conditioning: Mapping[str, object],
    ) -> None:
        if not callable(getattr(source, "state_dict", None)) or not callable(getattr(source, "load_state_dict", None)):
            raise TypeError("Self-Forcing prompt source must expose state_dict/load_state_dict")
        shape = tuple(int(size) for size in latent_shape)
        if len(shape) != 4 or any(size <= 0 for size in shape):
            raise ValueError("Self-Forcing latent_shape must be positive [C,T,H,W]")
        if not isinstance(dtype, torch.dtype) or not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("Self-Forcing latent dtype must be floating point")
        if not isinstance(shared_unconditional_conditioning, Mapping):
            raise TypeError("shared_unconditional_conditioning must be a mapping")
        self.source = source
        self.latent_shape = shape
        self.device = torch.device(device)
        self.dtype = dtype
        self.shared_unconditional_conditioning = dict(shared_unconditional_conditioning)

    def _batch(
        self,
        values: tuple[RolloutConditionedPrompt, ...],
    ) -> DMDTrainingBatch:
        if (
            not isinstance(values, tuple)
            or not values
            or not all(isinstance(value, RolloutConditionedPrompt) for value in values)
        ):
            raise TypeError(
                "Self-Forcing prompt source must emit non-empty conditioned prompt tuples"
            )
        conditioning = _batched_conditioning(
            values,
            device=self.device,
            dtype=self.dtype,
        )
        batch_size = len(values)
        return DMDTrainingBatch(
            sample_ids=tuple(value.record.prompt_id for value in values),
            clean_latents=torch.zeros(
                (batch_size, *self.latent_shape),
                device=self.device,
                dtype=self.dtype,
            ),
            conditioning=conditioning,
            unconditional_conditioning=batch_shared_conditioning(
                self.shared_unconditional_conditioning,
                conditioning,
                batch_size=batch_size,
            ),
        )

    def __iter__(self) -> Iterator[DMDTrainingBatch]:
        for values in self.source:
            yield self._batch(values)

    def state_dict(self) -> dict[str, object]:
        source_state = self.source.state_dict()  # type: ignore[attr-defined]
        if not isinstance(source_state, Mapping):
            raise TypeError("Self-Forcing prompt source state must be a mapping")
        return {
            "schema": SELF_FORCING_DATA_LOADER_STATE_SCHEMA,
            "source": dict(source_state),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping) or set(state_dict) != {"schema", "source"}:
            raise ValueError("Self-Forcing data-loader state fields differ")
        if state_dict["schema"] != SELF_FORCING_DATA_LOADER_STATE_SCHEMA:
            raise ValueError(f"unsupported Self-Forcing data-loader state: {state_dict['schema']!r}")
        source_state = state_dict["source"]
        if not isinstance(source_state, Mapping):
            raise TypeError("saved Self-Forcing prompt source state must be a mapping")
        self.source.load_state_dict(dict(source_state))  # type: ignore[attr-defined]


__all__ = [
    "SELF_FORCING_DATA_LOADER_STATE_SCHEMA",
    "NativeSelfForcingDataLoader",
]
