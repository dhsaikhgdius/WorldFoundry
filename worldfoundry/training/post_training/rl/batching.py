"""Prompt expansion and stateful data loading for native diffusion-policy RL."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from math import isfinite
from types import MappingProxyType

import torch

from worldfoundry.training.data.rollout_cache import (
    RolloutConditionedPrompt,
    resolve_rollout_generation_geometry,
)

from ..shared.batching import batch_shared_conditioning
from .contracts import FlowRolloutBatch, RolloutPrompt

FLOW_ROLLOUT_DATA_LOADER_STATE_SCHEMA = "worldfoundry-flow-rollout-data-loader"


def _expanded_rollout_conditioning(
    prompts: tuple[RolloutPrompt, ...],
    *,
    group_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    key_sets = {tuple(sorted(prompt.conditions)) for prompt in prompts}
    if len(key_sets) != 1 or not key_sets or not next(iter(key_sets)):
        raise ValueError("all rollout prompts must expose the same non-empty condition keys")
    result: dict[str, torch.Tensor] = {}
    for key in next(iter(key_sets)):
        values: list[torch.Tensor] = []
        expected_shape: tuple[int, ...] | None = None
        for prompt in prompts:
            value = prompt.conditions[key]
            if not isinstance(value, torch.Tensor) or value.ndim == 0:
                raise TypeError(f"rollout condition {key!r} must be a non-scalar tensor")
            tensor = value.detach()
            if tensor.ndim >= 2 and int(tensor.shape[0]) == 1:
                tensor = tensor[0]
            shape = tuple(int(size) for size in tensor.shape)
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                raise ValueError(f"rollout condition {key!r} shapes differ across prompts")
            values.extend(tensor.to(device=device) for _ in range(group_size))
        result[key] = torch.stack(values)
    return result


def flow_rollout_batch_from_prompts(
    prompts: Sequence[RolloutPrompt],
    *,
    group_size: int,
    policy_revision: str,
    latent_shape: Sequence[int],
    sigmas: Sequence[float],
    device: str | torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
    shared_negative_conditioning: Mapping[str, object] | None = None,
    init_same_noise: bool = False,
) -> FlowRolloutBatch:
    """Expand complete prompt groups and create their native initial-noise population."""

    values = tuple(prompts)
    if not values or not all(isinstance(prompt, RolloutPrompt) for prompt in values):
        raise ValueError("prompts must contain RolloutPrompt values")
    if len({prompt.prompt_id for prompt in values}) != len(values):
        raise ValueError("rollout prompt_id values must be unique before group expansion")
    if isinstance(group_size, bool) or int(group_size) < 2:
        raise ValueError("group_size must be an integer of at least two")
    shape = tuple(int(size) for size in latent_shape)
    if len(shape) < 2 or any(size <= 0 for size in shape):
        raise ValueError("latent_shape must contain positive channel/spatial dimensions")
    schedule = tuple(float(sigma) for sigma in sigmas)
    if (
        len(schedule) < 2
        or any(not isfinite(sigma) or not 0 <= sigma <= 1 for sigma in schedule)
        or any(left <= right for left, right in zip(schedule, schedule[1:]))
    ):
        raise ValueError("rollout sigmas must be finite and strictly descending in [0,1]")
    resolved_device = torch.device(device)
    if not isinstance(init_same_noise, bool):
        raise TypeError("init_same_noise must be a bool")
    count = len(values) * int(group_size)
    sample_ids = tuple(
        f"{prompt.prompt_id}::sample-{sample_index:04d}" for prompt in values for sample_index in range(int(group_size))
    )
    group_ids = tuple(prompt.prompt_id for prompt in values for _ in range(int(group_size)))
    conditioning = _expanded_rollout_conditioning(
        values,
        group_size=int(group_size),
        device=resolved_device,
    )
    if shared_negative_conditioning is not None:
        negative = batch_shared_conditioning(
            shared_negative_conditioning,
            conditioning,
            batch_size=count,
        )
        collisions = sorted(f"negative_{key}" for key in negative if f"negative_{key}" in conditioning)
        if collisions:
            raise ValueError(f"negative rollout conditioning keys collide: {collisions}")
        conditioning.update({f"negative_{key}": value for key, value in negative.items()})
    noise_count = len(values) if init_same_noise else count
    initial_latents = torch.randn(
        (noise_count, *shape),
        device=resolved_device,
        dtype=dtype,
        generator=generator,
    )
    if init_same_noise:
        initial_latents = initial_latents.repeat_interleave(int(group_size), dim=0)
    return FlowRolloutBatch(
        sample_ids=sample_ids,
        group_ids=group_ids,
        policy_revision=policy_revision,
        initial_latents=initial_latents,
        sigmas=torch.tensor(schedule, device=resolved_device, dtype=torch.float32),
        conditioning=conditioning,
        metadata={
            "prompt_by_group": {prompt.prompt_id: prompt.prompt for prompt in values},
            "generation_by_group": {prompt.prompt_id: dict(prompt.generation) for prompt in values},
        },
    )


class NativeFlowPolicyDataLoader(Iterable[FlowRolloutBatch]):
    """Expand stateful conditioned prompt batches at the active policy revision."""

    def __init__(
        self,
        source: Iterable[Sequence[RolloutConditionedPrompt]],
        *,
        group_size: int,
        policy_revision: Callable[[], str],
        latent_shape: Sequence[int],
        sigmas: Sequence[float],
        device: str | torch.device,
        dtype: torch.dtype,
        generator: torch.Generator,
        generation_defaults: Mapping[str, object] | None = None,
        group_namespace: str | None = None,
        shared_negative_conditioning: Mapping[str, object] | None = None,
        init_same_noise: bool = False,
    ) -> None:
        if not callable(getattr(source, "state_dict", None)) or not callable(getattr(source, "load_state_dict", None)):
            raise TypeError("flow-policy source loader must expose state_dict/load_state_dict")
        if not callable(policy_revision):
            raise TypeError("policy_revision must be callable")
        if not isinstance(generator, torch.Generator):
            raise TypeError("flow-policy data loader generator must be torch.Generator")
        if isinstance(group_size, bool) or int(group_size) < 2:
            raise ValueError("flow-policy group_size must be at least two")
        shape = tuple(int(size) for size in latent_shape)
        if len(shape) != 4 or any(size <= 0 for size in shape):
            raise ValueError("Wan flow-policy latent_shape must be [C,T,H,W]")
        self.source = source
        self.group_size = int(group_size)
        self.policy_revision = policy_revision
        self.latent_shape = shape
        self.sigmas = tuple(float(value) for value in sigmas)
        self.device = torch.device(device)
        self.dtype = dtype
        self.generator = generator
        self.generation_defaults = MappingProxyType(
            {str(key): value for key, value in dict(generation_defaults or {}).items()}
        )
        namespace = None if group_namespace is None else str(group_namespace).strip()
        if group_namespace is not None and not namespace:
            raise ValueError("flow-policy group_namespace cannot be empty")
        self.group_namespace = namespace
        if shared_negative_conditioning is not None and not isinstance(
            shared_negative_conditioning,
            Mapping,
        ):
            raise TypeError("shared_negative_conditioning must be a mapping")
        self.shared_negative_conditioning = (
            None if shared_negative_conditioning is None else MappingProxyType(dict(shared_negative_conditioning))
        )
        if not isinstance(init_same_noise, bool):
            raise TypeError("init_same_noise must be a bool")
        self.init_same_noise = init_same_noise

    def __iter__(self) -> Iterator[FlowRolloutBatch]:
        for batch in self.source:
            if (
                isinstance(batch, (str, bytes, bytearray))
                or not isinstance(batch, Sequence)
            ):
                raise TypeError("flow-policy source loader must emit conditioned prompt sequences")
            values = tuple(batch)
            if not values or not all(isinstance(value, RolloutConditionedPrompt) for value in values):
                raise TypeError("flow-policy source loader must emit non-empty conditioned prompt sequences")
            prompt_group_ids = tuple(
                value.record.prompt_id
                if self.group_namespace is None
                else f"{self.group_namespace}:{value.record.prompt_id}"
                for value in values
            )
            prompts = tuple(
                RolloutPrompt(
                    prompt_id=group_id,
                    prompt=value.record.prompt,
                    conditions=value.conditioning,
                    generation=dict(
                        zip(
                            ("height", "width", "num_frames"),
                            resolve_rollout_generation_geometry(
                                value.record,
                                self.generation_defaults,
                            ),
                            strict=True,
                        )
                    ),
                )
                for value, group_id in zip(values, prompt_group_ids, strict=True)
            )
            revision = str(self.policy_revision()).strip()
            if not revision:
                raise ValueError("active flow-policy revision cannot be empty")
            rollout = flow_rollout_batch_from_prompts(
                prompts,
                group_size=self.group_size,
                policy_revision=revision,
                latent_shape=self.latent_shape,
                sigmas=self.sigmas,
                device=self.device,
                dtype=self.dtype,
                generator=self.generator,
                shared_negative_conditioning=self.shared_negative_conditioning,
                init_same_noise=self.init_same_noise,
            )
            yield rollout

    def state_dict(self) -> dict[str, object]:
        source_state = self.source.state_dict()  # type: ignore[attr-defined]
        if not isinstance(source_state, Mapping):
            raise TypeError("flow-policy source loader state_dict must return a mapping")
        return {
            "schema": FLOW_ROLLOUT_DATA_LOADER_STATE_SCHEMA,
            "source": dict(source_state),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping) or set(state_dict) != {"schema", "source"}:
            raise ValueError("flow-policy data-loader state fields differ")
        if state_dict["schema"] != FLOW_ROLLOUT_DATA_LOADER_STATE_SCHEMA:
            raise ValueError(f"unsupported flow-policy data-loader state: {state_dict['schema']!r}")
        source_state = state_dict["source"]
        if not isinstance(source_state, Mapping):
            raise TypeError("saved flow-policy source state must be a mapping")
        self.source.load_state_dict(dict(source_state))  # type: ignore[attr-defined]


__all__ = [
    "FLOW_ROLLOUT_DATA_LOADER_STATE_SCHEMA",
    "NativeFlowPolicyDataLoader",
    "flow_rollout_batch_from_prompts",
]
