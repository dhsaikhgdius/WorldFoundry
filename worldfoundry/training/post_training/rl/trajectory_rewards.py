"""Adapters that decode native terminal latents before reward evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ..rewards.contracts import RewardEvaluator, RewardRequest, RewardResult
from ..shared.contracts import TensorLike, freeze_mapping


@dataclass(frozen=True, slots=True)
class TerminalLatentView:
    """The algorithm-independent fields needed to decode and score a terminal state."""

    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    policy_revision: str
    terminal_latents: TensorLike
    transition_count: int
    metadata: Mapping[str, object]
    request_metadata: Mapping[str, object]

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)


def terminal_latent_view(value: object) -> TerminalLatentView:
    """Extract a validated terminal-latent view from a supported native rollout."""

    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("terminal latent rewards require the 'train-core' extra") from error

    from .algorithms.diffusion_nft.contracts import DiffusionNFTTerminalLatents
    from .contracts import FlowTrajectory

    if isinstance(value, FlowTrajectory):
        terminal_latents = value.latents[:, -1].detach()
        request_metadata: Mapping[str, object] = {
            "step_indices": list(value.step_indices),
            "transition": dict(value.transition_identity),
        }
    elif isinstance(value, DiffusionNFTTerminalLatents):
        terminal_latents = value.clean_latents.detach()
        request_metadata = {"collection_id": value.collection_id}
    else:
        raise TypeError("terminal reward input must be FlowTrajectory or DiffusionNFTTerminalLatents")

    sample_ids = tuple(value.sample_ids)
    group_ids = tuple(value.group_ids)
    if not sample_ids or len(sample_ids) != len(group_ids):
        raise ValueError("terminal reward sample_ids and group_ids must have equal non-zero length")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("terminal reward sample_ids must be unique")
    if not isinstance(value.policy_revision, str) or not value.policy_revision.strip():
        raise ValueError("terminal reward policy_revision must be non-empty")
    if isinstance(value.transition_count, bool) or int(value.transition_count) <= 0:
        raise ValueError("terminal reward transition_count must be positive")
    if (
        not isinstance(terminal_latents, torch.Tensor)
        or terminal_latents.ndim < 2
        or int(terminal_latents.shape[0]) != len(sample_ids)
        or not terminal_latents.is_floating_point()
    ):
        raise TypeError("terminal reward latents must be a floating [B,...] torch.Tensor")
    if not bool(torch.isfinite(terminal_latents).all()):
        raise ValueError("terminal reward latents contain NaN or infinity")
    if not isinstance(value.metadata, Mapping):
        raise TypeError("terminal reward metadata must be a mapping")

    return TerminalLatentView(
        sample_ids=sample_ids,
        group_ids=group_ids,
        policy_revision=value.policy_revision,
        terminal_latents=terminal_latents,
        transition_count=int(value.transition_count),
        metadata=MappingProxyType(dict(value.metadata)),
        request_metadata=MappingProxyType(dict(request_metadata)),
    )


class DecodedTerminalRewardAdapter:
    """Decode a supported native terminal state and score it through a typed evaluator."""

    schema = "worldfoundry-decoded-terminal-reward"

    def __init__(
        self,
        decoder: object,
        evaluator: RewardEvaluator,
        *,
        reward_ids: tuple[str, ...],
        evaluator_identity: Mapping[str, object],
    ) -> None:
        if not callable(getattr(decoder, "decode", None)):
            raise TypeError("terminal reward decoder must expose decode")
        if not isinstance(evaluator, RewardEvaluator):
            raise TypeError("evaluator must implement RewardEvaluator")
        resolved_ids = tuple(str(value).strip() for value in reward_ids)
        if not resolved_ids or any(not value for value in resolved_ids):
            raise ValueError("terminal reward_ids must be non-empty")
        if len(resolved_ids) != len(set(resolved_ids)):
            raise ValueError("terminal reward_ids must be unique")
        self.decoder = decoder
        self.evaluator = evaluator
        self.reward_ids = resolved_ids
        self.evaluator_identity = MappingProxyType(
            dict(freeze_mapping(evaluator_identity, field_name="evaluator_identity"))
        )
        self.last_results: tuple[RewardResult, ...] = ()

    @property
    def identity(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "reward_ids": list(self.reward_ids),
            "evaluator": dict(self.evaluator_identity),
        }

    def score(self, terminal_state: object) -> Mapping[str, object]:
        try:
            import torch
        except ModuleNotFoundError as error:
            raise RuntimeError("decoded terminal rewards require the 'train-core' extra") from error
        from worldfoundry.base_models.diffusion_model.contracts import (
            DiffusionRequest,
            SamplingConfig,
        )

        terminal = terminal_latent_view(terminal_state)
        metadata = dict(terminal.metadata)
        prompt_by_group = metadata.get("prompt_by_group")
        generation_by_group = metadata.get("generation_by_group")
        if not isinstance(prompt_by_group, Mapping) or not isinstance(
            generation_by_group,
            Mapping,
        ):
            raise ValueError("terminal reward metadata requires prompt_by_group and generation_by_group")
        prompts: list[str] = []
        geometries: list[tuple[int, int, int]] = []
        for group_id in terminal.group_ids:
            prompt = str(prompt_by_group.get(group_id, "")).strip()
            generation = generation_by_group.get(group_id)
            if not prompt or not isinstance(generation, Mapping):
                raise ValueError(f"terminal reward metadata is incomplete for group {group_id!r}")
            try:
                geometry = (
                    int(generation["height"]),
                    int(generation["width"]),
                    int(generation["num_frames"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"terminal generation geometry is invalid for group {group_id!r}") from error
            if any(value <= 0 for value in geometry):
                raise ValueError("terminal generation geometry must be positive")
            prompts.append(prompt)
            geometries.append(geometry)
        if len(set(geometries)) != 1:
            raise ValueError("one decoded reward batch requires identical generation geometry")
        height, width, frames = geometries[0]
        request = DiffusionRequest(
            prompt=tuple(prompts),
            height=height,
            width=width,
            num_frames=frames,
            sampling=SamplingConfig(num_inference_steps=terminal.transition_count),
            metadata={"sample_ids": terminal.sample_ids},
        )
        with torch.inference_mode():
            videos = self.decoder.decode(terminal.terminal_latents, request)
        if not isinstance(videos, torch.Tensor) or videos.ndim != 5:
            raise TypeError("terminal reward decoder must return [B,C,T,H,W] video")
        expected_shape = (terminal.batch_size, 3, frames, height, width)
        if tuple(videos.shape) != expected_shape:
            raise ValueError(f"decoded reward video must have shape {expected_shape}; got {tuple(videos.shape)}")
        if not bool(torch.isfinite(videos).all()):
            raise ValueError("decoded reward video contains NaN or infinity")
        requests = tuple(
            RewardRequest(
                request_id=sample_id,
                rollout_id=f"{terminal.policy_revision}:{sample_id}",
                prompt=prompt,
                conditions={"group_id": group_id},
                artifacts={"video": videos[index]},
                reward_ids=self.reward_ids,
                metadata={
                    **terminal.request_metadata,
                    "generation": {
                        "height": height,
                        "width": width,
                        "num_frames": frames,
                    },
                },
            )
            for index, (sample_id, group_id, prompt) in enumerate(zip(terminal.sample_ids, terminal.group_ids, prompts))
        )
        results = self.evaluator.evaluate(requests)
        if not isinstance(results, tuple) or len(results) != terminal.batch_size:
            raise ValueError("reward evaluator must return one ordered result per terminal sample")
        components = {
            reward_id: torch.empty(
                terminal.batch_size,
                device=terminal.terminal_latents.device,
                dtype=torch.float32,
            )
            for reward_id in self.reward_ids
        }
        for index, (reward_request, result) in enumerate(zip(requests, results)):
            if not isinstance(result, RewardResult):
                raise TypeError("reward evaluator returned a non-RewardResult value")
            if (
                result.request_id != reward_request.request_id
                or result.rollout_id != reward_request.rollout_id
                or set(result.values) != set(self.reward_ids)
            ):
                raise ValueError("reward result identity/components differ from its request")
            for reward_id in self.reward_ids:
                components[reward_id][index] = result.values[reward_id] if result.valid[reward_id] else torch.nan
        self.last_results = results
        return MappingProxyType(components)


__all__ = [
    "DecodedTerminalRewardAdapter",
    "TerminalLatentView",
    "terminal_latent_view",
]
