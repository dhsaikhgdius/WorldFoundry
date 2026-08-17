"""Terminal audio-video reward decoding for the LTX-2.3 policy."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import torch

from worldfoundry.base_models.diffusion_model.contracts import (
    DiffusionRequest,
    ModalityState,
    SamplingConfig,
)
from worldfoundry.base_models.diffusion_model.models.representations.ltx.patchifiers import (
    VideoLatentPatchifier,
)
from worldfoundry.training.post_training.rewards.contracts import (
    RewardEvaluator,
    RewardRequest,
    RewardResult,
)
from worldfoundry.training.post_training.rl.contracts import FlowTrajectory
from worldfoundry.training.post_training.rl.trajectory_rewards import (
    terminal_latent_view,
)

from .trajectory import LTX_AUDIO_TRAJECTORY


def _decoder_state(latent: torch.Tensor) -> ModalityState:
    return ModalityState(
        latent=latent,
        denoise_mask=torch.ones((), device=latent.device, dtype=torch.bool),
        positions=torch.empty(0, device=latent.device),
        clean_latent=latent,
    )


class LTXAVTerminalRewardAdapter:
    """Decode synchronized final video/audio states and score both artifacts."""

    schema = "worldfoundry-ltx-av-terminal-reward"

    def __init__(
        self,
        decoder: object,
        evaluator: RewardEvaluator,
        *,
        reward_ids: tuple[str, ...],
        frame_rate: float = 24.0,
        evaluator_identity: Mapping[str, object] | None = None,
    ) -> None:
        if not callable(getattr(decoder, "decode_modalities", None)):
            raise TypeError("LTX AV reward decoder must expose decode_modalities")
        if not isinstance(evaluator, RewardEvaluator):
            raise TypeError("evaluator must implement RewardEvaluator")
        resolved_ids = tuple(str(value).strip() for value in reward_ids)
        if not resolved_ids or any(not value for value in resolved_ids):
            raise ValueError("LTX AV reward_ids must be non-empty")
        if len(set(resolved_ids)) != len(resolved_ids):
            raise ValueError("LTX AV reward_ids must be unique")
        resolved_rate = float(frame_rate)
        if resolved_rate <= 0:
            raise ValueError("LTX AV frame_rate must be positive")
        self.decoder = decoder
        self.evaluator = evaluator
        self.reward_ids = resolved_ids
        self.frame_rate = resolved_rate
        self.evaluator_identity = MappingProxyType(dict(evaluator_identity or {}))
        self.last_results: tuple[RewardResult, ...] = ()

    @property
    def identity(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "reward_ids": list(self.reward_ids),
            "evaluator": dict(self.evaluator_identity),
        }

    def _request_context(
        self,
        trajectory: FlowTrajectory,
    ) -> tuple[list[str], list[tuple[int, int, int]]]:
        prompt_by_group = trajectory.metadata.get("prompt_by_group")
        generation_by_group = trajectory.metadata.get("generation_by_group")
        if not isinstance(prompt_by_group, Mapping) or not isinstance(generation_by_group, Mapping):
            raise ValueError("LTX AV reward metadata requires prompt_by_group and generation_by_group")
        prompts: list[str] = []
        geometries: list[tuple[int, int, int]] = []
        for group_id in trajectory.group_ids:
            prompt = str(prompt_by_group.get(group_id, "")).strip()
            generation = generation_by_group.get(group_id)
            if not prompt or not isinstance(generation, Mapping):
                raise ValueError(f"LTX AV reward metadata is incomplete for group {group_id!r}")
            prompts.append(prompt)
            geometries.append(
                (
                    int(generation["height"]),
                    int(generation["width"]),
                    int(generation["num_frames"]),
                )
            )
        return prompts, geometries

    def score(self, terminal_state: object) -> Mapping[str, torch.Tensor]:
        if not isinstance(terminal_state, FlowTrajectory):
            raise TypeError("LTX AV rewards require a FlowTrajectory")
        terminal = terminal_latent_view(terminal_state)
        audio_trajectory = terminal_state.conditioning.get(LTX_AUDIO_TRAJECTORY)
        if not isinstance(audio_trajectory, torch.Tensor) or audio_trajectory.ndim != 4:
            raise ValueError("LTX AV rewards require an [B,S+1,T,C] audio trajectory")
        if tuple(audio_trajectory.shape[:2]) != (
            terminal.batch_size,
            terminal.transition_count + 1,
        ):
            raise ValueError("LTX audio and video trajectories must have the same batch and step counts")
        prompts, geometries = self._request_context(terminal_state)
        video_patchifier = VideoLatentPatchifier(1)
        requests: list[RewardRequest] = []
        for index, (sample_id, group_id, prompt, geometry) in enumerate(
            zip(
                terminal.sample_ids,
                terminal.group_ids,
                prompts,
                geometries,
                strict=True,
            )
        ):
            height, width, frames = geometry
            decode_request = DiffusionRequest(
                prompt=prompt,
                height=height,
                width=width,
                num_frames=frames,
                sampling=SamplingConfig(num_inference_steps=terminal.transition_count),
                inputs={"frame_rate": self.frame_rate},
                metadata={"sample_ids": (sample_id,)},
            )
            video_tokens = video_patchifier.patchify(terminal.terminal_latents[index : index + 1])
            audio_tokens = audio_trajectory[index : index + 1, -1].detach()
            with torch.inference_mode():
                decoded = self.decoder.decode_modalities(
                    {
                        "video": _decoder_state(video_tokens),
                        "audio": _decoder_state(audio_tokens),
                    },
                    decode_request,
                )
            if not isinstance(decoded, Mapping):
                raise TypeError("LTX AV decoder must return an artifact mapping")
            video = decoded.get("video")
            waveform = decoded.get("audio")
            sampling_rate = decoded.get("audio_sampling_rate")
            if not isinstance(video, torch.Tensor) or tuple(video.shape) != (frames, height, width, 3):
                raise ValueError("LTX AV decoder video must be [T,H,W,3]")
            if not isinstance(waveform, torch.Tensor) or waveform.ndim not in {1, 2}:
                raise ValueError("LTX AV decoder audio must be a waveform tensor")
            if isinstance(sampling_rate, bool) or int(sampling_rate) <= 0:
                raise ValueError("LTX AV decoder must return a positive audio sampling rate")
            requests.append(
                RewardRequest(
                    request_id=sample_id,
                    rollout_id=f"{terminal.policy_revision}:{sample_id}",
                    prompt=prompt,
                    conditions={"group_id": group_id},
                    artifacts={
                        "video": video.permute(3, 0, 1, 2).contiguous(),
                        "audio": waveform,
                    },
                    reward_ids=self.reward_ids,
                    metadata={
                        **terminal.request_metadata,
                        "generation": {
                            "height": height,
                            "width": width,
                            "num_frames": frames,
                            "frame_rate": self.frame_rate,
                        },
                        "audio_sampling_rate": int(sampling_rate),
                    },
                )
            )

        results = self.evaluator.evaluate(tuple(requests))
        if not isinstance(results, tuple) or len(results) != terminal.batch_size:
            raise ValueError("reward evaluator must return one ordered result per LTX AV sample")
        components = {
            reward_id: torch.empty(
                terminal.batch_size,
                device=terminal.terminal_latents.device,
                dtype=torch.float32,
            )
            for reward_id in self.reward_ids
        }
        for index, (request, result) in enumerate(zip(requests, results, strict=True)):
            if not isinstance(result, RewardResult):
                raise TypeError("reward evaluator returned a non-RewardResult value")
            if (
                result.request_id != request.request_id
                or result.rollout_id != request.rollout_id
                or set(result.values) != set(self.reward_ids)
            ):
                raise ValueError("reward result identity/components differ from its LTX AV request")
            for reward_id in self.reward_ids:
                components[reward_id][index] = result.values[reward_id] if result.valid[reward_id] else torch.nan
        self.last_results = results
        return MappingProxyType(components)


__all__ = ["LTXAVTerminalRewardAdapter"]
