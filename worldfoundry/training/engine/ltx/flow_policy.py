"""LTX-2 video policy rollout and native RL stack materialization."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal

import numpy as np
import torch

from worldfoundry.training.models.ltx import LTXTrainAdapter
from worldfoundry.training.post_training.rl.algorithms.diffusion_nft.builder import (
    NativeDiffusionNFTTrainingStack,
    build_native_diffusion_nft_training_stack,
)
from worldfoundry.training.post_training.rl.algorithms.diffusion_nft.contracts import (
    DiffusionNFTRewardAdapter,
)
from worldfoundry.training.post_training.rl.algorithms.flow_policy.builder import (
    NativeFlowPolicyTrainingStack,
    build_native_flow_policy_training_stack,
)
from worldfoundry.training.post_training.shared.prediction import NativeFlowPredictionAdapter
from worldfoundry.training.recipes.post_training.algorithms.diffusion_nft import (
    DiffusionNFTAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.algorithms.flow_policy import (
    FlowPolicyAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from .trajectory import (
    LTXAudioConditionedTrajectoryReplay,
    LTXAudioConditionedTrajectorySampler,
)

LTX_POLICY_MODELS = frozenset({"ltx-2-i2v", "ltx-2.3-i2v"})
LTX_POLICY_TIMESTEP_SHIFT = 2.05
LTX_POLICY_TERMINAL_SIGMA = 0.1


def ltx_flow_policy_sigmas(num_inference_steps: int = 10) -> tuple[float, ...]:
    """Return the LTX-2 constant-mu exponential schedule with terminal stretch."""

    steps = int(num_inference_steps)
    if steps < 2:
        raise ValueError("LTX policy rollout requires at least two inference steps")
    sigmas = np.linspace(1.0, LTX_POLICY_TERMINAL_SIGMA, steps, dtype=np.float32)
    shifted = math.exp(LTX_POLICY_TIMESTEP_SHIFT) / (math.exp(LTX_POLICY_TIMESTEP_SHIFT) + (1 / sigmas - 1) ** 1.0)
    one_minus_shifted = 1 - shifted
    scale = one_minus_shifted[-1] / (1 - LTX_POLICY_TERMINAL_SIGMA)
    shifted = 1 - one_minus_shifted / scale
    return tuple(float(value) for value in shifted) + (0.0,)


@dataclass(frozen=True, slots=True)
class LTXFlowPolicyProfile:
    """Released LTX-2.x policy geometry and optimization defaults."""

    model_recipe: str
    generation: Mapping[str, int]
    sigmas: tuple[float, ...]
    sde_step_indices: tuple[int, ...]
    guidance_scale: float = 1.0
    eta: float = 0.7
    group_size: int = 16
    trajectory_dtype: str = "float32"
    old_log_prob_source: str = "rollout"
    rollout_forward_batch_size: int = 1
    audio_joint_sde: bool = False
    lora_rank: int = 32
    lora_alpha: int = 256
    learning_rate: float = 1.0e-4
    global_prompt_batch_size: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(self, "generation", MappingProxyType(dict(self.generation)))


def ltx_flow_policy_profile(model_recipe: str = "ltx-2-i2v") -> LTXFlowPolicyProfile:
    recipe = str(model_recipe).strip().lower().replace("_", "-")
    if recipe not in LTX_POLICY_MODELS:
        raise ValueError(f"unsupported LTX policy model: {model_recipe!r}")
    sigmas = ltx_flow_policy_sigmas()
    if recipe == "ltx-2.3-i2v":
        return LTXFlowPolicyProfile(
            model_recipe=recipe,
            generation={"height": 512, "width": 768, "num_frames": 33},
            sigmas=sigmas,
            sde_step_indices=tuple(range(5)),
            group_size=4,
            trajectory_dtype="float16",
            old_log_prob_source="replay",
            rollout_forward_batch_size=2,
            audio_joint_sde=True,
            lora_alpha=64,
        )
    return LTXFlowPolicyProfile(
        model_recipe=recipe,
        generation={"height": 512, "width": 768, "num_frames": 9},
        sigmas=sigmas,
        sde_step_indices=tuple(range(5)),
    )


@dataclass(frozen=True, slots=True)
class LTXFlowPolicyDataPlan:
    generation: Mapping[str, int]
    target_fps: float
    audio_joint_sde: bool
    global_prompt_batch_size: int
    rollout_forward_batch_size: int | None
    replay_microbatch_size: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "generation", MappingProxyType(dict(self.generation)))


def _optional_positive_int(value: object | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive")
    return resolved


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


def _ltx_data_plan(recipe: PostTrainingRecipe) -> LTXFlowPolicyDataPlan:
    options = dict(recipe.data.options)
    raw_generation = options.get("generation")
    if not isinstance(raw_generation, Mapping):
        raise TypeError("LTX policy data.options.generation must be a mapping")
    generation = {name: int(raw_generation[name]) for name in ("height", "width", "num_frames")}
    if any(value <= 0 for value in generation.values()):
        raise ValueError("LTX policy generation dimensions must be positive")
    if generation["height"] % 32 or generation["width"] % 32:
        raise ValueError("LTX policy height and width must be divisible by 32")
    if (generation["num_frames"] - 1) % 8:
        raise ValueError("LTX policy num_frames must satisfy 1 + 8k")
    target_fps = float(options.get("target_fps", 24.0))
    if not math.isclose(target_fps, 24.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("LTX-2.x policy rollout requires 24 FPS audio-video alignment")
    audio_joint_sde = options.get("audio_joint_sde", False)
    if not isinstance(audio_joint_sde, bool):
        raise TypeError("LTX policy data.options.audio_joint_sde must be a bool")
    return LTXFlowPolicyDataPlan(
        generation=generation,
        target_fps=target_fps,
        audio_joint_sde=audio_joint_sde,
        global_prompt_batch_size=_positive_int(
            options.get("global_prompt_batch_size", 1),
            field_name="data.options.global_prompt_batch_size",
        ),
        rollout_forward_batch_size=_optional_positive_int(
            options.get("rollout_forward_batch_size"),
            field_name="data.options.rollout_forward_batch_size",
        ),
        replay_microbatch_size=_optional_positive_int(
            options.get("replay_microbatch_size"),
            field_name="data.options.replay_microbatch_size",
        ),
    )


def _validate_common_recipe(recipe: PostTrainingRecipe, sigmas: tuple[float, ...]) -> LTXFlowPolicyDataPlan:
    if recipe.model.recipe not in LTX_POLICY_MODELS:
        raise ValueError(f"LTX policy training cannot train {recipe.model.recipe!r}")
    if recipe.distributed.backend not in {"single", "fsdp2"}:
        raise ValueError("LTX policy training supports single-device and FSDP2 execution")
    if recipe.distributed.cp != 1 or recipe.distributed.tp != 1:
        raise ValueError("LTX policy context and tensor parallelism are not implemented")
    if recipe.tuning.mode == "lora" and recipe.tuning.preset != "ltx-policy":
        raise ValueError("LTX policy LoRA requires tuning.preset='ltx-policy'")
    expected = ltx_flow_policy_sigmas(len(sigmas) - 1)
    if len(sigmas) != len(expected) or any(
        not math.isclose(actual, target, rel_tol=0.0, abs_tol=2.0e-7) for actual, target in zip(sigmas, expected)
    ):
        raise ValueError("LTX policy sigmas differ from the native constant-mu schedule")
    plan = _ltx_data_plan(recipe)
    return plan


def validate_ltx_flow_policy_recipe(
    recipe: PostTrainingRecipe,
) -> tuple[FlowPolicyAlgorithmSpec, LTXFlowPolicyDataPlan]:
    if not isinstance(recipe.algorithm, FlowPolicyAlgorithmSpec):
        raise TypeError("LTX flow policy requires a flow-policy algorithm")
    if recipe.algorithm.guidance_scale != 1.0:
        raise ValueError("LTX video policy currently requires guidance_scale=1")
    if recipe.model.recipe == "ltx-2.3-i2v" and recipe.algorithm.type != "flow-grpo":
        raise NotImplementedError("LTX-2.3 joint AV transition means are not connected to non-GRPO objectives")
    expected_dtype = ltx_flow_policy_profile(recipe.model.recipe).trajectory_dtype
    if recipe.algorithm.trajectory_dtype != expected_dtype:
        raise ValueError(f"{recipe.model.recipe} policy trajectory_dtype must be {expected_dtype}")
    plan = _validate_common_recipe(recipe, recipe.algorithm.sigmas)
    expected_joint_audio = ltx_flow_policy_profile(recipe.model.recipe).audio_joint_sde
    if plan.audio_joint_sde != expected_joint_audio:
        mode = "joint audio-video SDE" if expected_joint_audio else "video SDE with audio ODE"
        raise ValueError(f"{recipe.model.recipe} policy requires {mode}")
    if plan.audio_joint_sde and recipe.algorithm.reference_kl_weight > 0:
        raise NotImplementedError("LTX-2.3 reference KL needs a joint AV transition-mean contract")
    return recipe.algorithm, plan


def validate_ltx_diffusion_nft_recipe(
    recipe: PostTrainingRecipe,
) -> tuple[DiffusionNFTAlgorithmSpec, LTXFlowPolicyDataPlan]:
    if not isinstance(recipe.algorithm, DiffusionNFTAlgorithmSpec):
        raise TypeError("LTX DiffusionNFT requires a diffusion-nft algorithm")
    raise NotImplementedError("LTX-2.x DiffusionNFT needs a collector that stores the co-denoised audio trajectory")


class LTXFlowPredictionAdapter(NativeFlowPredictionAdapter):
    """Add model-native LTX positions before shared flow rollout and replay."""

    adapter: LTXTrainAdapter

    def __init__(
        self,
        adapter: LTXTrainAdapter,
        *,
        target_fps: float = 24.0,
        autocast_dtype: torch.dtype | None = None,
        checkpoint_identity: str | None = None,
    ) -> None:
        if not isinstance(adapter, LTXTrainAdapter):
            raise TypeError("LTX flow prediction requires LTXTrainAdapter")
        super().__init__(
            adapter,
            autocast_dtype=autocast_dtype,
            checkpoint_identity=checkpoint_identity,
        )
        self.target_fps = float(target_fps)

    def initial_audio_latents(
        self,
        video_latents: torch.Tensor,
        *,
        generator: torch.Generator | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate the packed audio state co-denoised by the AV transformer."""

        channels = self.adapter.expected_audio_latent_channels
        if not self.adapter.supports_joint_audio or channels is None:
            raise ValueError("LTX policy rollout requires an audio-video transformer")
        pixel_frames = (int(video_latents.shape[2]) - 1) * self.adapter.temporal_compression + 1
        audio_frames = max(1, round(pixel_frames / self.target_fps * 25.0))
        return torch.randn(
            int(video_latents.shape[0]),
            audio_frames,
            channels,
            device=video_latents.device,
            dtype=dtype,
            generator=generator,
        )

    def predict_joint_velocity(
        self,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict both modalities while keeping trajectory arithmetic in FP32."""

        if int(video_latents.shape[0]) != len(sample_ids):
            raise ValueError("LTX sample_ids must match the AV latent batch")
        trajectory_dtype = video_latents.dtype
        model_dtype = self.autocast_dtype or next(self.module.parameters()).dtype
        model_video = video_latents.to(dtype=model_dtype)
        model_audio = audio_latents.to(dtype=model_dtype)
        device_type = video_latents.device.type
        with torch.autocast(
            device_type=device_type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_dtype is not None and device_type in {"cpu", "cuda"},
        ):
            video_velocity, audio_velocity = self.adapter.forward_joint_model(
                model_video,
                model_audio,
                sigmas,
                conditioning,
                training=training,
            )
        return (
            video_velocity.to(dtype=trajectory_dtype),
            audio_velocity.to(dtype=trajectory_dtype),
        )

    def predict_velocity(
        self,
        noisy_latents: object,
        sigmas: object,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> object:
        if not isinstance(noisy_latents, torch.Tensor):
            raise TypeError("LTX rollout latents must be a torch.Tensor")
        prepared = self.adapter.rollout_conditioning(
            noisy_latents,
            conditioning,
            fps=self.target_fps,
            dtype=self.autocast_dtype or noisy_latents.dtype,
        )
        return super().predict_velocity(
            noisy_latents,
            sigmas,
            sample_ids=sample_ids,
            conditioning=prepared,
            training=training,
            branch=branch,
        )


@dataclass(frozen=True, slots=True)
class LTXFlowPolicyRuntime:
    prediction: LTXFlowPredictionAdapter
    stack: NativeFlowPolicyTrainingStack
    data_plan: LTXFlowPolicyDataPlan
    latent_shape: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class LTXDiffusionNFTRuntime:
    policy_prediction: LTXFlowPredictionAdapter
    old_policy_prediction: LTXFlowPredictionAdapter
    stack: NativeDiffusionNFTTrainingStack
    data_plan: LTXFlowPolicyDataPlan
    latent_shape: tuple[int, int, int, int]


def _latent_shape(adapter: LTXTrainAdapter, plan: LTXFlowPolicyDataPlan) -> tuple[int, int, int, int]:
    generation = plan.generation
    return (
        adapter.expected_latent_channels,
        (generation["num_frames"] - 1) // adapter.temporal_compression + 1,
        generation["height"] // adapter.spatial_compression,
        generation["width"] // adapter.spatial_compression,
    )


def bind_ltx_trajectory_adapters(
    stack: NativeFlowPolicyTrainingStack,
    *,
    policy: LTXFlowPredictionAdapter,
    reference_policy: LTXFlowPredictionAdapter | None,
    data_plan: LTXFlowPolicyDataPlan,
    trajectory_dtype: str,
) -> NativeFlowPolicyTrainingStack:
    """Install the model-native audio-conditioned rollout and replay plane."""

    resolved_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
    }[trajectory_dtype]
    sampler = LTXAudioConditionedTrajectorySampler(
        policy,
        transition_strategy=stack.transition_strategy,
        trajectory_dtype=resolved_dtype,
        audio_joint_sde=data_plan.audio_joint_sde,
        init_same_noise=stack.init_same_noise,
        forward_batch_size=data_plan.rollout_forward_batch_size,
    )
    replay = LTXAudioConditionedTrajectoryReplay(
        policy,
        audio_joint_sde=data_plan.audio_joint_sde,
    )
    reference_replay = (
        None
        if reference_policy is None
        else LTXAudioConditionedTrajectoryReplay(
            reference_policy,
            audio_joint_sde=data_plan.audio_joint_sde,
        )
    )
    stack.engine.replay_adapter = replay
    stack.engine.reference_replay_adapter = reference_replay
    return replace(
        stack,
        sampler=sampler,
        replay=replay,
        reference_replay=reference_replay,
    )


def materialize_ltx_flow_policy_stack(
    recipe: PostTrainingRecipe,
    *,
    policy: LTXTrainAdapter,
    initial_policy_revision: str,
    reference_policy: LTXTrainAdapter | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> LTXFlowPolicyRuntime:
    """Connect a loaded native LTX policy to its supported flow-policy objective."""

    _, plan = validate_ltx_flow_policy_recipe(recipe)
    if not policy.supports_joint_audio:
        raise ValueError("LTX-2.x policy rollout requires the checkpoint's audio-video transformer")
    autocast_dtype = next(policy.trainable_module.parameters()).dtype
    if autocast_dtype is torch.float32:
        autocast_dtype = None
    prediction = LTXFlowPredictionAdapter(
        policy,
        target_fps=plan.target_fps,
        autocast_dtype=autocast_dtype,
        checkpoint_identity=recipe.model.checkpoint,
    )
    reference_prediction = None
    if reference_policy is not None:
        if not reference_policy.supports_joint_audio:
            raise ValueError("LTX-2.x reference policy must use an audio-video transformer")
        reference_dtype = next(reference_policy.trainable_module.parameters()).dtype
        reference_prediction = LTXFlowPredictionAdapter(
            reference_policy,
            target_fps=plan.target_fps,
            autocast_dtype=None if reference_dtype is torch.float32 else reference_dtype,
        )
    stack = build_native_flow_policy_training_stack(
        recipe,
        policy=prediction,
        initial_policy_revision=initial_policy_revision,
        reference_policy=reference_prediction,
        fused_adamw=fused_adamw,
        rollout_forward_batch_size=plan.rollout_forward_batch_size,
        replay_microbatch_size=plan.replay_microbatch_size,
    )
    stack = bind_ltx_trajectory_adapters(
        stack,
        policy=prediction,
        reference_policy=reference_prediction,
        data_plan=plan,
        trajectory_dtype=recipe.algorithm.trajectory_dtype,
    )
    return LTXFlowPolicyRuntime(
        prediction=prediction,
        stack=stack,
        data_plan=plan,
        latent_shape=_latent_shape(policy, plan),
    )


def materialize_ltx_diffusion_nft_stack(
    recipe: PostTrainingRecipe,
    *,
    policy: LTXTrainAdapter,
    old_policy: LTXTrainAdapter,
    initial_old_policy_revision: str,
    reward_adapter: DiffusionNFTRewardAdapter,
    reference_policy: LTXTrainAdapter | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> LTXDiffusionNFTRuntime:
    """Connect loaded LTX policy roles to native DiffusionNFT collection and updates."""

    _, plan = validate_ltx_diffusion_nft_recipe(recipe)

    def prediction(adapter: LTXTrainAdapter) -> LTXFlowPredictionAdapter:
        dtype = next(adapter.trainable_module.parameters()).dtype
        return LTXFlowPredictionAdapter(
            adapter,
            target_fps=plan.target_fps,
            autocast_dtype=None if dtype is torch.float32 else dtype,
        )

    policy_prediction = prediction(policy)
    old_prediction = prediction(old_policy)
    reference_prediction = None if reference_policy is None else prediction(reference_policy)
    stack = build_native_diffusion_nft_training_stack(
        recipe,
        policy=policy_prediction,
        old_policy=old_prediction,
        initial_old_policy_revision=initial_old_policy_revision,
        reward_adapter=reward_adapter,
        reference_policy=reference_prediction,
        fused_adamw=fused_adamw,
    )
    return LTXDiffusionNFTRuntime(
        policy_prediction=policy_prediction,
        old_policy_prediction=old_prediction,
        stack=stack,
        data_plan=plan,
        latent_shape=_latent_shape(policy, plan),
    )


__all__ = [
    "bind_ltx_trajectory_adapters",
    "LTXDiffusionNFTRuntime",
    "LTXFlowPolicyDataPlan",
    "LTXFlowPolicyProfile",
    "LTXFlowPolicyRuntime",
    "LTXFlowPredictionAdapter",
    "LTX_POLICY_MODELS",
    "ltx_flow_policy_profile",
    "ltx_flow_policy_sigmas",
    "materialize_ltx_diffusion_nft_stack",
    "materialize_ltx_flow_policy_stack",
    "validate_ltx_diffusion_nft_recipe",
    "validate_ltx_flow_policy_recipe",
]
