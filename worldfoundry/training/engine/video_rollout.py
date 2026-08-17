"""Actor-local policy construction for native video Ray rollouts."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from worldfoundry.training.distributed.ray_runtime import RayWorkerContext
from worldfoundry.training.post_training.shared.prediction import (
    NativeFlowPredictionAdapter,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

_HUNYUAN_MODELS = frozenset({"hunyuanvideo-t2v", "hunyuanvideo-1.5-t2v"})
_LTX_MODELS = frozenset({"ltx-2-i2v", "ltx-2.3-i2v"})
_WAN22_MODEL = "wan2.2-t2v-a14b"


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _rollout_device(device_type: str) -> torch.device:
    resolved = str(device_type).strip().lower()
    if resolved == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU Ray rollout worker has no visible CUDA device")
        return torch.device("cuda")
    if resolved == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported video rollout device type: {device_type!r}")


def materialize_video_ray_rollout_policy(
    *,
    context: RayWorkerContext,
    recipe: PostTrainingRecipe,
    checkpoint_overrides: Mapping[str, object],
    device_type: str,
) -> object:
    """Load one independently weighted actor policy using native family roles."""

    del context
    device = _rollout_device(device_type)
    dtype = _torch_dtype(recipe.runtime.param_dtype)
    overrides = dict(checkpoint_overrides)

    if recipe.model.recipe == _WAN22_MODEL:
        from worldfoundry.training.engine.video_policy import _wan22_checkpoints
        from worldfoundry.training.engine.wan22.flow_policy import (
            validate_wan22_flow_policy_recipe,
        )
        from worldfoundry.training.engine.wan22.roles import (
            load_wan22_role_adapter,
        )
        from worldfoundry.training.engine.wan22.tuning import apply_wan22_tuning

        algorithm, data_plan = validate_wan22_flow_policy_recipe(recipe)
        policy = load_wan22_role_adapter(
            checkpoints=_wan22_checkpoints(
                recipe,
                {key: value for key, value in overrides.items() if key in {"high-noise", "low-noise"}},
            ),
            device=device,
            dtype=dtype,
            boundary_ratio=data_plan.boundary_ratio,
            num_train_timesteps=algorithm.num_train_timesteps,
            gradient_checkpointing=False,
        )
        apply_wan22_tuning(recipe, policy)
        return NativeFlowPredictionAdapter(
            policy,
            autocast_dtype=None if dtype is torch.float32 else dtype,
            checkpoint_identity=recipe.model.checkpoint,
        )

    if recipe.model.recipe in _HUNYUAN_MODELS:
        from worldfoundry.training.engine.hunyuan_video import (
            apply_hunyuan_video_activation_checkpointing,
            apply_hunyuan_video_tuning,
            load_hunyuan_video_role_adapter,
            validate_hunyuan_video_flow_policy_recipe,
        )

        validate_hunyuan_video_flow_policy_recipe(recipe)
        policy_checkpoint = overrides.get("policy")
        if recipe.model.checkpoint != "default" and policy_checkpoint is None:
            raise ValueError("a non-default HunyuanVideo Ray rollout requires a policy checkpoint override")
        policy = load_hunyuan_video_role_adapter(
            model_recipe=recipe.model.recipe,
            checkpoint=policy_checkpoint,
            device=device,
            dtype=dtype,
        )
        apply_hunyuan_video_tuning(recipe, policy)
        if recipe.runtime.activation_checkpoint == "full":
            apply_hunyuan_video_activation_checkpointing(policy)
        return NativeFlowPredictionAdapter(
            policy,
            autocast_dtype=None if dtype is torch.float32 else dtype,
            checkpoint_identity=recipe.model.checkpoint,
        )

    if recipe.model.recipe in _LTX_MODELS:
        from worldfoundry.training.engine.ltx.flow_policy import (
            LTXFlowPredictionAdapter,
            validate_ltx_flow_policy_recipe,
        )
        from worldfoundry.training.engine.ltx.flow_policy_roles import (
            apply_ltx_policy_tuning,
            load_ltx_policy_adapter,
        )

        _, data_plan = validate_ltx_flow_policy_recipe(recipe)
        role_keys = {"model", "gemma", "tokenizer", "text_encoder"}
        policy = load_ltx_policy_adapter(
            recipe,
            device=device,
            dtype=dtype,
            checkpoint_overrides={key: value for key, value in overrides.items() if key in role_keys},
        )
        apply_ltx_policy_tuning(recipe, policy)
        return LTXFlowPredictionAdapter(
            policy,
            target_fps=data_plan.target_fps,
            autocast_dtype=None if dtype is torch.float32 else dtype,
            checkpoint_identity=recipe.model.checkpoint,
        )

    raise ValueError(f"video Ray rollout does not support {recipe.model.recipe!r}")


__all__ = ["materialize_video_ray_rollout_policy"]
