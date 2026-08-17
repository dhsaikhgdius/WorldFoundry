"""Released HunyuanVideo RL profiles expressed in native recipe terms."""

from __future__ import annotations

from dataclasses import dataclass

from worldfoundry.training.post_training.rl.transitions.flow_sde import (
    flow_match_sigma_schedule,
)


@dataclass(frozen=True, slots=True)
class HunyuanVideoRLProfile:
    model_recipe: str
    sigmas: tuple[float, ...]
    sde_timestep_fraction: tuple[float, float]
    num_sde_steps: int
    eta: float
    group_size: int
    updates_per_trajectory: int
    trajectory_dtype: str
    lora_rank: int
    lora_alpha: int
    learning_rate: float
    weight_decay: float
    generation: tuple[int, int, int]
    guidance_scale: float
    old_log_prob_source: str
    global_prompt_batch_size: int

    @property
    def sigma_max(self) -> float:
        return self.sigmas[1]


def hunyuan_video_rl_profile(model_recipe: str) -> HunyuanVideoRLProfile:
    """Return the current UniRL train-side profile without fixing GPU count."""

    if model_recipe == "hunyuanvideo-t2v":
        return HunyuanVideoRLProfile(
            model_recipe=model_recipe,
            sigmas=flow_match_sigma_schedule(10, shift=5.0),
            sde_timestep_fraction=(0.0, 0.5),
            num_sde_steps=4,
            eta=0.7,
            group_size=8,
            updates_per_trajectory=2,
            trajectory_dtype="bfloat16",
            lora_rank=64,
            lora_alpha=256,
            learning_rate=3.0e-4,
            weight_decay=0.0,
            generation=(720, 1280, 5),
            guidance_scale=1.0,
            old_log_prob_source="rollout",
            global_prompt_batch_size=8,
        )
    if model_recipe == "hunyuanvideo-1.5-t2v":
        return HunyuanVideoRLProfile(
            model_recipe=model_recipe,
            sigmas=flow_match_sigma_schedule(16, shift=5.0),
            sde_timestep_fraction=(0.0, 0.6),
            num_sde_steps=8,
            eta=0.25,
            group_size=24,
            updates_per_trajectory=2,
            trajectory_dtype="bfloat16",
            lora_rank=64,
            lora_alpha=64,
            learning_rate=1.0e-5,
            weight_decay=1.0e-4,
            generation=(480, 480, 5),
            guidance_scale=0.0,
            old_log_prob_source="rollout",
            global_prompt_batch_size=48,
        )
    raise ValueError(f"unsupported HunyuanVideo RL profile: {model_recipe!r}")


__all__ = ["HunyuanVideoRLProfile", "hunyuan_video_rl_profile"]
