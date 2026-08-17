"""Native construction of independently checkpointed Wan2.2 A14B roles."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from worldfoundry.base_models.diffusion_model.components import (
    BuildPurpose,
    ComponentBuildContext,
    ComponentKey,
    ComponentKind,
)
from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import (
    build_wan22_t2v_a14b_denoiser,
)
from worldfoundry.base_models.diffusion_model.optimizations import (
    AttentionBackend,
    RuntimePolicy,
)
from worldfoundry.training.data.wan22.assets import WAN22_T2V_A14B_REPOSITORY
from worldfoundry.training.models.wan import WanTrainAdapter
from worldfoundry.training.models.wan22 import (
    WAN22_A14B_BOUNDARY_RATIO,
    Wan22TrainAdapter,
)


@dataclass(frozen=True, slots=True)
class Wan22RoleCheckpoints:
    high_noise: CheckpointSpec
    low_noise: CheckpointSpec


def wan22_role_checkpoints(
    *,
    repository: str = WAN22_T2V_A14B_REPOSITORY,
    revision: str = "main",
) -> Wan22RoleCheckpoints:
    """Select the two official expert subtrees without integrity metadata."""

    return Wan22RoleCheckpoints(
        high_noise=CheckpointSpec(
            repo_id=repository,
            revision=revision,
            files=("high_noise_model/diffusion_pytorch_model.safetensors.index.json",),
            allow_patterns=("high_noise_model/*",),
        ),
        low_noise=CheckpointSpec(
            repo_id=repository,
            revision=revision,
            files=("low_noise_model/diffusion_pytorch_model.safetensors.index.json",),
            allow_patterns=("low_noise_model/*",),
        ),
    )


def _branch_adapter(
    checkpoint: CheckpointSpec,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    num_train_timesteps: int,
    gradient_checkpointing: bool,
    force_torch_attention: bool,
) -> WanTrainAdapter:
    context = ComponentBuildContext(
        model_id="wan2.2-t2v-a14b",
        key=ComponentKey(ComponentKind.DENOISER, name),
        purpose=BuildPurpose.TRAINING,
        policy=RuntimePolicy(
            device=device,
            dtype=dtype,
            attention=AttentionBackend.TORCH,
        ),
        checkpoints={"weights": checkpoint},
        component_options={"weight_dtype": dtype},
    )
    denoiser = build_wan22_t2v_a14b_denoiser(context)
    return WanTrainAdapter(
        denoiser,
        codec=None,
        conditioner=None,
        expected_latent_channels=16,
        temporal_compression=4,
        spatial_compression=8,
        model_timestep_scale=float(num_train_timesteps),
        num_train_timesteps=num_train_timesteps,
        gradient_checkpointing=gradient_checkpointing,
        attention_compatibility_mode=force_torch_attention,
    )


def load_wan22_role_adapter(
    *,
    checkpoints: Wan22RoleCheckpoints | None = None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    boundary_ratio: float = WAN22_A14B_BOUNDARY_RATIO,
    num_train_timesteps: int = 1000,
    gradient_checkpointing: bool = False,
    force_torch_attention: bool = True,
) -> Wan22TrainAdapter:
    """Materialize both expert weights into one WorldFoundry training role."""

    selected = wan22_role_checkpoints() if checkpoints is None else checkpoints
    resolved_device = torch.device(device)
    high_noise = _branch_adapter(
        selected.high_noise,
        name="high-noise",
        device=resolved_device,
        dtype=dtype,
        num_train_timesteps=num_train_timesteps,
        gradient_checkpointing=gradient_checkpointing,
        force_torch_attention=force_torch_attention,
    )
    low_noise = _branch_adapter(
        selected.low_noise,
        name="low-noise",
        device=resolved_device,
        dtype=dtype,
        num_train_timesteps=num_train_timesteps,
        gradient_checkpointing=gradient_checkpointing,
        force_torch_attention=force_torch_attention,
    )
    return Wan22TrainAdapter(
        high_noise,
        low_noise,
        boundary_ratio=boundary_ratio,
    )


__all__ = [
    "WAN22_T2V_A14B_REPOSITORY",
    "Wan22RoleCheckpoints",
    "load_wan22_role_adapter",
    "wan22_role_checkpoints",
]
