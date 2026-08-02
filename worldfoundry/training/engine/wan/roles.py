"""Wan role materialization, dtype validation, and tuning policy."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import nn

from worldfoundry.training.distributed.fsdp import FSDP2Application
from worldfoundry.training.models.wan import WanTrainAdapter
from worldfoundry.training.post_training.shared.role_checkpoints import (
    ResolvedRoleCheckpoint,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.tuning.peft import (
    PeftLoraApplication,
    apply_peft_lora_to_adapter,
)


def torch_dtype(value: str) -> torch.dtype:
    """Resolve the recipe's closed precision vocabulary."""

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[value]


def seed_initialization(seed: int | None) -> None:
    """Seed role construction consistently on CPU and every visible CUDA device."""

    if seed is None:
        return
    if isinstance(seed, bool):
        raise TypeError("initialization_seed must be an integer, not bool")
    resolved = int(seed) % (2**63 - 1)
    random.seed(resolved)
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)


def validate_model_dtype(adapter: WanTrainAdapter, expected_dtype: torch.dtype) -> None:
    """Reject silently mixed base-model precision before tuning or sharding."""

    base_dtypes = {
        parameter.dtype for parameter in adapter.trainable_module.parameters() if parameter.is_floating_point()
    }
    if base_dtypes != {expected_dtype}:
        raise ValueError(
            "loaded Wan parameter dtype differs from runtime.param_dtype: "
            f"loaded={sorted(map(str, base_dtypes))}, expected={expected_dtype}"
        )


def apply_wan_tuning(
    recipe: PostTrainingRecipe,
    adapter: WanTrainAdapter,
) -> PeftLoraApplication | None:
    """Apply the audited full-parameter or LoRA policy to one mutable role."""

    if recipe.tuning.mode == "full":
        adapter.trainable_module.requires_grad_(True)
        return None
    if recipe.tuning.mode != "lora":
        raise ValueError(f"unsupported Wan post-training tuning mode: {recipe.tuning.mode!r}")
    assert recipe.tuning.preset is not None
    assert recipe.tuning.rank is not None
    assert recipe.tuning.alpha is not None
    if recipe.tuning.preset != adapter.lora_target_preset:
        raise ValueError("recipe LoRA preset differs from the Wan adapter")
    return apply_peft_lora_to_adapter(
        adapter,
        preset=recipe.tuning.preset,
        rank=recipe.tuning.rank,
        alpha=recipe.tuning.alpha,
        dropout=recipe.tuning.dropout,
        modules_to_save=recipe.tuning.modules_to_save,
    )


def load_wan_role_adapter(
    *,
    assembler: object,
    native_recipe: object,
    checkpoint: ResolvedRoleCheckpoint,
    device: torch.device,
    dtype: torch.dtype,
    num_train_timesteps: int,
    gradient_checkpointing: bool,
    force_torch_attention: bool,
) -> WanTrainAdapter:
    """Build one independently checkpointed Wan denoiser role."""

    from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler
    from worldfoundry.base_models.diffusion_model.components import (
        BuildPurpose,
        ComponentKey,
        ComponentKind,
    )
    from worldfoundry.base_models.diffusion_model.optimizations import (
        AttentionBackend,
        RuntimePolicy,
    )
    from worldfoundry.base_models.diffusion_model.recipes.spec import NativeDiffusionRecipe
    from worldfoundry.training.models.wan import build_cached_wan_train_adapter

    if not isinstance(assembler, NativeDiffusionAssembler):
        raise TypeError("assembler must be NativeDiffusionAssembler")
    if not isinstance(native_recipe, NativeDiffusionRecipe):
        raise TypeError("native_recipe must be NativeDiffusionRecipe")
    denoiser_key = ComponentKey(ComponentKind.DENOISER)
    components = assembler.build_components(
        native_recipe,
        purpose=BuildPurpose.TRAINING,
        policy=RuntimePolicy(
            device=device,
            dtype=dtype,
            attention=AttentionBackend.TORCH,
        ),
        checkpoint_overrides={"dit": checkpoint.checkpoint},
        component_options={denoiser_key: {"weight_dtype": dtype}},
        component_keys=(denoiser_key,),
    )
    return build_cached_wan_train_adapter(
        components,
        expected_latent_channels=int(native_recipe.options.get("latent_channels", 16)),
        temporal_compression=int(native_recipe.options.get("temporal_compression", 4)),
        spatial_compression=int(native_recipe.options.get("spatial_compression", 8)),
        model_timestep_scale=float(num_train_timesteps),
        num_train_timesteps=num_train_timesteps,
        gradient_checkpointing=gradient_checkpointing,
        attention_compatibility_mode=force_torch_attention,
    )


def peft_identity(
    application: PeftLoraApplication | None,
) -> dict[str, object] | None:
    """Return the stable runtime identity of an optional LoRA application."""

    if application is None:
        return None
    return {
        "target_audit": application.target_audit.to_dict(),
        "trainable_parameter_names": list(application.trainable_parameter_names),
        "trainable_parameter_count": application.trainable_parameter_count,
    }


def fsdp_identity(
    application: FSDP2Application | None,
) -> dict[str, object] | None:
    """Return the stable runtime identity of an optional FSDP2 application."""

    if application is None:
        return None
    return {
        "digest": application.digest,
        "parameter_mode": application.parameter_mode,
    }


class DMDTrainableRoles(nn.Module):
    """One DCP model tree containing only mutable DMD roles."""

    def __init__(self, student: nn.Module, fake_score: nn.Module) -> None:
        super().__init__()
        if not isinstance(student, nn.Module) or not isinstance(fake_score, nn.Module):
            raise TypeError("DMD trainable roles must be nn.Module values")
        if student is fake_score:
            raise ValueError("DMD student and fake-score modules must be distinct")
        self.student = student
        self.fake_score = fake_score


@dataclass(frozen=True, slots=True)
class WanDMDRoleBundle:
    """Materialized Wan denoisers and their immutable runtime identities."""

    student: WanTrainAdapter
    real_score: WanTrainAdapter
    fake_score: WanTrainAdapter
    student_checkpoint: ResolvedRoleCheckpoint
    real_score_checkpoint: ResolvedRoleCheckpoint
    fake_score_checkpoint: ResolvedRoleCheckpoint
    student_peft: PeftLoraApplication | None
    fake_score_peft: PeftLoraApplication | None
    student_fsdp: FSDP2Application | None
    real_score_fsdp: FSDP2Application | None
    fake_score_fsdp: FSDP2Application | None

    def checkpoint_identity(self) -> dict[str, object]:
        return {
            "student": {
                **self.student_checkpoint.to_dict(),
                "digest": self.student_checkpoint.digest,
            },
            "real_score": {
                **self.real_score_checkpoint.to_dict(),
                "digest": self.real_score_checkpoint.digest,
            },
            "fake_score": {
                **self.fake_score_checkpoint.to_dict(),
                "digest": self.fake_score_checkpoint.digest,
            },
        }

    def runtime_identity(self) -> dict[str, object]:
        return {
            "checkpoints": self.checkpoint_identity(),
            "student": {
                "peft": peft_identity(self.student_peft),
                "fsdp2": fsdp_identity(self.student_fsdp),
            },
            "real_score": {
                "peft": None,
                "fsdp2": fsdp_identity(self.real_score_fsdp),
            },
            "fake_score": {
                "peft": peft_identity(self.fake_score_peft),
                "fsdp2": fsdp_identity(self.fake_score_fsdp),
            },
        }


__all__ = [
    "DMDTrainableRoles",
    "WanDMDRoleBundle",
    "apply_wan_tuning",
    "fsdp_identity",
    "load_wan_role_adapter",
    "peft_identity",
    "seed_initialization",
    "torch_dtype",
    "validate_model_dtype",
]
