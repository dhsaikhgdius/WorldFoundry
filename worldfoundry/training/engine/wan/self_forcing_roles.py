"""Causal-student and score-model roles for Wan Self-Forcing."""

from __future__ import annotations

from dataclasses import dataclass

from worldfoundry.training.distributed.fsdp import FSDP2Application
from worldfoundry.training.models.causal_wan import CausalWanTrainRole
from worldfoundry.training.models.wan import WanTrainAdapter
from worldfoundry.training.post_training.shared.role_checkpoints import (
    ResolvedRoleCheckpoint,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.tuning.peft import (
    PeftLoraApplication,
    apply_peft_lora,
)

from .roles import fsdp_identity, peft_identity


def apply_causal_wan_tuning(
    recipe: PostTrainingRecipe,
    role: CausalWanTrainRole,
) -> PeftLoraApplication | None:
    """Apply the recipe's full or audited Wan-attention LoRA policy."""

    if recipe.tuning.mode == "full":
        role.trainable_module.requires_grad_(True)
        role.trainable_module.eval()
        return None
    if recipe.tuning.mode != "lora":
        raise ValueError(f"unsupported causal Wan Self-Forcing tuning mode: {recipe.tuning.mode!r}")
    assert recipe.tuning.preset is not None
    assert recipe.tuning.rank is not None
    assert recipe.tuning.alpha is not None
    if recipe.tuning.preset != role.lora_target_preset:
        raise ValueError("recipe LoRA preset differs from the causal Wan role")
    application = apply_peft_lora(
        role.graph,
        preset=recipe.tuning.preset,
        rank=recipe.tuning.rank,
        alpha=recipe.tuning.alpha,
        dropout=recipe.tuning.dropout,
        modules_to_save=recipe.tuning.modules_to_save,
    )
    role.replace_trainable_module(application.model)
    role.trainable_module.train()
    return application


@dataclass(frozen=True, slots=True)
class WanSelfForcingRoleBundle:
    """One causal student plus independent real/fake Wan score roles."""

    student: CausalWanTrainRole
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
            name: {**checkpoint.to_dict(), "digest": checkpoint.digest}
            for name, checkpoint in (
                ("student", self.student_checkpoint),
                ("real_score", self.real_score_checkpoint),
                ("fake_score", self.fake_score_checkpoint),
            )
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


__all__ = ["WanSelfForcingRoleBundle", "apply_causal_wan_tuning"]
