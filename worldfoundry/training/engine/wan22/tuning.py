"""LoRA targeting for both native Wan2.2 A14B experts."""

from __future__ import annotations

import re

from torch import nn

from worldfoundry.training.models.wan22 import (
    WAN22_DUAL_ATTENTION,
    Wan22TrainAdapter,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.tuning.peft import (
    LoraTargetAudit,
    PeftLoraApplication,
    apply_peft_lora_with_audit,
)

_TARGET_PATTERN = re.compile(
    r"^(?P<expert>high_noise|low_noise)\.blocks\.(?P<block>\d+)\."
    r"(?P<role>(?:self|cross)_attn\.(?:q|k|v|o))$"
)
_ATTENTION_ROLES = frozenset(
    {
        "self_attn.q",
        "self_attn.k",
        "self_attn.v",
        "self_attn.o",
        "cross_attn.q",
        "cross_attn.k",
        "cross_attn.v",
        "cross_attn.o",
    }
)


def audit_wan22_lora_targets(model: nn.Module) -> LoraTargetAudit:
    """Resolve the attention projections in every block of both experts."""

    roles: dict[tuple[str, int], set[str]] = {}
    names: list[str] = []
    shapes: dict[str, tuple[int, int]] = {}
    for name, module in model.named_modules():
        match = _TARGET_PATTERN.fullmatch(name)
        if match is None:
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Wan2.2 LoRA target {name!r} must be nn.Linear")
        key = (match.group("expert"), int(match.group("block")))
        roles.setdefault(key, set()).add(match.group("role"))
        names.append(name)
        shapes[name] = (int(module.in_features), int(module.out_features))
    for expert in ("high_noise", "low_noise"):
        if not any(name == expert for name, _ in roles):
            raise ValueError(f"Wan2.2 LoRA found no {expert} transformer blocks")
    drift = {
        f"{expert}.blocks.{block}": sorted(_ATTENTION_ROLES - found)
        for (expert, block), found in roles.items()
        if found != _ATTENTION_ROLES
    }
    if drift:
        raise ValueError(f"Wan2.2 attention target graph differs: {drift}")
    names.sort()
    return LoraTargetAudit(
        preset=WAN22_DUAL_ATTENTION,
        target_pattern=_TARGET_PATTERN.pattern,
        module_names=tuple(names),
        module_shapes=shapes,
        block_count=len(roles),
        base_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )


def apply_wan22_tuning(
    recipe: PostTrainingRecipe,
    adapter: Wan22TrainAdapter,
) -> PeftLoraApplication | None:
    """Apply full or dual-expert LoRA tuning before optimizer construction."""

    if recipe.tuning.mode == "full":
        adapter.trainable_module.requires_grad_(True)
        return None
    if recipe.tuning.mode != "lora":
        raise ValueError("Wan2.2 policy training supports full or LoRA tuning")
    if recipe.tuning.preset != WAN22_DUAL_ATTENTION:
        raise ValueError(f"Wan2.2 LoRA requires tuning.preset={WAN22_DUAL_ATTENTION!r}")
    assert recipe.tuning.rank is not None
    assert recipe.tuning.alpha is not None
    application = apply_peft_lora_with_audit(
        adapter.trainable_module,
        audit=audit_wan22_lora_targets(adapter.trainable_module),
        rank=recipe.tuning.rank,
        alpha=recipe.tuning.alpha,
        dropout=recipe.tuning.dropout,
        modules_to_save=recipe.tuning.modules_to_save,
    )
    adapter.trainable_module = application.model
    return application


__all__ = ["apply_wan22_tuning", "audit_wan22_lora_targets"]
