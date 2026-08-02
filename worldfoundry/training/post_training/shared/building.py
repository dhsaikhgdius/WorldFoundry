"""Model-neutral construction helpers for native post-training stacks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import torch

from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection
from worldfoundry.training.optimization import build_adamw, build_came, trainable_parameters
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.recipes.spec import OptimizerSpec

from .contracts import FlowPredictionAdapter


def resolve_tensor_dtype(name: str) -> torch.dtype:
    """Resolve the recipe's supported post-training tensor dtype."""

    try:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[name]
    except KeyError as error:
        raise ValueError(f"unsupported post-training tensor dtype: {name!r}") from error


def validate_post_training_recipe(recipe: PostTrainingRecipe) -> None:
    """Validate execution invariants common to every native algorithm."""

    if not isinstance(recipe, PostTrainingRecipe):
        raise TypeError("recipe must be PostTrainingRecipe")
    if recipe.execution_owner != "worldfoundry-native":
        raise ValueError("post-training execution must be owned by WorldFoundry")
    if recipe.runtime.reduce_dtype != "float32":
        raise ValueError("native post-training reductions currently require float32")
    if recipe.runtime.compile:
        raise ValueError("torch.compile is not supported by native post-training")


def build_post_training_optimizer(
    spec: OptimizerSpec,
    module: torch.nn.Module,
    *,
    fused: bool | Literal["auto"],
    role: str,
) -> torch.optim.Optimizer:
    """Build a native optimizer from its behavior-bearing recipe spec."""

    if spec.gradient_accumulation_steps != 1:
        raise ValueError(f"native {role} gradient accumulation is not implemented; set gradient_accumulation_steps=1")
    parameters = trainable_parameters(module)
    if spec.type == "adamw":
        if not isinstance(spec.epsilon, float) or len(spec.betas) != 2:
            raise RuntimeError("validated AdamW optimizer fields have an invalid shape")
        return build_adamw(
            parameters,
            learning_rate=spec.learning_rate,
            weight_decay=spec.weight_decay,
            betas=(spec.betas[0], spec.betas[1]),
            epsilon=spec.epsilon,
            fused=fused,
        )
    if spec.type == "came":
        if not isinstance(spec.epsilon, tuple) or len(spec.epsilon) != 2 or len(spec.betas) != 3:
            raise RuntimeError("validated CAME optimizer fields have an invalid shape")
        if spec.update_clip_threshold is None:
            raise RuntimeError("validated CAME update clip threshold is missing")
        return build_came(
            parameters,
            learning_rate=spec.learning_rate,
            weight_decay=spec.weight_decay,
            betas=(spec.betas[0], spec.betas[1], spec.betas[2]),
            epsilons=spec.epsilon,
            update_clip_threshold=spec.update_clip_threshold,
        )
    raise ValueError(f"native {role} does not support optimizer.type={spec.type!r}")


def prediction_module(adapter: FlowPredictionAdapter, *, role: str) -> torch.nn.Module:
    """Return and validate the module owned by a flow prediction adapter."""

    if not isinstance(adapter, FlowPredictionAdapter):
        raise TypeError(f"{role} must implement FlowPredictionAdapter")
    module = adapter.module
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"{role}.module must be an nn.Module")
    return module


def require_checkpoint_identity(
    adapter: object,
    expected: str,
    *,
    role: str,
) -> str:
    """Bind a loaded role to the checkpoint selected by its recipe.

    This is an execution gate rather than run metadata: builders fail before
    allocating optimizers when a caller supplies an anonymous or different
    model role.
    """

    actual_value = getattr(adapter, "checkpoint_identity", None)
    if not isinstance(actual_value, str) or not actual_value.strip():
        raise ValueError(f"{role}.checkpoint_identity must be a non-empty string")
    actual = actual_value.strip()
    resolved_expected = str(expected).strip()
    if not resolved_expected:
        raise ValueError(f"{role} recipe checkpoint identity must be non-empty")
    if actual != resolved_expected:
        raise ValueError(
            f"{role} loaded checkpoint identity {actual!r} differs from recipe "
            f"{resolved_expected!r}"
        )
    return actual


def require_independent_modules(
    roles: Mapping[str, torch.nn.Module],
) -> None:
    """Reject aliased model roles and shared parameters before training."""

    normalized = {str(name): module for name, module in roles.items()}
    if len(normalized) < 2:
        raise ValueError("at least two model roles are required for independence checks")
    for name, module in normalized.items():
        if not name.strip():
            raise ValueError("model role names must be non-empty")
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"{name} must be an nn.Module")
    module_ids = [id(module) for module in normalized.values()]
    if len(set(module_ids)) != len(module_ids):
        raise ValueError("model roles must be independently materialized modules")
    inventories = {
        name: {id(parameter) for parameter in module.parameters()}
        for name, module in normalized.items()
    }
    names = tuple(inventories)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if inventories[left] & inventories[right]:
                raise ValueError(
                    f"model roles {left!r} and {right!r} share parameters"
                )


def require_disjoint_trainable_parameters(
    roles: Mapping[str, torch.nn.Module],
) -> None:
    """Allow a shared frozen base while rejecting shared optimizer state."""

    normalized = {str(name): module for name, module in roles.items()}
    if len(normalized) < 2:
        raise ValueError("at least two trainable roles are required")
    if len({id(module) for module in normalized.values()}) != len(normalized):
        raise ValueError("trainable roles must expose distinct module views")
    inventories: dict[str, set[int]] = {}
    for name, module in normalized.items():
        if not name.strip() or not isinstance(module, torch.nn.Module):
            raise TypeError("trainable role names and modules must be valid")
        inventories[name] = {
            id(parameter)
            for parameter in module.parameters()
            if parameter.requires_grad
        }
    names = tuple(inventories)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if inventories[left] & inventories[right]:
                raise ValueError(
                    f"model roles {left!r} and {right!r} share trainable parameters"
                )


def named_stateful_collection(**components: object | None) -> NamedStatefulCollection | None:
    """Collect present checkpoint components under stable role names."""

    active = {name: value for name, value in components.items() if value is not None}
    return None if not active else NamedStatefulCollection(active)


__all__ = [
    "build_post_training_optimizer",
    "named_stateful_collection",
    "prediction_module",
    "require_checkpoint_identity",
    "require_disjoint_trainable_parameters",
    "require_independent_modules",
    "resolve_tensor_dtype",
    "validate_post_training_recipe",
]
