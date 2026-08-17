"""Declarative auxiliary-optimizer requirements owned by algorithm specs.

Each post-training algorithm spec may expose ``auxiliary_optimizer_rules()``
returning the ordered presence/absence constraints (with their exact recipe
error messages) for ``fake_score_optimizer``, ``guidance_optimizer``, and
``discriminator_optimizer``.  Specs that do not declare rules fall back to
``DEFAULT_AUXILIARY_OPTIMIZER_RULES``, which rejects every auxiliary
optimizer.  ``PostTrainingRecipe`` validates uniformly against the
declaration, so adding an algorithm no longer edits a central isinstance
chain.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

AUXILIARY_OPTIMIZER_NAMES = (
    "fake_score_optimizer",
    "guidance_optimizer",
    "discriminator_optimizer",
)


@dataclass(frozen=True, slots=True)
class AuxiliaryOptimizerRule:
    """One ordered constraint with the exact error message it raises."""

    optimizers: tuple[str, ...]
    required: bool
    message: str

    def __post_init__(self) -> None:
        resolved = tuple(str(name) for name in self.optimizers)
        if not resolved:
            raise ValueError("auxiliary optimizer rule must name at least one optimizer")
        unknown = sorted(set(resolved) - set(AUXILIARY_OPTIMIZER_NAMES))
        if unknown:
            raise ValueError(f"unknown auxiliary optimizers: {unknown}")
        if not isinstance(self.required, bool):
            raise TypeError("required must be a bool")
        if not str(self.message).strip():
            raise ValueError("auxiliary optimizer rule message must be non-empty")
        object.__setattr__(self, "optimizers", resolved)
        object.__setattr__(self, "message", str(self.message))


def requires_auxiliary(name: str, message: str) -> AuxiliaryOptimizerRule:
    """Rule: the named optimizer section must be present."""

    return AuxiliaryOptimizerRule(optimizers=(name,), required=True, message=message)


def forbids_auxiliary(*names: str, message: str) -> AuxiliaryOptimizerRule:
    """Rule: none of the named optimizer sections may be present."""

    return AuxiliaryOptimizerRule(optimizers=tuple(names), required=False, message=message)


DEFAULT_AUXILIARY_OPTIMIZER_RULES = tuple(
    forbids_auxiliary(name, message=f"this algorithm cannot configure {name}")
    for name in AUXILIARY_OPTIMIZER_NAMES
)


def resolve_auxiliary_optimizer_rules(algorithm: object) -> tuple[AuxiliaryOptimizerRule, ...]:
    """Return the algorithm's declared rules or the reject-all default."""

    declaration = getattr(algorithm, "auxiliary_optimizer_rules", None)
    if declaration is None:
        return DEFAULT_AUXILIARY_OPTIMIZER_RULES
    if not callable(declaration):
        raise TypeError(f"{type(algorithm).__name__}.auxiliary_optimizer_rules must be callable")
    rules = declaration()
    if not isinstance(rules, Sequence) or isinstance(rules, (str, bytes)):
        raise TypeError(f"{type(algorithm).__name__} auxiliary optimizer rules must be a sequence")
    resolved = tuple(rules)
    for rule in resolved:
        if not isinstance(rule, AuxiliaryOptimizerRule):
            raise TypeError(f"{type(algorithm).__name__} declared a non-AuxiliaryOptimizerRule entry")
    return resolved


def validate_auxiliary_optimizers(
    algorithm: object,
    *,
    fake_score_optimizer: object | None,
    guidance_optimizer: object | None,
    discriminator_optimizer: object | None,
) -> None:
    """Apply the algorithm's declared rules in order, raising its messages."""

    values = {
        "fake_score_optimizer": fake_score_optimizer,
        "guidance_optimizer": guidance_optimizer,
        "discriminator_optimizer": discriminator_optimizer,
    }
    for rule in resolve_auxiliary_optimizer_rules(algorithm):
        if rule.required:
            if any(values[name] is None for name in rule.optimizers):
                raise ValueError(rule.message)
        elif any(values[name] is not None for name in rule.optimizers):
            raise ValueError(rule.message)


__all__ = [
    "AUXILIARY_OPTIMIZER_NAMES",
    "AuxiliaryOptimizerRule",
    "DEFAULT_AUXILIARY_OPTIMIZER_RULES",
    "forbids_auxiliary",
    "requires_auxiliary",
    "resolve_auxiliary_optimizer_rules",
    "validate_auxiliary_optimizers",
]
