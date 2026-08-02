"""Strict weight loading across diagonal distillation stages."""

from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from worldfoundry.core.checkpoint import validate_state_dict_compatibility
from worldfoundry.training.models.causal_wan import convert_self_forcing_causal_state_dict


def load_diagonal_ode_initialization(
    generator: nn.Module,
    checkpoint: Mapping[str, object],
) -> None:
    """Load either released causal ODE envelope into the native generator graph."""

    if not isinstance(generator, nn.Module):
        raise TypeError("generator must be nn.Module")
    converted = convert_self_forcing_causal_state_dict(checkpoint)
    validate_state_dict_compatibility(generator, converted, label="causal ODE initialization")
    incompatible = generator.load_state_dict(converted, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict causal ODE initialization returned incompatible keys")


def load_diagonal_stage_weights(
    student: nn.Module,
    checkpoint: Mapping[str, object],
) -> None:
    """Strictly carry identical trainable topology from stage one to stage two."""

    if not isinstance(student, nn.Module):
        raise TypeError("student must be nn.Module")
    converted = convert_self_forcing_causal_state_dict(checkpoint)
    validate_state_dict_compatibility(student, converted, label="diagonal stage checkpoint")
    incompatible = student.load_state_dict(converted, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict diagonal stage load returned incompatible keys")


__all__ = ["load_diagonal_ode_initialization", "load_diagonal_stage_weights"]
