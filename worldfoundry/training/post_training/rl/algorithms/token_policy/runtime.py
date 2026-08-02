"""Small runtime factory for selecting a token-policy loss stage."""

from __future__ import annotations

from collections.abc import Mapping

from .stages import (
    TokenCPPOStage,
    TokenDPPOStage,
    TokenDRPOStage,
    TokenGRPOStage,
    TokenGSPOStage,
    TokenPolicyStage,
)

_STAGE_TYPES = {
    "grpo": TokenGRPOStage,
    "gspo": TokenGSPOStage,
    "dppo": TokenDPPOStage,
    "drpo": TokenDRPOStage,
    "cppo": TokenCPPOStage,
}


def build_token_policy_stage(
    name: str,
    settings: Mapping[str, object] | None = None,
) -> TokenPolicyStage:
    """Construct one native algorithm stage without selecting a model backend."""

    normalized = str(name).strip().lower().replace("_", "-")
    stage_type = _STAGE_TYPES.get(normalized)
    if stage_type is None:
        raise ValueError(f"unsupported token-policy algorithm: {name!r}; expected one of {sorted(_STAGE_TYPES)}")
    kwargs = {} if settings is None else dict(settings)
    return stage_type(**kwargs)


__all__ = ["build_token_policy_stage"]
