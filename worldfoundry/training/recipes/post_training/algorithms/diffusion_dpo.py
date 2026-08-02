"""Pure recipe contract for paired Diffusion-DPO training."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..common import strict_mapping

DIFFUSION_DPO_ALGORITHM_FIELDS = {
    "type",
    "beta",
}


@dataclass(frozen=True, slots=True)
class DiffusionDPOAlgorithmSpec:
    """Preference strength for a caller-supplied frozen reference policy."""

    beta: float
    type: str = "diffusion-dpo"

    def __post_init__(self) -> None:
        resolved_type = str(self.type).strip().lower().replace("_", "-")
        if resolved_type != "diffusion-dpo":
            raise ValueError("Diffusion-DPO algorithm type must be 'diffusion-dpo'")
        beta = float(self.beta)
        if not isfinite(beta) or beta <= 0:
            raise ValueError("Diffusion-DPO beta must be finite and positive")
        object.__setattr__(self, "type", resolved_type)
        object.__setattr__(self, "beta", beta)


def parse_diffusion_dpo_algorithm(value: object) -> DiffusionDPOAlgorithmSpec:
    """Parse a strict Diffusion-DPO algorithm section."""

    payload = strict_mapping(
        value,
        field_name="algorithm",
        allowed=DIFFUSION_DPO_ALGORITHM_FIELDS,
    )
    missing = sorted({"beta"} - set(payload))
    if missing:
        raise ValueError(f"Diffusion-DPO algorithm is missing required fields: {missing}")
    return DiffusionDPOAlgorithmSpec(**payload)


__all__ = ["DiffusionDPOAlgorithmSpec"]
