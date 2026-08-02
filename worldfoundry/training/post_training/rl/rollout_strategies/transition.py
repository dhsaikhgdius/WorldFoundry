"""Pluggable stochastic transition strategies for flow-policy rollouts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from ..transitions.constant_diffusion import constant_diffusion_flow_transition
from ..transitions.flow_sde import FlowSDETransition, flow_sde_transition


@runtime_checkable
class FlowTransitionStrategy(Protocol):
    eta: float

    @property
    def identity(self) -> Mapping[str, object]: ...

    def step(
        self,
        velocity: object,
        sample: object,
        sigma: object,
        sigma_next: object,
        *,
        generator: object | None = None,
        next_sample: object | None = None,
        trajectory_dtype: object | None = None,
    ) -> FlowSDETransition: ...


@dataclass(frozen=True, slots=True)
class VariancePreservingFlowTransition:
    eta: float
    sigma_max: float

    def __post_init__(self) -> None:
        eta = float(self.eta)
        sigma_max = float(self.sigma_max)
        if not isfinite(eta) or eta <= 0:
            raise ValueError("eta must be finite and positive")
        if not isfinite(sigma_max) or not 0 < sigma_max < 1:
            raise ValueError("sigma_max must be finite and in (0,1)")
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "sigma_max", sigma_max)

    @property
    def identity(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "kind": "variance-preserving",
                "eta": self.eta,
                "sigma_max": self.sigma_max,
            }
        )

    def step(
        self,
        velocity: object,
        sample: object,
        sigma: object,
        sigma_next: object,
        *,
        generator: object | None = None,
        next_sample: object | None = None,
        trajectory_dtype: object | None = None,
    ) -> FlowSDETransition:
        return flow_sde_transition(
            velocity,
            sample,
            sigma,
            sigma_next,
            eta=self.eta,
            sigma_max=self.sigma_max,
            generator=generator,
            next_sample=next_sample,
            trajectory_dtype=trajectory_dtype,
        )


@dataclass(frozen=True, slots=True)
class ConstantDiffusionFlowTransition:
    eta: float

    def __post_init__(self) -> None:
        eta = float(self.eta)
        if not isfinite(eta) or eta <= 0:
            raise ValueError("eta must be finite and positive")
        object.__setattr__(self, "eta", eta)

    @property
    def identity(self) -> Mapping[str, object]:
        return MappingProxyType({"kind": "constant-diffusion", "eta": self.eta})

    def step(
        self,
        velocity: object,
        sample: object,
        sigma: object,
        sigma_next: object,
        *,
        generator: object | None = None,
        next_sample: object | None = None,
        trajectory_dtype: object | None = None,
    ) -> FlowSDETransition:
        return constant_diffusion_flow_transition(
            velocity,
            sample,
            sigma,
            sigma_next,
            eta=self.eta,
            generator=generator,
            next_sample=next_sample,
            trajectory_dtype=trajectory_dtype,
        )


def flow_transition_strategy_from_identity(
    value: Mapping[str, object],
) -> FlowTransitionStrategy:
    """Restore one strict transition strategy from trajectory state."""

    if not isinstance(value, Mapping):
        raise TypeError("transition identity must be a mapping")
    payload = {str(key): item for key, item in value.items()}
    kind = str(payload.get("kind", "")).strip().lower()
    if kind == "variance-preserving":
        if set(payload) != {"kind", "eta", "sigma_max"}:
            raise ValueError("variance-preserving transition identity fields differ")
        return VariancePreservingFlowTransition(
            eta=float(payload["eta"]),
            sigma_max=float(payload["sigma_max"]),
        )
    if kind == "constant-diffusion":
        if set(payload) != {"kind", "eta"}:
            raise ValueError("constant-diffusion transition identity fields differ")
        return ConstantDiffusionFlowTransition(eta=float(payload["eta"]))
    raise ValueError(f"unsupported flow transition strategy: {kind!r}")


__all__ = [
    "ConstantDiffusionFlowTransition",
    "FlowTransitionStrategy",
    "VariancePreservingFlowTransition",
    "flow_transition_strategy_from_identity",
]
