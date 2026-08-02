"""Declarative component and execution specifications.

Model packages provide checkpoint-compatible component implementations.  They
do not provide runners, pipelines, loaders, or optimization managers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping

from .loaders import CheckpointSpec
from .optimizations import AttentionBackend, OffloadMode, QuantizationMode, RuntimePolicy


class BuildPurpose(str, Enum):
    """Why a component graph is being constructed.

    Component factories default to inference for backwards compatibility.  A
    training assembler must opt in explicitly so factories can reject
    inference-only wrappers before parameters or optimizer state are created.
    """

    INFERENCE = "inference"
    TRAINING = "training"
    ROLLOUT = "rollout"
    REWARD = "reward"


class ComponentKind(str, Enum):
    DENOISER = "denoiser"
    CONDITIONER = "conditioner"
    LATENT_ENCODER = "latent_encoder"
    LATENT_INITIALIZER = "latent_initializer"
    LATENT_PROCESSOR = "latent_processor"
    SCHEDULER = "scheduler"
    DECODER = "decoder"


@dataclass(frozen=True, slots=True, order=True)
class ComponentKey:
    """Stable kind/name identity for a component within one recipe."""

    kind: ComponentKind
    name: str = "main"

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ComponentKind(self.kind))
        normalized_name = str(self.name).strip().lower().replace("_", "-")
        if not normalized_name:
            raise ValueError("component name cannot be empty")
        object.__setattr__(self, "name", normalized_name)

    def __str__(self) -> str:
        return f"{self.kind.value}:{self.name}"


@dataclass(frozen=True, slots=True)
class ComponentBuildContext:
    """Framework-owned input to a pure component factory."""

    model_id: str
    key: ComponentKey
    policy: RuntimePolicy
    checkpoints: Mapping[str, CheckpointSpec] = field(default_factory=dict)
    recipe_options: Mapping[str, object] = field(default_factory=dict)
    component_options: Mapping[str, object] = field(default_factory=dict)
    purpose: BuildPurpose = BuildPurpose.INFERENCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "purpose", BuildPurpose(self.purpose))
        validate_runtime_policy_for_purpose(self.policy, self.purpose)
        checkpoints = {str(name): value for name, value in self.checkpoints.items()}
        if any(not name.strip() for name in checkpoints):
            raise ValueError(f"checkpoint binding names for {self.key} cannot be empty")
        if not all(isinstance(value, CheckpointSpec) for value in checkpoints.values()):
            raise TypeError(f"checkpoint bindings for {self.key} must contain CheckpointSpec values")
        object.__setattr__(self, "checkpoints", MappingProxyType(checkpoints))
        object.__setattr__(self, "recipe_options", MappingProxyType(dict(self.recipe_options)))
        object.__setattr__(
            self,
            "component_options",
            MappingProxyType(dict(self.component_options)),
        )

    def checkpoint(self, name: str = "weights") -> CheckpointSpec | None:
        """Return one optional named checkpoint bound by the recipe."""

        return self.checkpoints.get(name)

    def require_checkpoint(self, name: str = "weights") -> CheckpointSpec:
        """Return one named checkpoint or fail before model construction."""

        try:
            return self.checkpoints[name]
        except KeyError as error:
            raise KeyError(f"component {self.key} requires checkpoint binding {name!r}") from error


_TRAINING_FORBIDDEN_OPTIONS = frozenset(
    {
        "block_swap",
        "cache_skip",
        "cuda_graph",
        "enable_cuda_graph",
        "inference_mode",
        "step_cache",
        "teacache",
    }
)


def validate_runtime_policy_for_purpose(policy: RuntimePolicy, purpose: BuildPurpose | str) -> None:
    """Fail closed when a training build requests inference-only behavior.

    P0 deliberately accepts a narrow policy.  Frozen-component offload and
    training-safe quantization can be added later through training-owned
    contracts instead of silently reusing inference wrappers.
    """

    resolved = BuildPurpose(purpose)
    if resolved is not BuildPurpose.TRAINING:
        return

    errors: list[str] = []
    if policy.offload.mode is not OffloadMode.NONE:
        errors.append(f"offload={policy.offload.mode.value}")
    if policy.quantization.mode is not QuantizationMode.NONE:
        errors.append(f"quantization={policy.quantization.mode.value}")
    if policy.attention is AttentionBackend.SAGE:
        errors.append("attention=sage")
    enabled_options = sorted(key for key in _TRAINING_FORBIDDEN_OPTIONS if bool(policy.options.get(key)))
    if enabled_options:
        errors.append(f"inference-only options={enabled_options}")
    if errors:
        raise ValueError("training component build rejects " + ", ".join(errors))


ComponentFactory = Callable[[ComponentBuildContext], object]


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """One architecture component selected by a model recipe."""

    key: ComponentKey
    factory: ComponentFactory
    checkpoints: Mapping[str, str] = field(default_factory=dict)
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not callable(self.factory):
            raise TypeError(f"component factory for {self.key} must be callable")
        checkpoints = {str(name): str(value) for name, value in self.checkpoints.items()}
        if any(not name.strip() or not value.strip() for name, value in checkpoints.items()):
            raise ValueError(f"checkpoint bindings for {self.key} cannot be empty")
        object.__setattr__(self, "checkpoints", MappingProxyType(checkpoints))
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


STANDARD_COMPONENT_BINDINGS = MappingProxyType(
    {
        "denoiser": ComponentKey(ComponentKind.DENOISER),
        "conditioner": ComponentKey(ComponentKind.CONDITIONER),
        "latent_initializer": ComponentKey(ComponentKind.LATENT_INITIALIZER),
        "scheduler": ComponentKey(ComponentKind.SCHEDULER),
        "decoder": ComponentKey(ComponentKind.DECODER),
    }
)


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """Selection of a framework-owned execution strategy and its bindings."""

    strategy: str = "standard"
    bindings: Mapping[str, ComponentKey] = field(default_factory=lambda: dict(STANDARD_COMPONENT_BINDINGS))
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        strategy = str(self.strategy).strip().lower().replace("_", "-")
        if not strategy:
            raise ValueError("execution strategy cannot be empty")
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "bindings", MappingProxyType(dict(self.bindings)))
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


__all__ = [
    "BuildPurpose",
    "ComponentBuildContext",
    "ComponentFactory",
    "ComponentKey",
    "ComponentKind",
    "ComponentSpec",
    "ExecutionSpec",
    "STANDARD_COMPONENT_BINDINGS",
    "validate_runtime_policy_for_purpose",
]
