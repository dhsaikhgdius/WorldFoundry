"""Framework-owned execution strategy registry.

Recipes select a strategy by ID. They cannot provide runner factories or
register strategies as an import side effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, runtime_checkable

from ..components import ComponentKey
from ..contracts import DiffusionOutput, DiffusionRequest
from ..extensions import DiffusionExtension, FrozenContextSuffixExtension, FrozenLatentMaskExtension
from ..optimizations import RuntimePolicy
from ..recipes.spec import NativeDiffusionRecipe
from .autoregressive import AutoregressiveWindowRunner
from .base import DualConditionGuidanceRunner, NativeDiffusionRunner, RunnerComponents
from .chunked import ChunkedKVCacheRunner
from .multistage import JointMultiStageDiffusionRunner, MultiStageComponents


@runtime_checkable
class DiffusionExecutor(Protocol):
    """Common execution surface consumed by the native public pipeline."""

    model_id: str

    def run(self, request: DiffusionRequest) -> DiffusionOutput:
        """Execute one normalized native diffusion request."""


@dataclass(frozen=True, slots=True)
class ExecutionBuildContext:
    """Complete framework-owned input for an execution strategy builder."""

    recipe: NativeDiffusionRecipe
    components: Mapping[ComponentKey, object]
    policy: RuntimePolicy
    extensions: tuple[DiffusionExtension, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))


ExecutionStrategyBuilder = Callable[[ExecutionBuildContext], DiffusionExecutor]


class UnsupportedExecutionStrategyError(ValueError):
    """Raised when a recipe selects an unknown framework strategy."""


class ExecutionStrategyRegistry:
    """Explicit instance-local registry for framework execution strategies."""

    def __init__(self) -> None:
        self._builders: dict[str, ExecutionStrategyBuilder] = {}
        self._frozen = False

    def register(self, strategy: str, builder: ExecutionStrategyBuilder) -> None:
        if self._frozen:
            raise RuntimeError("execution strategy registry is frozen")
        key = _strategy_key(strategy)
        if key in self._builders:
            raise ValueError(f"execution strategy is already registered: {key}")
        if not callable(builder):
            raise TypeError("execution strategy builder must be callable")
        self._builders[key] = builder

    def freeze(self) -> None:
        self._frozen = True

    def build(self, strategy: str, context: ExecutionBuildContext) -> DiffusionExecutor:
        key = _strategy_key(strategy)
        try:
            builder = self._builders[key]
        except KeyError as error:
            raise UnsupportedExecutionStrategyError(key) from error
        executor = builder(context)
        if not isinstance(executor, DiffusionExecutor):
            raise TypeError(
                f"execution strategy {key!r} returned {type(executor).__name__}; expected DiffusionExecutor"
            )
        return executor

    def strategy_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))


def build_standard_strategy(context: ExecutionBuildContext) -> NativeDiffusionRunner:
    """Build the shared condition-initialize-denoise-decode execution loop."""

    required_bindings = {
        "denoiser",
        "conditioner",
        "latent_initializer",
        "scheduler",
        "decoder",
    }
    optional_bindings = {"latent_encoder"}
    actual_bindings = set(context.recipe.execution.bindings)
    if not required_bindings.issubset(actual_bindings) or actual_bindings - required_bindings - optional_bindings:
        raise ValueError(
            "standard execution requires the canonical bindings and optionally latent_encoder: "
            f"required={sorted(required_bindings)}; got={sorted(actual_bindings)}"
        )
    bound = {role: context.components[key] for role, key in context.recipe.execution.bindings.items()}
    return NativeDiffusionRunner(
        model_id=context.recipe.model_id,
        components=RunnerComponents(
            denoiser=bound["denoiser"],  # type: ignore[arg-type]
            conditioner=bound["conditioner"],  # type: ignore[arg-type]
            latent_initializer=bound["latent_initializer"],  # type: ignore[arg-type]
            scheduler=bound["scheduler"],  # type: ignore[arg-type]
            decoder=bound["decoder"],  # type: ignore[arg-type]
            latent_encoder=bound.get("latent_encoder"),  # type: ignore[arg-type]
        ),
        device=context.policy.device,
        dtype=context.policy.dtype,
        extensions=context.extensions,
        guidance_mode=str(context.recipe.execution.options.get("guidance_mode", "standard")),
    )


def build_dual_condition_guidance_strategy(
    context: ExecutionBuildContext,
) -> DualConditionGuidanceRunner:
    """Build the standard lifecycle with framework-owned three-branch CFG."""

    required_bindings = {
        "denoiser",
        "conditioner",
        "latent_initializer",
        "scheduler",
        "decoder",
        "latent_encoder",
    }
    actual_bindings = set(context.recipe.execution.bindings)
    if actual_bindings != required_bindings:
        raise ValueError(
            "dual-condition-guidance execution requires canonical codec bindings: "
            f"required={sorted(required_bindings)}; got={sorted(actual_bindings)}"
        )
    bound = {role: context.components[key] for role, key in context.recipe.execution.bindings.items()}
    return DualConditionGuidanceRunner(
        model_id=context.recipe.model_id,
        components=RunnerComponents(
            denoiser=bound["denoiser"],  # type: ignore[arg-type]
            conditioner=bound["conditioner"],  # type: ignore[arg-type]
            latent_initializer=bound["latent_initializer"],  # type: ignore[arg-type]
            scheduler=bound["scheduler"],  # type: ignore[arg-type]
            decoder=bound["decoder"],  # type: ignore[arg-type]
            latent_encoder=bound["latent_encoder"],  # type: ignore[arg-type]
        ),
        device=context.policy.device,
        dtype=context.policy.dtype,
        extensions=context.extensions,
        secondary_guidance_scale=float(
            context.recipe.execution.options.get("secondary_guidance_scale", 1.0)
        ),
        secondary_guidance_input=str(
            context.recipe.execution.options.get(
                "secondary_guidance_input",
                "secondary_guidance_scale",
            )
        ),
    )


def build_frozen_context_strategy(
    context: ExecutionBuildContext,
) -> NativeDiffusionRunner:
    """Use the standard loop with framework-owned frozen suffix relocking."""

    extension = FrozenContextSuffixExtension()
    if any(item.extension_id == extension.extension_id for item in context.extensions):
        raise ValueError("frozen-context strategy installs its suffix extension automatically")
    enriched = ExecutionBuildContext(
        recipe=context.recipe,
        components=context.components,
        policy=context.policy,
        extensions=(*context.extensions, extension),
    )
    return build_standard_strategy(enriched)


def build_masked_latent_strategy(
    context: ExecutionBuildContext,
) -> NativeDiffusionRunner:
    """Use the standard loop while projecting frozen latent regions after each step."""

    extension = FrozenLatentMaskExtension()
    if any(item.extension_id == extension.extension_id for item in context.extensions):
        raise ValueError("masked-latent strategy installs its projection extension automatically")
    enriched = ExecutionBuildContext(
        recipe=context.recipe,
        components=context.components,
        policy=context.policy,
        extensions=(*context.extensions, extension),
    )
    return build_standard_strategy(enriched)


def build_joint_multistage_strategy(
    context: ExecutionBuildContext,
) -> JointMultiStageDiffusionRunner:
    """Build a generic joint-modality runner with declarative stage bindings."""

    bindings = context.recipe.execution.bindings
    required = {"denoiser", "conditioner", "latent_initializer", "decoder"}
    if not required.issubset(bindings):
        missing = sorted(required - set(bindings))
        raise ValueError(f"joint-multistage execution is missing bindings: {missing}")
    scheduler_roles = tuple(sorted(role for role in bindings if role.startswith("scheduler-")))
    if not scheduler_roles:
        raise ValueError("joint-multistage execution requires scheduler-* bindings")
    optional = {"processor"}
    unexpected = sorted(set(bindings) - required - optional - set(scheduler_roles))
    if unexpected:
        raise ValueError(f"joint-multistage execution has unsupported bindings: {unexpected}")
    stage_steps = tuple(int(value) for value in context.recipe.execution.options.get("stage_steps", ()))
    if len(stage_steps) != len(scheduler_roles):
        raise ValueError("joint-multistage stage_steps must match scheduler binding count")
    if len(stage_steps) > 1 and "processor" not in bindings:
        raise ValueError("multi-stage execution requires a processor binding")
    bound = {role: context.components[key] for role, key in bindings.items()}
    return JointMultiStageDiffusionRunner(
        model_id=context.recipe.model_id,
        components=MultiStageComponents(
            denoiser=bound["denoiser"],
            conditioner=bound["conditioner"],  # type: ignore[arg-type]
            latent_initializer=bound["latent_initializer"],  # type: ignore[arg-type]
            schedulers=tuple(bound[role] for role in scheduler_roles),  # type: ignore[arg-type]
            processor=bound.get("processor"),  # type: ignore[arg-type]
            decoder=bound["decoder"],  # type: ignore[arg-type]
        ),
        stage_steps=stage_steps,
        device=context.policy.device,
        dtype=context.policy.dtype,
    )


def build_autoregressive_window_strategy(
    context: ExecutionBuildContext,
) -> AutoregressiveWindowRunner:
    """Build the generic multi-view/block autoregressive diffusion lifecycle."""

    required = {
        "denoiser",
        "conditioner",
        "latent_initializer",
        "latent_encoder",
        "scheduler",
        "decoder",
    }
    bindings = context.recipe.execution.bindings
    if set(bindings) != required:
        raise ValueError(
            "autoregressive-window execution requires the canonical encoded-latent bindings: "
            f"required={sorted(required)}; got={sorted(bindings)}"
        )
    bound = {role: context.components[key] for role, key in bindings.items()}
    options = context.recipe.execution.options
    return AutoregressiveWindowRunner(
        model_id=context.recipe.model_id,
        components=RunnerComponents(
            denoiser=bound["denoiser"],  # type: ignore[arg-type]
            conditioner=bound["conditioner"],  # type: ignore[arg-type]
            latent_initializer=bound["latent_initializer"],  # type: ignore[arg-type]
            scheduler=bound["scheduler"],  # type: ignore[arg-type]
            decoder=bound["decoder"],  # type: ignore[arg-type]
            latent_encoder=bound["latent_encoder"],  # type: ignore[arg-type]
        ),
        device=context.policy.device,
        dtype=context.policy.dtype,
        extensions=context.extensions,
        prediction_mode=str(options.get("prediction_mode", "flow")),
        fixed_timesteps=tuple(int(value) for value in options.get("fixed_timesteps", ())),
        context_timestep=int(options.get("context_timestep", 0)),
    )


def build_chunked_kv_cache_strategy(
    context: ExecutionBuildContext,
) -> ChunkedKVCacheRunner:
    """Build generic temporal chunk traversal with persistent attention caches."""

    required = {
        "denoiser",
        "conditioner",
        "latent_initializer",
        "latent_encoder",
        "scheduler",
        "decoder",
    }
    bindings = context.recipe.execution.bindings
    if set(bindings) != required:
        raise ValueError(
            "chunked-kv-cache execution requires canonical encoded-latent bindings: "
            f"required={sorted(required)}; got={sorted(bindings)}"
        )
    bound = {role: context.components[key] for role, key in bindings.items()}
    options = context.recipe.execution.options
    return ChunkedKVCacheRunner(
        model_id=context.recipe.model_id,
        components=RunnerComponents(
            denoiser=bound["denoiser"],  # type: ignore[arg-type]
            conditioner=bound["conditioner"],  # type: ignore[arg-type]
            latent_initializer=bound["latent_initializer"],  # type: ignore[arg-type]
            scheduler=bound["scheduler"],  # type: ignore[arg-type]
            decoder=bound["decoder"],  # type: ignore[arg-type]
            latent_encoder=bound["latent_encoder"],  # type: ignore[arg-type]
        ),
        device=context.policy.device,
        dtype=context.policy.dtype,
        extensions=context.extensions,
        base_chunk_frames=int(options.get("base_chunk_frames", 3)),
        num_cached_chunks=int(options.get("num_cached_chunks", 2)),
        sink_token=bool(options.get("sink_token", True)),
    )


def default_execution_strategy_registry() -> ExecutionStrategyRegistry:
    """Create the framework's built-in strategy set without global mutation."""

    registry = ExecutionStrategyRegistry()
    registry.register("standard", build_standard_strategy)
    registry.register("dual-condition-guidance", build_dual_condition_guidance_strategy)
    registry.register("frozen-context", build_frozen_context_strategy)
    registry.register("masked-latent", build_masked_latent_strategy)
    registry.register("joint-multistage", build_joint_multistage_strategy)
    registry.register("autoregressive-window", build_autoregressive_window_strategy)
    registry.register("chunked-kv-cache", build_chunked_kv_cache_strategy)
    registry.freeze()
    return registry


def _strategy_key(value: str) -> str:
    key = str(value).strip().lower().replace("_", "-")
    if not key:
        raise ValueError("execution strategy cannot be empty")
    return key


__all__ = [
    "DiffusionExecutor",
    "ExecutionBuildContext",
    "ExecutionStrategyBuilder",
    "ExecutionStrategyRegistry",
    "UnsupportedExecutionStrategyError",
    "build_dual_condition_guidance_strategy",
    "build_autoregressive_window_strategy",
    "build_chunked_kv_cache_strategy",
    "build_standard_strategy",
    "build_frozen_context_strategy",
    "build_joint_multistage_strategy",
    "build_masked_latent_strategy",
    "default_execution_strategy_registry",
]
