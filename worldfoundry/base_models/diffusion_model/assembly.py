"""Framework-owned construction of components and execution strategies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from .components import BuildPurpose, ComponentBuildContext, ComponentKey, ComponentKind
from .contracts import (
    ConditionEncoder,
    Denoiser,
    DiffusionScheduler,
    LatentDecoder,
    LatentEncoder,
    LatentInitializer,
    LatentProcessor,
)
from .extensions import DiffusionExtension
from .loaders import CheckpointSpec
from .optimizations import RuntimePolicy
from .recipes.spec import NativeDiffusionRecipe
from .runners import (
    DiffusionExecutor,
    ExecutionBuildContext,
    ExecutionStrategyRegistry,
    UnsupportedExecutionStrategyError,
    default_execution_strategy_registry,
)


class NativeDiffusionAssembler:
    """Build model recipes without permitting model-owned runtime factories."""

    _EXPECTED_PROTOCOLS = {
        ComponentKind.DENOISER: Denoiser,
        ComponentKind.CONDITIONER: ConditionEncoder,
        ComponentKind.LATENT_ENCODER: LatentEncoder,
        ComponentKind.LATENT_INITIALIZER: LatentInitializer,
        ComponentKind.LATENT_PROCESSOR: LatentProcessor,
        ComponentKind.SCHEDULER: DiffusionScheduler,
        ComponentKind.DECODER: LatentDecoder,
    }

    def __init__(
        self,
        strategies: ExecutionStrategyRegistry | None = None,
    ) -> None:
        self.strategies = strategies or default_execution_strategy_registry()

    def build(
        self,
        recipe: NativeDiffusionRecipe,
        *,
        policy: RuntimePolicy | None = None,
        checkpoint_overrides: Mapping[str, CheckpointSpec | str] | None = None,
        component_options: Mapping[str | ComponentKey, Mapping[str, object]] | None = None,
        extensions: Iterable[DiffusionExtension] = (),
    ) -> DiffusionExecutor:
        components = self.build_components(
            recipe,
            purpose=BuildPurpose.INFERENCE,
            policy=policy,
            checkpoint_overrides=checkpoint_overrides,
            component_options=component_options,
        )
        runtime_policy = policy or RuntimePolicy()

        return self.strategies.build(
            recipe.execution.strategy,
            ExecutionBuildContext(
                recipe=recipe,
                components=components,
                policy=runtime_policy,
                extensions=tuple(extensions),
            ),
        )

    def build_components(
        self,
        recipe: NativeDiffusionRecipe,
        *,
        purpose: BuildPurpose,
        policy: RuntimePolicy | None = None,
        checkpoint_overrides: Mapping[str, CheckpointSpec | str] | None = None,
        component_options: Mapping[str | ComponentKey, Mapping[str, object]] | None = None,
        component_keys: Iterable[ComponentKey] | None = None,
    ) -> dict[ComponentKey, object]:
        """Construct a recipe's components without creating an inference runner.

        Training code must call this seam with ``BuildPurpose.TRAINING``.  The
        ordinary :meth:`build` path remains inference-only and backwards
        compatible.
        """

        resolved_purpose = BuildPurpose(purpose)
        runtime_policy = policy or RuntimePolicy()
        checkpoints = self.resolve_checkpoints(recipe, checkpoint_overrides or {})
        options = self._component_options(recipe, component_options or {})

        specs = recipe.components
        if component_keys is not None:
            requested = tuple(component_keys)
            if not requested:
                raise ValueError("component_keys cannot be empty")
            if not all(isinstance(key, ComponentKey) for key in requested):
                raise TypeError("component_keys must contain only ComponentKey values")
            if len(requested) != len(set(requested)):
                raise ValueError("component_keys cannot contain duplicates")
            available = {spec.key for spec in recipe.components}
            unknown = sorted(str(key) for key in set(requested) - available)
            if unknown:
                raise KeyError(f"unknown component keys for {recipe.model_id}: {unknown}")
            selected = set(requested)
            specs = tuple(spec for spec in recipe.components if spec.key in selected)

        components: dict[ComponentKey, object] = {}
        for spec in specs:
            component_checkpoints = {
                name: checkpoints[checkpoint_key] for name, checkpoint_key in spec.checkpoints.items()
            }
            context = ComponentBuildContext(
                model_id=recipe.model_id,
                key=spec.key,
                policy=runtime_policy,
                purpose=resolved_purpose,
                checkpoints=component_checkpoints,
                recipe_options=recipe.options,
                component_options={**spec.options, **options.get(spec.key, {})},
            )
            component = spec.factory(context)
            expected_protocol = self._EXPECTED_PROTOCOLS[spec.key.kind]
            if not isinstance(component, expected_protocol):
                raise TypeError(
                    f"factory for {spec.key} returned {type(component).__name__}; expected {expected_protocol.__name__}"
                )
            components[spec.key] = component
        return components

    @staticmethod
    def resolve_checkpoints(
        recipe: NativeDiffusionRecipe,
        overrides: Mapping[str, CheckpointSpec | str],
    ) -> dict[str, CheckpointSpec]:
        """Resolve audited recipe checkpoint overrides for training or inference."""

        return NativeDiffusionAssembler._checkpoints(recipe, overrides)

    @staticmethod
    def _checkpoints(
        recipe: NativeDiffusionRecipe,
        overrides: Mapping[str, CheckpointSpec | str],
    ) -> dict[str, CheckpointSpec]:
        unknown = sorted(set(overrides) - set(recipe.checkpoints))
        if unknown:
            raise KeyError(f"unknown checkpoint override keys for {recipe.model_id}: {unknown}")
        checkpoints = dict(recipe.checkpoints)
        for key, value in overrides.items():
            if isinstance(value, CheckpointSpec):
                checkpoints[key] = value
                continue
            resolved = Path(value).expanduser()
            if resolved.is_dir():
                default = checkpoints[key]
                checkpoints[key] = CheckpointSpec(
                    source=value,
                    files=default.files,
                    allow_patterns=default.allow_patterns,
                    metadata=default.metadata,
                    file_sha256=default.file_sha256,
                    file_size_bytes=default.file_size_bytes,
                    resource_sha256=default.resource_sha256,
                    resource_size_bytes=default.resource_size_bytes,
                )
            else:
                default = checkpoints[key]
                if len(default.files) == 1 and (
                    default.file_sha256 or default.file_size_bytes
                ):
                    original_name = default.files[0]
                    local_name = resolved.name
                    checkpoints[key] = CheckpointSpec(
                        source=resolved.parent,
                        files=(local_name,),
                        metadata=default.metadata,
                        file_sha256=(
                            {local_name: default.file_sha256[original_name]}
                            if original_name in default.file_sha256
                            else {}
                        ),
                        file_size_bytes=(
                            {local_name: default.file_size_bytes[original_name]}
                            if original_name in default.file_size_bytes
                            else {}
                        ),
                        resource_sha256=default.resource_sha256,
                        resource_size_bytes=default.resource_size_bytes,
                    )
                else:
                    checkpoints[key] = CheckpointSpec(value)
        return checkpoints

    @staticmethod
    def _component_options(
        recipe: NativeDiffusionRecipe,
        values: Mapping[str | ComponentKey, Mapping[str, object]],
    ) -> dict[ComponentKey, Mapping[str, object]]:
        recipe_keys = {component.key for component in recipe.components}
        by_string = {str(key): key for key in recipe_keys}
        normalized: dict[ComponentKey, Mapping[str, object]] = {}
        for raw_key, options in values.items():
            key = raw_key if isinstance(raw_key, ComponentKey) else by_string.get(raw_key)
            if key is None or key not in recipe_keys:
                raise KeyError(f"unknown component options key for {recipe.model_id}: {raw_key}")
            normalized[key] = dict(options)
        return normalized


__all__ = ["NativeDiffusionAssembler", "UnsupportedExecutionStrategyError"]
