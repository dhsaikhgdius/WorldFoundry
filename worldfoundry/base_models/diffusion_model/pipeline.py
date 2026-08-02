"""Small public construction surface for canonical diffusion runners."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .components import ComponentKey
from .contracts import DiffusionOutput, DiffusionRequest
from .extensions import DiffusionExtension
from .loaders import CheckpointSpec
from .optimizations import RuntimePolicy
from .registry import NativeDiffusionRegistry
from .runners import DiffusionExecutor


class NativeDiffusionPipeline:
    """Thin request boundary around one fully constructed native runner."""

    def __init__(self, runner: DiffusionExecutor) -> None:
        self.runner = runner

    @property
    def model_id(self) -> str:
        return self.runner.model_id

    @property
    def components(self):
        """Expose the assembled role bundle for higher-level orchestration."""

        try:
            return self.runner.components
        except AttributeError as error:
            raise AttributeError("this diffusion execution strategy does not expose a component bundle") from error

    @property
    def device(self):
        return self.runner.device

    @property
    def dtype(self):
        return self.runner.dtype

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        registry: NativeDiffusionRegistry | None = None,
        policy: RuntimePolicy | None = None,
        checkpoint_overrides: Mapping[str, CheckpointSpec | str] | None = None,
        component_options: Mapping[str | ComponentKey, Mapping[str, object]] | None = None,
        extensions: Iterable[DiffusionExtension] = (),
    ) -> "NativeDiffusionPipeline":
        """Construct a built-in model ID through the canonical assembler."""

        if registry is None:
            from .registry import default_native_diffusion_registry

            registry = default_native_diffusion_registry()
        return cls.from_registry(
            registry,
            model_id,
            policy=policy,
            checkpoint_overrides=checkpoint_overrides,
            component_options=component_options,
            extensions=extensions,
        )

    @classmethod
    def from_registry(
        cls,
        registry: NativeDiffusionRegistry,
        model_id: str,
        *,
        policy: RuntimePolicy | None = None,
        checkpoint_overrides: Mapping[str, CheckpointSpec | str] | None = None,
        component_options: Mapping[str | ComponentKey, Mapping[str, object]] | None = None,
        extensions: Iterable[DiffusionExtension] = (),
    ) -> "NativeDiffusionPipeline":
        return cls(
            registry.build_runner(
                model_id,
                policy=policy,
                checkpoint_overrides=checkpoint_overrides,
                component_options=component_options,
                extensions=extensions,
            )
        )

    def __call__(self, request: DiffusionRequest) -> DiffusionOutput:
        if not isinstance(request, DiffusionRequest):
            raise TypeError(f"native diffusion pipeline expects DiffusionRequest, got {type(request).__name__}")
        return self.runner.run(request)


__all__ = ["NativeDiffusionPipeline"]
