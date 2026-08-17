"""Small structured configuration primitives shared by in-tree inference runtimes."""

from __future__ import annotations

from typing import Any, TypeVar

import attrs

from worldfoundry.core.configuration.lazy_config import LazyDict


@attrs.define(slots=False)
class _FallbackModelParallelConfig:
    """Fallback for configuration discovery without Megatron installed."""

    context_parallel_size: int = 1


def _default_model_parallel() -> Any:
    """Build the default ``model_parallel`` field value.

    Megatron is imported lazily here (not at module import) so that importing
    ``worldfoundry.core.configuration`` never pulls the megatron/torch stack;
    environments with megatron installed still get the real
    ``ModelParallelConfig`` the first time a ``Config`` is constructed.
    Without megatron the attrs fallback above only carries the single field
    the inference runtimes consume (``context_parallel_size``).
    """

    try:
        from megatron.core import ModelParallelConfig
    except ImportError:
        return _FallbackModelParallelConfig()
    return ModelParallelConfig()


T = TypeVar("T")


def _is_attrs_instance(value: object) -> bool:
    return attrs.has(type(value))


def make_freezable(cls: T) -> T:
    """Add a recursive runtime ``freeze`` operation to an attrs class.

    Freezing is *shallow* with respect to containers: it blocks attribute
    assignment on the instance (and recursively freezes nested attrs values
    that expose ``freeze``), but the contents of list/dict fields remain
    mutable. Decorating the same class twice is a no-op.
    """

    if not hasattr(cls, "__dict__"):
        raise TypeError("make_freezable requires attrs classes declared with slots=False")
    if cls.__dict__.get("_worldfoundry_freezable", False):
        return cls
    original_setattr = cls.__setattr__

    def setattr_override(self, key, value) -> None:  # noqa: ANN001
        if getattr(self, "_is_frozen", False) and key != "_is_frozen":
            raise AttributeError("Cannot modify frozen instance")
        original_setattr(self, key, value)

    def freeze(self) -> None:  # noqa: ANN001
        for value in attrs.asdict(self, recurse=False).values():
            if _is_attrs_instance(value) and hasattr(value, "freeze"):
                value.freeze()
        self._is_frozen = True

    cls.__setattr__ = setattr_override
    cls.freeze = freeze
    cls._worldfoundry_freezable = True
    return cls


@make_freezable
@attrs.define(slots=False)
class JobConfig:
    project: str = ""
    group: str = ""
    name: str = ""


@make_freezable
@attrs.define(slots=False)
class ObjectStoreConfig:
    """Object-store location used to read inference checkpoints."""

    enabled: bool = False
    credentials: str = ""
    bucket: str = ""


@make_freezable
@attrs.define(slots=False)
class CuDNNConfig:
    deterministic: bool = False
    benchmark: bool = True


@make_freezable
@attrs.define(slots=False)
class InferenceRuntimeConfig:
    cudnn: CuDNNConfig = attrs.field(factory=CuDNNConfig)


@make_freezable
@attrs.define(slots=False)
class CheckpointConfig:
    """Checkpoint source and strictness controls for inference construction."""

    load_path: str = ""
    load_from_object_store: ObjectStoreConfig = attrs.field(factory=ObjectStoreConfig)
    strict_resume: bool = True
    dcp_allow_mismatched_size: bool = False
    load_ema_to_reg: bool = False


@make_freezable
@attrs.define(slots=False)
class EMAConfig:
    """Exponential moving-average settings used while loading inference models."""

    enabled: bool = False
    rate: float = 0.1
    iteration_shift: int = 0


@make_freezable
@attrs.define(slots=False)
class Config:
    """Fields used to compose and instantiate a released inference model."""

    model: LazyDict | None
    job: JobConfig = attrs.field(factory=JobConfig)
    trainer: InferenceRuntimeConfig = attrs.field(factory=InferenceRuntimeConfig)
    # megatron's ModelParallelConfig when available, attrs fallback otherwise
    # (resolved lazily; see _default_model_parallel).
    model_parallel: Any = attrs.field(factory=_default_model_parallel)
    checkpoint: CheckpointConfig = attrs.field(factory=CheckpointConfig)

    def to_dict(self) -> dict[str, Any]:
        return attrs.asdict(self)

    def validate(self) -> None:
        """Validate the small set of fields needed during inference."""

        if self.model is None:
            raise ValueError("model configuration is required")


__all__ = [
    "CheckpointConfig",
    "Config",
    "CuDNNConfig",
    "EMAConfig",
    "InferenceRuntimeConfig",
    "JobConfig",
    "ObjectStoreConfig",
    "make_freezable",
]
