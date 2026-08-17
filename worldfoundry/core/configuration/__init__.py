"""Reusable structured configuration primitives for model inference.

Several configuration systems coexist here (CF-11); pick by use case:

- ``lazy_config`` (``LazyCall``/``LazyConfig``/``instantiate``): detectron2
  lineage, ``_target_``-based deferred object graphs. Default choice for
  new inference configs.
- ``cosmos_config`` (``Config`` + ``make_freezable`` attrs classes): the
  structured top-level config consumed by Cosmos-family runtimes; composed
  and overridden through ``hydra.override``.
- ``flags``: process-wide boolean switches snapshotted from environment
  variables at import.
- ``model_config`` (``ModelConfig``/``DiTConfig``...): DiT architecture
  hyperparameter dataclasses. NOTE: this ``ModelConfig`` (architecture) is
  unrelated to ``worldfoundry.core.model_loading.config.ModelConfig``
  (checkpoint download/placement), which is what ``worldfoundry.core``
  re-exports as ``ModelConfig``.

Legacy instantiate/registry vocabularies live in
``worldfoundry.core.io.config_utils`` (``cls``/``class`` keys) and
``worldfoundry.core.model_loading.factory``; do not adopt them in new code.
"""

from .cosmos_config import CheckpointConfig, Config, EMAConfig, ObjectStoreConfig, make_freezable
from .flags import FLAGS, INTERNAL, VALIDATION, VERBOSE
from .lazy_config import LazyCall, LazyConfig, LazyDict, instantiate
from .model_config import (
    ArchConfig,
    DiTArchConfig,
    DiTConfig,
    ModelConfig,
    build_kwargs_from_config,
    require_config_value,
)

__all__ = [
    "Config",
    "CheckpointConfig",
    "ArchConfig",
    "DiTArchConfig",
    "DiTConfig",
    "EMAConfig",
    "FLAGS",
    "INTERNAL",
    "LazyCall",
    "LazyConfig",
    "LazyDict",
    "ObjectStoreConfig",
    "ModelConfig",
    "VALIDATION",
    "VERBOSE",
    "build_kwargs_from_config",
    "instantiate",
    "make_freezable",
    "require_config_value",
]
