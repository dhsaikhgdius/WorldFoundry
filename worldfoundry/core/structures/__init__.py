"""Lazy data-structure utility exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_SUBMODULES = {
    "shape_utils": "worldfoundry.core.structures.shape_utils",
    "tree_utils": "worldfoundry.core.structures.tree_utils",
}

_EXPORT_MODULES = {
    "broadcast_structures": "worldfoundry.core.structures.tree_utils",
    "check_shape": "worldfoundry.core.structures.shape_utils",
    "copy_non_leaf": "worldfoundry.core.structures.tree_utils",
    "fast_map_structure": "worldfoundry.core.structures.tree_utils",
    "is_mapping": "worldfoundry.core.structures.predicates",
    "is_sequence": "worldfoundry.core.structures.predicates",
    "Bool": "worldfoundry.core.structures.validator",
    "Float": "worldfoundry.core.structures.validator",
    "Int": "worldfoundry.core.structures.validator",
    "JsonDict": "worldfoundry.core.structures.validator",
    "OneOf": "worldfoundry.core.structures.validator",
    "String": "worldfoundry.core.structures.validator",
    "Validator": "worldfoundry.core.structures.validator",
    "shape_avgpool1d": "worldfoundry.core.structures.shape_utils",
    "shape_avgpool2d": "worldfoundry.core.structures.shape_utils",
    "shape_avgpool3d": "worldfoundry.core.structures.shape_utils",
    "shape_conv1d": "worldfoundry.core.structures.shape_utils",
    "shape_conv2d": "worldfoundry.core.structures.shape_utils",
    "shape_conv3d": "worldfoundry.core.structures.shape_utils",
    "shape_convnd": "worldfoundry.core.structures.shape_utils",
    "shape_maxpool1d": "worldfoundry.core.structures.shape_utils",
    "shape_maxpool2d": "worldfoundry.core.structures.shape_utils",
    "shape_maxpool3d": "worldfoundry.core.structures.shape_utils",
    "shape_poolnd": "worldfoundry.core.structures.shape_utils",
    "shape_slice": "worldfoundry.core.structures.shape_utils",
    "shape_transpose_conv1d": "worldfoundry.core.structures.shape_utils",
    "shape_transpose_conv2d": "worldfoundry.core.structures.shape_utils",
    "shape_transpose_conv3d": "worldfoundry.core.structures.shape_utils",
    "shape_transpose_convnd": "worldfoundry.core.structures.shape_utils",
    "stack_sequence_fields": "worldfoundry.core.structures.tree_utils",
    "tree_assign_at_path": "worldfoundry.core.structures.tree_utils",
    "tree_value_at_path": "worldfoundry.core.structures.tree_utils",
    "unstack_sequence_fields": "worldfoundry.core.structures.tree_utils",
}


def __getattr__(name: str) -> Any:
    module_name = _SUBMODULES.get(name) or _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = module if name in _SUBMODULES else getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


__all__ = sorted({*_SUBMODULES, *_EXPORT_MODULES})
