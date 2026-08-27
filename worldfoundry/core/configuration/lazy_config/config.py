# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Loading and serialization for inference-oriented lazy configurations.

Trust boundary: LazyConfig is code. Loading a ``.py`` config executes the
file (detectron2-style LazyConfig, by design), including its relative
config imports -- so config files must come from trusted sources and be
treated exactly like source code under review. YAML configs are parsed
with ``yaml.safe_load`` and cannot instantiate arbitrary Python objects.
"""

from __future__ import annotations

import ast
import builtins
import importlib.machinery
import importlib.util
import inspect
import threading
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict as dataclass_asdict
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

import attrs
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

from .lazy_call import get_default_params


def _cast_to_config(value: Any) -> Any:
    return DictConfig(value, flags={"allow_objects": True}) if isinstance(value, dict) else value


def _validate_python(path: Path) -> None:
    try:
        ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        raise SyntaxError(f"Config file {path} has invalid Python syntax.") from exc


_CONFIG_PACKAGE_PREFIX = "worldfoundry._lazy_config_"


def _module_name(path: Path) -> str:
    return f"{_CONFIG_PACKAGE_PREFIX}{path.stem}_{uuid.uuid4().hex[:8]}"


# Serializes _patch_relative_imports so concurrent/nested config loads cannot
# interleave the save/restore of builtins.__import__ (which would leave the
# patched hook installed permanently). Loads themselves become sequential;
# imports from other threads during a load still pass through patched_import,
# which delegates every non-config import to the original untouched.
_IMPORT_PATCH_LOCK = threading.RLock()


@contextmanager
def _patch_relative_imports():
    original_import = builtins.__import__

    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        package = "" if globals is None else globals.get("__package__", "") or ""
        if level and package.startswith(_CONFIG_PACKAGE_PREFIX):
            if not name:
                raise ImportError("Relative config imports must name a Python config file.")
            source = Path(globals["__file__"])
            target = source.parent
            for _ in range(level - 1):
                target = target.parent
            target = target.joinpath(*name.split(".")).with_suffix(".py")
            if not target.is_file():
                raise ImportError(f"Relative config import does not exist: {target}")
            _validate_python(target)
            spec = importlib.machinery.ModuleSpec(_module_name(target), loader=None, origin=str(target))
            module = importlib.util.module_from_spec(spec)
            module.__file__ = str(target)
            module.__package__ = spec.name
            exec(compile(target.read_text(encoding="utf-8"), str(target), "exec"), module.__dict__)
            for imported_name in fromlist:
                module.__dict__[imported_name] = _cast_to_config(module.__dict__[imported_name])
            return module
        return original_import(name, globals, locals, fromlist=fromlist, level=level)

    with _IMPORT_PATCH_LOCK:
        builtins.__import__ = patched_import
        try:
            yield
        finally:
            builtins.__import__ = original_import


def _plain_config(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True, enum_to_str=False)
    if attrs.has(type(value)):
        return attrs.asdict(value, recurse=True)
    if is_dataclass(value) and not isinstance(value, type):
        return dataclass_asdict(value)
    if isinstance(value, Mapping):
        return {key: _plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_config(item) for item in value]
    try:
        yaml.safe_dump(value)
    except Exception:
        return str(value)
    return value


def _sort_recursive(value: Any) -> Any:
    if isinstance(value, dict):
        return OrderedDict((key, _sort_recursive(item)) for key, item in sorted(value.items()))
    if isinstance(value, list):
        return [_sort_recursive(item) for item in value]
    return value


class LazyConfig:
    """Load local Python/YAML lazy configs and save resolved inference configs.

    Trust boundary: ``.py`` configs are executed (detectron2 LazyConfig
    design) -- only load Python configs you would run as code. YAML configs
    are parsed with ``yaml.safe_load`` and cannot instantiate arbitrary
    Python objects; ``save_yaml`` only ever emits plain YAML, so framework
    round-trips are unaffected. Configs needing Python objects belong in
    ``.py`` files.
    """

    @staticmethod
    def load_rel(filename: str, keys: str | tuple[str, ...] | None = None):
        caller = Path(inspect.stack()[1].filename)
        if str(caller) == "<string>":
            raise RuntimeError("LazyConfig.load_rel cannot resolve a caller for <string>.")
        return LazyConfig.load(str(caller.parent / filename), keys)

    @staticmethod
    def load(filename: str, keys: str | tuple[str, ...] | None = None):
        path = Path(filename).expanduser().resolve()
        if path.suffix not in {".py", ".yaml", ".yml"}:
            raise ValueError(f"Config file must be Python or YAML: {path}")

        if path.suffix == ".py":
            _validate_python(path)
            namespace = {"__file__": str(path), "__package__": _module_name(path)}
            with _patch_relative_imports():
                exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
            result: Any = namespace
        else:
            # safe_load: YAML is data, not code. ``save_yaml`` emits plain
            # YAML only, and no repository config uses python object tags;
            # anything needing Python objects must be a ``.py`` config.
            result = OmegaConf.create(yaml.safe_load(path.read_text(encoding="utf-8")), flags={"allow_objects": True})

        if keys is not None:
            if isinstance(keys, str):
                return _cast_to_config(result[keys])
            return tuple(_cast_to_config(result[key]) for key in keys)
        if path.suffix == ".py":
            result = DictConfig(
                {
                    name: _cast_to_config(value)
                    for name, value in result.items()
                    if not name.startswith("_") and isinstance(value, (dict, DictConfig, ListConfig))
                },
                flags={"allow_objects": True},
            )
        return result

    @staticmethod
    def save_yaml(config: Any, filename: str | Path) -> str:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            config = deepcopy(config)
        except Exception:
            pass
        value = _sort_recursive(_plain_config(config))
        yaml.add_representer(
            OrderedDict,
            lambda dumper, data: dumper.represent_mapping("tag:yaml.org,2002:map", data.items()),
        )
        path.write_text(yaml.dump(value, default_flow_style=False), encoding="utf-8")
        return str(path)


__all__ = ["LazyConfig", "get_default_params"]
