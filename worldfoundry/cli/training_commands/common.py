"""Small, accelerator-free helpers shared by training CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from worldfoundry.training.recipes import PostTrainingRecipe, TrainingRecipe


def training_family(model_recipe: str) -> str:
    model_id = str(model_recipe).strip().lower().replace("_", "-")
    if model_id.startswith("qwen3-"):
        return "qwen3"
    if model_id.startswith("sana-"):
        return "sana"
    if model_id == "wan2.1-t2v-1.3b":
        return "wan"
    if model_id == "wan2.2-t2v-a14b":
        return "wan22"
    if model_id in {"ltx-video-i2v", "ltx-2-i2v", "ltx-2.3-i2v"}:
        return "ltx"
    if model_id in {"hunyuanvideo-t2v", "hunyuanvideo-1.5-t2v"}:
        return "hunyuan-video"
    if model_id == "lvdm-short-unconditional":
        return "lvdm"
    if model_id in {"dynamicrafter-512-i2v", "dynamicrafter-1024-i2v"}:
        return "dynamicrafter"
    if model_id == "t2v-turbo":
        return "t2v-turbo"
    if model_id in {
        "cosmos-predict2-2b-video2world",
        "cosmos-predict2-14b-video2world",
        "cosmos-predict2.5-2b",
        "cosmos-predict2.5-14b",
        "cosmos3-nano",
        "cosmos3-super",
    }:
        return "cosmos"
    raise ValueError(f"native training does not support model recipe {model_recipe!r}")


def training_checkpoint_key(model_recipe: str) -> str:
    """Return the denoiser asset key owned by one native model recipe."""

    family = training_family(model_recipe)
    if family in {"sana", "wan"}:
        return "dit"
    if family == "ltx":
        return "model"
    if family == "lvdm":
        return "denoiser"
    if family == "dynamicrafter":
        return "model"
    if family == "t2v-turbo":
        return "student"
    return "transformer"


def training_base_dir(value: Path | None) -> Path:
    """Resolve recipe paths against an explicit root or the launch directory."""

    if value is None:
        return Path.cwd().resolve()
    return value.expanduser().resolve()


def checkpoint_overrides(
    declarations: list[str] | None,
    *,
    base_dir: Path,
) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for declaration in declarations or ():
        key, separator, raw_path = str(declaration).partition("=")
        key = key.strip()
        raw_path = raw_path.strip()
        if not separator or not key or not raw_path:
            raise ValueError("checkpoint overrides must use NAME=PATH")
        if key in overrides:
            raise ValueError(f"duplicate checkpoint override: {key!r}")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        overrides[key] = str(path.resolve())
    return overrides


def load_cache_recipe(path: Path) -> TrainingRecipe | PostTrainingRecipe:
    """Load either native pre-training/SFT or post-training cache contracts."""

    import yaml

    from worldfoundry.training.recipes import (
        POST_TRAINING_RECIPE_SCHEMA,
        TRAINING_RECIPE_SCHEMA,
        PostTrainingRecipe,
        TrainingRecipe,
    )

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"training recipe must be .json, .yaml, or .yml: {path}")
    if not isinstance(payload, dict):
        raise TypeError(f"training recipe must contain a mapping: {path}")
    schema = payload.get("schema")
    if schema == TRAINING_RECIPE_SCHEMA:
        return TrainingRecipe.from_mapping(payload)
    if schema == POST_TRAINING_RECIPE_SCHEMA:
        return PostTrainingRecipe.from_mapping(payload)
    raise ValueError(f"unsupported native training recipe schema: {schema!r}")


__all__ = [
    "checkpoint_overrides",
    "load_cache_recipe",
    "training_base_dir",
    "training_checkpoint_key",
    "training_family",
]
