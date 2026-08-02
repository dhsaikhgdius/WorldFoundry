"""Strict semantic configuration schema for WorldFoundry-native training."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from worldfoundry.core.io.integrity import canonical_json, canonical_sha256

TRAINING_RECIPE_SCHEMA = "worldfoundry-training"
_DTYPES = frozenset({"bfloat16", "float16", "float32"})
NATIVE_EXECUTION_OWNER = "worldfoundry-native"


def _nonempty(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    return normalized


def _normalize_id(value: object, *, field_name: str) -> str:
    normalized = _nonempty(value, field_name=field_name).lower().replace("_", "-")
    if any(character.isspace() for character in normalized):
        raise ValueError(f"{field_name} cannot contain whitespace: {value!r}")
    return normalized


def _options(value: Mapping[str, object], *, field_name: str = "options") -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    normalized = {str(key): item for key, item in value.items()}
    if any(not key.strip() for key in normalized):
        raise ValueError(f"{field_name} keys cannot be empty")
    return MappingProxyType(normalized)


def _mapping(value: object, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _strict_section(
    payload: Mapping[str, Any],
    *,
    section: str,
    allowed: set[str],
    aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    normalized = dict(payload)
    for source, destination in (aliases or {}).items():
        if source not in normalized:
            continue
        if destination in normalized:
            raise ValueError(f"{section} cannot specify both {source!r} and {destination!r}")
        normalized[destination] = normalized.pop(source)
    unknown = sorted(set(normalized) - allowed)
    if unknown:
        raise ValueError(f"{section} contains unknown fields: {unknown}")
    return normalized


def _positive_int(value: object, *, field_name: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    resolved = int(value)
    minimum = 0 if allow_zero else 1
    if resolved < minimum:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{field_name} must be {relation}; got {resolved}")
    return resolved


def _strict_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool")
    return value


def _plain(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        result: dict[str, object] = {}
        for item in fields(value):
            key = "async" if item.name == "async_save" else item.name
            result[key] = _plain(getattr(value, item.name))
        return result
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class RunSpec:
    id: str
    output_dir: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _normalize_id(self.id, field_name="run.id"))
        object.__setattr__(self, "output_dir", _nonempty(self.output_dir, field_name="run.output_dir"))


@dataclass(frozen=True, slots=True)
class ModelSpec:
    recipe: str
    checkpoint: str = "default"
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "recipe", _normalize_id(self.recipe, field_name="model.recipe"))
        object.__setattr__(self, "checkpoint", _nonempty(self.checkpoint, field_name="model.checkpoint"))
        object.__setattr__(self, "options", _options(self.options, field_name="model.options"))


@dataclass(frozen=True, slots=True)
class TuningSpec:
    mode: str
    preset: str | None = None
    rank: int | None = None
    alpha: int | None = None
    dropout: float = 0.0
    modules_to_save: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        mode = _normalize_id(self.mode, field_name="tuning.mode")
        if mode not in {"full", "lora", "partial"}:
            raise ValueError(f"unsupported tuning.mode: {mode!r}")
        preset = None if self.preset is None else _normalize_id(self.preset, field_name="tuning.preset")
        rank = None if self.rank is None else _positive_int(self.rank, field_name="tuning.rank")
        alpha = None if self.alpha is None else _positive_int(self.alpha, field_name="tuning.alpha")
        dropout = float(self.dropout)
        if not 0.0 <= dropout < 1.0:
            raise ValueError("tuning.dropout must be in [0, 1)")
        if mode == "lora" and (preset is None or rank is None or alpha is None):
            raise ValueError("LoRA tuning requires preset, rank, and alpha")
        if mode != "lora" and any(value is not None for value in (rank, alpha)):
            raise ValueError(f"tuning.mode={mode!r} cannot set LoRA rank/alpha")
        modules = tuple(_nonempty(value, field_name="tuning.modules_to_save") for value in self.modules_to_save)
        if len(modules) != len(set(modules)):
            raise ValueError("tuning.modules_to_save cannot contain duplicates")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "preset", preset)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "dropout", dropout)
        object.__setattr__(self, "modules_to_save", modules)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    manifest: str
    cache: str | None = None
    max_latent_tokens_per_microbatch: int | None = None
    split: str = "train"
    shuffle: bool = True
    shuffle_seed: int = 42
    tail_policy: str = "drop"
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", _nonempty(self.manifest, field_name="data.manifest"))
        if self.cache is not None:
            object.__setattr__(self, "cache", _nonempty(self.cache, field_name="data.cache"))
        if self.max_latent_tokens_per_microbatch is not None:
            object.__setattr__(
                self,
                "max_latent_tokens_per_microbatch",
                _positive_int(
                    self.max_latent_tokens_per_microbatch,
                    field_name="data.max_latent_tokens_per_microbatch",
                ),
            )
        split = _normalize_id(self.split, field_name="data.split")
        tail_policy = _normalize_id(self.tail_policy, field_name="data.tail_policy")
        if tail_policy not in {"drop", "pad", "uneven"}:
            raise ValueError("data.tail_policy must be drop, pad, or uneven")
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "shuffle", _strict_bool(self.shuffle, field_name="data.shuffle"))
        object.__setattr__(self, "shuffle_seed", int(self.shuffle_seed))
        object.__setattr__(self, "tail_policy", tail_policy)
        object.__setattr__(self, "options", _options(self.options, field_name="data.options"))


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    type: str
    prediction_type: str = "flow_velocity"
    timestep_sampler: str = "logit_normal"
    conditioning_dropout: float = 0.0
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", _normalize_id(self.type, field_name="objective.type"))
        object.__setattr__(
            self,
            "prediction_type",
            _normalize_id(
                self.prediction_type,
                field_name="objective.prediction_type",
            ).replace("-", "_"),
        )
        object.__setattr__(
            self,
            "timestep_sampler",
            _normalize_id(self.timestep_sampler, field_name="objective.timestep_sampler"),
        )
        dropout = float(self.conditioning_dropout)
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("objective.conditioning_dropout must be in [0, 1]")
        object.__setattr__(self, "conditioning_dropout", dropout)
        object.__setattr__(self, "options", _options(self.options, field_name="objective.options"))


@dataclass(frozen=True, slots=True)
class OptimizerSpec:
    type: str
    learning_rate: float
    weight_decay: float = 0.0
    betas: tuple[float, ...] = (0.9, 0.999)
    epsilon: float | tuple[float, float] = 1.0e-8
    update_clip_threshold: float | None = None
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    def __post_init__(self) -> None:
        optimizer_type = _normalize_id(self.type, field_name="optimizer.type")
        if optimizer_type not in {"adamw", "came"}:
            raise ValueError(f"unsupported optimizer.type: {optimizer_type!r}")
        object.__setattr__(self, "type", optimizer_type)
        learning_rate = float(self.learning_rate)
        weight_decay = float(self.weight_decay)
        max_grad_norm = float(self.max_grad_norm)
        accumulation = _positive_int(
            self.gradient_accumulation_steps,
            field_name="optimizer.gradient_accumulation_steps",
        )
        betas = tuple(float(value) for value in self.betas)
        if learning_rate <= 0 or max_grad_norm <= 0:
            raise ValueError("optimizer learning_rate and max_grad_norm must be positive")
        if weight_decay < 0:
            raise ValueError("optimizer.weight_decay must be non-negative")
        if optimizer_type == "adamw":
            if len(betas) != 2 or any(not 0.0 <= value < 1.0 for value in betas):
                raise ValueError("AdamW optimizer.betas must contain two values in [0, 1)")
            if isinstance(self.epsilon, (tuple, list)):
                raise TypeError("AdamW optimizer.epsilon must be one number")
            epsilon: float | tuple[float, float] = float(self.epsilon)
            if epsilon <= 0:
                raise ValueError("AdamW optimizer.epsilon must be positive")
            if self.update_clip_threshold is not None:
                raise ValueError("optimizer.update_clip_threshold is only valid for CAME")
            update_clip_threshold = None
        else:
            if len(betas) != 3 or any(not 0.0 <= value < 1.0 for value in betas):
                raise ValueError("CAME optimizer.betas must contain three values in [0, 1)")
            if not isinstance(self.epsilon, (tuple, list)):
                raise TypeError("CAME optimizer.epsilon must contain square-gradient and instability epsilons")
            epsilon = tuple(float(value) for value in self.epsilon)
            if len(epsilon) != 2 or any(value <= 0 for value in epsilon):
                raise ValueError("CAME optimizer.epsilon must contain two positive values")
            update_clip_threshold = (
                1.0 if self.update_clip_threshold is None else float(self.update_clip_threshold)
            )
            if update_clip_threshold <= 0:
                raise ValueError("CAME optimizer.update_clip_threshold must be positive")
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "weight_decay", weight_decay)
        object.__setattr__(self, "betas", betas)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "update_clip_threshold", update_clip_threshold)
        object.__setattr__(self, "max_grad_norm", max_grad_norm)
        object.__setattr__(self, "gradient_accumulation_steps", accumulation)


@dataclass(frozen=True, slots=True)
class TrainingRuntimeSpec:
    param_dtype: str = "bfloat16"
    reduce_dtype: str = "float32"
    activation_checkpoint: str = "none"
    compile: bool = False

    def __post_init__(self) -> None:
        param_dtype = str(self.param_dtype).lower().removeprefix("torch.")
        reduce_dtype = str(self.reduce_dtype).lower().removeprefix("torch.")
        aliases = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
        param_dtype = aliases.get(param_dtype, param_dtype)
        reduce_dtype = aliases.get(reduce_dtype, reduce_dtype)
        if param_dtype not in _DTYPES or reduce_dtype not in _DTYPES:
            raise ValueError(f"unsupported runtime dtype pair: {param_dtype}/{reduce_dtype}")
        object.__setattr__(self, "param_dtype", param_dtype)
        object.__setattr__(self, "reduce_dtype", reduce_dtype)
        object.__setattr__(
            self,
            "activation_checkpoint",
            _normalize_id(self.activation_checkpoint, field_name="runtime.activation_checkpoint"),
        )
        object.__setattr__(self, "compile", _strict_bool(self.compile, field_name="runtime.compile"))


@dataclass(frozen=True, slots=True)
class DistributedSpec:
    backend: str = "single"
    dp_replicate: int = 1
    dp_shard: str | int = "auto"
    cp: int = 1
    tp: int = 1

    def __post_init__(self) -> None:
        backend = _normalize_id(self.backend, field_name="distributed.backend")
        if backend not in {"single", "ddp", "fsdp2"}:
            raise ValueError(f"unsupported distributed.backend: {backend!r}")
        dp_replicate = _positive_int(
            self.dp_replicate,
            field_name="distributed.dp_replicate",
        )
        dp_shard: str | int
        if isinstance(self.dp_shard, str):
            dp_shard = _normalize_id(self.dp_shard, field_name="distributed.dp_shard")
            if dp_shard != "auto":
                raise ValueError("distributed.dp_shard string value must be 'auto'")
        else:
            dp_shard = _positive_int(self.dp_shard, field_name="distributed.dp_shard")
        cp = _positive_int(self.cp, field_name="distributed.cp")
        tp = _positive_int(self.tp, field_name="distributed.tp")
        if backend == "single" and (dp_replicate != 1 or cp != 1 or tp != 1 or dp_shard != "auto"):
            raise ValueError("distributed.backend='single' requires dp_replicate=cp=tp=1 and dp_shard=auto")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "dp_replicate", dp_replicate)
        object.__setattr__(self, "dp_shard", dp_shard)
        object.__setattr__(self, "cp", cp)
        object.__setattr__(self, "tp", tp)


@dataclass(frozen=True, slots=True)
class CheckpointSpec:
    save_every_steps: int = 0
    async_save: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "save_every_steps",
            _positive_int(self.save_every_steps, field_name="checkpoint.save_every_steps", allow_zero=True),
        )
        object.__setattr__(
            self,
            "async_save",
            _strict_bool(self.async_save, field_name="checkpoint.async"),
        )
        if self.async_save and self.save_every_steps == 0:
            raise ValueError("checkpoint.async requires save_every_steps > 0")


@dataclass(frozen=True, slots=True)
class PostTrainingCheckpointSpec(CheckpointSpec):
    export_every_steps: int = 0

    def __post_init__(self) -> None:
        CheckpointSpec.__post_init__(self)
        object.__setattr__(
            self,
            "export_every_steps",
            _positive_int(
                self.export_every_steps,
                field_name="checkpoint.export_every_steps",
                allow_zero=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class ExportSpec:
    format: str = "peft"
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "format", _normalize_id(self.format, field_name="export.format"))
        object.__setattr__(self, "options", _options(self.options, field_name="export.options"))


@dataclass(frozen=True, slots=True)
class TrainingRecipe:
    """Canonical run request owned and executed only by WorldFoundry."""

    run: RunSpec
    model: ModelSpec
    tuning: TuningSpec
    data: DatasetSpec
    objective: ObjectiveSpec
    optimizer: OptimizerSpec
    execution_owner: str = NATIVE_EXECUTION_OWNER
    runtime: TrainingRuntimeSpec = TrainingRuntimeSpec()
    distributed: DistributedSpec = DistributedSpec()
    checkpoint: CheckpointSpec = CheckpointSpec()
    schema: str = TRAINING_RECIPE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRAINING_RECIPE_SCHEMA:
            raise ValueError(f"unsupported training recipe schema {self.schema!r}; expected {TRAINING_RECIPE_SCHEMA!r}")
        owner = _normalize_id(self.execution_owner, field_name="execution_owner")
        if owner != NATIVE_EXECUTION_OWNER:
            raise ValueError(
                f"execution_owner must be {NATIVE_EXECUTION_OWNER!r}; external training loops are unsupported"
            )
        object.__setattr__(self, "execution_owner", owner)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TrainingRecipe":
        root = _strict_section(
            _mapping(value, field_name="training recipe"),
            section="training recipe",
            allowed={
                "schema",
                "run",
                "execution_owner",
                "model",
                "tuning",
                "data",
                "objective",
                "optimizer",
                "runtime",
                "distributed",
                "checkpoint",
            },
        )
        required = {"run", "model", "tuning", "data", "objective", "optimizer"}
        missing = sorted(required - set(root))
        if missing:
            raise ValueError(f"training recipe is missing required sections: {missing}")

        def section(name: str, allowed: set[str], *, aliases: Mapping[str, str] | None = None) -> dict[str, Any]:
            return _strict_section(
                _mapping(root.get(name, {}), field_name=name),
                section=name,
                allowed=allowed,
                aliases=aliases,
            )

        return cls(
            schema=str(root.get("schema", TRAINING_RECIPE_SCHEMA)),
            run=RunSpec(**section("run", {"id", "output_dir"})),
            execution_owner=str(root.get("execution_owner", NATIVE_EXECUTION_OWNER)),
            model=ModelSpec(**section("model", {"recipe", "checkpoint", "options"})),
            tuning=TuningSpec(**section("tuning", {"mode", "preset", "rank", "alpha", "dropout", "modules_to_save"})),
            data=DatasetSpec(
                **section(
                    "data",
                    {
                        "manifest",
                        "cache",
                        "max_latent_tokens_per_microbatch",
                        "split",
                        "shuffle",
                        "shuffle_seed",
                        "tail_policy",
                        "options",
                    },
                )
            ),
            objective=ObjectiveSpec(
                **section(
                    "objective",
                    {"type", "prediction_type", "timestep_sampler", "conditioning_dropout", "options"},
                )
            ),
            optimizer=OptimizerSpec(
                **section(
                    "optimizer",
                    {
                        "type",
                        "learning_rate",
                        "weight_decay",
                        "betas",
                        "epsilon",
                        "update_clip_threshold",
                        "max_grad_norm",
                        "gradient_accumulation_steps",
                    },
                )
            ),
            runtime=TrainingRuntimeSpec(
                **section("runtime", {"param_dtype", "reduce_dtype", "activation_checkpoint", "compile"})
            ),
            distributed=DistributedSpec(
                **section(
                    "distributed",
                    {"backend", "dp_replicate", "dp_shard", "cp", "tp"},
                )
            ),
            checkpoint=CheckpointSpec(
                **section(
                    "checkpoint",
                    {"save_every_steps", "async_save"},
                    aliases={"async": "async_save"},
                )
            ),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "TrainingRecipe":
        source = Path(path)
        if source.suffix.lower() == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
        elif source.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ModuleNotFoundError as error:
                raise RuntimeError("loading YAML training recipes requires the base 'pyyaml' dependency") from error
            payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        else:
            raise ValueError(f"training recipe must be .json, .yaml, or .yml: {source}")
        return cls.from_mapping(_mapping(payload, field_name=str(source)))

    def to_dict(self) -> dict[str, object]:
        return _plain(self)  # type: ignore[return-value]

    def canonical_json(self) -> str:
        try:
            return canonical_json(self.to_dict())
        except (TypeError, ValueError) as error:
            raise TypeError("training recipe values must be JSON serializable") from error

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


__all__ = [
    "CheckpointSpec",
    "DatasetSpec",
    "DistributedSpec",
    "ExportSpec",
    "ModelSpec",
    "ObjectiveSpec",
    "OptimizerSpec",
    "PostTrainingCheckpointSpec",
    "NATIVE_EXECUTION_OWNER",
    "RunSpec",
    "TRAINING_RECIPE_SCHEMA",
    "TrainingRecipe",
    "TrainingRuntimeSpec",
    "TuningSpec",
]
