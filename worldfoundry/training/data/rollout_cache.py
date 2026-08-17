"""Cached prompt conditioning for native policy rollouts."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import overload

import torch

from worldfoundry.core.io.integrity import sync_directory, write_exclusive_json

from .rollout_manifest import RolloutPromptDataset, RolloutPromptRecord
from .shared_conditioning import SharedConditioningArtifact, SharedConditioningSample, SharedConditioningStore

ROLLOUT_CONDITIONING_CACHE_SCHEMA = "worldfoundry-rollout-conditioning-cache"
ROLLOUT_CONDITIONING_INDEX_SCHEMA = "worldfoundry-rollout-conditioning-index"
_INDEX_NAME = "rollout-conditioning-index.json"


def _json_mapping(value: Mapping[str, object], *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    try:
        payload = json.loads(json.dumps(dict(value), sort_keys=True))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} must contain JSON values") from error
    return MappingProxyType(payload)


def resolve_rollout_generation_geometry(
    record: RolloutPromptRecord,
    defaults: Mapping[str, object] | None,
) -> tuple[int, int, int]:
    """Merge recipe defaults with one prompt record and validate H/W/T."""

    if not isinstance(record, RolloutPromptRecord):
        raise TypeError("record must be RolloutPromptRecord")
    values = dict(defaults or {})
    values.update(record.generation)
    expected = {"height", "width", "num_frames"}
    if set(values) != expected:
        raise ValueError(f"rollout generation geometry requires {sorted(expected)}")
    resolved = tuple(int(values[name]) for name in ("height", "width", "num_frames"))
    if any(value <= 0 for value in resolved):
        raise ValueError("rollout generation geometry must be positive")
    return resolved


@dataclass(frozen=True, slots=True)
class RolloutConditioningEntry:
    record: RolloutPromptRecord
    branch: str
    artifact: SharedConditioningArtifact

    def __post_init__(self) -> None:
        if not isinstance(self.record, RolloutPromptRecord):
            raise TypeError("record must be RolloutPromptRecord")
        branch = str(self.branch).strip().lower()
        if not branch:
            raise ValueError("rollout conditioning branch cannot be empty")
        if not isinstance(self.artifact, SharedConditioningArtifact):
            raise TypeError("artifact must be SharedConditioningArtifact")
        if self.artifact.identity.branch != branch or self.artifact.identity.prompt != self.record.prompt:
            raise ValueError("rollout conditioning artifact identity differs from its prompt record")
        object.__setattr__(self, "branch", branch)

    @property
    def prompt_id(self) -> str:
        return self.record.prompt_id

    def to_dict(self) -> dict[str, object]:
        return {"record": self.record.to_dict(), "branch": self.branch, "artifact": self.artifact.to_dict()}

    @classmethod
    def from_mapping(cls, value: object) -> RolloutConditioningEntry:
        if not isinstance(value, Mapping) or set(value) != {"record", "branch", "artifact"}:
            raise ValueError("rollout conditioning entry fields differ from the active schema")
        return cls(
            record=RolloutPromptRecord.from_mapping(value["record"]),
            branch=str(value["branch"]),
            artifact=SharedConditioningArtifact.from_mapping(value["artifact"]),
        )


@dataclass(frozen=True, slots=True)
class RolloutConditioningIndex:
    model_recipe: str
    conditioner: Mapping[str, object]
    tokenizer: Mapping[str, object]
    entries: tuple[RolloutConditioningEntry, ...]
    schema: str = ROLLOUT_CONDITIONING_INDEX_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ROLLOUT_CONDITIONING_INDEX_SCHEMA:
            raise ValueError(f"unsupported rollout conditioning index: {self.schema!r}")
        recipe = str(self.model_recipe).strip().lower().replace("_", "-")
        if not recipe:
            raise ValueError("model_recipe cannot be empty")
        entries = tuple(self.entries)
        if not entries or not all(isinstance(item, RolloutConditioningEntry) for item in entries):
            raise ValueError("rollout conditioning index requires entries")
        if len({entry.prompt_id for entry in entries}) != len(entries):
            raise ValueError("rollout conditioning prompt IDs must be unique")
        conditioner = _json_mapping(self.conditioner, field_name="conditioner")
        tokenizer = _json_mapping(self.tokenizer, field_name="tokenizer")
        for entry in entries:
            identity = entry.artifact.identity
            if (
                identity.model_recipe != recipe
                or identity.conditioner != conditioner
                or identity.tokenizer != tokenizer
            ):
                raise ValueError("rollout conditioning entry uses another model or encoder configuration")
        object.__setattr__(self, "model_recipe", recipe)
        object.__setattr__(self, "conditioner", conditioner)
        object.__setattr__(self, "tokenizer", tokenizer)
        object.__setattr__(self, "entries", entries)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "model_recipe": self.model_recipe,
            "conditioner": dict(self.conditioner),
            "tokenizer": dict(self.tokenizer),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_mapping(cls, value: object) -> RolloutConditioningIndex:
        if not isinstance(value, Mapping):
            raise TypeError("rollout conditioning index must be a mapping")
        payload = {str(key): item for key, item in value.items()}
        expected = {"schema", "model_recipe", "conditioner", "tokenizer", "entries"}
        if set(payload) != expected:
            raise ValueError("rollout conditioning index fields differ from the active schema")
        entries = payload.pop("entries")
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)):
            raise TypeError("rollout conditioning entries must be a sequence")
        return cls(**payload, entries=tuple(RolloutConditioningEntry.from_mapping(item) for item in entries))


class RolloutConditioningStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.shared = SharedConditioningStore(self.root)
        self.index_path = self.root / _INDEX_NAME

    def write_index(self, index: RolloutConditioningIndex) -> RolloutConditioningIndex:
        if self.index_path.exists():
            existing = self.read_index()
            if existing != index:
                raise FileExistsError(f"rollout conditioning index already stores different entries: {self.index_path}")
            return existing
        write_exclusive_json(
            self.index_path,
            {"schema": ROLLOUT_CONDITIONING_CACHE_SCHEMA, "index": index.to_dict()},
            root=self.root,
        )
        sync_directory(self.root)
        return index

    def read_index(self) -> RolloutConditioningIndex:
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid rollout conditioning index: {self.index_path}") from error
        if not isinstance(payload, Mapping) or set(payload) != {"schema", "index"}:
            raise ValueError("rollout conditioning cache envelope fields differ")
        if payload["schema"] != ROLLOUT_CONDITIONING_CACHE_SCHEMA:
            raise ValueError(f"unsupported rollout conditioning cache: {payload['schema']!r}")
        index = RolloutConditioningIndex.from_mapping(payload["index"])
        for entry in index.entries:
            if self.shared.read(entry.branch, load_tensors=False) != entry.artifact:
                raise ValueError("rollout conditioning object differs from its index")
        return index


@dataclass(frozen=True, slots=True)
class RolloutConditionedPrompt:
    record: RolloutPromptRecord
    conditioning: Mapping[str, torch.Tensor]
    artifact: SharedConditioningArtifact

    def __post_init__(self) -> None:
        if not isinstance(self.record, RolloutPromptRecord):
            raise TypeError("record must be RolloutPromptRecord")
        if not isinstance(self.artifact, SharedConditioningArtifact):
            raise TypeError("artifact must be SharedConditioningArtifact")
        values = {str(key): value for key, value in self.conditioning.items()}
        if not values or not all(isinstance(value, torch.Tensor) for value in values.values()):
            raise TypeError("rollout conditioning must contain named tensors")
        object.__setattr__(self, "conditioning", MappingProxyType(values))


class RolloutConditioningDataset(Sequence[RolloutConditionedPrompt]):
    def __init__(self, prompts: RolloutPromptDataset, cache_root: str | Path) -> None:
        if not isinstance(prompts, RolloutPromptDataset):
            raise TypeError("prompts must be RolloutPromptDataset")
        self.prompts = prompts
        self.store = RolloutConditioningStore(cache_root)
        self.index = self.store.read_index()
        if tuple(entry.record for entry in self.index.entries) != tuple(prompts):
            raise ValueError("rollout conditioning cache belongs to another prompt dataset")
        self.sample_ids = prompts.sample_ids

    def __len__(self) -> int:
        return len(self.prompts)

    @overload
    def __getitem__(self, index: int) -> RolloutConditionedPrompt: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[RolloutConditionedPrompt, ...]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> RolloutConditionedPrompt | tuple[RolloutConditionedPrompt, ...]:
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(len(self))))
        record = self.prompts[index]
        entry = self.index.entries[index]
        sample = self.store.shared.read(entry.branch)
        if not isinstance(sample, SharedConditioningSample):
            raise RuntimeError("rollout conditioning store failed to load tensors")
        return RolloutConditionedPrompt(record=record, conditioning=sample.tensors, artifact=sample.artifact)

    def __iter__(self) -> Iterator[RolloutConditionedPrompt]:
        for index in range(len(self)):
            yield self[index]


@dataclass(frozen=True, slots=True)
class RolloutConditioningPreparationResult:
    index: RolloutConditioningIndex
    entries: tuple[RolloutConditioningEntry, ...]
    unconditional_conditioning: SharedConditioningArtifact | None = None


def prepare_rollout_conditioning_cache(
    prompts: RolloutPromptDataset,
    *,
    cache_root: str | Path,
    encoder: object,
    model_recipe: str,
    conditioner: Mapping[str, object],
    tokenizer: Mapping[str, object],
    generation_defaults: Mapping[str, object] | None = None,
    encoder_options: Mapping[str, object] | None = None,
    tensor_layouts: Mapping[str, str] | None = None,
) -> RolloutConditioningPreparationResult:
    """Encode every prompt once and record the model/encoder configuration.

    Encoders may return Wan's single context tensor or a named tensor mapping
    for model families whose denoisers consume multiple text streams.
    """

    if not isinstance(prompts, RolloutPromptDataset):
        raise TypeError("prompts must be RolloutPromptDataset")
    encode = getattr(encoder, "encode", None)
    if not callable(encode):
        raise TypeError("rollout conditioning encoder must expose encode")
    recipe = str(model_recipe).strip().lower().replace("_", "-")
    encode_kwargs = dict(encoder_options or {})
    layouts = None if tensor_layouts is None else {str(key): str(value) for key, value in tensor_layouts.items()}
    store = RolloutConditioningStore(cache_root)
    entries: list[RolloutConditioningEntry] = []
    for index, record in enumerate(prompts):
        height, width, frames = resolve_rollout_generation_geometry(record, generation_defaults)
        encoded = encode(
            sample_id=record.prompt_id,
            prompt=record.prompt,
            frames=frames,
            height=height,
            width=width,
            **encode_kwargs,
        )
        if isinstance(encoded, torch.Tensor):
            tensors = {"context": encoded}
            resolved_layouts = layouts or {"context": "sequence-features"}
        elif isinstance(encoded, Mapping):
            tensors = {str(key): value for key, value in encoded.items()}
            if not tensors or not all(isinstance(value, torch.Tensor) for value in tensors.values()):
                raise TypeError("rollout conditioning encoder mappings must contain named tensors")
            if layouts is None:
                raise ValueError("multi-tensor rollout conditioning requires tensor_layouts")
            resolved_layouts = layouts
        else:
            raise TypeError("rollout conditioning encoder must return a tensor or tensor mapping")
        branch = f"rollout-{index:08d}"
        artifact = store.shared.write(
            branch=branch,
            prompt=record.prompt,
            model_recipe=recipe,
            conditioner=conditioner,
            tokenizer=tokenizer,
            tensors=tensors,
            layouts=resolved_layouts,
        )
        entries.append(RolloutConditioningEntry(record=record, branch=branch, artifact=artifact))
    index = store.write_index(
        RolloutConditioningIndex(
            model_recipe=recipe,
            conditioner=conditioner,
            tokenizer=tokenizer,
            entries=tuple(entries),
        )
    )
    return RolloutConditioningPreparationResult(index=index, entries=tuple(entries))


def collate_rollout_conditioned_prompts(
    values: list[RolloutConditionedPrompt],
) -> tuple[RolloutConditionedPrompt, ...]:
    if not values or not all(isinstance(value, RolloutConditionedPrompt) for value in values):
        raise ValueError("conditioned rollout batches require RolloutConditionedPrompt values")
    return tuple(values)


__all__ = [
    "ROLLOUT_CONDITIONING_CACHE_SCHEMA",
    "ROLLOUT_CONDITIONING_INDEX_SCHEMA",
    "RolloutConditionedPrompt",
    "RolloutConditioningDataset",
    "RolloutConditioningEntry",
    "RolloutConditioningIndex",
    "RolloutConditioningPreparationResult",
    "RolloutConditioningStore",
    "collate_rollout_conditioned_prompts",
    "prepare_rollout_conditioning_cache",
    "resolve_rollout_generation_geometry",
]
