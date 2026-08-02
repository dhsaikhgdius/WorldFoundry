"""Content-addressed prompt conditioning for native policy rollouts."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import torch

from worldfoundry.core.io.integrity import (
    canonical_sha256,
    sync_directory,
    write_exclusive_json,
)

from .rollout_manifest import RolloutPromptDataset, RolloutPromptRecord
from .shared_conditioning import (
    SharedConditioningArtifact,
    SharedConditioningSample,
    SharedConditioningStore,
)

ROLLOUT_CONDITIONING_CACHE_SCHEMA = "worldfoundry-rollout-conditioning-cache"
ROLLOUT_CONDITIONING_INDEX_SCHEMA = "worldfoundry-rollout-conditioning-index"
_INDEX_NAME = "rollout-conditioning-index.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256(value: object, *, field_name: str) -> str:
    resolved = str(value).strip().lower()
    if _SHA256.fullmatch(resolved) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return resolved


def resolve_rollout_generation_geometry(
    record: RolloutPromptRecord,
    defaults: Mapping[str, object] | None,
) -> tuple[int, int, int]:
    """Merge recipe defaults with one audited record and validate H/W/T."""

    if not isinstance(record, RolloutPromptRecord):
        raise TypeError("record must be RolloutPromptRecord")
    if defaults is not None and not isinstance(defaults, Mapping):
        raise TypeError("rollout generation defaults must be a mapping")
    values = dict(defaults or {})
    values.update(record.generation)
    unknown = sorted(set(values) - {"height", "width", "num_frames"})
    missing = sorted({"height", "width", "num_frames"} - set(values))
    if unknown or missing:
        raise ValueError(f"rollout generation geometry mismatch; missing={missing}, unknown={unknown}")
    resolved: list[int] = []
    for name in ("height", "width", "num_frames"):
        value = values[name]
        if isinstance(value, bool) or int(value) <= 0:
            raise ValueError(f"rollout generation {name} must be positive")
        resolved.append(int(value))
    return tuple(resolved)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class RolloutConditioningEntry:
    prompt_id: str
    prompt_sha256: str
    prompt_record_sha256: str
    safety_audit_sha256: str
    branch: str
    artifact: SharedConditioningArtifact

    def __post_init__(self) -> None:
        prompt_id = str(self.prompt_id).strip()
        if not prompt_id:
            raise ValueError("rollout conditioning prompt_id cannot be empty")
        prompt_sha = _sha256(self.prompt_sha256, field_name="prompt_sha256")
        record_sha = _sha256(
            self.prompt_record_sha256,
            field_name="prompt_record_sha256",
        )
        safety_sha = _sha256(
            self.safety_audit_sha256,
            field_name="safety_audit_sha256",
        )
        branch = str(self.branch).strip().lower()
        expected_branch = f"rollout-{prompt_sha}"
        if branch != expected_branch:
            raise ValueError(f"rollout conditioning branch must be {expected_branch!r}")
        if not isinstance(self.artifact, SharedConditioningArtifact):
            raise TypeError("rollout conditioning artifact must be SharedConditioningArtifact")
        if self.artifact.identity.branch != branch or self.artifact.identity.prompt_sha256 != prompt_sha:
            raise ValueError("rollout conditioning artifact identity differs from its entry")
        object.__setattr__(self, "prompt_id", prompt_id)
        object.__setattr__(self, "prompt_sha256", prompt_sha)
        object.__setattr__(self, "prompt_record_sha256", record_sha)
        object.__setattr__(self, "safety_audit_sha256", safety_sha)
        object.__setattr__(self, "branch", branch)

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_id": self.prompt_id,
            "prompt_sha256": self.prompt_sha256,
            "prompt_record_sha256": self.prompt_record_sha256,
            "safety_audit_sha256": self.safety_audit_sha256,
            "branch": self.branch,
            "artifact": self.artifact.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: object) -> RolloutConditioningEntry:
        if not isinstance(value, Mapping):
            raise TypeError("rollout conditioning entry must be a mapping")
        payload = {str(key): item for key, item in value.items()}
        expected = {
            "prompt_id",
            "prompt_sha256",
            "prompt_record_sha256",
            "safety_audit_sha256",
            "branch",
            "artifact",
        }
        if set(payload) != expected:
            raise ValueError("rollout conditioning entry fields differ from the active schema")
        artifact = SharedConditioningArtifact.from_mapping(payload.pop("artifact"))
        return cls(**payload, artifact=artifact)


@dataclass(frozen=True, slots=True)
class RolloutConditioningIndex:
    dataset_digest: str
    manifest_sha256: str
    model_recipe: str
    model_recipe_digest: str
    conditioner_digest: str
    tokenizer_digest: str
    entries: tuple[RolloutConditioningEntry, ...]
    schema: str = ROLLOUT_CONDITIONING_INDEX_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ROLLOUT_CONDITIONING_INDEX_SCHEMA:
            raise ValueError(f"unsupported rollout conditioning index: {self.schema!r}")
        for name in (
            "dataset_digest",
            "manifest_sha256",
            "model_recipe_digest",
            "conditioner_digest",
            "tokenizer_digest",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), field_name=name))
        recipe = str(self.model_recipe).strip().lower().replace("_", "-")
        if not recipe:
            raise ValueError("model_recipe cannot be empty")
        entries = tuple(self.entries)
        if not entries or not all(isinstance(item, RolloutConditioningEntry) for item in entries):
            raise ValueError("rollout conditioning index requires entries")
        ids = tuple(entry.prompt_id for entry in entries)
        if len(ids) != len(set(ids)):
            raise ValueError("rollout conditioning entry prompt_id values must be unique")
        for entry in entries:
            identity = entry.artifact.identity
            if (
                identity.model_recipe_digest != self.model_recipe_digest
                or identity.conditioner_digest != self.conditioner_digest
                or identity.tokenizer_digest != self.tokenizer_digest
            ):
                raise ValueError("rollout conditioning entry uses another model/encoder identity")
        object.__setattr__(self, "model_recipe", recipe)
        object.__setattr__(self, "entries", entries)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "dataset_digest": self.dataset_digest,
            "manifest_sha256": self.manifest_sha256,
            "model_recipe": self.model_recipe,
            "model_recipe_digest": self.model_recipe_digest,
            "conditioner_digest": self.conditioner_digest,
            "tokenizer_digest": self.tokenizer_digest,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_mapping(cls, value: object) -> RolloutConditioningIndex:
        if not isinstance(value, Mapping):
            raise TypeError("rollout conditioning index must be a mapping")
        payload = {str(key): item for key, item in value.items()}
        expected = {
            "schema",
            "dataset_digest",
            "manifest_sha256",
            "model_recipe",
            "model_recipe_digest",
            "conditioner_digest",
            "tokenizer_digest",
            "entries",
        }
        if set(payload) != expected:
            raise ValueError("rollout conditioning index fields differ from the active schema")
        raw_entries = payload.pop("entries")
        if not isinstance(raw_entries, Sequence) or isinstance(
            raw_entries,
            (str, bytes, bytearray),
        ):
            raise TypeError("rollout conditioning entries must be a sequence")
        return cls(
            **payload,
            entries=tuple(RolloutConditioningEntry.from_mapping(item) for item in raw_entries),
        )


class RolloutConditioningStore:
    """Own the strict index while reusing shared tensor object storage."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.shared = SharedConditioningStore(self.root)
        self.index_path = self.root / _INDEX_NAME

    def write_index(self, index: RolloutConditioningIndex) -> RolloutConditioningIndex:
        if not isinstance(index, RolloutConditioningIndex):
            raise TypeError("index must be RolloutConditioningIndex")
        if self.index_path.exists():
            existing = self.read_index()
            if existing != index:
                raise FileExistsError(f"rollout conditioning index already binds different content: {self.index_path}")
            return existing
        write_exclusive_json(
            self.index_path,
            {
                "schema": ROLLOUT_CONDITIONING_CACHE_SCHEMA,
                "index": index.to_dict(),
                "index_sha256": index.digest,
            },
            root=self.root,
        )
        sync_directory(self.root)
        return index

    def read_index(self) -> RolloutConditioningIndex:
        if self.index_path.is_symlink():
            raise ValueError("rollout conditioning index cannot be a symlink")
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid rollout conditioning index: {self.index_path}") from error
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema",
            "index",
            "index_sha256",
        }:
            raise ValueError("rollout conditioning cache envelope fields differ")
        if payload["schema"] != ROLLOUT_CONDITIONING_CACHE_SCHEMA:
            raise ValueError(f"unsupported rollout conditioning cache: {payload['schema']!r}")
        index = RolloutConditioningIndex.from_mapping(payload["index"])
        if _sha256(payload["index_sha256"], field_name="index_sha256") != index.digest:
            raise ValueError("rollout conditioning index digest mismatch")
        for entry in index.entries:
            artifact = self.shared.read(entry.branch, load_tensors=False)
            if artifact != entry.artifact:
                raise ValueError("rollout conditioning object differs from its strict index")
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
    """Join a safe prompt manifest to its exact cached encoder outputs."""

    def __init__(
        self,
        prompts: RolloutPromptDataset,
        cache_root: str | Path,
    ) -> None:
        if not isinstance(prompts, RolloutPromptDataset):
            raise TypeError("prompts must be RolloutPromptDataset")
        self.prompts = prompts
        self.store = RolloutConditioningStore(cache_root)
        self.index = self.store.read_index()
        if self.index.dataset_digest != prompts.dataset_digest or self.index.manifest_sha256 != prompts.manifest_sha256:
            raise ValueError("rollout conditioning cache belongs to another prompt dataset")
        if tuple(entry.prompt_id for entry in self.index.entries) != prompts.sample_ids:
            raise ValueError("rollout conditioning cache order differs from the prompt manifest")
        for entry, record in zip(self.index.entries, prompts):
            if (
                entry.prompt_sha256 != record.safety_audit.prompt_sha256
                or entry.prompt_record_sha256 != record.digest
                or entry.safety_audit_sha256 != record.safety_audit.digest
            ):
                raise ValueError("rollout conditioning entry differs from its prompt record")
        self.sample_ids = prompts.sample_ids
        self.dataset_digest = canonical_sha256(
            {
                "schema": "worldfoundry-conditioned-rollout-prompt-dataset",
                "prompt_dataset_digest": prompts.dataset_digest,
                "conditioning_index_sha256": self.index.digest,
            }
        )

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> RolloutConditionedPrompt:
        record = self.prompts[index]
        entry = self.index.entries[index]
        sample = self.store.shared.read(entry.branch)
        if not isinstance(sample, SharedConditioningSample):
            raise RuntimeError("rollout conditioning store failed to load tensors")
        return RolloutConditionedPrompt(
            record=record,
            conditioning=sample.tensors,
            artifact=sample.artifact,
        )

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
    model_recipe_digest: str,
    conditioner_digest: str,
    tokenizer_digest: str,
    generation_defaults: Mapping[str, object] | None = None,
) -> RolloutConditioningPreparationResult:
    """Encode each unique prompt once and bind it to encoder/model identities."""

    if not isinstance(prompts, RolloutPromptDataset):
        raise TypeError("prompts must be RolloutPromptDataset")
    encode = getattr(encoder, "encode", None)
    if not callable(encode):
        raise TypeError("rollout conditioning encoder must expose encode")
    store = RolloutConditioningStore(cache_root)
    entries: list[RolloutConditioningEntry] = []
    for record in prompts:
        height, width, frames = resolve_rollout_generation_geometry(
            record,
            generation_defaults,
        )
        context = encode(
            sample_id=record.prompt_id,
            prompt=record.prompt,
            frames=frames,
            height=height,
            width=width,
        )
        if not isinstance(context, torch.Tensor):
            raise TypeError("rollout conditioning encoder must return a context tensor")
        branch = f"rollout-{record.safety_audit.prompt_sha256}"
        artifact = store.shared.write(
            branch=branch,
            prompt_sha256=record.safety_audit.prompt_sha256,
            model_recipe_digest=model_recipe_digest,
            conditioner_digest=conditioner_digest,
            tokenizer_digest=tokenizer_digest,
            tensors={"context": context},
            layouts={"context": "sequence-features"},
        )
        entries.append(
            RolloutConditioningEntry(
                prompt_id=record.prompt_id,
                prompt_sha256=record.safety_audit.prompt_sha256,
                prompt_record_sha256=record.digest,
                safety_audit_sha256=record.safety_audit.digest,
                branch=branch,
                artifact=artifact,
            )
        )
    index = store.write_index(
        RolloutConditioningIndex(
            dataset_digest=prompts.dataset_digest,
            manifest_sha256=prompts.manifest_sha256,
            model_recipe=model_recipe,
            model_recipe_digest=model_recipe_digest,
            conditioner_digest=conditioner_digest,
            tokenizer_digest=tokenizer_digest,
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
