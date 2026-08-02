"""Strict prompt-only manifests for native diffusion-policy rollouts."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from worldfoundry.core.io.file_utils import file_sha256
from worldfoundry.core.io.integrity import canonical_json, canonical_sha256
from worldfoundry.training.safety.shieldgemma import PromptSafetyAudit

ROLLOUT_PROMPT_SCHEMA = "worldfoundry-rollout-prompt"
ROLLOUT_PROMPT_DATASET_SCHEMA = "worldfoundry-rollout-prompt-dataset"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")


def _text(value: object, *, field_name: str) -> str:
    resolved = str(value).strip()
    if not resolved:
        raise ValueError(f"{field_name} cannot be empty")
    return resolved


def _identifier(value: object, *, field_name: str) -> str:
    resolved = _text(value, field_name=field_name)
    if _IDENTIFIER.fullmatch(resolved) is None:
        raise ValueError(f"{field_name} contains unsupported characters: {value!r}")
    return resolved


def _mapping(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    resolved = {str(key): item for key, item in value.items()}
    if any(not key.strip() for key in resolved):
        raise ValueError(f"{field_name} keys cannot be empty")
    # Reject non-canonical values, NaN, and hidden unserializable state now.
    canonical_json(resolved)
    return resolved


@dataclass(frozen=True, slots=True)
class RolloutPromptRecord:
    """One safe prompt group before stochastic sample expansion."""

    prompt_id: str
    prompt: str
    safety_audit: PromptSafetyAudit
    split: str = "train"
    generation: Mapping[str, object] = field(default_factory=dict)
    schema: str = ROLLOUT_PROMPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ROLLOUT_PROMPT_SCHEMA:
            raise ValueError(f"unsupported rollout prompt schema: {self.schema!r}")
        prompt_id = _identifier(self.prompt_id, field_name="prompt_id")
        prompt = _text(self.prompt, field_name="prompt")
        split = _identifier(self.split, field_name="split").lower()
        if not isinstance(self.safety_audit, PromptSafetyAudit):
            raise TypeError("safety_audit must be PromptSafetyAudit")
        from worldfoundry.core.io.integrity import text_sha256

        if self.safety_audit.prompt_sha256 != text_sha256(prompt):
            raise ValueError("rollout prompt text differs from its safety audit")
        if not self.safety_audit.safe:
            raise ValueError("unsafe rollout prompts cannot enter a training manifest")
        object.__setattr__(self, "prompt_id", prompt_id)
        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "split", split)
        object.__setattr__(
            self,
            "generation",
            MappingProxyType(_mapping(self.generation, field_name="generation")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "split": self.split,
            "generation": dict(self.generation),
            "safety_audit": self.safety_audit.to_dict(),
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_mapping(cls, value: object) -> RolloutPromptRecord:
        if not isinstance(value, Mapping):
            raise TypeError("rollout prompt row must be a mapping")
        payload = {str(key): item for key, item in value.items()}
        expected = {
            "schema",
            "prompt_id",
            "prompt",
            "split",
            "generation",
            "safety_audit",
        }
        if set(payload) != expected:
            raise ValueError(
                "rollout prompt fields mismatch; "
                f"missing={sorted(expected - set(payload))}, "
                f"unknown={sorted(set(payload) - expected)}"
            )
        audit = PromptSafetyAudit.from_mapping(payload.pop("safety_audit"))
        return cls(**payload, safety_audit=audit)


class RolloutPromptDataset(Sequence[RolloutPromptRecord]):
    """A selected immutable split with byte- and content-level identity."""

    def __init__(
        self,
        records: Sequence[RolloutPromptRecord],
        *,
        split: str,
        manifest_path: Path,
        manifest_sha256: str,
    ) -> None:
        values = tuple(records)
        if not values or not all(isinstance(item, RolloutPromptRecord) for item in values):
            raise ValueError("rollout prompt dataset requires non-empty prompt records")
        selected_split = _identifier(split, field_name="split").lower()
        if any(record.split != selected_split for record in values):
            raise ValueError("rollout prompt dataset contains records from another split")
        ids = tuple(record.prompt_id for record in values)
        hashes = tuple(record.safety_audit.prompt_sha256 for record in values)
        if len(ids) != len(set(ids)):
            raise ValueError("rollout prompt_id values must be unique")
        if len(hashes) != len(set(hashes)):
            raise ValueError("duplicate prompt text would create ambiguous reward groups")
        self._records = values
        self.split = selected_split
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest_sha256 = str(manifest_sha256).lower()
        self.sample_ids = ids
        self.dataset_digest = canonical_sha256(
            {
                "schema": ROLLOUT_PROMPT_DATASET_SCHEMA,
                "manifest_sha256": self.manifest_sha256,
                "split": self.split,
                "records": [record.to_dict() for record in values],
            }
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        split: str = "train",
    ) -> RolloutPromptDataset:
        source = Path(path).expanduser().resolve()
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"rollout prompt manifest does not exist: {source}")
        if source.suffix.lower() != ".jsonl":
            raise ValueError("rollout prompt manifest must use JSONL")
        records: list[RolloutPromptRecord] = []
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                record = RolloutPromptRecord.from_mapping(payload)
            except Exception as error:
                raise ValueError(f"invalid rollout prompt manifest row {source}:{line_number}") from error
            if record.split == str(split).strip().lower():
                records.append(record)
        if not records:
            raise ValueError(f"rollout prompt split {split!r} is empty")
        return cls(
            records,
            split=split,
            manifest_path=source,
            manifest_sha256=file_sha256(source),
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int | slice) -> RolloutPromptRecord | tuple[RolloutPromptRecord, ...]:
        return self._records[index]

    def __iter__(self) -> Iterator[RolloutPromptRecord]:
        return iter(self._records)


__all__ = [
    "ROLLOUT_PROMPT_DATASET_SCHEMA",
    "ROLLOUT_PROMPT_SCHEMA",
    "RolloutPromptDataset",
    "RolloutPromptRecord",
]
