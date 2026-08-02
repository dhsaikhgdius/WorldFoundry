"""Build strict rollout manifests from prompt sources after safety filtering."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from worldfoundry.core.io.file_utils import file_sha256
from worldfoundry.core.io.integrity import canonical_json, write_exclusive_text
from worldfoundry.training.safety.shieldgemma import (
    PromptSafetyAudit,
    ShieldGemmaPromptFilter,
)

from .rollout_manifest import RolloutPromptRecord, _identifier, _mapping, _text


@dataclass(frozen=True, slots=True)
class _RolloutPromptSource:
    prompt_id: str
    prompt: str
    split: str
    generation: Mapping[str, object]

    @classmethod
    def from_mapping(cls, value: object) -> _RolloutPromptSource:
        if not isinstance(value, Mapping):
            raise TypeError("rollout prompt source row must be a mapping")
        payload = {str(key): item for key, item in value.items()}
        allowed = {"prompt_id", "prompt", "split", "generation"}
        required = {"prompt_id", "prompt"}
        missing = sorted(required - set(payload))
        unknown = sorted(set(payload) - allowed)
        if missing or unknown:
            raise ValueError(f"rollout prompt source fields mismatch; missing={missing}, unknown={unknown}")
        return cls(
            prompt_id=_identifier(payload["prompt_id"], field_name="prompt_id"),
            prompt=_text(payload["prompt"], field_name="prompt"),
            split=_identifier(payload.get("split", "train"), field_name="split").lower(),
            generation=_mapping(payload.get("generation", {}), field_name="generation"),
        )

    def audited(self, audit: PromptSafetyAudit) -> RolloutPromptRecord:
        return RolloutPromptRecord(
            prompt_id=self.prompt_id,
            prompt=self.prompt,
            split=self.split,
            generation=self.generation,
            safety_audit=audit,
        )


@dataclass(frozen=True, slots=True)
class RolloutPromptAuditResult:
    """The immutable manifest emitted by one complete prompt audit."""

    manifest_path: Path
    manifest_sha256: str
    records: tuple[RolloutPromptRecord, ...]


def _read_sources(path: Path) -> tuple[_RolloutPromptSource, ...]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"rollout prompt source does not exist: {path}")
    if path.suffix.lower() != ".jsonl":
        raise ValueError("rollout prompt source must use JSONL")
    records: list[_RolloutPromptSource] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(_RolloutPromptSource.from_mapping(json.loads(line)))
        except Exception as error:
            raise ValueError(f"invalid rollout prompt source row {path}:{line_number}") from error
    if not records:
        raise ValueError("rollout prompt source is empty")
    prompt_ids = tuple(record.prompt_id for record in records)
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("rollout prompt source prompt_id values must be unique")
    prompts = tuple(record.prompt for record in records)
    if len(prompts) != len(set(prompts)):
        raise ValueError("rollout prompt source text must be unique")
    return tuple(records)


def audit_rollout_prompts(
    *,
    source_path: str | Path,
    output_manifest_path: str | Path,
    prompt_filter: ShieldGemmaPromptFilter,
    batch_size: int = 4,
) -> RolloutPromptAuditResult:
    """Filter prompt-only input and emit the manifest consumed by rollout caches."""

    if not isinstance(prompt_filter, ShieldGemmaPromptFilter):
        raise TypeError("prompt_filter must be a ShieldGemmaPromptFilter")
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("batch_size must be a positive integer")
    size = int(batch_size)
    source = Path(source_path).expanduser().resolve()
    destination = Path(output_manifest_path).expanduser().resolve()
    if source == destination:
        raise ValueError("rollout prompt source and output manifest paths must differ")
    if destination.exists():
        raise FileExistsError(f"rollout prompt manifest already exists: {destination}")
    candidates = _read_sources(source)
    audits: list[PromptSafetyAudit] = []
    for offset in range(0, len(candidates), size):
        prompts = tuple(candidate.prompt for candidate in candidates[offset : offset + size])
        batch = prompt_filter.require_safe(prompts)
        if not isinstance(batch, tuple) or len(batch) != len(prompts):
            raise RuntimeError("ShieldGemma returned a different number of prompt audits")
        audits.extend(batch)
    records = tuple(candidate.audited(audit) for candidate, audit in zip(candidates, audits, strict=True))
    rows = "\n".join(canonical_json(record.to_dict()) for record in records) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_exclusive_text(destination, rows, root=destination.parent)
    return RolloutPromptAuditResult(
        manifest_path=destination,
        manifest_sha256=file_sha256(destination),
        records=records,
    )


__all__ = ["RolloutPromptAuditResult", "audit_rollout_prompts"]
