"""Immutable ShieldGemma audit sidecars and audited manifest construction."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from worldfoundry.training.safety.shieldgemma import PromptSafetyAudit, ShieldGemmaPromptFilter

from .dataset import TrainingManifestDataset
from .manifest import resolve_local_media_path

PROMPT_AUDIT_SET_SCHEMA = "worldfoundry-training-prompt-audit-set"
def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive")
    return resolved


@dataclass(frozen=True, slots=True)
class PromptAuditSet:
    """Prompt audits keyed by training sample ID."""

    records: tuple[tuple[str, PromptSafetyAudit], ...]
    schema: str = PROMPT_AUDIT_SET_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PROMPT_AUDIT_SET_SCHEMA:
            raise ValueError(f"unsupported prompt audit set schema: {self.schema!r}")
        records = tuple(self.records)
        normalized: list[tuple[str, PromptSafetyAudit]] = []
        for sample_id, audit in records:
            resolved_id = str(sample_id).strip()
            if not resolved_id:
                raise ValueError("prompt audit sample_id cannot be empty")
            if not isinstance(audit, PromptSafetyAudit):
                raise TypeError("prompt audit records must contain PromptSafetyAudit values")
            normalized.append((resolved_id, audit))
        sample_ids = [sample_id for sample_id, _ in normalized]
        if not sample_ids or len(sample_ids) != len(set(sample_ids)):
            raise ValueError("prompt audit records require unique non-empty sample ids")
        object.__setattr__(self, "records", tuple(normalized))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "records": [{"sample_id": sample_id, "audit": audit.to_dict()} for sample_id, audit in self.records],
        }

    @classmethod
    def from_mapping(cls, value: object) -> PromptAuditSet:
        if not isinstance(value, Mapping):
            raise TypeError("prompt audit set must be a mapping")
        fields = {str(key): item for key, item in value.items()}
        expected = {"schema", "records"}
        if set(fields) != expected:
            raise ValueError(
                "prompt audit set fields mismatch; "
                f"missing={sorted(expected - set(fields))}, "
                f"unknown={sorted(set(fields) - expected)}"
            )
        raw_records = fields["records"]
        if not isinstance(raw_records, Sequence) or isinstance(
            raw_records,
            (str, bytes, bytearray),
        ):
            raise TypeError("prompt audit set records must be a sequence")
        records: list[tuple[str, PromptSafetyAudit]] = []
        for raw in raw_records:
            if not isinstance(raw, Mapping) or set(raw) != {"sample_id", "audit"}:
                raise ValueError("each prompt audit record must contain sample_id and audit")
            records.append(
                (
                    str(raw["sample_id"]),
                    PromptSafetyAudit.from_mapping(raw["audit"]),
                )
            )
        return cls(
            schema=str(fields["schema"]),
            records=tuple(records),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> PromptAuditSet:
        source = Path(path).expanduser().resolve()
        return cls.from_mapping(json.loads(source.read_text(encoding="utf-8")))

    def select_for_manifest(
        self,
        manifest: TrainingManifestDataset,
    ) -> tuple[PromptSafetyAudit, ...]:
        if not isinstance(manifest, TrainingManifestDataset):
            raise TypeError("manifest must be a TrainingManifestDataset")
        by_id = dict(self.records)
        missing = sorted(set(manifest.sample_ids) - set(by_id))
        if missing:
            raise ValueError(f"prompt audit set lacks selected samples: {missing}")
        audits = tuple(by_id[sample_id] for sample_id in manifest.sample_ids)
        for sample, audit in zip(manifest, audits):
            if audit.prompt != sample.prompt:
                raise ValueError(f"prompt audit text differs for sample {sample.sample_id!r}")
            if sample.safety.get("prompt_safe") is not True:
                raise ValueError(f"manifest prompt safety decision differs for sample {sample.sample_id!r}")
            if sample.safety.get("model_revision") != audit.model_revision:
                raise ValueError(f"manifest safety model revision differs for sample {sample.sample_id!r}")
        return audits


@dataclass(frozen=True, slots=True)
class PromptAuditPreparationResult:
    manifest_path: Path
    audit_path: Path
    audit_set: PromptAuditSet


def audit_training_manifest_prompts(
    *,
    manifest_path: str | Path,
    output_manifest_path: str | Path,
    output_audit_path: str | Path,
    prompt_filter: ShieldGemmaPromptFilter,
    batch_size: int = 4,
    verify_media_files: bool = True,
) -> PromptAuditPreparationResult:
    """Filter every prompt and write a new manifest plus its exact audit sidecar."""

    if not isinstance(prompt_filter, ShieldGemmaPromptFilter):
        raise TypeError("prompt_filter must be a ShieldGemmaPromptFilter")
    size = _positive_int(batch_size, field_name="batch_size")
    source_path = Path(manifest_path).expanduser().resolve()
    destination = Path(output_manifest_path).expanduser().resolve()
    audit_destination = Path(output_audit_path).expanduser().resolve()
    if source_path in {destination, audit_destination} or destination == audit_destination:
        raise ValueError("input manifest, audited manifest, and audit sidecar paths must differ")
    for path in (destination, audit_destination):
        if path.exists():
            raise FileExistsError(f"audit output already exists: {path}")

    source = TrainingManifestDataset.from_file(
        source_path,
        split=None,
        verify_files=verify_media_files,
    )
    audits: list[PromptSafetyAudit] = []
    for offset in range(0, len(source), size):
        audits.extend(prompt_filter.require_safe(tuple(sample.prompt for sample in source[offset : offset + size])))

    destination.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    relocate_media = destination.parent != source_path.parent
    for sample, audit in zip(source, audits):
        payload = sample.to_dict()
        safety = dict(payload["safety"])
        safety.update(
            {
                "filter": "ShieldGemma-2B",
                "prompt_safe": audit.safe,
                "model_revision": audit.model_revision,
            }
        )
        payload["safety"] = safety
        if relocate_media:
            media_path = resolve_local_media_path(sample.media, manifest_path=source_path)
            if media_path is not None:
                media = dict(payload["media"])
                media["uri"] = str(media_path)
                payload["media"] = media
        rows.append(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    with destination.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(rows) + "\n")

    audited_manifest = TrainingManifestDataset.from_file(
        destination,
        split=None,
        verify_files=verify_media_files,
    )
    audit_set = PromptAuditSet(
        records=tuple(zip(audited_manifest.sample_ids, audits)),
    )
    audit_destination.parent.mkdir(parents=True, exist_ok=True)
    with audit_destination.open("x", encoding="utf-8") as handle:
        json.dump(
            audit_set.to_dict(),
            handle,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return PromptAuditPreparationResult(destination, audit_destination, audit_set)


__all__ = [
    "PROMPT_AUDIT_SET_SCHEMA",
    "PromptAuditPreparationResult",
    "PromptAuditSet",
    "audit_training_manifest_prompts",
]
