"""Audited PEFT LoRA injection for custom WorldFoundry model graphs."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from types import MappingProxyType

from torch import nn

from worldfoundry.core.io.file_utils import file_sha256 as _file_sha256
from worldfoundry.core.io.integrity import canonical_json as _core_canonical_json
from worldfoundry.core.io.integrity import write_exclusive_text

SANA_ATTENTION = "sana-attention"
WAN_ATTENTION = "wan-attention"
PEFT_ADAPTER_ARTIFACT_SCHEMA = "worldfoundry-peft-adapter"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SANA_ATTENTION_PATTERN = re.compile(
    r"^blocks\.(?P<block>\d+)\."
    r"(?P<role>attn\.(?:qkv|proj)|cross_attn\.(?:q_linear|kv_linear|proj))$"
)
_SANA_ATTENTION_ROLES = frozenset(
    {
        "attn.qkv",
        "attn.proj",
        "cross_attn.q_linear",
        "cross_attn.kv_linear",
        "cross_attn.proj",
    }
)
_WAN_ATTENTION_PATTERN = re.compile(
    r"^blocks\.(?P<block>\d+)\."
    r"(?P<role>(?:self|cross)_attn\.(?:q|k|v|o))$"
)
_WAN_ATTENTION_ROLES = frozenset(
    {
        "self_attn.q",
        "self_attn.k",
        "self_attn.v",
        "self_attn.o",
        "cross_attn.q",
        "cross_attn.k",
        "cross_attn.v",
        "cross_attn.o",
    }
)


@dataclass(frozen=True, slots=True)
class LoraTargetAudit:
    preset: str
    target_pattern: str
    module_names: tuple[str, ...]
    module_shapes: Mapping[str, tuple[int, int]]
    block_count: int
    base_parameter_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "module_shapes", MappingProxyType(dict(self.module_shapes)))

    def expected_trainable_parameters(self, rank: int) -> int:
        if isinstance(rank, bool) or int(rank) <= 0:
            raise ValueError("LoRA rank must be a positive integer")
        return sum(
            int(rank) * (in_features + out_features) for in_features, out_features in self.module_shapes.values()
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "preset": self.preset,
            "target_pattern": self.target_pattern,
            "module_names": list(self.module_names),
            "module_shapes": {name: list(shape) for name, shape in self.module_shapes.items()},
            "module_count": len(self.module_names),
            "block_count": self.block_count,
            "base_parameter_count": self.base_parameter_count,
        }


@dataclass(frozen=True, slots=True)
class PeftLoraApplication:
    model: nn.Module
    target_audit: LoraTargetAudit
    targeted_module_names: tuple[str, ...]
    trainable_parameter_names: tuple[str, ...]
    trainable_parameter_count: int


@dataclass(frozen=True, slots=True)
class PeftAdapterArtifact:
    path: Path
    manifest_sha256: str
    file_digests: Mapping[str, str]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "file_digests", MappingProxyType(dict(self.file_digests)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def _canonical_json(value: object) -> str:
    try:
        return _core_canonical_json(value)
    except (TypeError, ValueError) as error:
        raise TypeError("PEFT adapter metadata must be JSON serializable without NaN or infinity") from error


def _json_metadata(value: Mapping[str, object] | None) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("PEFT adapter metadata must be a mapping")
    normalized = json.loads(_canonical_json(dict(value)))
    if not isinstance(normalized, dict):
        raise TypeError("PEFT adapter metadata must resolve to a JSON object")
    return normalized


def _artifact_files(path: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for candidate in sorted(path.rglob("*")):
        if candidate.name == "worldfoundry_adapter.json":
            continue
        if candidate.is_symlink():
            raise ValueError(f"PEFT adapter artifacts cannot contain symlinks: {candidate}")
        if candidate.is_file():
            relative = candidate.relative_to(path).as_posix()
            files[relative] = _file_sha256(candidate)
    if "adapter_config.json" not in files:
        raise ValueError("PEFT adapter artifact is missing adapter_config.json")
    if "adapter_model.safetensors" not in files:
        raise ValueError("PEFT adapter artifact is missing adapter_model.safetensors")
    return files


def audit_lora_targets(model: nn.Module, preset: str) -> LoraTargetAudit:
    """Resolve a semantic target preset and fail closed on graph drift."""

    if not isinstance(model, nn.Module):
        raise TypeError("LoRA target audit requires an nn.Module")
    normalized = str(preset).strip().lower().replace("_", "-")
    if normalized not in {SANA_ATTENTION, WAN_ATTENTION}:
        raise ValueError(f"unsupported LoRA target preset: {preset!r}")

    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise ValueError(f"{normalized} requires a non-empty ModuleList named 'blocks'")

    if normalized == SANA_ATTENTION:
        target_pattern = _SANA_ATTENTION_PATTERN
        expected_roles = _SANA_ATTENTION_ROLES
    else:
        target_pattern = _WAN_ATTENTION_PATTERN
        expected_roles = _WAN_ATTENTION_ROLES

    names: list[str] = []
    shapes: dict[str, tuple[int, int]] = {}
    roles_by_block: dict[int, set[str]] = {index: set() for index in range(len(blocks))}
    for name, module in model.named_modules():
        match = target_pattern.fullmatch(name)
        if match is None:
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"LoRA target {name!r} is {type(module).__name__}, expected nn.Linear")
        block = int(match.group("block"))
        if block not in roles_by_block:
            raise ValueError(f"LoRA target {name!r} refers to an unknown block")
        role = match.group("role")
        roles_by_block[block].add(role)
        names.append(name)
        shapes[name] = (int(module.in_features), int(module.out_features))

    drift = {
        block: {
            "missing": sorted(expected_roles - roles),
            "unexpected": sorted(roles - expected_roles),
        }
        for block, roles in roles_by_block.items()
        if roles != expected_roles
    }
    if drift:
        raise ValueError(f"{normalized} target graph drifted: {drift}")

    names.sort()
    return LoraTargetAudit(
        preset=normalized,
        target_pattern=target_pattern.pattern,
        module_names=tuple(names),
        module_shapes=shapes,
        block_count=len(blocks),
        base_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )


def apply_peft_lora(
    model: nn.Module,
    *,
    preset: str,
    rank: int,
    alpha: int,
    dropout: float = 0.0,
    modules_to_save: Sequence[str] = (),
) -> PeftLoraApplication:
    """Inject PEFT LoRA after an exact target audit.

    PEFT is imported lazily so manifest inspection and full-parameter training
    do not require it.  The returned wrapper is the module that must be passed
    to the optimizer and checkpoint/export code.
    """

    audit = audit_lora_targets(model, preset)
    if isinstance(rank, bool) or int(rank) <= 0:
        raise ValueError("LoRA rank must be a positive integer")
    if isinstance(alpha, bool) or int(alpha) <= 0:
        raise ValueError("LoRA alpha must be a positive integer")
    resolved_dropout = float(dropout)
    if not 0.0 <= resolved_dropout < 1.0:
        raise ValueError("LoRA dropout must be in [0, 1)")
    save_modules = tuple(str(name).strip() for name in modules_to_save)
    if any(not name for name in save_modules) or len(save_modules) != len(set(save_modules)):
        raise ValueError("modules_to_save must contain unique non-empty names")

    try:
        from peft import LoraConfig, get_peft_model
    except ModuleNotFoundError as error:
        raise RuntimeError("LoRA tuning requires the 'train-core' PEFT dependency") from error

    config = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=resolved_dropout,
        target_modules=audit.target_pattern,
        modules_to_save=list(save_modules) or None,
        bias="none",
        init_lora_weights=True,
    )
    peft_model = get_peft_model(model, config)
    if not isinstance(peft_model, nn.Module):
        raise TypeError(f"PEFT returned {type(peft_model).__name__}, expected nn.Module")

    targeted = tuple(str(name) for name in getattr(peft_model, "targeted_module_names", ()))
    targeted_set = set(targeted)
    audited_set = set(audit.module_names)
    if len(targeted) != len(targeted_set) or targeted_set != audited_set:
        raise RuntimeError(
            "PEFT target names differ from the audited graph: "
            f"missing={sorted(audited_set - targeted_set)}, "
            f"unexpected={sorted(targeted_set - audited_set)}, "
            f"duplicates={len(targeted) - len(targeted_set)}"
        )
    trainable = tuple(name for name, parameter in peft_model.named_parameters() if parameter.requires_grad)
    allowed_saved = tuple(f".{name}." for name in save_modules)
    unexpected = tuple(
        name for name in trainable if "lora_" not in name and not any(marker in f".{name}." for marker in allowed_saved)
    )
    if unexpected:
        raise RuntimeError(f"PEFT left unexpected base parameters trainable: {unexpected}")
    trainable_count = sum(parameter.numel() for parameter in peft_model.parameters() if parameter.requires_grad)
    expected_count = audit.expected_trainable_parameters(int(rank))
    if not save_modules and trainable_count != expected_count:
        raise RuntimeError(f"PEFT trainable parameter count {trainable_count} does not match audited {expected_count}")
    return PeftLoraApplication(
        model=peft_model,
        target_audit=audit,
        targeted_module_names=targeted,
        trainable_parameter_names=trainable,
        trainable_parameter_count=trainable_count,
    )


def apply_peft_lora_to_adapter(
    adapter: object,
    *,
    preset: str,
    rank: int,
    alpha: int,
    dropout: float = 0.0,
    modules_to_save: Sequence[str] = (),
) -> PeftLoraApplication:
    """Inject LoRA and reconnect a mutable native model adapter."""

    model = getattr(adapter, "trainable_module", None)
    denoiser = getattr(adapter, "denoiser", None)
    if not isinstance(model, nn.Module) or getattr(denoiser, "model", None) is not model:
        raise TypeError("adapter must expose the same trainable_module through denoiser.model")
    application = apply_peft_lora(
        model,
        preset=preset,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        modules_to_save=modules_to_save,
    )
    denoiser.model = application.model
    adapter.trainable_module = application.model
    return application


def save_peft_adapter(
    application: PeftLoraApplication,
    output_dir: str | Path,
    *,
    metadata: Mapping[str, object] | None = None,
    model_state_dict: Mapping[str, object] | None = None,
) -> PeftAdapterArtifact:
    """Atomically save one audited PEFT adapter and content digest manifest."""

    if not isinstance(application, PeftLoraApplication):
        raise TypeError("save_peft_adapter requires a PeftLoraApplication")
    save_pretrained = getattr(application.model, "save_pretrained", None)
    if not callable(save_pretrained):
        raise TypeError("PEFT adapter model must expose save_pretrained")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"PEFT adapter output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}-incomplete-",
            dir=destination.parent,
        )
    )
    try:
        save_pretrained(
            str(temporary),
            safe_serialization=True,
            state_dict=model_state_dict,
        )
        file_digests = _artifact_files(temporary)
        manifest = {
            "schema": PEFT_ADAPTER_ARTIFACT_SCHEMA,
            "format": "peft",
            "target_audit": application.target_audit.to_dict(),
            "trainable_parameter_names": list(application.trainable_parameter_names),
            "trainable_parameter_count": application.trainable_parameter_count,
            "files": file_digests,
            "metadata": _json_metadata(metadata),
        }
        manifest_payload = _canonical_json(manifest) + "\n"
        write_exclusive_text(
            temporary / "worldfoundry_adapter.json",
            manifest_payload,
            root=temporary,
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return inspect_peft_adapter(destination)


def inspect_peft_adapter(input_dir: str | Path) -> PeftAdapterArtifact:
    """Validate the WorldFoundry manifest and every file digest in an adapter."""

    source = Path(input_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"PEFT adapter directory does not exist: {source}")
    manifest_path = source / "worldfoundry_adapter.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(f"PEFT adapter manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid PEFT adapter manifest: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise TypeError("PEFT adapter manifest must contain one JSON object")
    expected_fields = {
        "schema",
        "format",
        "target_audit",
        "trainable_parameter_names",
        "trainable_parameter_count",
        "files",
        "metadata",
    }
    if set(manifest) != expected_fields:
        raise ValueError(
            "PEFT adapter manifest fields differ: "
            f"missing={sorted(expected_fields - set(manifest))}, "
            f"unknown={sorted(set(manifest) - expected_fields)}"
        )
    if manifest["schema"] != PEFT_ADAPTER_ARTIFACT_SCHEMA or manifest["format"] != "peft":
        raise ValueError(
            f"unsupported PEFT adapter artifact: schema={manifest['schema']!r}, format={manifest['format']!r}"
        )
    if not isinstance(manifest["target_audit"], dict):
        raise TypeError("PEFT adapter target_audit must be an object")
    trainable_names = manifest["trainable_parameter_names"]
    if (
        not isinstance(trainable_names, list)
        or not trainable_names
        or any(not isinstance(name, str) or not name for name in trainable_names)
        or len(trainable_names) != len(set(trainable_names))
    ):
        raise ValueError("PEFT adapter trainable_parameter_names must be unique non-empty strings")
    trainable_count = manifest["trainable_parameter_count"]
    if isinstance(trainable_count, bool) or not isinstance(trainable_count, int) or trainable_count <= 0:
        raise ValueError("PEFT adapter trainable_parameter_count must be a positive integer")
    expected_digests = manifest["files"]
    if not isinstance(expected_digests, dict) or not expected_digests:
        raise TypeError("PEFT adapter manifest files must be a non-empty object")
    normalized_digests: dict[str, str] = {}
    for name, digest in expected_digests.items():
        relative = Path(str(name))
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != str(name):
            raise ValueError(f"unsafe PEFT adapter artifact path: {name!r}")
        normalized_digest = str(digest).lower()
        if _SHA256_PATTERN.fullmatch(normalized_digest) is None:
            raise ValueError(f"invalid PEFT adapter SHA-256 for {name!r}")
        normalized_digests[str(name)] = normalized_digest
    actual_digests = _artifact_files(source)
    if actual_digests != normalized_digests:
        missing = sorted(set(normalized_digests) - set(actual_digests))
        unexpected = sorted(set(actual_digests) - set(normalized_digests))
        changed = sorted(
            name
            for name in set(actual_digests) & set(normalized_digests)
            if actual_digests[name] != normalized_digests[name]
        )
        raise ValueError(
            f"PEFT adapter digest audit failed: missing={missing}, unexpected={unexpected}, changed={changed}"
        )
    metadata = manifest["metadata"]
    if not isinstance(metadata, dict):
        raise TypeError("PEFT adapter manifest metadata must be an object")
    return PeftAdapterArtifact(
        path=source,
        manifest_sha256=_file_sha256(manifest_path),
        file_digests=normalized_digests,
        metadata=metadata,
    )


def _read_adapter_json(path: Path, *, name: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid PEFT adapter {name}: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"PEFT adapter {name} must contain one JSON object")
    return payload


def _audit_peft_base_compatibility(
    base_model: nn.Module,
    source: Path,
    *,
    expected_preset: str,
    expected_base_model_id: str | None,
) -> None:
    normalized_preset = str(expected_preset).strip().lower().replace("_", "-")
    if not normalized_preset:
        raise ValueError("expected_preset must be a non-empty string")
    manifest = _read_adapter_json(
        source / "worldfoundry_adapter.json",
        name="manifest",
    )
    target_audit = manifest.get("target_audit")
    if not isinstance(target_audit, Mapping):
        raise TypeError("PEFT adapter target_audit must be an object")
    manifest_preset = target_audit.get("preset")
    if manifest_preset != normalized_preset:
        raise ValueError(
            "PEFT adapter target preset differs from the requested base model: "
            f"adapter={manifest_preset!r}, expected={normalized_preset!r}"
        )

    base_audit = audit_lora_targets(base_model, normalized_preset)
    expected_target_audit = base_audit.to_dict()
    if dict(target_audit) != expected_target_audit:
        changed = sorted(
            key
            for key in set(target_audit) | set(expected_target_audit)
            if target_audit.get(key) != expected_target_audit.get(key)
        )
        raise ValueError(
            "PEFT adapter target audit is incompatible with the loaded base model: "
            f"changed={changed}"
        )

    config = _read_adapter_json(source / "adapter_config.json", name="config")
    if config.get("peft_type") != "LORA":
        raise ValueError(f"unsupported PEFT adapter type: {config.get('peft_type')!r}")
    if config.get("target_modules") != base_audit.target_pattern:
        raise ValueError("PEFT adapter config target_modules differs from the audited base graph")
    rank = config.get("r")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("PEFT adapter config r must be a positive integer")
    alpha = config.get("lora_alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not isfinite(float(alpha)) or alpha <= 0:
        raise ValueError("PEFT adapter config lora_alpha must be finite and positive")
    if config.get("bias") != "none":
        raise ValueError("WorldFoundry PEFT adapters require bias='none'")

    modules_to_save = config.get("modules_to_save")
    if modules_to_save is not None and (
        not isinstance(modules_to_save, list)
        or not modules_to_save
        or any(not isinstance(name, str) or not name for name in modules_to_save)
        or len(modules_to_save) != len(set(modules_to_save))
    ):
        raise ValueError("PEFT adapter config modules_to_save must be null or unique non-empty strings")
    if modules_to_save is None:
        trainable_count = manifest.get("trainable_parameter_count")
        expected_count = base_audit.expected_trainable_parameters(rank)
        if trainable_count != expected_count:
            raise ValueError(
                "PEFT adapter trainable parameter count differs from its rank and audited base graph: "
                f"adapter={trainable_count!r}, expected={expected_count}"
            )

    auto_mapping = config.get("auto_mapping")
    if auto_mapping is not None:
        if not isinstance(auto_mapping, Mapping):
            raise TypeError("PEFT adapter config auto_mapping must be null or an object")
        configured_class = auto_mapping.get("base_model_class")
        if configured_class not in (None, "", type(base_model).__name__):
            raise ValueError(
                "PEFT adapter base model class differs from the loaded model: "
                f"adapter={configured_class!r}, loaded={type(base_model).__name__!r}"
            )
    configured_base_id = config.get("base_model_name_or_path")
    if configured_base_id not in (None, "") and expected_base_model_id is not None:
        if str(configured_base_id) != expected_base_model_id:
            raise ValueError(
                "PEFT adapter base_model_name_or_path differs from the selected model: "
                f"adapter={configured_base_id!r}, selected={expected_base_model_id!r}"
            )


def load_peft_adapter(
    base_model: nn.Module,
    input_dir: str | Path,
    *,
    is_trainable: bool = False,
    expected_preset: str | None = None,
    expected_base_model_id: str | None = None,
) -> nn.Module:
    """Load a digest-verified local adapter onto a caller-supplied base model."""

    if not isinstance(base_model, nn.Module):
        raise TypeError("PEFT base model must be an nn.Module")
    if not isinstance(is_trainable, bool):
        raise TypeError("is_trainable must be a bool")
    artifact = inspect_peft_adapter(input_dir)
    if expected_base_model_id is not None:
        expected_base_model_id = str(expected_base_model_id).strip()
        if not expected_base_model_id:
            raise ValueError("expected_base_model_id must be a non-empty string")
        if expected_preset is None:
            raise ValueError("expected_base_model_id requires expected_preset")
    if expected_preset is not None:
        _audit_peft_base_compatibility(
            base_model,
            artifact.path,
            expected_preset=expected_preset,
            expected_base_model_id=expected_base_model_id,
        )
    try:
        from peft import PeftModel
    except ModuleNotFoundError as error:
        raise RuntimeError("loading a LoRA adapter requires the 'train-core' PEFT dependency") from error
    model = PeftModel.from_pretrained(
        base_model,
        str(artifact.path),
        is_trainable=is_trainable,
    )
    if not isinstance(model, nn.Module):
        raise TypeError(f"PEFT returned {type(model).__name__}, expected nn.Module")
    return model


def merge_peft_adapter(model: nn.Module) -> nn.Module:
    """Safely merge a loaded LoRA adapter into its base model."""

    if not isinstance(model, nn.Module):
        raise TypeError("PEFT adapter model must be an nn.Module")
    merge = getattr(model, "merge_and_unload", None)
    if not callable(merge):
        raise TypeError("PEFT adapter model must expose merge_and_unload")
    get_base_model = getattr(model, "get_base_model", None)
    base_model = get_base_model() if callable(get_base_model) else model
    had_config = hasattr(base_model, "config")
    original_config = getattr(base_model, "config", None)
    injected_empty_config = original_config is None
    if injected_empty_config:
        # PEFT 0.20 assumes an existing config mapping while checking tied
        # embeddings during merge.  Custom diffusion nn.Modules may either
        # omit config or explicitly set it to None.
        base_model.config = {}
    try:
        merged = merge(safe_merge=True)
    except Exception:
        if injected_empty_config:
            if had_config:
                base_model.config = original_config
            else:
                delattr(base_model, "config")
        raise
    if not isinstance(merged, nn.Module):
        raise TypeError(f"PEFT merge returned {type(merged).__name__}, expected nn.Module")
    if injected_empty_config:
        if had_config:
            merged.config = original_config
        elif hasattr(merged, "config"):
            delattr(merged, "config")
    residual = tuple(name for name, _ in merged.named_parameters() if "lora_" in name.lower())
    if residual:
        raise RuntimeError(f"merged model still contains LoRA parameters: {residual}")
    return merged


__all__ = [
    "LoraTargetAudit",
    "PEFT_ADAPTER_ARTIFACT_SCHEMA",
    "PeftAdapterArtifact",
    "PeftLoraApplication",
    "SANA_ATTENTION",
    "WAN_ATTENTION",
    "apply_peft_lora",
    "apply_peft_lora_to_adapter",
    "audit_lora_targets",
    "inspect_peft_adapter",
    "load_peft_adapter",
    "merge_peft_adapter",
    "save_peft_adapter",
]
