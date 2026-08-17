"""PEFT LoRA injection for custom WorldFoundry model graphs."""

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
from types import MappingProxyType, SimpleNamespace

from torch import nn

SANA_ATTENTION = "sana-attention"
WAN_ATTENTION = "wan-attention"
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
    file_size_bytes: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        sizes = {str(name): int(size) for name, size in self.file_size_bytes.items()}
        object.__setattr__(self, "file_size_bytes", MappingProxyType(sizes))


def _artifact_files(path: Path) -> dict[str, int]:
    files: dict[str, int] = {}
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file():
            relative = candidate.relative_to(path).as_posix()
            files[relative] = candidate.stat().st_size
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

    return apply_peft_lora_with_audit(
        model,
        audit=audit_lora_targets(model, preset),
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        modules_to_save=modules_to_save,
    )


def apply_peft_lora_with_audit(
    model: nn.Module,
    *,
    audit: LoraTargetAudit,
    rank: int,
    alpha: int,
    dropout: float = 0.0,
    modules_to_save: Sequence[str] = (),
) -> PeftLoraApplication:
    """Inject LoRA from a model-family audit of its actual module graph."""

    if not isinstance(model, nn.Module):
        raise TypeError("LoRA target model must be an nn.Module")
    if not isinstance(audit, LoraTargetAudit):
        raise TypeError("audit must be a LoraTargetAudit")
    if audit.base_parameter_count != sum(parameter.numel() for parameter in model.parameters()):
        raise ValueError("LoRA target audit was created for a different model graph")
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
    model_state_dict: Mapping[str, object] | None = None,
) -> PeftAdapterArtifact:
    """Atomically save one standard PEFT adapter."""

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
    base_model = application.model.get_base_model()
    native_config = getattr(base_model, "config", None)
    replace_native_config = isinstance(native_config, SimpleNamespace)
    try:
        if replace_native_config:
            base_model.config = vars(native_config)
        save_pretrained(
            str(temporary),
            safe_serialization=True,
            state_dict=model_state_dict,
        )
        _artifact_files(temporary)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        if replace_native_config:
            base_model.config = native_config
    return inspect_peft_adapter(destination)


def inspect_peft_adapter(input_dir: str | Path) -> PeftAdapterArtifact:
    """Inspect a standard local PEFT adapter."""

    source = Path(input_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"PEFT adapter directory does not exist: {source}")
    config = _read_adapter_json(source / "adapter_config.json", name="config")
    if config.get("peft_type") != "LORA":
        raise ValueError(f"unsupported PEFT adapter type: {config.get('peft_type')!r}")
    file_sizes = _artifact_files(source)
    return PeftAdapterArtifact(
        path=source,
        file_size_bytes=file_sizes,
    )


def _read_adapter_json(path: Path, *, name: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid PEFT adapter {name}: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"PEFT adapter {name} must contain one JSON object")
    return payload


def _adapter_target_names(source: Path) -> set[str]:
    try:
        from safetensors import safe_open
    except ModuleNotFoundError as error:
        raise RuntimeError("loading a PEFT adapter requires Safetensors") from error
    targets: set[str] = set()
    with safe_open(str(source / "adapter_model.safetensors"), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            name = key.removeprefix("base_model.model.")
            for suffix in (".lora_A.weight", ".lora_B.weight"):
                if name.endswith(suffix):
                    targets.add(name.removesuffix(suffix))
    return targets


def _validate_peft_base_config(
    base_model: nn.Module,
    source: Path,
    *,
    expected_preset: str,
    expected_base_model_id: str | None,
) -> LoraTargetAudit:
    normalized_preset = str(expected_preset).strip().lower().replace("_", "-")
    if not normalized_preset:
        raise ValueError("expected_preset must be a non-empty string")
    base_audit = audit_lora_targets(base_model, normalized_preset)
    if _adapter_target_names(source) != set(base_audit.module_names):
        raise ValueError("PEFT adapter targets are incompatible with the loaded base model")
    config = _read_adapter_json(source / "adapter_config.json", name="config")
    if config.get("peft_type") != "LORA":
        raise ValueError(f"unsupported PEFT adapter type: {config.get('peft_type')!r}")
    target_modules = config.get("target_modules")
    if not target_modules or not isinstance(target_modules, (str, list)):
        raise ValueError("PEFT adapter config must declare target_modules")
    rank = config.get("r")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("PEFT adapter config r must be a positive integer")
    alpha = config.get("lora_alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not isfinite(float(alpha)) or alpha <= 0:
        raise ValueError("PEFT adapter config lora_alpha must be finite and positive")
    if config.get("bias") != "none":
        raise ValueError("WorldFoundry PEFT adapters require bias='none'")

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
    return base_audit


def load_peft_adapter(
    base_model: nn.Module,
    input_dir: str | Path,
    *,
    is_trainable: bool = False,
    expected_preset: str | None = None,
    expected_base_model_id: str | None = None,
) -> nn.Module:
    """Load a local adapter onto a caller-supplied base model."""

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
    target_audit = None
    if expected_preset is not None:
        target_audit = _validate_peft_base_config(
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
    if target_audit is not None:
        targeted = set(getattr(model, "targeted_module_names", ()))
        if targeted != set(target_audit.module_names):
            raise ValueError("PEFT adapter targets are incompatible with the loaded model")
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
    "PeftAdapterArtifact",
    "PeftLoraApplication",
    "SANA_ATTENTION",
    "WAN_ATTENTION",
    "apply_peft_lora",
    "apply_peft_lora_with_audit",
    "apply_peft_lora_to_adapter",
    "audit_lora_targets",
    "inspect_peft_adapter",
    "load_peft_adapter",
    "merge_peft_adapter",
    "save_peft_adapter",
]
