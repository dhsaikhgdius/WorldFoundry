"""LTX LoRA application and author-compatible Safetensors export."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from worldfoundry.training.tuning.application import ExportedAdapterArtifact
from worldfoundry.training.tuning.peft import (
    LoraTargetAudit,
    PeftLoraApplication,
)


@dataclass(frozen=True, slots=True)
class LTXLoraApplication:
    """PEFT-backed training state with the artifact format released by LTX."""

    peft: PeftLoraApplication

    @property
    def model(self) -> torch.nn.Module:
        return self.peft.model

    @property
    def target_audit(self) -> LoraTargetAudit:
        return self.peft.target_audit

    @property
    def targeted_module_names(self) -> tuple[str, ...]:
        return self.peft.targeted_module_names

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        return self.peft.trainable_parameter_names

    @property
    def trainable_parameter_count(self) -> int:
        return self.peft.trainable_parameter_count

    def export_adapter(
        self,
        output_dir: str | Path,
        *,
        model_state_dict: Mapping[str, object] | None = None,
    ) -> ExportedAdapterArtifact:
        """Write the single BF16 ``diffusion_model.*`` file used by LTX tools."""

        try:
            from peft import get_peft_model_state_dict
            from safetensors.torch import save_file
        except ModuleNotFoundError as error:
            raise RuntimeError("LTX LoRA export requires PEFT and Safetensors") from error

        adapter_state = get_peft_model_state_dict(
            self.model,
            state_dict=model_state_dict,
        )
        converted: dict[str, torch.Tensor] = {}
        for name, value in adapter_state.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"LTX adapter value {name!r} is not a tensor")
            native_name = name.removeprefix("base_model.model.")
            converted[f"diffusion_model.{native_name}"] = value.detach().to(
                device="cpu",
                dtype=torch.bfloat16,
            )

        destination = Path(output_dir)
        if destination.exists():
            raise FileExistsError(f"LTX adapter output already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-incomplete-", dir=destination.parent))
        filename = "lora_weights.safetensors"
        try:
            save_file(dict(sorted(converted.items())), temporary / filename)
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        artifact_path = destination / filename
        return ExportedAdapterArtifact(
            path=destination,
            file_size_bytes={filename: artifact_path.stat().st_size},
        )


__all__ = ["LTXLoraApplication"]
