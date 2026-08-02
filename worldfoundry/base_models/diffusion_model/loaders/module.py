"""Native PyTorch module loading shared by every diffusion family."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import torch

from ..optimizations import OffloadMode, QuantizationMode, RuntimePolicy
from .checkpoints import CheckpointSpec
from .materialize import MaterializedCheckpoint, NativeCheckpointResolver

StateDictConverter = Callable[[Mapping[str, object]], Mapping[str, object]]
CheckpointConfigResolver = Callable[[MaterializedCheckpoint], Mapping[str, object]]
PostLoadHook = Callable[[torch.nn.Module], None]


@dataclass(frozen=True, slots=True)
class ModuleLoadSpec:
    """Construction and weight-mapping rules for one PyTorch component."""

    module_class: type[torch.nn.Module]
    config: Mapping[str, object] = field(default_factory=dict)
    config_resolver: CheckpointConfigResolver | None = None
    state_dict_converter: StateDictConverter | None = None
    vram_module_map: Mapping[type[torch.nn.Module], type[torch.nn.Module]] | None = None
    layer_container: str | None = None
    vram_limit_gib: float | None = None
    post_load_hook: PostLoadHook | None = None


class NativeModuleLoader:
    """Thin diffusion-facing adapter over WorldFoundry's shared model loader."""

    def load(
        self,
        spec: ModuleLoadSpec,
        checkpoint: CheckpointSpec,
        policy: RuntimePolicy,
    ) -> torch.nn.Module:
        materialized = NativeCheckpointResolver().materialize(checkpoint)
        sources = tuple(str(path) for path in materialized.paths)
        config = dict(spec.config)
        if spec.config_resolver is not None:
            resolved_config = dict(spec.config_resolver(materialized))
            overlap = sorted(set(config) & set(resolved_config))
            if overlap:
                raise ValueError(f"static and checkpoint-derived module config overlap: {overlap}")
            config.update(resolved_config)

        if policy.quantization.mode is not QuantizationMode.NONE:
            raise NotImplementedError(
                "quantized module construction requires a registered core "
                f"quantization pass; requested {policy.quantization.mode.value}"
            )

        # Lazy imports keep the canonical package importable without optional
        # Transformers/VRAM dependencies until a real module is constructed.
        from worldfoundry.core.model_loading import (
            load_model,
            load_model_with_disk_offload,
        )

        source: str | list[str]
        source = sources[0] if len(sources) == 1 else list(sources)
        if policy.offload.mode is OffloadMode.DISK:
            if spec.post_load_hook is not None:
                raise ValueError("post-load mutations are not supported with disk offload")
            if spec.vram_module_map is None:
                raise ValueError("disk offload requires ModuleLoadSpec.vram_module_map")
            module = load_model_with_disk_offload(
                spec.module_class,
                source,
                config=config,
                torch_dtype=policy.dtype,
                device=policy.device,
                state_dict_converter=spec.state_dict_converter,
                module_map=dict(spec.vram_module_map),
            )
        else:
            vram_config = None
            module_map = None
            if spec.vram_module_map is not None and policy.offload.mode in {
                OffloadMode.COMPONENT,
                OffloadMode.BLOCK,
            }:
                module_map = dict(spec.vram_module_map)
                vram_config = {
                    "offload_dtype": policy.dtype,
                    "offload_device": policy.offload.target,
                    "onload_dtype": policy.dtype,
                    "onload_device": policy.device,
                    "preparing_dtype": policy.dtype,
                    "preparing_device": policy.device,
                    "computation_dtype": policy.dtype,
                    "computation_device": policy.device,
                }
            module = load_model(
                spec.module_class,
                source,
                config=config,
                torch_dtype=policy.dtype,
                device=policy.device,
                state_dict_converter=spec.state_dict_converter,
                module_map=module_map,
                vram_config=vram_config,
                vram_limit=spec.vram_limit_gib,
            )
            # Adapter fusion and other in-place mutations must run while the
            # checkpoint weights are still materialized. Layerwise CPU
            # offload replaces inactive parameters with zero-sized CUDA
            # placeholders, so applying a LoRA after installing its hooks
            # corrupts shape-based traversal and fails during the merge.
            if spec.post_load_hook is not None:
                spec.post_load_hook(module)
            if policy.offload.mode is OffloadMode.BLOCK and module_map is None:
                from worldfoundry.core.vram import enable_layerwise_cpu_offload

                enable_layerwise_cpu_offload(
                    module,
                    layer_container=spec.layer_container,
                    device=policy.device,
                    pin_memory=policy.offload.pin_memory,
                )
            elif policy.offload.mode is OffloadMode.COMPONENT and module_map is None:
                raise ValueError("component offload requires ModuleLoadSpec.vram_module_map")

        if policy.compile:
            module = torch.compile(module)
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"module loader returned {type(module).__name__}, expected torch.nn.Module")
        return module


__all__ = [
    "CheckpointConfigResolver",
    "ModuleLoadSpec",
    "NativeModuleLoader",
    "PostLoadHook",
    "StateDictConverter",
]
