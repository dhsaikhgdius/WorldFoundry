"""Materialize native AnyFlow model roles from immutable checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Literal

import torch
from packaging.version import Version
from torch import nn

from worldfoundry.base_models.diffusion_model.loaders import (
    CheckpointSpec,
    MaterializedCheckpoint,
    ModuleLoadSpec,
    NativeModuleLoader,
    checkpoint_json_config,
)
from worldfoundry.base_models.diffusion_model.optimizations import (
    AttentionBackend,
    OffloadMode,
    QuantizationMode,
    RuntimePolicy,
)
from worldfoundry.core.attention.chunk_partition import TemporalChunkPartition
from worldfoundry.training.post_training.distillation.anyflow.adapters import (
    NativeAnyFlowBidirectionalAdapter,
    NativeAnyFlowFARAdapter,
    NativeAnyFlowScoreAdapter,
)
from worldfoundry.training.post_training.shared.role_checkpoints import (
    ResolvedRoleCheckpoint,
)

ANYFLOW_FAR_WAN_SMALL_CHECKPOINT = CheckpointSpec(
    repo_id="nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers",
    revision="915af337434035df8545797ecc910d79fa78cf29",
    files=("transformer/diffusion_pytorch_model.safetensors",),
    allow_patterns=(
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
    ),
    file_sha256={
        "transformer/diffusion_pytorch_model.safetensors": (
            "cb52e9899ac7ec0cbe4687594b950600aa58f2f72588c8d16bcb5f8035a90f75"
        ),
    },
    file_size_bytes={
        "transformer/diffusion_pytorch_model.safetensors": 2_844_379_104,
    },
    resource_sha256={
        "transformer/config.json": (
            "a6ad22169dc90bbf6c38621d64e7acf838bb6ed1a8ccf3b78ee15ce5c12b4caf"
        ),
    },
    resource_size_bytes={"transformer/config.json": 671},
)

ANYFLOW_BIDIRECTIONAL_WAN_SMALL_CHECKPOINT = CheckpointSpec(
    repo_id="nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers",
    revision="4c2ec05c7fa4dbafbca131ad32430905c7ff2974",
    files=("transformer/diffusion_pytorch_model.safetensors",),
    allow_patterns=(
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
    ),
    file_sha256={
        "transformer/diffusion_pytorch_model.safetensors": (
            "2b6fee76a341e425da9f916f78ca0b5376ad4d4eaf8dbb341fb92f960a40ba26"
        ),
    },
    file_size_bytes={
        "transformer/diffusion_pytorch_model.safetensors": 2_843_589_400,
    },
    resource_sha256={
        "transformer/config.json": (
            "a72bc7169c3316888718f0315d5825a49e55a2bda10c7e7eefc5cefd4b4c624e"
        ),
    },
    resource_size_bytes={"transformer/config.json": 499},
)

AnyFlowCheckpoint = CheckpointSpec | ResolvedRoleCheckpoint
AnyFlowArchitecture = Literal["far", "bidirectional"]


def _native_model_class() -> type[nn.Module]:
    """Import the copied model graph only at the materialization boundary."""

    try:
        installed_diffusers = Version(package_version("diffusers"))
    except PackageNotFoundError as error:
        raise RuntimeError(
            "native AnyFlow model materialization requires "
            "diffusers>=0.35.1,<0.40"
        ) from error
    if not Version("0.35.1") <= installed_diffusers < Version("0.40"):
        raise RuntimeError(
            "native AnyFlow model materialization requires "
            "diffusers>=0.35.1,<0.40; "
            f"installed {installed_diffusers}"
        )
    module_name = (
        "worldfoundry.base_models.diffusion_model.models.networks.wan."
        "variants.anyflow"
    )
    try:
        module = import_module(module_name)
    except ImportError as error:
        missing = getattr(error, "name", None) or "an AnyFlow model dependency"
        raise RuntimeError(
            "native AnyFlow model materialization requires torch>=2.7, "
            "diffusers>=0.35.1,<0.40, and einops>=0.8; "
            f"failed while importing {missing!r}"
        ) from error
    model_class = getattr(module, "AnyFlowWanTransformer3DModel", None)
    if not isinstance(model_class, type) or not issubclass(model_class, nn.Module):
        raise RuntimeError("native AnyFlow model class is unavailable")
    return model_class


def _checkpoint_and_identity(
    checkpoint: AnyFlowCheckpoint,
    checkpoint_identity: str | None,
) -> tuple[CheckpointSpec, str]:
    if isinstance(checkpoint, ResolvedRoleCheckpoint):
        resolved = checkpoint.checkpoint
        inherited = checkpoint.requested_reference
    elif isinstance(checkpoint, CheckpointSpec):
        resolved = checkpoint
        inherited = None
    else:
        raise TypeError("AnyFlow checkpoint must be CheckpointSpec or ResolvedRoleCheckpoint")

    identity = inherited if checkpoint_identity is None else str(checkpoint_identity).strip()
    if not isinstance(identity, str) or not identity:
        raise ValueError(
            "raw AnyFlow CheckpointSpec materialization requires checkpoint_identity"
        )
    if inherited is not None and checkpoint_identity is not None and identity != inherited:
        raise ValueError(
            "explicit AnyFlow checkpoint_identity differs from the resolved role"
        )
    return resolved, identity


def _config_value(config: Mapping[str, object], name: str) -> object:
    if name not in config:
        raise ValueError(f"AnyFlow checkpoint config must define {name!r}")
    return config[name]


def _tuple3(value: object, *, name: str) -> tuple[int, int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"AnyFlow {name} must contain three integers")
    if any(isinstance(item, bool) or int(item) <= 0 for item in value):
        raise ValueError(f"AnyFlow {name} entries must be positive integers")
    return tuple(int(item) for item in value)


def _audit_flowmap_config(config: Mapping[str, object]) -> None:
    if _config_value(config, "deltatime_type") != "r":
        raise ValueError("AnyFlow checkpoint must use deltatime_type='r'")
    gate_value = _config_value(config, "gate_value")
    if isinstance(gate_value, bool) or float(gate_value) != 0.25:
        raise ValueError("AnyFlow checkpoint must use gate_value=0.25")
    _tuple3(_config_value(config, "patch_size"), name="patch_size")


def _audit_checkpoint_config(
    checkpoint: MaterializedCheckpoint,
    *,
    architecture: AnyFlowArchitecture,
    partition: TemporalChunkPartition | None,
) -> Mapping[str, object]:
    raw = dict(checkpoint_json_config(checkpoint, "transformer/config.json"))
    _audit_flowmap_config(raw)
    declared_class = str(raw.get("_class_name", "")).strip()
    allowed_classes = {
        "AnyFlowFARTransformer3DModel",
        "FAR_Wan_Transformer3DModel",
    }
    if architecture == "bidirectional":
        allowed_classes.add("AnyFlowTransformer3DModel")
    if declared_class not in allowed_classes:
        raise ValueError(
            "AnyFlow checkpoint config declares an incompatible transformer class: "
            f"{declared_class!r}"
        )

    if architecture == "far":
        if not isinstance(partition, TemporalChunkPartition):
            raise TypeError("FAR materialization requires TemporalChunkPartition")
        configured_partition = tuple(
            int(value) for value in _config_value(raw, "chunk_partition")
        )
        if configured_partition != partition.chunks:
            raise ValueError("AnyFlow checkpoint chunk_partition differs from training")
        if int(_config_value(raw, "full_chunk_limit")) != partition.full_chunk_limit:
            raise ValueError("AnyFlow checkpoint full_chunk_limit differs from training")
        if _tuple3(
            _config_value(raw, "patch_size"),
            name="patch_size",
        ) != partition.patch_size:
            raise ValueError("AnyFlow checkpoint patch_size differs from training")
        if _tuple3(
            _config_value(raw, "compressed_patch_size"),
            name="compressed_patch_size",
        ) != partition.compressed_patch_size:
            raise ValueError(
                "AnyFlow checkpoint compressed_patch_size differs from training"
            )
    else:
        for name in ("chunk_partition", "full_chunk_limit", "compressed_patch_size"):
            if raw.get(name) is not None:
                raise ValueError(
                    f"bidirectional AnyFlow checkpoint cannot enable FAR field {name!r}"
                )

    return {name: value for name, value in raw.items() if not name.startswith("_")}


def _module_config_value(module: nn.Module, name: str) -> object:
    config = getattr(module, "config", None)
    if config is None or not hasattr(config, name):
        raise TypeError(f"materialized AnyFlow model config has no {name!r}")
    return getattr(config, name)


def _audit_loaded_module(
    module: nn.Module,
    *,
    architecture: AnyFlowArchitecture,
    partition: TemporalChunkPartition | None,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if _module_config_value(module, "deltatime_type") != "r":
        raise ValueError("materialized AnyFlow model lost destination-time conditioning")
    if float(_module_config_value(module, "gate_value")) != 0.25:
        raise ValueError("materialized AnyFlow model lost the released 0.25 time gate")
    condition_embedder = getattr(module, "condition_embedder", None)
    if not isinstance(
        getattr(condition_embedder, "delta_embedder", None),
        nn.Module,
    ):
        raise TypeError("materialized AnyFlow model has no destination-time embedder")

    if architecture == "far":
        assert isinstance(partition, TemporalChunkPartition)
        if not isinstance(getattr(module, "far_patch_embedding", None), nn.Module):
            raise TypeError("materialized AnyFlow FAR model has no compressed patch embedding")
        if tuple(_module_config_value(module, "chunk_partition")) != partition.chunks:
            raise ValueError("materialized AnyFlow FAR partition differs from training")
        for name in (
            "_forward_train",
            "_forward_cache",
            "_forward_inference",
            "_forward_bidirection",
        ):
            if not callable(getattr(module, name, None)):
                raise TypeError(f"materialized AnyFlow FAR model has no {name} path")
    elif not callable(getattr(module, "_forward_bidirection", None)):
        raise TypeError("materialized bidirectional AnyFlow model has no execution path")

    parameter_dtypes = {
        parameter.dtype
        for parameter in module.parameters()
        if parameter.is_floating_point()
    }
    if parameter_dtypes != {dtype}:
        raise ValueError(
            "materialized AnyFlow parameter dtype differs from runtime policy: "
            f"{sorted(map(str, parameter_dtypes))}"
        )
    parameter_devices = {parameter.device for parameter in module.parameters()}
    device_matches = (
        len(parameter_devices) == 1
        and next(iter(parameter_devices)).type == device.type
        and (
            device.index is None
            or next(iter(parameter_devices)).index == device.index
        )
    )
    if not device_matches:
        raise ValueError(
            "materialized AnyFlow parameter device differs from runtime policy: "
            f"{sorted(map(str, parameter_devices))}"
        )


def _validate_training_policy(policy: RuntimePolicy) -> None:
    if not isinstance(policy, RuntimePolicy):
        raise TypeError("AnyFlow materialization policy must be RuntimePolicy")
    if policy.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("AnyFlow training supports float16, bfloat16, or float32")
    if policy.attention not in {AttentionBackend.AUTO, AttentionBackend.TORCH}:
        raise ValueError("native AnyFlow FAR owns its PyTorch flex-attention kernel")
    if policy.offload.mode is not OffloadMode.NONE:
        raise ValueError("AnyFlow training roles cannot use inference-time weight offload")
    if policy.quantization.mode is not QuantizationMode.NONE:
        raise ValueError("AnyFlow training roles cannot be materialized quantized")
    if policy.compile:
        raise ValueError(
            "compile the AnyFlow role after distributed wrapping, not while loading"
        )
    if policy.options:
        raise ValueError(f"unsupported AnyFlow RuntimePolicy options: {sorted(policy.options)}")


class NativeAnyFlowModelMaterializer:
    """Build independent student and score roles through WorldFoundry core I/O."""

    def __init__(self, loader: NativeModuleLoader | None = None) -> None:
        if loader is not None and not isinstance(loader, NativeModuleLoader):
            raise TypeError("AnyFlow loader must be NativeModuleLoader")
        self.loader = loader or NativeModuleLoader()

    def _load(
        self,
        checkpoint: AnyFlowCheckpoint,
        *,
        checkpoint_identity: str | None,
        architecture: AnyFlowArchitecture,
        partition: TemporalChunkPartition | None,
        policy: RuntimePolicy,
        trainable: bool,
        gradient_checkpointing: bool,
    ) -> tuple[nn.Module, str]:
        _validate_training_policy(policy)
        if not isinstance(trainable, bool) or not isinstance(gradient_checkpointing, bool):
            raise TypeError("AnyFlow role flags must be bool")
        if gradient_checkpointing and not trainable:
            raise ValueError("frozen AnyFlow roles cannot enable gradient checkpointing")
        checkpoint_spec, identity = _checkpoint_and_identity(
            checkpoint,
            checkpoint_identity,
        )
        module_class = _native_model_class()

        def resolve_config(materialized: MaterializedCheckpoint) -> Mapping[str, object]:
            return _audit_checkpoint_config(
                materialized,
                architecture=architecture,
                partition=partition,
            )

        def post_load(module: nn.Module) -> None:
            _audit_loaded_module(
                module,
                architecture=architecture,
                partition=partition,
                dtype=policy.dtype,
                device=policy.device,
            )

        module = self.loader.load(
            ModuleLoadSpec(
                module_class=module_class,
                config_resolver=resolve_config,
                layer_container="blocks",
                post_load_hook=post_load,
            ),
            checkpoint_spec,
            policy,
        )
        if gradient_checkpointing:
            enable = getattr(module, "enable_gradient_checkpointing", None)
            if not callable(enable):
                raise TypeError("AnyFlow model cannot enable gradient checkpointing")
            enable()
            if not bool(getattr(module, "gradient_checkpointing", False)):
                raise RuntimeError("AnyFlow model ignored gradient checkpointing")
        module.requires_grad_(trainable)
        module.train(trainable)
        return module, identity

    def far_student(
        self,
        checkpoint: AnyFlowCheckpoint,
        *,
        checkpoint_identity: str | None = None,
        partition: TemporalChunkPartition,
        policy: RuntimePolicy,
        gradient_checkpointing: bool = False,
    ) -> NativeAnyFlowFARAdapter:
        """Materialize one mutable FAR student with causal and auxiliary paths."""

        module, identity = self._load(
            checkpoint,
            checkpoint_identity=checkpoint_identity,
            architecture="far",
            partition=partition,
            policy=policy,
            trainable=True,
            gradient_checkpointing=gradient_checkpointing,
        )
        adapter = NativeAnyFlowFARAdapter(
            module,
            checkpoint_identity=identity,
        )
        adapter._audit(partition)
        return adapter

    def bidirectional_student(
        self,
        checkpoint: AnyFlowCheckpoint,
        *,
        checkpoint_identity: str | None = None,
        policy: RuntimePolicy,
        gradient_checkpointing: bool = False,
    ) -> NativeAnyFlowBidirectionalAdapter:
        """Materialize one mutable full-video FlowMap student."""

        module, identity = self._load(
            checkpoint,
            checkpoint_identity=checkpoint_identity,
            architecture="bidirectional",
            partition=None,
            policy=policy,
            trainable=True,
            gradient_checkpointing=gradient_checkpointing,
        )
        return NativeAnyFlowBidirectionalAdapter(
            module,
            checkpoint_identity=identity,
        )

    def real_score(
        self,
        checkpoint: AnyFlowCheckpoint,
        *,
        checkpoint_identity: str | None = None,
        policy: RuntimePolicy,
    ) -> NativeAnyFlowScoreAdapter:
        """Materialize one independent frozen bidirectional real-score role."""

        module, identity = self._load(
            checkpoint,
            checkpoint_identity=checkpoint_identity,
            architecture="bidirectional",
            partition=None,
            policy=policy,
            trainable=False,
            gradient_checkpointing=False,
        )
        return NativeAnyFlowScoreAdapter(
            module,
            checkpoint_identity=identity,
        )

    def fake_score(
        self,
        checkpoint: AnyFlowCheckpoint,
        *,
        checkpoint_identity: str | None = None,
        policy: RuntimePolicy,
        gradient_checkpointing: bool = False,
    ) -> NativeAnyFlowScoreAdapter:
        """Materialize one independent mutable bidirectional fake-score role."""

        module, identity = self._load(
            checkpoint,
            checkpoint_identity=checkpoint_identity,
            architecture="bidirectional",
            partition=None,
            policy=policy,
            trainable=True,
            gradient_checkpointing=gradient_checkpointing,
        )
        return NativeAnyFlowScoreAdapter(
            module,
            checkpoint_identity=identity,
        )


__all__ = [
    "ANYFLOW_BIDIRECTIONAL_WAN_SMALL_CHECKPOINT",
    "ANYFLOW_FAR_WAN_SMALL_CHECKPOINT",
    "AnyFlowArchitecture",
    "AnyFlowCheckpoint",
    "NativeAnyFlowModelMaterializer",
]
