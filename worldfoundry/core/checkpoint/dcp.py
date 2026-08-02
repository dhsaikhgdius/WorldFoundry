# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared inference-only helpers for loading PyTorch distributed checkpoints."""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.checkpoint.stateful import Stateful

from worldfoundry.core.distributed.logging import log
from worldfoundry.core.io.s3_filesystem import S3StorageReader

try:
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner as _DefaultLoadPlanner
    from torch.distributed.checkpoint.default_planner import (
        DTensor,
        LoadPlan,
        _create_read_items,
        _version,
        flatten_state_dict,
    )
    from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata

    def _create_local_load_plan(
        state_dict: dict[str, Any],
        metadata: Metadata,
        *,
        strict: bool,
        allow_mismatched_size: bool,
    ) -> LoadPlan:
        requests = []
        for name, value in state_dict.items():
            if name.endswith("._extra_state"):
                continue
            if name not in metadata.state_dict_metadata:
                if strict:
                    raise RuntimeError(f"Missing key in checkpoint state_dict: {name}.")
                continue

            saved = metadata.state_dict_metadata[name]
            if (
                not allow_mismatched_size
                and isinstance(saved, TensorStorageMetadata)
                and getattr(value, "size", None) is not None
                and saved.size != value.size()
            ):
                if strict:
                    raise ValueError(f"Size mismatch for {name}: saved {saved.size}, current {value.size()}")
                log.warning("Skipping mismatched checkpoint tensor {}", name)
                continue

            if not isinstance(value, DTensor) or value.device_mesh.get_coordinate() is not None:
                requests += _create_read_items(name, saved, value)
        return LoadPlan(requests)

    class DefaultLoadPlanner(_DefaultLoadPlanner):
        """Load planner compatible with old DCP key layouts and partial loads."""

        def set_partial_channel_weight(self, allow_mismatched_size: bool) -> None:
            self.allow_mismatched_size = allow_mismatched_size

        def create_local_plan(self) -> LoadPlan:
            if self.metadata is None:
                raise RuntimeError("Checkpoint metadata has not been initialized")
            if self.flatten_state_dict:
                current_keys = set(self.state_dict)
                missing_keys = set(self.metadata.state_dict_metadata) - current_keys
                if missing_keys:
                    _version._derived_version = "2_3"
                    try:
                        old_state_dict, old_mappings = flatten_state_dict(self.original_state_dict)
                        if set(old_state_dict) & missing_keys:
                            self.state_dict, self.mappings = old_state_dict, old_mappings
                    finally:
                        _version._derived_version = None
            return _create_local_load_plan(
                self.state_dict,
                self.metadata,
                strict=not self.allow_partial_load,
                allow_mismatched_size=getattr(self, "allow_mismatched_size", False),
            )

except ImportError:
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner


def dcp_load_state_dict(state_dict: dict[str, Any], storage_reader, load_planner) -> None:
    """Load a DCP state dictionary and reject tensor shape mismatches."""

    dcp.load(state_dict, storage_reader=storage_reader, planner=load_planner)
    if getattr(load_planner, "metadata", None) is not None:
        checkpoint_keys = set(load_planner.metadata.state_dict_metadata)
        model_keys = set(state_dict)
        missing = sorted(model_keys - checkpoint_keys)
        unexpected = sorted(key for key in checkpoint_keys - model_keys if "_extra_state" not in key)
        if missing:
            log.warning("Checkpoint is missing {} model keys", len(missing))
        if unexpected:
            log.warning("Checkpoint has {} unexpected keys", len(unexpected))

    metadata = storage_reader.read_metadata().state_dict_metadata
    mismatches = []
    for key, tensor in state_dict.items():
        if key.endswith("_extra_state") or key not in metadata or not hasattr(tensor, "shape"):
            continue
        checkpoint_shape = torch.Size(metadata[key].size)
        if tensor.shape != checkpoint_shape:
            mismatches.append(f"{key}: model {tuple(tensor.shape)}, checkpoint {tuple(checkpoint_shape)}")
    if mismatches:
        raise RuntimeError("Checkpoint tensor shape mismatches:\n" + "\n".join(mismatches))


class ModelWrapper(Stateful):
    """Adapt one or more modules to PyTorch DCP's ``Stateful`` protocol."""

    def __init__(self, model: nn.Module | Sequence[nn.Module], load_ema_to_reg: bool = False) -> None:
        self.models = [model] if isinstance(model, nn.Module) else list(model)
        self.load_ema_to_reg = load_ema_to_reg
        self.checkpoint_to_model_key: dict[str, str] = {}

    def state_dict(self, mapping_keys: Mapping[str, str] | None = None) -> dict[str, Any]:
        state_dict = {key: value for model in self.models for key, value in get_model_state_dict(model).items()}
        if self.load_ema_to_reg:
            state_dict = {key.replace("net.", "net_ema.", 1): value for key, value in state_dict.items()}

        config = getattr(self.models[0], "config", None)
        if getattr(config, "use_lora", False):
            replacements = {"base_layer.": "", "base_model.model.": "", **(mapping_keys or {})}
            for old_key in list(state_dict):
                new_key = old_key
                for source, target in replacements.items():
                    new_key = new_key.replace(source, target)
                if new_key != old_key:
                    self.checkpoint_to_model_key[new_key] = old_key
                    state_dict[new_key] = state_dict.pop(old_key)
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state_dict = dict(state_dict)
        for checkpoint_key, model_key in self.checkpoint_to_model_key.items():
            if checkpoint_key in state_dict:
                state_dict[model_key] = state_dict.pop(checkpoint_key)
        if self.load_ema_to_reg:
            state_dict = {key.replace("net_ema.", "net.", 1): value for key, value in state_dict.items()}

        setter = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(setter, self.models))


class DistributedCheckpointer:
    """Compatibility facade that exposes only inference-time checkpoint loading."""

    def __init__(self, config_checkpoint, config_job=None, callbacks=None, disable_async: bool = True) -> None:
        del config_job, callbacks, disable_async
        self.config_checkpoint = config_checkpoint

    def get_storage_reader(self, checkpoint_path: str) -> S3StorageReader | FileSystemReader:
        object_store = getattr(self.config_checkpoint, "load_from_object_store", None)
        if object_store is not None and object_store.enabled:
            return S3StorageReader(credential_path=object_store.credentials, path=checkpoint_path)
        return FileSystemReader(checkpoint_path)

    def load(self, model: nn.Module) -> int:
        """Load the model entry from the configured checkpoint path."""

        checkpoint_path = getattr(self.config_checkpoint, "load_path", "")
        if not checkpoint_path:
            raise ValueError("checkpoint.load_path is required for DCP loading")
        wrapper = ModelWrapper(model)
        state_dict = wrapper.state_dict()
        planner = DefaultLoadPlanner(allow_partial_load=True)
        planner.set_partial_channel_weight(
            getattr(self.config_checkpoint, "dcp_allow_mismatched_size", False)
        ) if hasattr(planner, "set_partial_channel_weight") else None
        dcp_load_state_dict(state_dict, self.get_storage_reader(checkpoint_path), planner)
        wrapper.load_state_dict(state_dict)
        return 0


__all__ = ["DefaultLoadPlanner", "DistributedCheckpointer", "ModelWrapper", "dcp_load_state_dict"]
