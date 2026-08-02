import json
import os
from collections.abc import Mapping

import torch
from safetensors import safe_open

from worldfoundry.core.model_loading import load_torch_state_dict
from worldfoundry.core.io.storage import read_text_uri


def _expand_safetensors_indexes(paths):
    """Expand sharded Safetensors indexes into unique shard paths."""

    expanded = []
    seen = set()
    for raw_path in paths:
        path = str(raw_path)
        if path.endswith(".safetensors.index.json"):
            index = json.loads(read_text_uri(path))
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict):
                raise ValueError(f"Safetensors index {path!r} does not contain a weight_map")
            root = path.rsplit("/", 1)[0]
            candidates = [f"{root}/{name}" for name in sorted(set(weight_map.values()))]
        else:
            candidates = [path]
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                expanded.append(candidate)
    return expanded


class SafetensorsCompatibleTensor:
    def __init__(self, tensor):
        self.tensor = tensor

    def get_shape(self):
        return list(self.tensor.shape)


class SafetensorsCompatibleBinaryLoader:
    def __init__(self, path, device):
        print(
            "Detected non-safetensors files, which may cause slower loading. It's recommended to convert it to a safetensors file."
        )
        self.state_dict = load_torch_state_dict(path, map_location=device)

    def keys(self):
        return self.state_dict.keys()

    def get_tensor(self, name):
        return self.state_dict[name]

    def get_slice(self, name):
        return SafetensorsCompatibleTensor(self.state_dict[name])


class DiskMap(Mapping):
    """Lazy mapping from checkpoint parameter names to materialized tensors.

    Safetensors files remain memory-mapped and individual weights are loaded on
    lookup. PyTorch binary checkpoints use an in-memory compatibility loader.
    The mapping may apply a state-dict converter and periodically reopen files
    after ``buffer_size`` tensor elements have been materialized.
    """

    def __init__(self, path, device, torch_dtype=None, state_dict_converter=None, buffer_size=10**9):
        """Index one or more checkpoint files.

        Args:
            path: Checkpoint path or list of paths.
            device: Device on which fetched tensors are materialized.
            torch_dtype: Optional dtype conversion applied on lookup.
            state_dict_converter: Optional callable that remaps public model
                keys to keys stored in the checkpoint.
            buffer_size: Number of fetched tensor elements after which file
                handles are refreshed.
        """
        paths = path if isinstance(path, list) else [path]
        self.path = _expand_safetensors_indexes(paths)
        self.device = device
        self.torch_dtype = torch_dtype
        if os.environ.get("WORLDFOUNDRY_DISK_MAP_BUFFER_SIZE") is not None:
            self.buffer_size = int(os.environ["WORLDFOUNDRY_DISK_MAP_BUFFER_SIZE"])
        else:
            self.buffer_size = buffer_size
        self.files = []
        self.flush_files()
        self.name_map = {}
        for file_id, file in enumerate(self.files):
            for name in file.keys():
                self.name_map[name] = file_id
        self.rename_dict = self.fetch_rename_dict(state_dict_converter)

    def flush_files(self):
        """Open or refresh checkpoint readers and reset the materialized-element count."""
        if len(self.files) == 0:
            for path in self.path:
                if path.endswith(".safetensors"):
                    self.files.append(safe_open(path, framework="pt", device=str(self.device)))
                else:
                    self.files.append(SafetensorsCompatibleBinaryLoader(path, device=self.device))
        else:
            for i, path in enumerate(self.path):
                if path.endswith(".safetensors"):
                    self.files[i] = safe_open(path, framework="pt", device=str(self.device))
        self.num_params = 0

    def __getitem__(self, name):
        if self.rename_dict is not None:
            name = self.rename_dict[name]
        file_id = self.name_map[name]
        param = self.files[file_id].get_tensor(name)
        if self.torch_dtype is not None and isinstance(param, torch.Tensor):
            param = param.to(self.torch_dtype)
        if isinstance(param, torch.Tensor) and param.device == "cpu":
            param = param.clone()
        if isinstance(param, torch.Tensor):
            self.num_params += param.numel()
        if self.num_params > self.buffer_size:
            self.flush_files()
        return param

    def fetch_rename_dict(self, state_dict_converter):
        """Build the optional model-key to checkpoint-key mapping."""
        if state_dict_converter is None:
            return None
        state_dict = {}
        for file in self.files:
            for name in file.keys():
                state_dict[name] = name
        state_dict = state_dict_converter(state_dict)
        return state_dict

    def __iter__(self):
        if self.rename_dict is not None:
            return self.rename_dict.__iter__()
        else:
            return self.name_map.__iter__()

    def __len__(self):
        if self.rename_dict is not None:
            return len(self.rename_dict)
        return len(self.name_map)

    def __contains__(self, x):
        if self.rename_dict is not None:
            return x in self.rename_dict
        else:
            return x in self.name_map


__all__ = ["DiskMap", "SafetensorsCompatibleBinaryLoader", "SafetensorsCompatibleTensor"]
