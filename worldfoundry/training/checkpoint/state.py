"""Exact-resume training state serialized through PyTorch DCP."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from worldfoundry.core.io.integrity import canonical_json as _core_canonical_json

from .artifacts import (
    OPTIONAL_TRAINING_STATE_NAMES,
    normalize_non_negative_int,
)
from .errors import TrainingCheckpointCompatibilityError

TRAINING_RUNTIME_STATE_SCHEMA = "worldfoundry-training-runtime-state-json-loader"
TRAINING_PROGRESS_SCHEMA = "worldfoundry-training-progress"


def _canonical_json(value: object) -> str:
    try:
        return _core_canonical_json(value)
    except (TypeError, ValueError) as error:
        raise TypeError("training checkpoint metadata must be canonical JSON") from error


def _canonical_mapping(value: Mapping[str, object], *, field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    normalized = json.loads(_canonical_json({str(key): item for key, item in value.items()}))
    if not isinstance(normalized, dict):
        raise TypeError(f"{field_name} must resolve to a JSON object")
    return normalized


def _distributed_context() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _rank_key(rank: int) -> str:
    return f"rank-{rank:08d}"


def _optimizer_key(index: int) -> str:
    return f"optimizer-{index:04d}"


@dataclass(slots=True)
class TrainingProgress:
    """Mutable counters committed only at optimizer-step boundaries."""

    optimizer_steps: int = 0
    microbatches_seen: int = 0
    samples_seen: int = 0
    latent_tokens_seen: int = 0
    # Always 0 by construction: sessions only checkpoint at optimizer-step
    # boundaries and ``record_step`` never mutates it.  The field is kept in
    # the schema as a second guard -- ``__post_init__`` rejects any non-zero
    # value a corrupted or hand-edited checkpoint might carry.
    gradient_accumulation_phase: int = 0

    def __post_init__(self) -> None:
        for name in (
            "optimizer_steps",
            "microbatches_seen",
            "samples_seen",
            "latent_tokens_seen",
            "gradient_accumulation_phase",
        ):
            setattr(self, name, normalize_non_negative_int(getattr(self, name), field_name=name))
        if self.gradient_accumulation_phase != 0:
            raise ValueError("checkpoints are only supported at optimizer-step boundaries")

    def record_step(
        self,
        *,
        microbatches: int,
        samples: int,
        latent_tokens: int,
    ) -> None:
        self.optimizer_steps += 1
        self.microbatches_seen += normalize_non_negative_int(microbatches, field_name="microbatches")
        self.samples_seen += normalize_non_negative_int(samples, field_name="samples")
        self.latent_tokens_seen += normalize_non_negative_int(latent_tokens, field_name="latent_tokens")

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": TRAINING_PROGRESS_SCHEMA,
            "optimizer_steps": self.optimizer_steps,
            "microbatches_seen": self.microbatches_seen,
            "samples_seen": self.samples_seen,
            "latent_tokens_seen": self.latent_tokens_seen,
            "gradient_accumulation_phase": self.gradient_accumulation_phase,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("training progress state must be a mapping")
        expected = {
            "schema",
            "optimizer_steps",
            "microbatches_seen",
            "samples_seen",
            "latent_tokens_seen",
            "gradient_accumulation_phase",
        }
        if set(state_dict) != expected:
            raise ValueError("training progress state fields differ from the active schema")
        if state_dict["schema"] != TRAINING_PROGRESS_SCHEMA:
            raise ValueError(f"unsupported training progress schema: {state_dict['schema']!r}")
        candidate = TrainingProgress(
            optimizer_steps=state_dict["optimizer_steps"],
            microbatches_seen=state_dict["microbatches_seen"],
            samples_seen=state_dict["samples_seen"],
            latent_tokens_seen=state_dict["latent_tokens_seen"],
            gradient_accumulation_phase=state_dict["gradient_accumulation_phase"],
        )
        self.optimizer_steps = candidate.optimizer_steps
        self.microbatches_seen = candidate.microbatches_seen
        self.samples_seen = candidate.samples_seen
        self.latent_tokens_seen = candidate.latent_tokens_seen
        self.gradient_accumulation_phase = candidate.gradient_accumulation_phase


class TrainingState(Stateful):
    """DCP Stateful combining tensors with exact rank-local runtime state."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | Iterable[torch.optim.Optimizer],
        engine: object,
        dataloader: object,
        objective_generator: torch.Generator,
        progress: TrainingProgress,
        identity: Mapping[str, object],
        ignore_frozen_parameters: bool = False,
        lr_scheduler: object | None = None,
        ema: object | None = None,
        grad_scaler: object | None = None,
        algorithm_state: object | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("training checkpoint model must be an nn.Module")
        if isinstance(optimizer, torch.optim.Optimizer):
            optimizers = (optimizer,)
        else:
            optimizers = tuple(optimizer)
            if not optimizers or not all(isinstance(value, torch.optim.Optimizer) for value in optimizers):
                raise TypeError("training checkpoint optimizers must be a non-empty torch optimizer iterable")
            if len({id(value) for value in optimizers}) != len(optimizers):
                raise ValueError("training checkpoint optimizers cannot contain duplicates")
        if not callable(getattr(engine, "state_dict", None)) or not callable(getattr(engine, "load_state_dict", None)):
            raise TypeError("training checkpoint engine must be stateful")
        if not callable(getattr(dataloader, "state_dict", None)) or not callable(
            getattr(dataloader, "load_state_dict", None)
        ):
            raise TypeError("training checkpoint dataloader must be stateful")
        if not isinstance(objective_generator, torch.Generator):
            raise TypeError("objective_generator must be a torch.Generator")
        if not isinstance(progress, TrainingProgress):
            raise TypeError("progress must be TrainingProgress")
        if not isinstance(ignore_frozen_parameters, bool):
            raise TypeError("ignore_frozen_parameters must be a bool")
        optional_stateful = {
            "lr_scheduler": lr_scheduler,
            "ema": ema,
            "grad_scaler": grad_scaler,
            "algorithm_state": algorithm_state,
        }
        for name, component in optional_stateful.items():
            if component is not None and (
                not callable(getattr(component, "state_dict", None))
                or not callable(getattr(component, "load_state_dict", None))
            ):
                raise TypeError(f"{name} must expose state_dict/load_state_dict or be None")

        self.model = model
        # ``optimizer`` remains the scalar object for existing callers while
        # post-training uses the explicit tuple and DCP's multi-optimizer API.
        self.optimizer = optimizers[0] if len(optimizers) == 1 else optimizers
        self.optimizers = optimizers
        self.engine = engine
        self.dataloader = dataloader
        self.objective_generator = objective_generator
        self.progress = progress
        self.identity = MappingProxyType(_canonical_mapping(identity, field_name="training resume identity"))
        self.ignore_frozen_parameters = ignore_frozen_parameters
        self._optional_stateful = optional_stateful
        self._options = StateDictOptions(
            ignore_frozen_params=ignore_frozen_parameters,
            # Adapter-only checkpoints intentionally omit frozen base tensors.
            # Their base identity is bound separately by ``identity``;
            # the explicit model-state key inventory below keeps adapter loads
            # strict without asking nn.Module.load_state_dict for absent base keys.
            strict=not ignore_frozen_parameters,
        )

    @property
    def optional_state_presence(self) -> dict[str, bool]:
        return {name: component is not None for name, component in self._optional_stateful.items()}

    def _local_runtime_state(self) -> dict[str, object]:
        engine_state = self.engine.state_dict()
        if int(engine_state.get("global_step", -1)) != self.progress.optimizer_steps:
            raise RuntimeError("engine global_step differs from committed training progress")
        dataloader_state = self.dataloader.state_dict()
        if not isinstance(dataloader_state, Mapping):
            raise TypeError("training dataloader state_dict must return a mapping")
        try:
            # DCP flattens nested Python containers into separate storage keys.
            # Token-bucket queues legitimately change list lengths after the
            # first batch, so exposing the raw mapping makes a fresh loader
            # lack saved leaf keys.  One canonical JSON scalar preserves the
            # strict loader schema while giving DCP a fixed key inventory.
            serialized_dataloader = _canonical_json(dataloader_state)
        except TypeError as error:
            raise TypeError("training dataloader state must contain canonical JSON values") from error
        cuda_states = {
            f"device-{index:04d}": value.clone() for index, value in enumerate(torch.cuda.get_rng_state_all())
        }
        return {
            "schema": TRAINING_RUNTIME_STATE_SCHEMA,
            "engine": engine_state,
            "progress": self.progress.state_dict(),
            "dataloader": serialized_dataloader,
            "torch_cpu_rng_state": torch.get_rng_state().clone(),
            "torch_cuda_rng_states": cuda_states,
            "objective_generator_state": self.objective_generator.get_state().clone(),
            "python_random_state": random.getstate(),
        }

    def _runtime_by_rank(self) -> dict[str, object]:
        local = self._local_runtime_state()
        rank, world_size = _distributed_context()
        if world_size == 1:
            return {_rank_key(rank): local}
        gathered: list[object] = [None] * world_size
        dist.all_gather_object(gathered, local)
        if any(not isinstance(value, Mapping) for value in gathered):
            raise RuntimeError("failed to gather rank-local training runtime state")
        return {_rank_key(index): value for index, value in enumerate(gathered)}

    def state_dict(self) -> dict[str, object]:
        model_state = get_model_state_dict(
            self.model,
            options=self._options,
        )
        optimizer_state = {
            _optimizer_key(index): get_optimizer_state_dict(
                self.model,
                optimizer,
                options=self._options,
            )
            for index, optimizer in enumerate(self.optimizers)
        }
        _, world_size = _distributed_context()
        return {
            "model": model_state,
            "model_state_keys": tuple(sorted(model_state)),
            "optimizer": optimizer_state,
            "optimizer_count": len(self.optimizers),
            "identity": dict(self.identity),
            "world_size": world_size,
            "ignore_frozen_parameters": self.ignore_frozen_parameters,
            "optional_state": {
                name: None if component is None else component.state_dict()
                for name, component in self._optional_stateful.items()
            },
            "runtime_by_rank": self._runtime_by_rank(),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("loaded training state must be a mapping")
        expected = {
            "model",
            "model_state_keys",
            "optimizer",
            "optimizer_count",
            "identity",
            "world_size",
            "ignore_frozen_parameters",
            "optional_state",
            "runtime_by_rank",
        }
        if set(state_dict) != expected:
            raise TrainingCheckpointCompatibilityError("loaded training state fields differ from the active schema")
        saved_identity = _canonical_mapping(
            state_dict["identity"],
            field_name="saved training resume identity",
        )
        if saved_identity != dict(self.identity):
            raise TrainingCheckpointCompatibilityError(
                "saved training identity differs from the active recipe/data/model/runtime"
            )
        if int(state_dict["optimizer_count"]) != len(self.optimizers):
            raise TrainingCheckpointCompatibilityError(
                "saved optimizer inventory differs from the active training stack"
            )
        loaded_optimizer_state = state_dict["optimizer"]
        expected_optimizer_keys = {_optimizer_key(index) for index in range(len(self.optimizers))}
        if not isinstance(loaded_optimizer_state, Mapping) or set(loaded_optimizer_state) != expected_optimizer_keys:
            raise TrainingCheckpointCompatibilityError("saved optimizer state inventory is incomplete")
        rank, world_size = _distributed_context()
        if int(state_dict["world_size"]) != world_size:
            raise TrainingCheckpointCompatibilityError("exact data/RNG resume requires the same world size")
        saved_ignore_frozen = state_dict["ignore_frozen_parameters"]
        if not isinstance(saved_ignore_frozen, bool):
            raise TrainingCheckpointCompatibilityError("saved frozen-parameter checkpoint policy is invalid")
        if saved_ignore_frozen != self.ignore_frozen_parameters:
            raise TrainingCheckpointCompatibilityError("frozen-parameter checkpoint policy differs from the active run")
        saved_optional_state = state_dict["optional_state"]
        if not isinstance(saved_optional_state, Mapping) or set(saved_optional_state) != set(
            OPTIONAL_TRAINING_STATE_NAMES
        ):
            raise TrainingCheckpointCompatibilityError("saved optional training-state inventory is invalid")
        for name, component in self._optional_stateful.items():
            loaded_component_state = saved_optional_state[name]
            if (component is None) != (loaded_component_state is None):
                raise TrainingCheckpointCompatibilityError(
                    f"saved {name} presence differs from the active training stack"
                )
            if component is not None and not isinstance(loaded_component_state, Mapping):
                raise TrainingCheckpointCompatibilityError(f"saved {name} state is invalid")
        active_model_state = get_model_state_dict(
            self.model,
            options=self._options,
        )
        saved_model_keys = state_dict["model_state_keys"]
        if not isinstance(saved_model_keys, (tuple, list)) or any(
            not isinstance(value, str) for value in saved_model_keys
        ):
            raise TrainingCheckpointCompatibilityError("saved model-state key inventory is invalid")
        expected_model_keys = tuple(sorted(active_model_state))
        if tuple(saved_model_keys) != expected_model_keys:
            raise TrainingCheckpointCompatibilityError("saved trainable model-state keys differ from the active model")
        loaded_model = state_dict["model"]
        if not isinstance(loaded_model, Mapping) or tuple(sorted(loaded_model)) != expected_model_keys:
            raise TrainingCheckpointCompatibilityError(
                "loaded model tensors differ from the saved trainable key inventory"
            )
        runtime_by_rank = state_dict["runtime_by_rank"]
        if not isinstance(runtime_by_rank, Mapping) or set(runtime_by_rank) != {
            _rank_key(index) for index in range(world_size)
        }:
            raise TrainingCheckpointCompatibilityError("saved rank runtime topology is incomplete")
        local_runtime = runtime_by_rank[_rank_key(rank)]
        if not isinstance(local_runtime, Mapping):
            raise TrainingCheckpointCompatibilityError("saved local runtime state is invalid")
        runtime_expected = {
            "schema",
            "engine",
            "progress",
            "dataloader",
            "torch_cpu_rng_state",
            "torch_cuda_rng_states",
            "objective_generator_state",
            "python_random_state",
        }
        if set(local_runtime) != runtime_expected:
            raise TrainingCheckpointCompatibilityError("saved local runtime fields differ from the active schema")
        if local_runtime["schema"] != TRAINING_RUNTIME_STATE_SCHEMA:
            raise TrainingCheckpointCompatibilityError(
                f"unsupported training runtime schema: {local_runtime['schema']!r}"
            )

        loaded_progress = TrainingProgress()
        loaded_progress.load_state_dict(local_runtime["progress"])
        loaded_engine = local_runtime["engine"]
        if (
            not isinstance(loaded_engine, Mapping)
            or int(loaded_engine.get("global_step", -1)) != loaded_progress.optimizer_steps
        ):
            raise TrainingCheckpointCompatibilityError("saved engine step differs from saved training progress")
        cpu_rng_state = local_runtime["torch_cpu_rng_state"]
        generator_state = local_runtime["objective_generator_state"]
        cuda_rng_states = local_runtime["torch_cuda_rng_states"]
        if not isinstance(cpu_rng_state, torch.Tensor) or cpu_rng_state.dtype is not torch.uint8:
            raise TrainingCheckpointCompatibilityError("saved CPU RNG state is invalid")
        if not isinstance(generator_state, torch.Tensor) or generator_state.dtype is not torch.uint8:
            raise TrainingCheckpointCompatibilityError("saved objective generator state is invalid")
        if not isinstance(cuda_rng_states, Mapping) or len(cuda_rng_states) != torch.cuda.device_count():
            raise TrainingCheckpointCompatibilityError("saved CUDA RNG device count differs from the active process")
        ordered_cuda_states: list[torch.Tensor] = []
        for index in range(torch.cuda.device_count()):
            value = cuda_rng_states.get(f"device-{index:04d}")
            if not isinstance(value, torch.Tensor) or value.dtype is not torch.uint8:
                raise TrainingCheckpointCompatibilityError("saved CUDA RNG state is invalid")
            ordered_cuda_states.append(value.cpu())

        set_model_state_dict(
            self.model,
            state_dict["model"],
            options=self._options,
        )
        for index, optimizer in enumerate(self.optimizers):
            set_optimizer_state_dict(
                self.model,
                optimizer,
                loaded_optimizer_state[_optimizer_key(index)],
                options=self._options,
            )
        for name, component in self._optional_stateful.items():
            if component is not None:
                component.load_state_dict(saved_optional_state[name])
        serialized_dataloader = local_runtime["dataloader"]
        if not isinstance(serialized_dataloader, str):
            raise TrainingCheckpointCompatibilityError("saved dataloader state is not canonical JSON")
        try:
            dataloader_state = json.loads(serialized_dataloader)
        except json.JSONDecodeError as error:
            raise TrainingCheckpointCompatibilityError("saved dataloader state JSON is invalid") from error
        if not isinstance(dataloader_state, Mapping):
            raise TrainingCheckpointCompatibilityError("saved dataloader state must decode to a mapping")
        self.dataloader.load_state_dict(dataloader_state)
        self.engine.load_state_dict(loaded_engine)
        self.progress.load_state_dict(local_runtime["progress"])
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng_state.cpu())
        if ordered_cuda_states:
            torch.cuda.set_rng_state_all(ordered_cuda_states)
        self.objective_generator.set_state(generator_state.cpu())
        random.setstate(local_runtime["python_random_state"])


__all__ = [
    "TRAINING_PROGRESS_SCHEMA",
    "TRAINING_RUNTIME_STATE_SCHEMA",
    "TrainingProgress",
    "TrainingState",
]
