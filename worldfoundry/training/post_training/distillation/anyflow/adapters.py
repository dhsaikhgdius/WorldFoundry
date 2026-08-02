"""Strict native model adapters for AnyFlow FAR and bidirectional scores."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from worldfoundry.core.attention.chunk_partition import (
    DualResolutionKVCache,
    TemporalChunkPartition,
)


def _underlying_module(module: nn.Module) -> nn.Module:
    """Reach the behavior-bearing module through DDP and optional PEFT."""

    from torch.nn.parallel import DistributedDataParallel

    base = module.module if isinstance(module, DistributedDataParallel) else module
    get_base_model = getattr(base, "get_base_model", None)
    if callable(get_base_model):
        candidate = get_base_model()
        if not isinstance(candidate, nn.Module):
            raise TypeError("AnyFlow PEFT get_base_model() must return nn.Module")
        base = candidate
    return base


def _checkpoint_identity(value: str | None) -> str | None:
    if value is None:
        return None
    resolved = str(value).strip()
    if not resolved:
        raise ValueError("AnyFlow checkpoint identity cannot be empty")
    return resolved


def _configuration_value(module: nn.Module, name: str) -> object:
    base = _underlying_module(module)
    config = getattr(base, "config", None)
    if config is None or not hasattr(config, name):
        raise TypeError(f"native AnyFlow module config must expose {name!r}")
    return getattr(config, name)


def _audit_far_capabilities(
    module: nn.Module,
    partition: TemporalChunkPartition,
) -> None:
    """Reject a standard Wan denoiser that has not been converted to FAR/FlowMap."""

    expected = {
        "patch_size": partition.patch_size,
        "compressed_patch_size": partition.compressed_patch_size,
        "full_chunk_limit": partition.full_chunk_limit,
    }
    for name, value in expected.items():
        actual = _configuration_value(module, name)
        normalized = tuple(actual) if isinstance(value, tuple) else int(actual)
        if normalized != value:
            raise ValueError(
                f"native AnyFlow module {name}={normalized!r} differs from the active FAR partition {value!r}"
            )
    configured_partition = tuple(int(value) for value in _configuration_value(module, "chunk_partition"))
    if configured_partition != partition.chunks:
        raise ValueError("native AnyFlow module chunk_partition differs from training")
    base = _underlying_module(module)
    if not isinstance(getattr(base, "far_patch_embedding", None), nn.Module):
        raise TypeError("native AnyFlow FAR module has no compressed patch embedding")
    _audit_flowmap_capability(module)


def _audit_flowmap_capability(module: nn.Module) -> None:
    """Require the destination-time conditioning used by released AnyFlow."""

    destination_type = _configuration_value(module, "deltatime_type")
    if destination_type != "r":
        raise ValueError("native AnyFlow module must condition on destination time r")
    gate_value = _configuration_value(module, "gate_value")
    if isinstance(gate_value, bool) or float(gate_value) != 0.25:
        raise ValueError("native AnyFlow module must use the released 0.25 time gate")
    base = _underlying_module(module)
    condition_embedder = getattr(base, "condition_embedder", None)
    if not isinstance(getattr(condition_embedder, "delta_embedder", None), nn.Module):
        raise TypeError("native AnyFlow module has no FlowMap destination-time embedder")


def _conditioning_kwargs(conditioning: Mapping[str, object]) -> dict[str, object]:
    """Translate WorldFoundry conditioning names into the native model call."""

    if not isinstance(conditioning, Mapping):
        raise TypeError("AnyFlow conditioning must be a mapping")
    values = dict(conditioning)
    aliases = {
        "context": "encoder_hidden_states",
        "image_context": "encoder_hidden_states_image",
    }
    for source, destination in aliases.items():
        if source not in values:
            continue
        if destination in values:
            raise ValueError(f"AnyFlow conditioning cannot contain both {source!r} and {destination!r}")
        values[destination] = values.pop(source)
    allowed = {
        "encoder_hidden_states",
        "encoder_hidden_states_image",
        "attention_kwargs",
    }
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unsupported AnyFlow conditioning fields: {unknown}")
    if "encoder_hidden_states" not in values:
        raise ValueError("AnyFlow conditioning requires text encoder hidden states")
    return values


def _model_output(value: object, *, reference: Tensor) -> Tensor:
    if isinstance(value, Tensor):
        result = value
    elif isinstance(value, (tuple, list)) and value and isinstance(value[0], Tensor):
        result = value[0]
    else:
        result = getattr(value, "sample", None)
    if not isinstance(result, Tensor):
        raise TypeError("native AnyFlow module must return a tensor, tensor tuple, or .sample")
    if result.ndim != 5:
        raise ValueError("native AnyFlow module output must be BFCHW")
    converted = result.permute(0, 2, 1, 3, 4)
    if converted.shape != reference.shape:
        raise ValueError("native AnyFlow module output does not match the requested BCTHW latent shape")
    return converted.to(dtype=reference.dtype)


@dataclass(slots=True)
class _FARRolloutState:
    cache: DualResolutionKVCache

    @property
    def flags(self) -> dict[str, object]:
        return {
            "num_cached_chunks": self.cache.num_cached_chunks,
            "is_cache_step": self.cache.is_cache_step,
        }

    @property
    def model_cache(self) -> dict[int, dict[str, Tensor]]:
        return {
            index: {
                "full_cache": self.cache.layer(index)[0],
                "compressed_cache": self.cache.layer(index)[1],
            }
            for index in range(int(self.cache.full.shape[0]))
        }


def _validate_call(
    latents: Tensor,
    timesteps: Tensor,
    destination_timesteps: Tensor,
    *,
    sample_ids: tuple[str, ...],
    training: bool,
    branch: str,
) -> None:
    if not isinstance(latents, Tensor) or latents.ndim != 5 or not latents.is_floating_point():
        raise TypeError("AnyFlow latents must be floating BCTHW tensors")
    batch, frames = int(latents.shape[0]), int(latents.shape[2])
    if len(sample_ids) != batch:
        raise ValueError("AnyFlow sample IDs and latents must share a batch dimension")
    if timesteps.shape != (batch, frames) or destination_timesteps.shape != (
        batch,
        frames,
    ):
        raise ValueError("AnyFlow model timesteps must have shape [B,T]")
    if not isinstance(training, bool):
        raise TypeError("training must be a bool")
    if branch not in {"positive", "negative"}:
        raise ValueError("branch must be positive or negative")
    if not training and torch.is_grad_enabled():
        raise RuntimeError("non-training AnyFlow model evaluations must run under no_grad")


class NativeAnyFlowFARAdapter:
    """Call a local FAR/FlowMap module using the released tensor convention.

    The adapter imports no upstream package.  It only defines the model-facing
    ABI that a WorldFoundry-loaded module must satisfy and audits the behavior-
    bearing FAR configuration before every distinct partition is used.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        checkpoint_identity: str | None = None,
    ) -> None:
        if not isinstance(module, nn.Module):
            raise TypeError("AnyFlow FAR module must be nn.Module")
        self.module = module
        self.checkpoint_identity = _checkpoint_identity(checkpoint_identity)
        self._audited_partitions: set[TemporalChunkPartition] = set()

    def _audit(self, partition: TemporalChunkPartition) -> None:
        if partition not in self._audited_partitions:
            _audit_far_capabilities(self.module, partition)
            self._audited_partitions.add(partition)

    def create_rollout_state(
        self,
        *,
        partition: TemporalChunkPartition,
        reference: Tensor,
    ) -> object:
        if not isinstance(reference, Tensor) or reference.ndim != 5:
            raise TypeError("AnyFlow rollout reference must be BCTHW")
        if int(reference.shape[2]) != partition.frame_count:
            raise ValueError("AnyFlow rollout reference and FAR partition differ")
        self._audit(partition)
        geometry = partition.token_geometry(
            latent_height=int(reference.shape[3]),
            latent_width=int(reference.shape[4]),
        )
        return _FARRolloutState(
            cache=DualResolutionKVCache(
                geometry,
                batch_size=int(reference.shape[0]),
                num_layers=int(_configuration_value(self.module, "num_layers")),
                num_heads=int(_configuration_value(self.module, "num_attention_heads")),
                head_dim=int(_configuration_value(self.module, "attention_head_dim")),
                device=reference.device,
                dtype=reference.dtype,
            )
        )

    def predict_flow_map(
        self,
        noisy_latents: Tensor,
        timesteps: Tensor,
        destination_timesteps: Tensor,
        *,
        clean_latents: Tensor,
        context_latents: Tensor,
        partition: TemporalChunkPartition,
        sampled_chunk_count: int,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> Tensor:
        _validate_call(
            noisy_latents,
            timesteps,
            destination_timesteps,
            sample_ids=sample_ids,
            training=training,
            branch=branch,
        )
        if clean_latents.shape != noisy_latents.shape:
            raise ValueError("AnyFlow clean and noisy target latents must have equal shapes")
        if (
            context_latents.ndim != 5
            or context_latents.shape[:2] != noisy_latents.shape[:2]
            or context_latents.shape[3:] != noisy_latents.shape[3:]
        ):
            raise ValueError("AnyFlow FAR context must match target batch/channel/spatial shape")
        chunks = partition.prefix(sampled_chunk_count)
        context_frames, target_frames = partition.context_target_frames(sampled_chunk_count)
        if int(context_latents.shape[2]) != context_frames or int(noisy_latents.shape[2]) != target_frames:
            raise ValueError("AnyFlow FAR context/target frames differ from the sampled partition")
        self._audit(partition)
        context_time = torch.zeros(
            (int(noisy_latents.shape[0]), context_frames),
            device=noisy_latents.device,
            dtype=timesteps.dtype,
        )
        model_input = torch.cat((context_latents, noisy_latents), dim=2).permute(
            0,
            2,
            1,
            3,
            4,
        )
        kwargs = _conditioning_kwargs(conditioning)
        value = self.module(
            model_input,
            chunk_partition=list(chunks),
            timestep=torch.cat((context_time, timesteps), dim=1),
            r_timestep=torch.cat((context_time, destination_timesteps), dim=1),
            clean_hidden_states=clean_latents.permute(0, 2, 1, 3, 4),
            clean_timestep=torch.zeros_like(timesteps),
            return_dict=False,
            is_causal=True,
            **kwargs,
        )
        return _model_output(value, reference=noisy_latents)

    def rollout_velocity(
        self,
        noisy_latents: Tensor,
        timesteps: Tensor,
        destination_timesteps: Tensor,
        *,
        partition: TemporalChunkPartition,
        chunk_index: int,
        rollout_state: object,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
    ) -> Tensor:
        _validate_call(
            noisy_latents,
            timesteps,
            destination_timesteps,
            sample_ids=sample_ids,
            training=training,
            branch="positive",
        )
        if isinstance(chunk_index, bool) or not 0 <= int(chunk_index) < partition.chunk_count:
            raise ValueError("AnyFlow rollout chunk_index is out of range")
        if int(noisy_latents.shape[2]) != partition.chunks[int(chunk_index)]:
            raise ValueError("AnyFlow rollout latent frames differ from the selected chunk")
        if not isinstance(rollout_state, _FARRolloutState):
            raise TypeError("rollout_state was not created by this AnyFlow adapter")
        if rollout_state.cache.num_cached_chunks != int(chunk_index):
            raise RuntimeError("AnyFlow KV cache and rollout chunk index diverged")
        self._audit(partition)
        value = self.module(
            noisy_latents.permute(0, 2, 1, 3, 4),
            chunk_partition=list(partition.chunks[: int(chunk_index) + 1]),
            timestep=timesteps,
            r_timestep=destination_timesteps,
            return_dict=False,
            is_causal=True,
            kv_cache=rollout_state.model_cache,
            kv_cache_flag=rollout_state.flags,
            **_conditioning_kwargs(conditioning),
        )
        return _model_output(value, reference=noisy_latents)

    def predict_bidirectional_velocity(
        self,
        noisy_latents: Tensor,
        timesteps: Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> Tensor:
        """Execute the auxiliary full-video objective from FAR pretraining."""

        _validate_call(
            noisy_latents,
            timesteps,
            timesteps,
            sample_ids=sample_ids,
            training=training,
            branch=branch,
        )
        _audit_flowmap_capability(self.module)
        value = self.module(
            noisy_latents.permute(0, 2, 1, 3, 4),
            timestep=timesteps,
            r_timestep=timesteps,
            return_dict=False,
            is_causal=False,
            **_conditioning_kwargs(conditioning),
        )
        return _model_output(value, reference=noisy_latents)

    @torch.no_grad()
    def commit_rollout_chunk(
        self,
        clean_prefix: Tensor,
        *,
        partition: TemporalChunkPartition,
        chunk_index: int,
        rollout_state: object,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
    ) -> None:
        if not isinstance(rollout_state, _FARRolloutState):
            raise TypeError("rollout_state was not created by this AnyFlow adapter")
        index = int(chunk_index)
        if not 0 <= index < partition.chunk_count - 1:
            raise ValueError("only non-final AnyFlow chunks are committed to KV cache")
        expected_frames = sum(partition.chunks[: index + 1])
        if (
            not isinstance(clean_prefix, Tensor)
            or clean_prefix.ndim != 5
            or int(clean_prefix.shape[0]) != len(sample_ids)
            or int(clean_prefix.shape[2]) != expected_frames
        ):
            raise ValueError("AnyFlow clean prefix differs from the committed FAR chunks")
        if rollout_state.cache.num_cached_chunks != index:
            raise RuntimeError("AnyFlow KV cache commit order diverged")
        rollout_state.cache.is_cache_step = True
        zeros = torch.zeros(
            (int(clean_prefix.shape[0]), expected_frames),
            device=clean_prefix.device,
            dtype=torch.float32,
        )
        try:
            value = self.module(
                clean_prefix.permute(0, 2, 1, 3, 4),
                chunk_partition=list(partition.chunks[: index + 1]),
                timestep=zeros,
                r_timestep=zeros,
                return_dict=False,
                is_causal=True,
                kv_cache=rollout_state.model_cache,
                kv_cache_flag=rollout_state.flags,
                **_conditioning_kwargs(conditioning),
            )
            if not (isinstance(value, (tuple, list)) and len(value) == 2 and value[0] is None):
                raise TypeError("AnyFlow cache commit must return (None, cache)")
        finally:
            rollout_state.cache.is_cache_step = False
        rollout_state.cache.num_cached_chunks += 1


class NativeAnyFlowScoreAdapter:
    """Expose a local bidirectional FlowMap/Wan module as an RF score."""

    def __init__(
        self,
        module: nn.Module,
        *,
        checkpoint_identity: str | None = None,
    ) -> None:
        if not isinstance(module, nn.Module):
            raise TypeError("AnyFlow score module must be nn.Module")
        self.module = module
        self.checkpoint_identity = _checkpoint_identity(checkpoint_identity)

    def predict_velocity(
        self,
        noisy_latents: Tensor,
        timesteps: Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> Tensor:
        _validate_call(
            noisy_latents,
            timesteps,
            timesteps,
            sample_ids=sample_ids,
            training=training,
            branch=branch,
        )
        value = self.module(
            noisy_latents.permute(0, 2, 1, 3, 4),
            timestep=timesteps,
            r_timestep=timesteps,
            return_dict=False,
            is_causal=False,
            **_conditioning_kwargs(conditioning),
        )
        return _model_output(value, reference=noisy_latents)


class NativeAnyFlowBidirectionalAdapter:
    """Call a local full-video FlowMap student with released Wan semantics."""

    def __init__(
        self,
        module: nn.Module,
        *,
        checkpoint_identity: str | None = None,
    ) -> None:
        if not isinstance(module, nn.Module):
            raise TypeError("AnyFlow bidirectional module must be nn.Module")
        self.module = module
        self.checkpoint_identity = _checkpoint_identity(checkpoint_identity)
        _audit_flowmap_capability(module)

    def predict_flow_map(
        self,
        noisy_latents: Tensor,
        timesteps: Tensor,
        destination_timesteps: Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> Tensor:
        _validate_call(
            noisy_latents,
            timesteps,
            destination_timesteps,
            sample_ids=sample_ids,
            training=training,
            branch=branch,
        )
        value = self.module(
            noisy_latents.permute(0, 2, 1, 3, 4),
            timestep=timesteps,
            r_timestep=destination_timesteps,
            return_dict=False,
            is_causal=False,
            **_conditioning_kwargs(conditioning),
        )
        return _model_output(value, reference=noisy_latents)

    def rollout_velocity(
        self,
        noisy_latents: Tensor,
        timesteps: Tensor,
        destination_timesteps: Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
    ) -> Tensor:
        return self.predict_flow_map(
            noisy_latents,
            timesteps,
            destination_timesteps,
            sample_ids=sample_ids,
            conditioning=conditioning,
            training=training,
            branch="positive",
        )


__all__ = [
    "NativeAnyFlowBidirectionalAdapter",
    "NativeAnyFlowFARAdapter",
    "NativeAnyFlowScoreAdapter",
]
