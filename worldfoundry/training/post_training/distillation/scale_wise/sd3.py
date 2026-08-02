"""Native SD3 execution adapters for scale-wise distillation."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import Protocol

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class _ContextManagerFactory(Protocol):
    def __call__(self) -> AbstractContextManager[object]: ...


def _conditioning_tensors(
    conditioning: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(conditioning, Mapping):
        raise TypeError("SD3 conditioning must be a mapping")
    prompt = conditioning.get("prompt_embeds")
    pooled = conditioning.get("pooled_prompt_embeds")
    if not isinstance(prompt, torch.Tensor) or not isinstance(pooled, torch.Tensor):
        raise TypeError(
            "SD3 conditioning requires prompt_embeds and pooled_prompt_embeds tensors"
        )
    if prompt.ndim != 3 or pooled.ndim != 2 or prompt.shape[0] != pooled.shape[0]:
        raise ValueError("SD3 prompt and pooled embeddings have incompatible shapes")
    return prompt, pooled


def _sd3_graph(module: nn.Module) -> nn.Module:
    candidate = module
    get_base_model = getattr(candidate, "get_base_model", None)
    if callable(get_base_model):
        candidate = get_base_model()
    required = (
        "pos_embed",
        "time_text_embed",
        "context_embedder",
        "transformer_blocks",
        "norm_out",
        "proj_out",
        "config",
    )
    if not all(hasattr(candidate, name) for name in required):
        raise TypeError("module does not expose an SD3 transformer graph")
    return candidate


def _block_forward(
    block: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    output = block(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        joint_attention_kwargs={},
    )
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError("SD3 transformer block must return context and latent tokens")
    context, latent = output
    if (context is not None and not isinstance(context, torch.Tensor)) or not isinstance(
        latent,
        torch.Tensor,
    ):
        raise TypeError("SD3 transformer block returned invalid token values")
    return output


def sd3_velocity_and_features(
    module: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    pooled_projections: torch.Tensor,
    timestep: torch.Tensor,
    *,
    block_indices: tuple[int, ...],
    return_velocity: bool,
) -> tuple[torch.Tensor | None, tuple[torch.Tensor, ...]]:
    """Execute the SD3 graph once and expose selected latent-token features."""

    if not isinstance(module, nn.Module):
        raise TypeError("module must be nn.Module")
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 4:
        raise TypeError("hidden_states must have shape [B,C,H,W]")
    if not all(
        isinstance(value, torch.Tensor)
        for value in (encoder_hidden_states, pooled_projections, timestep)
    ):
        raise TypeError("SD3 conditioning and timestep values must be tensors")
    indices = tuple(int(value) for value in block_indices)
    if not indices or indices != tuple(sorted(set(indices))) or indices[0] < 0:
        raise ValueError("block_indices must be non-empty, sorted, and unique")
    graph = _sd3_graph(module)
    blocks = tuple(graph.transformer_blocks)
    if indices[-1] >= len(blocks):
        raise ValueError("requested SD3 feature block does not exist")
    height, width = hidden_states.shape[-2:]
    latent_tokens = graph.pos_embed(hidden_states)
    temb = graph.time_text_embed(timestep, pooled_projections)
    context_tokens: torch.Tensor | None = graph.context_embedder(
        encoder_hidden_states
    )
    collected: list[torch.Tensor] = []
    gradient_checkpointing = bool(
        getattr(graph, "gradient_checkpointing", False)
    )
    for index, block in enumerate(blocks):
        if context_tokens is None:
            raise RuntimeError("SD3 context became unavailable before the final block")
        if torch.is_grad_enabled() and gradient_checkpointing:

            def custom_forward(
                latent: torch.Tensor,
                context: torch.Tensor,
                embedding: torch.Tensor,
                *,
                active_block: nn.Module = block,
            ) -> tuple[torch.Tensor | None, torch.Tensor]:
                return _block_forward(
                    active_block,
                    latent,
                    context,
                    embedding,
                )

            context_tokens, latent_tokens = checkpoint(
                custom_forward,
                latent_tokens,
                context_tokens,
                temb,
                use_reentrant=False,
            )
        else:
            context_tokens, latent_tokens = _block_forward(
                block,
                latent_tokens,
                context_tokens,
                temb,
            )
        if index in indices:
            collected.append(latent_tokens)
            if index == indices[-1] and not return_velocity:
                return None, tuple(collected)
    latent_tokens = graph.norm_out(latent_tokens, temb)
    latent_tokens = graph.proj_out(latent_tokens)
    patch_size = int(graph.config.patch_size)
    output_channels = int(graph.out_channels)
    patch_height = height // patch_size
    patch_width = width // patch_size
    latent_tokens = latent_tokens.reshape(
        hidden_states.shape[0],
        patch_height,
        patch_width,
        patch_size,
        patch_size,
        output_channels,
    )
    latent_tokens = torch.einsum("nhwpqc->nchpwq", latent_tokens)
    velocity = latent_tokens.reshape(
        hidden_states.shape[0],
        output_channels,
        patch_height * patch_size,
        patch_width * patch_size,
    )
    return velocity, tuple(collected)


class _SD3PredictionBase:
    def __init__(
        self,
        module: nn.Module,
        *,
        checkpoint_identity: str,
        num_train_timesteps: int = 1000,
        adapter_context: _ContextManagerFactory | None = None,
    ) -> None:
        if not isinstance(module, nn.Module):
            raise TypeError("module must be nn.Module")
        identity = str(checkpoint_identity).strip()
        if not identity:
            raise ValueError("checkpoint_identity must be non-empty")
        if isinstance(num_train_timesteps, bool) or int(num_train_timesteps) < 2:
            raise ValueError("num_train_timesteps must be at least two")
        if adapter_context is not None and not callable(adapter_context):
            raise TypeError("adapter_context must be callable")
        self.module = module
        self.checkpoint_identity = identity
        self.num_train_timesteps = int(num_train_timesteps)
        self._adapter_context = adapter_context

    def _context(self):
        if self._adapter_context is None:
            return nullcontext()
        return self._adapter_context()

    def _execution_module(self) -> nn.Module:
        return self.module

    def predict_velocity(
        self,
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> torch.Tensor:
        del training, branch
        prompt, pooled = _conditioning_tensors(conditioning)
        if noisy_latents.shape[0] != len(sample_ids):
            raise ValueError("SD3 latent batch and sample ids differ")
        if sigmas.ndim != 1 or sigmas.shape[0] != noisy_latents.shape[0]:
            raise ValueError("SD3 sigmas must have shape [B]")
        timestep = sigmas.to(
            device=noisy_latents.device,
            dtype=torch.float32,
        ) * float(self.num_train_timesteps)
        with self._context():
            output = self._execution_module()(
                hidden_states=noisy_latents,
                encoder_hidden_states=prompt,
                pooled_projections=pooled,
                timestep=timestep,
                return_dict=False,
            )
        if not isinstance(output, tuple) or not isinstance(output[0], torch.Tensor):
            raise TypeError("SD3 transformer must return a velocity tensor tuple")
        return output[0]


class SD3ScaleWisePredictionAdapter(_SD3PredictionBase):
    """Student or independently materialized teacher SD3 velocity adapter."""


class _TeacherModuleView(nn.Module):
    """Parameterless engine role for an adapter-disabled shared base."""

    def forward(self, *args, **kwargs):  # pragma: no cover - adapter owns execution
        raise RuntimeError("teacher module view is not a direct execution entrypoint")


class SD3AdapterDisabledTeacherAdapter(_SD3PredictionBase):
    """Reuse a student's frozen base under PEFT's disable-adapter context."""

    def __init__(
        self,
        student_module: nn.Module,
        *,
        checkpoint_identity: str,
        num_train_timesteps: int = 1000,
    ) -> None:
        disable_adapter = getattr(student_module, "disable_adapter", None)
        if not callable(disable_adapter):
            raise TypeError("shared SD3 teacher requires module.disable_adapter()")
        self.student_module = student_module
        super().__init__(
            _TeacherModuleView(),
            checkpoint_identity=checkpoint_identity,
            num_train_timesteps=num_train_timesteps,
            adapter_context=disable_adapter,
        )

    def _execution_module(self) -> nn.Module:
        return self.student_module


class _FeatureClassifier(nn.Sequential):
    def __init__(self, input_dim: int, layers: int) -> None:
        dimensions = torch.linspace(input_dim, 1, layers + 1, dtype=torch.int64)
        modules: list[nn.Module] = []
        for input_value, output_value in zip(
            dimensions[:-1],
            dimensions[1:],
            strict=True,
        ):
            modules.extend(
                (
                    nn.LayerNorm(int(input_value.item())),
                    nn.GELU(),
                    nn.Linear(int(input_value.item()), int(output_value.item())),
                )
            )
        super().__init__(*modules)


class SD3ScaleWiseCriticModule(nn.Module):
    """Fake SD3 score network and the released progressively narrowing head."""

    def __init__(self, transformer: nn.Module, *, discriminator_layers: int) -> None:
        super().__init__()
        if not isinstance(transformer, nn.Module):
            raise TypeError("transformer must be nn.Module")
        if isinstance(discriminator_layers, bool) or int(discriminator_layers) <= 0:
            raise ValueError("discriminator_layers must be positive")
        graph = _sd3_graph(transformer)
        inner_dim = int(getattr(graph, "inner_dim", 0))
        if inner_dim <= 0:
            raise ValueError("SD3 transformer must expose a positive inner_dim")
        self.transformer = transformer
        self.classifier_head = _FeatureClassifier(
            inner_dim,
            int(discriminator_layers),
        )
        self.discriminator_layers = int(discriminator_layers)

    def forward(
        self,
        mode: str,
        hidden_states: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        pooled_projections: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        block_indices: tuple[int, ...] = (),
        pooled_features: tuple[torch.Tensor, ...] = (),
    ):
        if mode == "classify":
            if not pooled_features:
                raise ValueError("pooled_features cannot be empty")
            return tuple(self.classifier_head(value) for value in pooled_features)
        if not all(
            isinstance(value, torch.Tensor)
            for value in (
                hidden_states,
                encoder_hidden_states,
                pooled_projections,
                timestep,
            )
        ):
            raise TypeError("SD3 critic forward requires latent and conditioning tensors")
        assert hidden_states is not None
        assert encoder_hidden_states is not None
        assert pooled_projections is not None
        assert timestep is not None
        if mode == "velocity":
            output = self.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                return_dict=False,
            )
            if not isinstance(output, tuple) or not isinstance(output[0], torch.Tensor):
                raise TypeError("SD3 transformer must return a velocity tensor tuple")
            return output[0]
        if mode not in {"velocity-and-features", "features"}:
            raise ValueError(f"unsupported SD3 critic forward mode: {mode!r}")
        velocity, features = sd3_velocity_and_features(
            self.transformer,
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            block_indices=block_indices,
            return_velocity=mode == "velocity-and-features",
        )
        return (velocity, features) if mode == "velocity-and-features" else features


def _critic_module(module: nn.Module) -> SD3ScaleWiseCriticModule:
    candidate = module
    wrapped = getattr(candidate, "module", None)
    if isinstance(wrapped, SD3ScaleWiseCriticModule):
        candidate = wrapped
    if not isinstance(candidate, SD3ScaleWiseCriticModule):
        raise TypeError("module must wrap SD3ScaleWiseCriticModule")
    return candidate


class SD3ScaleWiseCriticAdapter:
    """Feature-aware fake score adapter without an upstream trainer backend."""

    def __init__(
        self,
        module: nn.Module,
        *,
        checkpoint_identity: str,
        num_train_timesteps: int = 1000,
    ) -> None:
        _critic_module(module)
        identity = str(checkpoint_identity).strip()
        if not identity:
            raise ValueError("checkpoint_identity must be non-empty")
        if isinstance(num_train_timesteps, bool) or int(num_train_timesteps) < 2:
            raise ValueError("num_train_timesteps must be at least two")
        self.module = module
        self.checkpoint_identity = identity
        self.num_train_timesteps = int(num_train_timesteps)

    def audit_scale_wise_critic(
        self,
        *,
        classifier_blocks: tuple[int, ...],
        mmd_blocks: tuple[int, ...],
        discriminator_layers: int,
    ) -> None:
        critic = _critic_module(self.module)
        blocks = tuple(_sd3_graph(critic.transformer).transformer_blocks)
        requested = (*classifier_blocks, *mmd_blocks)
        if not requested or min(requested) < 0 or max(requested) >= len(blocks):
            raise ValueError("scale-wise feature block does not exist in SD3 graph")
        if int(discriminator_layers) != critic.discriminator_layers:
            raise ValueError("scale-wise discriminator depth differs from loaded head")

    def _inputs(
        self,
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt, pooled = _conditioning_tensors(conditioning)
        if noisy_latents.shape[0] != len(sample_ids):
            raise ValueError("SD3 latent batch and sample ids differ")
        if sigmas.ndim != 1 or sigmas.shape[0] != noisy_latents.shape[0]:
            raise ValueError("SD3 sigmas must have shape [B]")
        timestep = sigmas.to(
            device=noisy_latents.device,
            dtype=torch.float32,
        ) * float(self.num_train_timesteps)
        return prompt, pooled, timestep

    def predict_velocity(
        self,
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        training: bool,
        branch: str = "positive",
    ) -> torch.Tensor:
        del training, branch
        prompt, pooled, timestep = self._inputs(
            noisy_latents,
            sigmas,
            sample_ids,
            conditioning,
        )
        output = self.module(
            "velocity",
            noisy_latents,
            prompt,
            pooled,
            timestep,
        )
        if not isinstance(output, torch.Tensor):
            raise TypeError("SD3 critic must return a velocity tensor")
        return output

    def predict_velocity_and_features(
        self,
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        block_indices: tuple[int, ...],
        training: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        del training
        prompt, pooled, timestep = self._inputs(
            noisy_latents,
            sigmas,
            sample_ids,
            conditioning,
        )
        output = self.module(
            "velocity-and-features",
            noisy_latents,
            prompt,
            pooled,
            timestep,
            block_indices=block_indices,
        )
        if (
            not isinstance(output, tuple)
            or len(output) != 2
            or not isinstance(output[0], torch.Tensor)
            or not isinstance(output[1], tuple)
        ):
            raise TypeError("SD3 critic combined forward returned invalid values")
        velocity, features = output
        return velocity, features

    def extract_features(
        self,
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        sample_ids: tuple[str, ...],
        conditioning: Mapping[str, object],
        block_indices: tuple[int, ...],
        training: bool,
    ) -> tuple[torch.Tensor, ...]:
        del training
        prompt, pooled, timestep = self._inputs(
            noisy_latents,
            sigmas,
            sample_ids,
            conditioning,
        )
        features = self.module(
            "features",
            noisy_latents,
            prompt,
            pooled,
            timestep,
            block_indices=block_indices,
        )
        if not isinstance(features, tuple):
            raise TypeError("SD3 critic feature forward must return a tuple")
        return features

    def classify_features(
        self,
        pooled_features: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        output = self.module("classify", pooled_features=pooled_features)
        if not isinstance(output, tuple):
            raise TypeError("SD3 critic classifier must return a tuple")
        return output


__all__ = [
    "SD3AdapterDisabledTeacherAdapter",
    "SD3ScaleWiseCriticAdapter",
    "SD3ScaleWiseCriticModule",
    "SD3ScaleWisePredictionAdapter",
    "sd3_velocity_and_features",
]
