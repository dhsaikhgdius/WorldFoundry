"""Reusable layers and tensor operations for diffusion transformers."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from math import prod

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Checkpoint-stable RMS normalization with optional affine weight."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        weight = torch.ones(int(dim), device=device, dtype=dtype) if elementwise_affine else None
        self.weight = nn.Parameter(weight) if weight is not None else None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        input_dtype = value.dtype
        normalized = value * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + self.eps)
        normalized = normalized.to(input_dtype)
        return normalized if self.weight is None else normalized * self.weight


class AdaLayerNorm(nn.Module):
    """Checkpoint-stable adaptive LayerNorm for diffusion transformers."""

    def __init__(
        self,
        dim: int,
        single: bool = False,
        dual: bool = False,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if single and dual:
            raise ValueError("single and dual AdaLayerNorm modes are mutually exclusive")
        self.single = bool(single)
        self.dual = bool(dual)
        factor = 9 if self.dual else (2 if self.single else 6)
        self.linear = nn.Linear(dim, dim * factor, device=device, dtype=dtype)
        self.norm = nn.LayerNorm(
            dim,
            elementwise_affine=False,
            eps=1e-6,
            device=device,
            dtype=dtype,
        )

    def forward(self, value: torch.Tensor, embedding: torch.Tensor):
        parameters = self.linear(torch.nn.functional.silu(embedding)).unsqueeze(1)
        if self.single:
            scale, shift = parameters.chunk(2, dim=2)
            return self.norm(value) * (1 + scale) + shift

        normalized = self.norm(value)
        if self.dual:
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                shift_msa_2,
                scale_msa_2,
                gate_msa_2,
            ) = parameters.chunk(9, dim=2)
            first = normalized * (1 + scale_msa) + shift_msa
            second = normalized * (1 + scale_msa_2) + shift_msa_2
            return first, gate_msa, shift_mlp, scale_mlp, gate_mlp, second, gate_msa_2

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = parameters.chunk(6, dim=2)
        normalized = normalized * (1 + scale_msa) + shift_msa
        return normalized, gate_msa, shift_mlp, scale_mlp, gate_mlp


def activation_layer(name: str) -> Callable[[], nn.Module]:
    """Resolve the small activation vocabulary shared by native DiTs."""

    normalized = name.strip().lower().replace("-", "_")
    if normalized == "gelu":
        return nn.GELU
    if normalized in {"gelu_tanh", "gelu_approximate"}:
        return lambda: nn.GELU(approximate="tanh")
    if normalized == "relu":
        return nn.ReLU
    if normalized in {"silu", "swish"}:
        return nn.SiLU
    raise ValueError(f"unknown activation type: {name}")


def normalization_layer(name: str) -> type[nn.Module]:
    """Resolve LayerNorm or RMSNorm without a model-local factory."""

    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"layer", "layer_norm", "layernorm"}:
        return nn.LayerNorm
    if normalized in {"rms", "rms_norm", "rmsnorm"}:
        return RMSNorm
    raise ValueError(f"unknown normalization type: {name}")


class DiTModulation(nn.Module):
    """Zero-initialized activation-plus-linear AdaLN parameter projection."""

    def __init__(
        self,
        hidden_size: int,
        factor: int,
        act_layer: Callable[[], nn.Module] = nn.SiLU,
        *,
        zero_init: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.act = act_layer()
        self.linear = nn.Linear(
            int(hidden_size),
            int(factor) * int(hidden_size),
            bias=True,
            dtype=dtype,
            device=device,
        )
        if zero_init:
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        value: torch.Tensor,
        *,
        secondary_value: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        output = self.linear(self.act(value))
        if secondary_value is None:
            return output
        return output, self.linear(self.act(secondary_value))


class TransformerMLP(nn.Module):
    """Checkpoint-stable two-layer transformer MLP shared by native DiTs."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] | None = None,
        bias: bool | tuple[bool, bool] = True,
        drop: float | tuple[float, float] = 0.0,
        use_conv: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        hidden_channels = int(hidden_channels or in_channels)
        out_features = int(out_features or in_channels)
        bias_pair = bias if isinstance(bias, tuple) else (bias, bias)
        drop_pair = drop if isinstance(drop, tuple) else (drop, drop)
        factory_kwargs = {"device": device, "dtype": dtype}
        linear = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.fc1 = linear(in_channels, hidden_channels, bias=bias_pair[0], **factory_kwargs)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_pair[0])
        self.norm = norm_layer(hidden_channels, **factory_kwargs) if norm_layer is not None else nn.Identity()
        self.fc2 = linear(hidden_channels, out_features, bias=bias_pair[1], **factory_kwargs)
        self.drop2 = nn.Dropout(drop_pair[1])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.fc1(value)
        value = self.act(value)
        value = self.drop1(value)
        value = self.norm(value)
        value = self.fc2(value)
        return self.drop2(value)


class MLPEmbedder(nn.Module):
    """Two-layer SiLU conditioning projection with stable checkpoint names."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True, **factory_kwargs)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True, **factory_kwargs)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(value)))


class ConditioningProjection(nn.Module):
    """Two-layer activation projection for text or auxiliary conditioning."""

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        act_layer: Callable[[], nn.Module],
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"dtype": dtype, "device": device}
        self.linear_1 = nn.Linear(in_channels, hidden_size, bias=True, **factory_kwargs)
        self.act_1 = act_layer()
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True, **factory_kwargs)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act_1(self.linear_1(value)))


class SinusoidalTimestepEmbedder(nn.Module):
    """Project cosine/sine timestep features with a checkpoint-stable MLP."""

    def __init__(
        self,
        hidden_size: int,
        act_layer: Callable[[], nn.Module],
        frequency_embedding_size: int = 256,
        max_period: float = 10_000.0,
        out_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.max_period = float(max_period)
        out_size = hidden_size if out_size is None else int(out_size)
        factory_kwargs = {"dtype": dtype, "device": device}
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, hidden_size, bias=True, **factory_kwargs),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True, **factory_kwargs),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        from worldfoundry.core.nn.transformer import sinusoidal_embedding_1d

        frequencies = sinusoidal_embedding_1d(
            self.frequency_embedding_size,
            timestep,
            self.max_period,
        ).type(self.mlp[0].weight.dtype)
        return self.mlp(frequencies)


class ConcatenatedLinear(nn.Module):
    """Concatenate two token streams before a checkpoint-visible linear map."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias, device=device, dtype=dtype)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat((first.contiguous(), second.contiguous()), dim=2).contiguous())


class DiTFinalLayer(nn.Module):
    """Shared AdaLN plus output projection used by patchified DiTs."""

    def __init__(
        self,
        hidden_size: int,
        patch_size: int | tuple[int, ...] | list[int],
        out_channels: int,
        act_layer: Callable[[], nn.Module],
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        patch_volume = patch_size * patch_size if isinstance(patch_size, int) else prod(patch_size)
        self.linear = nn.Linear(hidden_size, patch_volume * out_channels, bias=True, **factory_kwargs)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True, **factory_kwargs),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
        return self.linear(modulate_sequence(self.norm_final(value), shift=shift, scale=scale))


def modulate_sequence(
    value: torch.Tensor,
    shift: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply broadcast AdaLN shift/scale to a sequence tensor."""

    if scale is not None:
        value = value * (1 + scale.unsqueeze(1))
    if shift is not None:
        value = value + shift.unsqueeze(1)
    return value


def scale_shift(value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply an already-broadcast affine ``value * (1 + scale) + shift``."""

    return value * (1 + scale) + shift


def apply_gate(
    value: torch.Tensor,
    gate: torch.Tensor | None = None,
    tanh: bool = False,
) -> torch.Tensor:
    """Apply a broadcast residual gate to a sequence tensor."""

    if gate is None:
        return value
    gate = gate.tanh() if tanh else gate
    return value * gate.unsqueeze(1)


def modulate_sequence_with_prefix(
    value: torch.Tensor,
    shift: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    prefix_shift: torch.Tensor | None = None,
    prefix_scale: torch.Tensor | None = None,
    prefix_length: int | None = None,
) -> torch.Tensor:
    """Apply separate AdaLN parameters to a sequence prefix and its remainder."""

    if prefix_shift is None and prefix_scale is None:
        return modulate_sequence(value, shift=shift, scale=scale)
    if prefix_length is None:
        raise ValueError("prefix_length is required when prefix modulation is supplied")
    prefix = modulate_sequence(value[:, :prefix_length], shift=prefix_shift, scale=prefix_scale)
    remainder = modulate_sequence(value[:, prefix_length:], shift=shift, scale=scale)
    return torch.cat((prefix, remainder), dim=1)


def apply_gate_with_prefix(
    value: torch.Tensor,
    gate: torch.Tensor | None = None,
    prefix_gate: torch.Tensor | None = None,
    prefix_length: int | None = None,
    *,
    tanh: bool = False,
) -> torch.Tensor:
    """Apply separate residual gates to a sequence prefix and its remainder."""

    if prefix_gate is None:
        return apply_gate(value, gate, tanh=tanh)
    if prefix_length is None:
        raise ValueError("prefix_length is required when prefix_gate is supplied")
    prefix = apply_gate(value[:, :prefix_length], prefix_gate, tanh=tanh)
    remainder = apply_gate(value[:, prefix_length:], gate, tanh=tanh)
    return torch.cat((prefix, remainder), dim=1)


def velocity_to_denoised(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: float | torch.Tensor,
    *,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert a flow/velocity prediction to its denoised sample."""

    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(device=sample.device, dtype=calc_dtype)
    return (sample.to(calc_dtype) - velocity.to(calc_dtype) * sigma).to(sample.dtype)


__all__ = [
    "AdaLayerNorm",
    "ConditioningProjection",
    "ConcatenatedLinear",
    "DiTFinalLayer",
    "DiTModulation",
    "MLPEmbedder",
    "RMSNorm",
    "SinusoidalTimestepEmbedder",
    "TransformerMLP",
    "activation_layer",
    "apply_gate",
    "apply_gate_with_prefix",
    "modulate_sequence",
    "modulate_sequence_with_prefix",
    "normalization_layer",
    "scale_shift",
    "velocity_to_denoised",
]
