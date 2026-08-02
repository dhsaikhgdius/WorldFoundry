"""Reusable convolutional feed-forward blocks for image and video transformers."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _triple(value):
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 3:
            raise ValueError("expected three convolution options")
        return tuple(value)
    return (value, value, value)


def _activation(name):
    if name is None:
        return nn.Identity()
    if isinstance(name, type) and issubclass(name, nn.Module):
        return name()
    normalized = str(name).lower()
    if normalized in {"silu", "swish"}:
        return nn.SiLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "relu":
        return nn.ReLU()
    raise ValueError(f"unsupported activation {name!r}")


class _Conv2dLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        bias: bool = False,
        activation=None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2 if padding is None else padding,
            groups=groups,
            bias=bias,
        )
        self.activation = _activation(activation)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(value))


class GLUMBConvTemp(nn.Module):
    """Spatial gated depthwise MLP followed by a zero-initialized temporal residual."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_feature: int | None = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        use_bias=False,
        norm=(None, None, None),
        act=("silu", "silu", None),
        t_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if any(item is not None for item in _triple(norm)):
            raise ValueError("GLUMBConvTemp currently supports unnormalized checkpoint blocks")
        bias = _triple(use_bias)
        activations = _triple(act)
        out_features = out_feature or in_features
        self.inverted_conv = _Conv2dLayer(
            in_features,
            hidden_features * 2,
            1,
            bias=bool(bias[0]),
            activation=activations[0],
        )
        self.depth_conv = _Conv2dLayer(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=hidden_features * 2,
            bias=bool(bias[1]),
        )
        self.glu_act = _activation(activations[1])
        self.point_conv = _Conv2dLayer(
            hidden_features,
            out_features,
            1,
            bias=bool(bias[2]),
            activation=activations[2],
        )
        self.t_conv = nn.Conv2d(
            out_features,
            out_features,
            kernel_size=(t_kernel_size, 1),
            padding=(t_kernel_size // 2, 0),
            bias=False,
        )
        nn.init.zeros_(self.t_conv.weight)

    def _spatial(self, value: torch.Tensor) -> torch.Tensor:
        value = self.depth_conv(self.inverted_conv(value))
        values, gates = value.chunk(2, dim=1)
        return self.point_conv(values * self.glu_act(gates))

    def forward(self, value: torch.Tensor, HW=None, **kwargs) -> torch.Tensor:
        del kwargs
        if HW is None or len(HW) != 3:
            raise ValueError("GLUMBConvTemp requires HW=(frames, height, width)")
        batch, tokens, channels = value.shape
        frames, height, width = (int(item) for item in HW)
        if tokens != frames * height * width:
            raise ValueError("token count does not match HW")
        spatial = value.reshape(batch * frames, height, width, channels).permute(0, 3, 1, 2)
        spatial = self._spatial(spatial)
        temporal = spatial.view(batch, frames, channels, height * width).permute(0, 2, 1, 3)
        temporal = temporal + self.t_conv(temporal)
        return temporal.permute(0, 2, 3, 1).reshape(batch, tokens, channels)


class Mlp(nn.Module):
    """Vision-transformer MLP accepting the common optional ``HW`` argument."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer=nn.GELU,
        bias: bool = True,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, value: torch.Tensor, HW=None) -> torch.Tensor:
        del HW
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(value)))))


__all__ = ["GLUMBConvTemp", "Mlp"]
