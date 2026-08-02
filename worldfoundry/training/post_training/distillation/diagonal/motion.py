"""Spatial residual head used by diagonal temporal regression."""

from __future__ import annotations

from torch import Tensor, nn


class SpatialMotionHead(nn.Module):
    """Apply the released residual Conv2d head independently to each frame."""

    def __init__(
        self,
        channels: int,
        *,
        num_layers: int = 2,
        kernel_size: int = 1,
        hidden_dim: int = 64,
        norm_num_groups: int = 32,
        norm_epsilon: float = 1.0e-5,
    ) -> None:
        super().__init__()
        for name, value, minimum in (
            ("channels", channels, 1),
            ("num_layers", num_layers, 2),
            ("kernel_size", kernel_size, 1),
            ("hidden_dim", hidden_dim, 1),
            ("norm_num_groups", norm_num_groups, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if hidden_dim % norm_num_groups:
            raise ValueError("hidden_dim must be divisible by norm_num_groups")
        self.channels = channels
        self.in_activation = nn.SiLU()
        padding = (kernel_size - 1) // 2
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    channels if index == 0 else hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                ),
                nn.GroupNorm(
                    num_groups=norm_num_groups,
                    num_channels=hidden_dim,
                    eps=float(norm_epsilon),
                ),
                nn.SiLU(),
            )
            for index in range(num_layers - 1)
        )
        self.output = nn.Conv2d(hidden_dim, channels, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, value: Tensor) -> Tensor:
        if not isinstance(value, Tensor) or value.ndim != 5:
            raise ValueError("SpatialMotionHead requires [B,F,C,H,W] input")
        batch, frames, channels, height, width = value.shape
        if int(channels) != self.channels:
            raise ValueError("motion-head input channel count differs from its construction")
        residual = value
        hidden = self.in_activation(value.reshape(batch * frames, channels, height, width))
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.output(hidden).reshape(batch, frames, channels, height, width)
        return hidden + residual


def register_motion_head(
    student_module: nn.Module,
    motion_head: nn.Module,
    *,
    name: str = "diagonal_motion_head",
) -> nn.Module:
    """Register the trainable head inside the student role before wrapping."""

    if not isinstance(student_module, nn.Module) or not isinstance(motion_head, nn.Module):
        raise TypeError("student_module and motion_head must be nn.Module values")
    if not isinstance(name, str) or not name or "." in name:
        raise ValueError("motion-head registration name must be a non-empty child name")
    if hasattr(student_module, name):
        raise ValueError(f"student module already has child or attribute {name!r}")
    student_module.add_module(name, motion_head)
    return motion_head


__all__ = ["SpatialMotionHead", "register_motion_head"]
