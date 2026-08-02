# adopted from
# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# and
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# and
# https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py
#
# thanks!

"""Classic convolutional diffusion layers shared by latent video models."""

import torch.nn as nn
import torch
from worldfoundry.core.model_loading.factory import instantiate_from_config


class CausalConv1d(torch.nn.Module):
    """1D causal convolution compatible with conv_nd(..., causal=True)."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=None):
        super().__init__()
        del padding
        self.padding = (kernel_size - 1) * dilation
        self.conv1d = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

    def forward(self, x):
        x = self.conv1d(x)
        if self.padding > 0:
            x = x[:, :, : -self.padding]
        return x


class CausalConv2d(torch.nn.Module):
    """2D causal convolution over the first spatial axis."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=None):
        super().__init__()
        del padding
        self.padding1 = (kernel_size - 1) * dilation
        self.padding2 = (kernel_size - 1) * dilation // 2
        self.conv2d = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(self.padding1, self.padding2),
            dilation=dilation,
        )

    def forward(self, x):
        x = self.conv2d(x)
        if self.padding1 > 0:
            x = x[:, :, : -self.padding1, :]
        return x


class CausalConv3d(torch.nn.Module):
    """3D causal convolution over the temporal axis."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=None):
        super().__init__()
        del padding
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            assert len(kernel_size) == 3, "kernel_size must be a tuple of length 3"
        self.padding1 = (kernel_size[0] - 1) * dilation
        self.padding2 = (kernel_size[1] - 1) * dilation // 2
        self.padding3 = (kernel_size[2] - 1) * dilation // 2
        self.conv3d = torch.nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(self.padding1, self.padding2, self.padding3),
            dilation=dilation,
        )

    @property
    def weight(self):
        return self.conv3d.weight

    @property
    def bias(self):
        return self.conv3d.bias

    def forward(self, x):
        x = self.conv3d(x)
        if self.padding1 > 0:
            x = x[:, :, : -self.padding1, :, :]
        return x


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def conv_nd(dims, *args, causal=False, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if causal:
        if dims == 1:
            return CausalConv1d(*args, **kwargs)
        elif dims == 2:
            return CausalConv2d(*args, **kwargs)
        elif dims == 3:
            return CausalConv3d(*args, **kwargs)
        raise ValueError(f"unsupported dimensions: {dims}")
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def nonlinearity(type='silu'):
    """Nonlinearity.

    Args:
        type: The type.
    """
    if type == 'silu':
        return nn.SiLU()
    elif type == 'leaky_relu':
        return nn.LeakyReLU()


class GroupNormSpecific(nn.GroupNorm):
    """Group norm specific implementation."""
    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        return super().forward(x)


def normalization(channels, num_groups=32):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNormSpecific(num_groups, channels)


class HybridConditioner(nn.Module):
    """Hybrid conditioner implementation."""

    def __init__(self, c_concat_config, c_crossattn_config):
        """Init.

        Args:
            c_concat_config: The c concat config.
            c_crossattn_config: The c crossattn config.
        """
        super().__init__()
        self.concat_conditioner = instantiate_from_config(c_concat_config)
        self.crossattn_conditioner = instantiate_from_config(c_crossattn_config)

    def forward(self, c_concat, c_crossattn):
        """Forward.

        Args:
            c_concat: The c concat.
            c_crossattn: The c crossattn.
        """
        c_concat = self.concat_conditioner(c_concat)
        c_crossattn = self.crossattn_conditioner(c_crossattn)
        return {'c_concat': [c_concat], 'c_crossattn': [c_crossattn]}
