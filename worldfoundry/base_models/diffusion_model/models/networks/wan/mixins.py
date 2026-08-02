"""State-neutral methods shared by Wan transformer variants."""

from __future__ import annotations

import math

import torch
from torch import nn


class WanTransformerMethodsMixin:
    def unpatchify(self, values, grid_sizes, channels=None):
        channels = self.out_dim if channels is None else channels
        output = []
        for value, grid in zip(values, grid_sizes.tolist()):
            value = value[: math.prod(grid)].view(
                *grid,
                *self.patch_size,
                channels,
            )
            value = torch.einsum("fhwpqrc->cfphqwr", value)
            value = value.reshape(
                channels,
                *[
                    grid_size * patch_size
                    for grid_size, patch_size in zip(grid, self.patch_size)
                ],
            )
            output.append(value)
        return output

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for module in self.text_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
        for module in self.time_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
        nn.init.zeros_(self.head.head.weight)


__all__ = ["WanTransformerMethodsMixin"]
