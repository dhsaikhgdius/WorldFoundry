# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""CAME optimizer used by the official SANA training recipes.

The update is adapted from ``CAMEWrapper`` in NVlabs/Sana.  It intentionally
keeps SANA's factorization policy: matrices and 1x1 convolutions use row/column
statistics, while all other tensors retain an unfactored second moment.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import nn


class CAME(torch.optim.Optimizer):
    """Confidence-guided adaptive memory-efficient optimization."""

    def __init__(
        self,
        params: Iterable[nn.Parameter] | Iterable[dict[str, object]],
        *,
        lr: float,
        eps: tuple[float, float] = (1.0e-30, 1.0e-16),
        clip_threshold: float = 1.0,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        weight_decay: float = 0.0,
    ) -> None:
        learning_rate = float(lr)
        resolved_eps = tuple(float(value) for value in eps)
        resolved_betas = tuple(float(value) for value in betas)
        threshold = float(clip_threshold)
        decay = float(weight_decay)
        if learning_rate <= 0.0:
            raise ValueError("CAME learning rate must be positive")
        if len(resolved_eps) != 2 or any(value <= 0.0 for value in resolved_eps):
            raise ValueError("CAME eps must contain two positive values")
        if len(resolved_betas) != 3 or any(not 0.0 <= value < 1.0 for value in resolved_betas):
            raise ValueError("CAME betas must contain three values in [0, 1)")
        if threshold <= 0.0:
            raise ValueError("CAME clip_threshold must be positive")
        if decay < 0.0:
            raise ValueError("CAME weight_decay must be non-negative")
        defaults = {
            "lr": learning_rate,
            "eps": resolved_eps,
            "clip_threshold": threshold,
            "betas": resolved_betas,
            "weight_decay": decay,
        }
        super().__init__(params, defaults)
        self._materialize_checkpointable_state()

    @property
    def supports_memory_efficient_fp16(self) -> bool:
        return True

    @property
    def supports_flat_params(self) -> bool:
        return False

    @staticmethod
    def _factorization(shape: torch.Size) -> tuple[bool, str]:
        if len(shape) == 4:
            return (True, "1x1_conv") if shape[2:] == torch.Size((1, 1)) else (False, "conv")
        if len(shape) == 2:
            return True, "linear"
        return False, "other"

    @staticmethod
    def _rms(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approximate_inverse_root(
        row_moment: torch.Tensor,
        column_moment: torch.Tensor,
    ) -> torch.Tensor:
        row_factor = (row_moment / row_moment.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        column_factor = column_moment.unsqueeze(-2).rsqrt()
        return row_factor.mul(column_factor)

    @staticmethod
    def _initialize_state(
        state: dict[str, object],
        gradient: torch.Tensor,
        *,
        factored: bool,
    ) -> None:
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(gradient)
        if factored:
            state["exp_avg_sq_row"] = torch.zeros(gradient.shape[0], dtype=gradient.dtype, device=gradient.device)
            state["exp_avg_sq_col"] = torch.zeros(gradient.shape[1], dtype=gradient.dtype, device=gradient.device)
            state["exp_avg_res_row"] = torch.zeros(gradient.shape[0], dtype=gradient.dtype, device=gradient.device)
            state["exp_avg_res_col"] = torch.zeros(gradient.shape[1], dtype=gradient.dtype, device=gradient.device)
        else:
            state["exp_avg_sq"] = torch.zeros_like(gradient)
        state["RMS"] = 0

    def _materialize_checkpointable_state(self) -> None:
        """Create deterministic state slots so a fresh optimizer can load DCP."""

        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state[parameter]
                if state:
                    continue
                template = parameter.detach()
                if template.dtype in {torch.float16, torch.bfloat16}:
                    template = template.float()
                factored, _ = self._factorization(template.shape)
                self._initialize_state(state, template, factored=factored)
                state["RMS"] = self._rms(parameter.detach())

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore optimizer state while preserving SANA's fp32 low-precision moments."""

        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.dtype not in {torch.float16, torch.bfloat16}:
                    continue
                state = self.state[parameter]
                for name, value in tuple(state.items()):
                    if isinstance(value, torch.Tensor) and value.is_floating_point():
                        state[name] = value.float()

    @staticmethod
    def _matrix_view(tensor: torch.Tensor, *, layer_type: str) -> torch.Tensor:
        return tensor.squeeze(-1).squeeze(-1) if layer_type == "1x1_conv" else tensor

    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        """Perform one SANA-compatible CAME update."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]
            eps_square, eps_instability = group["eps"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach()
                if gradient.is_sparse:
                    raise RuntimeError("CAME does not support sparse gradients")
                if gradient.dtype in {torch.float16, torch.bfloat16}:
                    gradient = gradient.float()

                state = self.state[parameter]
                factored, layer_type = self._factorization(gradient.shape)
                if not state:
                    self._initialize_state(state, gradient, factored=factored)

                state["step"] += 1
                state["RMS"] = self._rms(parameter.data)
                squared_gradient = gradient.square().add(eps_square)

                if factored:
                    squared_matrix = self._matrix_view(squared_gradient, layer_type=layer_type)
                    row_moment = state["exp_avg_sq_row"]
                    column_moment = state["exp_avg_sq_col"]
                    row_moment.mul_(beta2).add_(squared_matrix.mean(dim=1), alpha=1.0 - beta2)
                    column_moment.mul_(beta2).add_(squared_matrix.mean(dim=0), alpha=1.0 - beta2)
                    update = self._approximate_inverse_root(row_moment, column_moment)
                    if layer_type == "1x1_conv":
                        update = update.view(gradient.shape)
                    update.mul_(gradient)
                else:
                    second_moment = state["exp_avg_sq"]
                    second_moment.mul_(beta2).add_(squared_gradient, alpha=1.0 - beta2)
                    update = second_moment.rsqrt().mul_(gradient)

                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
                first_moment = state["exp_avg"]
                first_moment.mul_(beta1).add_(update, alpha=1.0 - beta1)

                residual = (update - first_moment).square().add_(eps_instability)
                if factored:
                    residual_matrix = self._matrix_view(residual, layer_type=layer_type)
                    residual_row_moment = state["exp_avg_res_row"]
                    residual_column_moment = state["exp_avg_res_col"]
                    residual_row_moment.mul_(beta3).add_(residual_matrix.mean(dim=1), alpha=1.0 - beta3)
                    residual_column_moment.mul_(beta3).add_(residual_matrix.mean(dim=0), alpha=1.0 - beta3)
                    confidence = self._approximate_inverse_root(
                        residual_row_moment,
                        residual_column_moment,
                    )
                    if layer_type == "1x1_conv":
                        confidence = confidence.view(gradient.shape)
                    final_update = confidence.mul_(first_moment)
                else:
                    final_update = first_moment.clone()

                if group["weight_decay"] != 0.0:
                    parameter.data.add_(parameter.data, alpha=-group["weight_decay"] * group["lr"])
                parameter.data.add_(final_update, alpha=-group["lr"])

        return loss


__all__ = ["CAME"]
