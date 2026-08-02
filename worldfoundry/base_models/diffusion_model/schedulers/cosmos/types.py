# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Cosmos component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LabelImageCondition:
    """Label image condition implementation."""

    label: torch.Tensor

    def get_classifier_free_guidance_condition(self) -> "LabelImageCondition":
        """Get classifier free guidance condition.

        Returns:
            The return value.
        """
        return LabelImageCondition(torch.zeros_like(self.label))


@dataclass
class DenoisePrediction:
    """Denoise prediction implementation."""

    x0: torch.Tensor
    eps: Optional[torch.Tensor] = None
    logvar: Optional[torch.Tensor] = None
