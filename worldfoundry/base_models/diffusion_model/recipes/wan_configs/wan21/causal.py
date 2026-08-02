# SPDX-License-Identifier: Apache-2.0
"""Module for base_models -> diffusion_model -> video -> wan -> wan_2p1 -> configs -> causal_wan_config.py functionality."""

from dataclasses import dataclass, field

from worldfoundry.core.configuration.model_config import DiTArchConfig, DiTConfig


def is_causal_block(n: str, m) -> bool:
    """Is causal block.

    Args:
        n: The n.
        m: The m.

    Returns:
        The return value.
    """
    parts = n.split(".")
    return len(parts) >= 2 and parts[0] == "blocks" and parts[1].isdigit()


@dataclass
class CausalWanArchConfig(DiTArchConfig):
    """Causal wan arch config implementation."""
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [is_causal_block]
    )


@dataclass
class CausalWanConfig(DiTConfig):
    """Causal wan config implementation."""
    arch_config: DiTArchConfig = field(default_factory=CausalWanArchConfig)

    prefix: str = "CausalWan"
