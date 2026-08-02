"""Checkpoint-compatible LTX prompt encoder role."""

from .component import LTXPromptConditioner, build_ltx_prompt_conditioner

__all__ = [
    "LTXPromptConditioner",
    "build_ltx_prompt_conditioner",
]
