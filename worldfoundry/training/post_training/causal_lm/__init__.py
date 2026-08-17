"""Lazy causal-language-model post-training integrations."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "QWEN3_CALCULATOR_TOOL_SCHEMA": ".qwen3",
    "Qwen3ActorCritic": ".qwen3",
    "Qwen3ChatCodec": ".qwen3",
    "Qwen3PostTrainingRun": ".qwen3",
    "Qwen3TokenPPOAdapter": ".qwen3",
    "Qwen3TokenPPORewardAdapter": ".qwen3",
    "materialize_qwen3_agentic_training_run": ".qwen3",
    "materialize_qwen3_post_training_run": ".qwen3",
    "materialize_qwen3_token_ppo_training_run": ".qwen3",
    "parse_qwen3_hermes_response": ".qwen3",
    "qwen3_turn_end_token_ids": ".qwen3",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
