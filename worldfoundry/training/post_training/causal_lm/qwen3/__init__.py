"""Lazy public surface for native Qwen3 post-training."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "QWEN3_CALCULATOR_TOOL_SCHEMA": ".codec",
    "Qwen3ChatCodec": ".codec",
    "parse_qwen3_hermes_response": ".codec",
    "qwen3_turn_end_token_ids": ".codec",
    "Qwen3ActorCritic": ".models",
    "Qwen3TokenPPOAdapter": ".models",
    "Qwen3TokenPPORewardAdapter": ".models",
    "Qwen3PostTrainingRun": ".materializer",
    "Qwen3ActorHostedTrainingRun": ".materializer",
    "materialize_qwen3_agentic_training_run": ".materializer",
    "materialize_qwen3_post_training_run": ".materializer",
    "materialize_qwen3_token_ppo_training_run": ".materializer",
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
