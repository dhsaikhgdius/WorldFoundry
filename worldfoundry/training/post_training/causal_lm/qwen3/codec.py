"""Qwen3 chat-template and Hermes tool-call semantics."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

from ...agentic.causal_lm import TokenizerAgenticChatCodec
from ...agentic.contracts import AgentMessage, AgentToolCall

QWEN3_CALCULATOR_TOOL_SCHEMA: Mapping[str, object] = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a basic arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression to evaluate.",
                }
            },
            "required": ["expression"],
        },
    },
}

_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TRAILING_SPECIAL = re.compile(r"(?:<\|im_end\|>|<\|endoftext\|>)+\s*$")


def parse_qwen3_hermes_response(text: str) -> AgentMessage:
    """Parse Qwen3's Hermes JSON objects, including parallel tool calls."""

    rendered = _TRAILING_SPECIAL.sub("", str(text)).strip()
    calls: list[AgentToolCall] = []
    for index, match in enumerate(_TOOL_CALL.finditer(rendered)):
        payload = json.loads(match.group(1))
        if not isinstance(payload, Mapping):
            raise TypeError("Qwen3 tool call must contain a JSON object")
        name = str(payload.get("name", "")).strip()
        arguments = payload.get("arguments", {})
        if not name or not isinstance(arguments, Mapping):
            raise ValueError("Qwen3 tool call requires a name and object arguments")
        call_id = str(payload.get("id", "")).strip() or f"tool-call-{index:04d}"
        calls.append(
            AgentToolCall(
                call_id=call_id,
                name=name,
                arguments=arguments,
            )
        )
    content = _TOOL_CALL.sub("", rendered).strip()
    return AgentMessage(
        role="assistant",
        content=content,
        tool_calls=tuple(calls),
    )


def qwen3_turn_end_token_ids(tokenizer: object) -> tuple[int, ...]:
    """Return configured EOS ids plus Qwen3's assistant-turn boundary."""

    configured = getattr(tokenizer, "eos_token_id", None)
    eos_values = configured if isinstance(configured, Sequence) else (configured,)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        raise TypeError("Qwen3 tokenizer must expose convert_tokens_to_ids")
    im_end = convert("<|im_end|>")
    values = tuple(int(value) for value in (*eos_values, im_end) if value is not None)
    if not values:
        raise ValueError("Qwen3 tokenizer exposes no turn-end token ids")
    return tuple(dict.fromkeys(values))


class Qwen3ChatCodec(TokenizerAgenticChatCodec):
    """Bind the Transformers Qwen3 template to Hermes response parsing."""

    def __init__(
        self,
        tokenizer: object,
        *,
        tool_schemas: Sequence[Mapping[str, object]] = (QWEN3_CALCULATOR_TOOL_SCHEMA,),
        enable_thinking: bool = False,
    ) -> None:
        super().__init__(
            tokenizer,
            response_parser=parse_qwen3_hermes_response,
            tool_schemas=tool_schemas,
            eos_token_ids=qwen3_turn_end_token_ids(tokenizer),
            template_options={"enable_thinking": bool(enable_thinking)},
        )


__all__ = [
    "QWEN3_CALCULATOR_TOOL_SCHEMA",
    "Qwen3ChatCodec",
    "parse_qwen3_hermes_response",
    "qwen3_turn_end_token_ids",
]
