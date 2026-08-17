"""Local callable tools for agentic training environments."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .contracts import AgentMessage, AgentToolCall


@dataclass(frozen=True, slots=True)
class LocalAgentTool:
    """A named pure-Python callable exposed to the rollout environment."""

    name: str
    function: Callable[[Mapping[str, object]], object]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("local tool name must be a non-empty string")
        if not callable(self.function):
            raise TypeError("local tool function must be callable")


@runtime_checkable
class AgentToolExecutor(Protocol):
    """Environment-owned execution seam for structured policy tool calls."""

    def execute(self, call: AgentToolCall) -> AgentMessage: ...


class LocalToolExecutor:
    """Expose explicitly registered Python callables to an agent rollout."""

    def __init__(self, tools: tuple[LocalAgentTool, ...]) -> None:
        resolved = tuple(tools)
        names = tuple(tool.name for tool in resolved)
        if not resolved or len(set(names)) != len(names):
            raise ValueError("local tools must be non-empty with unique names")
        self._tools = {tool.name: tool.function for tool in resolved}

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def execute(self, call: AgentToolCall) -> AgentMessage:
        function = self._tools.get(call.name)
        if function is None:
            return AgentMessage(
                role="tool",
                content=f"unknown tool: {call.name}",
                tool_call_id=call.call_id,
                name=call.name,
                tool_failed=True,
            )
        try:
            content = str(function(call.arguments))
            failed = False
        except Exception as error:
            content = f"{type(error).__name__}: {error}"
            failed = True
        return AgentMessage(
            role="tool",
            content=content,
            tool_call_id=call.call_id,
            name=call.name,
            tool_failed=failed,
        )


__all__ = ["AgentToolExecutor", "LocalAgentTool", "LocalToolExecutor"]
