"""Stateful prompt expansion for grouped agentic policy rollouts."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .contracts import AgenticRolloutRequest, AgenticSampleRequest, AgentMessage

AGENTIC_PROMPT_LOADER_STATE_SCHEMA = "worldfoundry-agentic-prompt-loader"


@dataclass(frozen=True, slots=True)
class AgenticPrompt:
    """One initial conversation expanded into a complete policy group."""

    prompt_id: str
    messages: tuple[AgentMessage, ...]
    conditioning: Mapping[str, object] = field(default_factory=dict)
    split: str = "train"

    def __post_init__(self) -> None:
        validated = AgenticSampleRequest(
            sample_id=self.prompt_id,
            group_id=self.prompt_id,
            messages=tuple(self.messages),
            conditioning=self.conditioning,
        )
        split = str(self.split).strip().lower().replace("_", "-")
        if not split:
            raise ValueError("agentic prompt split cannot be empty")
        object.__setattr__(self, "prompt_id", validated.sample_id)
        object.__setattr__(self, "messages", validated.messages)
        object.__setattr__(self, "conditioning", validated.conditioning)
        object.__setattr__(self, "split", split)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> AgenticPrompt:
        if not isinstance(value, Mapping):
            raise TypeError("agentic prompt record must be a mapping")
        raw_messages = value.get("messages")
        if not isinstance(raw_messages, Sequence) or isinstance(
            raw_messages,
            (str, bytes, bytearray),
        ):
            raise TypeError("agentic prompt messages must be a sequence")
        messages: list[AgentMessage] = []
        for raw in raw_messages:
            if not isinstance(raw, Mapping):
                raise TypeError("agentic prompt message must be a mapping")
            messages.append(
                AgentMessage(
                    role=str(raw.get("role", "")),  # type: ignore[arg-type]
                    content=str(raw.get("content", "")),
                )
            )
        conditioning = value.get("conditioning", {})
        if not isinstance(conditioning, Mapping):
            raise TypeError("agentic prompt conditioning must be a mapping")
        return cls(
            prompt_id=str(value.get("prompt_id", "")),
            messages=tuple(messages),
            conditioning=conditioning,
            split=str(value.get("split", "train")),
        )


def load_agentic_prompts(
    path: str | Path,
    *,
    split: str = "train",
) -> tuple[AgenticPrompt, ...]:
    """Load JSONL prompt records for one configured split."""

    source = Path(path).expanduser().resolve()
    selected_split = str(split).strip().lower().replace("_", "-")
    prompts = tuple(
        prompt
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(),
            start=1,
        )
        if line.strip()
        for prompt in (AgenticPrompt.from_mapping(_prompt_record(line, source=source, line_number=line_number)),)
        if prompt.split == selected_split
    )
    if not prompts:
        raise ValueError(f"agentic prompt manifest has no records for split {selected_split!r}")
    return prompts


def _prompt_record(
    line: str,
    *,
    source: Path,
    line_number: int,
) -> Mapping[str, object]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid agentic prompt JSON at {source}:{line_number}") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"agentic prompt record must be an object at {source}:{line_number}")
    return value


class NativeAgenticPromptLoader(Iterator[AgenticRolloutRequest]):
    """Expand prompt groups while preserving exact epoch and batch position."""

    def __init__(
        self,
        prompts: Sequence[AgenticPrompt],
        *,
        group_size: int,
        groups_per_batch: int,
        policy_revision: Callable[[], str],
        sampling_temperature: float,
        max_turns: int,
        shuffle: bool = True,
        shuffle_seed: int = 42,
        tail_policy: str = "drop",
    ) -> None:
        resolved = tuple(prompts)
        if not resolved or not all(isinstance(prompt, AgenticPrompt) for prompt in resolved):
            raise ValueError("agentic prompt loader requires AgenticPrompt values")
        prompt_ids = tuple(prompt.prompt_id for prompt in resolved)
        if len(set(prompt_ids)) != len(prompt_ids):
            raise ValueError("agentic prompt ids must be unique")
        if isinstance(group_size, bool) or int(group_size) < 2:
            raise ValueError("agentic group_size must be at least two")
        if isinstance(groups_per_batch, bool) or int(groups_per_batch) <= 0:
            raise ValueError("agentic groups_per_batch must be positive")
        if not callable(policy_revision):
            raise TypeError("policy_revision must be callable")
        if isinstance(max_turns, bool) or int(max_turns) <= 0:
            raise ValueError("agentic max_turns must be positive")
        resolved_tail = str(tail_policy).strip().lower().replace("_", "-")
        if resolved_tail not in {"drop", "pad", "uneven"}:
            raise ValueError("agentic tail_policy must be drop, pad, or uneven")
        if resolved_tail == "drop" and len(resolved) < int(groups_per_batch):
            raise ValueError("drop tail policy requires at least one complete prompt batch")

        self.prompts = resolved
        self.group_size = int(group_size)
        self.groups_per_batch = int(groups_per_batch)
        self.policy_revision = policy_revision
        self.sampling_temperature = float(sampling_temperature)
        self.max_turns = int(max_turns)
        self.shuffle = bool(shuffle)
        self.shuffle_seed = int(shuffle_seed)
        self.tail_policy = resolved_tail
        self.epoch = 0
        self.cursor = 0
        self.completed_batches = 0
        self.order = self._order_for_epoch(self.epoch)

    def _order_for_epoch(self, epoch: int) -> list[int]:
        order = list(range(len(self.prompts)))
        if self.shuffle:
            random.Random(self.shuffle_seed + epoch).shuffle(order)
        return order

    def _next_epoch(self) -> None:
        self.epoch += 1
        self.cursor = 0
        self.order = self._order_for_epoch(self.epoch)

    def _take_prompt_indices(self) -> tuple[int, ...]:
        remaining = len(self.order) - self.cursor
        if remaining == 0:
            self._next_epoch()
            remaining = len(self.order)
        if self.tail_policy == "drop" and remaining < self.groups_per_batch:
            self._next_epoch()
            remaining = len(self.order)

        take = min(remaining, self.groups_per_batch)
        indices = list(self.order[self.cursor : self.cursor + take])
        self.cursor += take
        if self.tail_policy == "pad" and take < self.groups_per_batch:
            while len(indices) < self.groups_per_batch:
                self._next_epoch()
                needed = self.groups_per_batch - len(indices)
                epoch_take = min(needed, len(self.order))
                indices.extend(self.order[:epoch_take])
                self.cursor = epoch_take
        return tuple(indices)

    def __iter__(self) -> NativeAgenticPromptLoader:
        return self

    def __next__(self) -> AgenticRolloutRequest:
        prompt_indices = self._take_prompt_indices()
        batch_index = self.completed_batches
        samples: list[AgenticSampleRequest] = []
        for group_index, prompt_index in enumerate(prompt_indices):
            prompt = self.prompts[prompt_index]
            group_id = f"{prompt.prompt_id}::batch-{batch_index:08d}-group-{group_index:04d}"
            samples.extend(
                AgenticSampleRequest(
                    sample_id=f"{group_id}::sample-{sample_index:04d}",
                    group_id=group_id,
                    messages=prompt.messages,
                    conditioning=prompt.conditioning,
                )
                for sample_index in range(self.group_size)
            )
        revision = str(self.policy_revision()).strip()
        if not revision:
            raise ValueError("active agentic policy revision cannot be empty")
        self.completed_batches += 1
        return AgenticRolloutRequest(
            samples=tuple(samples),
            policy_revision=revision,
            sampling_temperature=self.sampling_temperature,
            max_turns=self.max_turns,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": AGENTIC_PROMPT_LOADER_STATE_SCHEMA,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "completed_batches": self.completed_batches,
            "order": list(self.order),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping) or set(state_dict) != {
            "schema",
            "epoch",
            "cursor",
            "completed_batches",
            "order",
        }:
            raise ValueError("agentic prompt-loader state fields differ")
        if state_dict["schema"] != AGENTIC_PROMPT_LOADER_STATE_SCHEMA:
            raise ValueError("unsupported agentic prompt-loader state schema")
        epoch = int(state_dict["epoch"])
        cursor = int(state_dict["cursor"])
        completed_batches = int(state_dict["completed_batches"])
        order = [int(index) for index in state_dict["order"]]  # type: ignore[union-attr]
        if (
            epoch < 0
            or completed_batches < 0
            or not 0 <= cursor <= len(self.prompts)
            or sorted(order) != list(range(len(self.prompts)))
        ):
            raise ValueError("saved agentic prompt-loader position is invalid")
        self.epoch = epoch
        self.cursor = cursor
        self.completed_batches = completed_batches
        self.order = order


__all__ = [
    "AGENTIC_PROMPT_LOADER_STATE_SCHEMA",
    "AgenticPrompt",
    "NativeAgenticPromptLoader",
    "load_agentic_prompts",
]
