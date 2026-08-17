"""Causal-language-model generation and replay for agentic token policies."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import torch
from torch import nn

from ..rl.algorithms.token_policy.contracts import (
    PackedTokenReplayBatch,
    PackedTokenTrajectory,
    TokenReplayResult,
)
from .contracts import (
    AgenticAssistantTurn,
    AgentMessage,
    agentic_trajectory_from_packed,
)


@runtime_checkable
class AgenticChatCodec(Protocol):
    """Model-family chat serialization and structured assistant parsing."""

    eos_token_ids: tuple[int, ...]

    def encode_prompt(
        self,
        messages: tuple[AgentMessage, ...],
        *,
        conditioning: Mapping[str, object],
        device: torch.device,
    ) -> Mapping[str, torch.Tensor]: ...

    def decode_assistant(self, token_ids: torch.Tensor) -> AgentMessage: ...


def _chat_message(message: AgentMessage) -> dict[str, object]:
    value: dict[str, object] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        value["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": dict(call.arguments)},
            }
            for call in message.tool_calls
        ]
    if message.role == "tool":
        value["tool_call_id"] = message.tool_call_id
        value["name"] = message.name
    return value


class TokenizerAgenticChatCodec:
    """Use a tokenizer chat template and a model-specific response parser."""

    def __init__(
        self,
        tokenizer: object,
        *,
        response_parser: Callable[[str], AgentMessage] | None = None,
        tool_schemas: Sequence[Mapping[str, object]] = (),
        eos_token_ids: Sequence[int] | None = None,
        template_options: Mapping[str, object] | None = None,
    ) -> None:
        if not callable(getattr(tokenizer, "apply_chat_template", None)):
            raise TypeError("agentic tokenizer must expose apply_chat_template")
        if not callable(getattr(tokenizer, "decode", None)):
            raise TypeError("agentic tokenizer must expose decode")
        raw_eos = eos_token_ids
        if raw_eos is None:
            configured = getattr(tokenizer, "eos_token_id", None)
            raw_eos = configured if isinstance(configured, Sequence) else (configured,)
        self.eos_token_ids = tuple(int(value) for value in raw_eos if value is not None)
        if not self.eos_token_ids:
            raise ValueError("agentic tokenizer requires at least one EOS token id")
        self.tokenizer = tokenizer
        self.response_parser = response_parser or (lambda text: AgentMessage(role="assistant", content=text))
        self.tool_schemas = tuple(MappingProxyType(dict(schema)) for schema in tool_schemas)
        self.template_options = MappingProxyType(dict(template_options or {}))

    def encode_prompt(
        self,
        messages: tuple[AgentMessage, ...],
        *,
        conditioning: Mapping[str, object],
        device: torch.device,
    ) -> Mapping[str, torch.Tensor]:
        del conditioning
        options: dict[str, object] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        if self.tool_schemas:
            options["tools"] = [dict(schema) for schema in self.tool_schemas]
        options.update(self.template_options)
        encoded = self.tokenizer.apply_chat_template(
            [_chat_message(message) for message in messages],
            **options,
        )
        if isinstance(encoded, torch.Tensor):
            input_ids = encoded
            attention_mask = torch.ones_like(input_ids)
        elif isinstance(encoded, Mapping):
            input_ids = encoded.get("input_ids")
            attention_mask = encoded.get("attention_mask")
            if not isinstance(input_ids, torch.Tensor):
                raise TypeError("chat template output is missing input_ids")
            if not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.ones_like(input_ids)
        else:
            raise TypeError("chat template must return a tensor or mapping")
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        if tuple(input_ids.shape) != tuple(attention_mask.shape) or int(input_ids.shape[0]) != 1:
            raise ValueError("agentic chat prompt must have one input_ids/attention_mask row")
        return {
            "input_ids": input_ids.to(device=device, dtype=torch.int64),
            "attention_mask": attention_mask.to(device=device),
        }

    def decode_assistant(self, token_ids: torch.Tensor) -> AgentMessage:
        text = self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=False)
        message = self.response_parser(str(text))
        if not isinstance(message, AgentMessage) or message.role != "assistant":
            raise TypeError("agentic response parser must return an assistant AgentMessage")
        return message


@dataclass(frozen=True, slots=True)
class CausalLMGenerationConfig:
    max_new_tokens: int = 512

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")


def _module_device(module: nn.Module) -> torch.device:
    parameter = next(module.parameters(), None)
    return torch.device("cpu") if parameter is None else parameter.device


def _causal_lm_autocast(module: nn.Module, compute_dtype: torch.dtype):
    device_type = _module_device(module).type
    enabled = (compute_dtype is torch.bfloat16 and device_type in {"cpu", "cuda"}) or (
        compute_dtype is torch.float16 and device_type == "cuda"
    )
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device_type, dtype=compute_dtype)


class CausalLMAgenticPolicyAdapter:
    """Generate tool-capable turns and replay their sampled tokens with one LM."""

    def __init__(
        self,
        module: nn.Module,
        codec: AgenticChatCodec,
        *,
        generation: CausalLMGenerationConfig | None = None,
        compute_dtype: torch.dtype = torch.float32,
    ) -> None:
        if not isinstance(module, nn.Module):
            raise TypeError("agentic causal LM must be an nn.Module")
        if not isinstance(codec, AgenticChatCodec):
            raise TypeError("codec must implement AgenticChatCodec")
        if compute_dtype not in {torch.float32, torch.bfloat16, torch.float16}:
            raise ValueError("compute_dtype must be float32, bfloat16, or float16")
        self.module = module
        self.codec = codec
        self.generation = generation or CausalLMGenerationConfig()
        self.compute_dtype = compute_dtype

    @staticmethod
    def _logits(output: object) -> torch.Tensor:
        logits = getattr(output, "logits", None)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise TypeError("causal LM output must expose logits with shape [B,L,V]")
        return logits

    def generate_turn(
        self,
        *,
        sample_id: str,
        messages: tuple[AgentMessage, ...],
        policy_revision: str,
        sampling_temperature: float,
        rollout_index: int,
        turn_index: int,
        conditioning: Mapping[str, object],
        generator: torch.Generator | None,
    ) -> AgenticAssistantTurn:
        del sample_id, policy_revision, rollout_index, turn_index
        device = _module_device(self.module)
        encoded = self.codec.encode_prompt(messages, conditioning=conditioning, device=device)
        sequence = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        sampled: list[torch.Tensor] = []
        log_probs: list[torch.Tensor] = []
        hit_eos = False
        self.module.eval()
        with torch.no_grad(), _causal_lm_autocast(self.module, self.compute_dtype):
            for _ in range(self.generation.max_new_tokens):
                output = self.module(
                    input_ids=sequence,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                logits = self._logits(output)[:, -1].float() / sampling_temperature
                probabilities = torch.softmax(logits, dim=-1)
                token = torch.multinomial(probabilities, 1, generator=generator)
                log_prob = torch.log_softmax(logits, dim=-1).gather(1, token)
                sampled.append(token.squeeze(0))
                log_probs.append(log_prob.squeeze(0))
                sequence = torch.cat((sequence, token), dim=1)
                attention_mask = torch.cat(
                    (attention_mask, attention_mask.new_ones((1, 1))),
                    dim=1,
                )
                if int(token.item()) in self.codec.eos_token_ids:
                    hit_eos = True
                    break
        token_ids = torch.cat(sampled).to(dtype=torch.int64)
        message = self.codec.decode_assistant(token_ids)
        if message.tool_calls:
            finish_reason = "tool_calls"
        elif hit_eos:
            finish_reason = "stop"
        else:
            finish_reason = "length"
        return AgenticAssistantTurn(
            message=message,
            token_ids=token_ids,
            old_log_probs=torch.cat(log_probs).to(dtype=torch.float32),
            finish_reason=finish_reason,
        )

    def _turn_log_probs(
        self,
        messages: tuple[AgentMessage, ...],
        response_ids: torch.Tensor,
        *,
        conditioning: Mapping[str, object],
        sampling_temperature: float,
    ) -> torch.Tensor:
        device = _module_device(self.module)
        encoded = self.codec.encode_prompt(messages, conditioning=conditioning, device=device)
        prompt_ids = encoded["input_ids"]
        prompt_mask = encoded["attention_mask"]
        response = response_ids.to(device=device, dtype=torch.int64).unsqueeze(0)
        input_ids = torch.cat((prompt_ids, response), dim=1)
        attention_mask = torch.cat(
            (prompt_mask, prompt_mask.new_ones(response.shape)),
            dim=1,
        )
        with _causal_lm_autocast(self.module, self.compute_dtype):
            output = self.module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        prompt_length = int(prompt_ids.shape[1])
        logits = self._logits(output)[
            0,
            prompt_length - 1 : prompt_length + int(response.shape[1]) - 1,
        ].float()
        return (
            torch.log_softmax(logits / sampling_temperature, dim=-1)
            .gather(
                1,
                response.squeeze(0).unsqueeze(1),
            )
            .squeeze(1)
        )

    def replay(
        self,
        trajectory: PackedTokenTrajectory | PackedTokenReplayBatch,
        *,
        training: bool,
    ) -> TokenReplayResult:
        agentic = agentic_trajectory_from_packed(trajectory)
        # Rollout is sampled with dropout disabled, so replay must evaluate the
        # same policy distribution. Evaluation mode still permits gradients.
        self.module.eval()
        replayed: list[torch.Tensor] = []
        with torch.set_grad_enabled(training):
            for sample in agentic.samples:
                messages = list(sample.request.messages)
                for turn in sample.turns:
                    replayed.append(
                        self._turn_log_probs(
                            tuple(messages),
                            turn.assistant.token_ids,
                            conditioning=sample.request.conditioning,
                            sampling_temperature=trajectory.sampling_temperature,
                        )
                    )
                    messages.append(turn.assistant.message)
                    messages.extend(turn.tool_results)
        log_probs = torch.cat(replayed)
        if int(log_probs.numel()) != trajectory.token_count:
            raise ValueError("agentic replay token count differs from the packed trajectory")
        return TokenReplayResult(
            log_probs=log_probs,
            sampling_temperature=trajectory.sampling_temperature,
        )


__all__ = [
    "AgenticChatCodec",
    "CausalLMAgenticPolicyAdapter",
    "CausalLMGenerationConfig",
    "TokenizerAgenticChatCodec",
]
