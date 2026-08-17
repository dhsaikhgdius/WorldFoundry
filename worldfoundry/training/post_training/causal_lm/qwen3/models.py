"""Transformers-backed Qwen3 actor-critic adapters for native token PPO."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from contextlib import nullcontext

import torch
from torch import nn

from ...agentic.contracts import AgentMessage
from ...rl.algorithms.token_ppo.contracts import (
    PackedTokenPPOReplayBatch,
    PackedTokenPPOTrajectory,
    TokenPPOReplayResult,
    TokenPPORolloutRequest,
)
from .codec import Qwen3ChatCodec


def _module_device(module: nn.Module) -> torch.device:
    parameter = next(module.parameters(), None)
    return torch.device("cpu") if parameter is None else parameter.device


def _qwen3_autocast(module: nn.Module, compute_dtype: torch.dtype):
    device_type = _module_device(module).type
    enabled = (compute_dtype is torch.bfloat16 and device_type in {"cpu", "cuda"}) or (
        compute_dtype is torch.float16 and device_type == "cuda"
    )
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device_type, dtype=compute_dtype)


def _logits(output: object) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise TypeError("Qwen3 causal LM output must expose logits with shape [B,L,V]")
    return logits


def _hidden(output: object) -> torch.Tensor:
    hidden_states = getattr(output, "hidden_states", None)
    if not isinstance(hidden_states, Sequence) or not hidden_states:
        raise TypeError("Qwen3 PPO replay requires output_hidden_states")
    hidden = hidden_states[-1]
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise TypeError("Qwen3 final hidden state must have shape [B,L,H]")
    return hidden


def _messages(sample: Mapping[str, object]) -> tuple[AgentMessage, ...]:
    messages = sample.get("messages")
    if not isinstance(messages, tuple) or not all(isinstance(message, AgentMessage) for message in messages):
        raise TypeError("Qwen3 PPO samples must contain an AgentMessage tuple")
    return messages


class Qwen3ActorCritic(nn.Module):
    """Qwen3 causal policy with a zero-initialized FP32 scalar value head."""

    def __init__(
        self,
        policy: nn.Module,
        *,
        hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(policy, nn.Module):
            raise TypeError("Qwen3 policy must be an nn.Module")
        resolved_hidden = hidden_size
        if resolved_hidden is None:
            resolved_hidden = getattr(getattr(policy, "config", None), "hidden_size", None)
        if isinstance(resolved_hidden, bool) or not isinstance(resolved_hidden, int) or resolved_hidden <= 0:
            raise ValueError("Qwen3 policy config must expose a positive hidden_size")
        self.policy = policy
        self.value_head = nn.Linear(
            resolved_hidden,
            1,
            bias=True,
            dtype=torch.float32,
            device=_module_device(policy),
        )
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def values(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(
            hidden.float(),
            self.value_head.weight.float(),
            self.value_head.bias.float(),
        ).squeeze(-1)


class Qwen3TokenPPOAdapter:
    """Sample and teacher-force Qwen3 responses for native actor-critic PPO."""

    def __init__(
        self,
        module: Qwen3ActorCritic,
        codec: Qwen3ChatCodec,
        *,
        max_new_tokens: int,
        compute_dtype: torch.dtype = torch.float32,
    ) -> None:
        if not isinstance(module, Qwen3ActorCritic):
            raise TypeError("module must be Qwen3ActorCritic")
        if not isinstance(codec, Qwen3ChatCodec):
            raise TypeError("codec must be Qwen3ChatCodec")
        if isinstance(max_new_tokens, bool) or int(max_new_tokens) <= 0:
            raise ValueError("max_new_tokens must be positive")
        if compute_dtype not in {torch.float32, torch.bfloat16, torch.float16}:
            raise ValueError("compute_dtype must be float32, bfloat16, or float16")
        self.module = module
        self.codec = codec
        self.max_new_tokens = int(max_new_tokens)
        self.compute_dtype = compute_dtype

    def _sample_response(
        self,
        messages: tuple[AgentMessage, ...],
        *,
        temperature: float,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = _module_device(self.module)
        encoded = self.codec.encode_prompt(messages, conditioning={}, device=device)
        sequence = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        tokens: list[torch.Tensor] = []
        log_probs: list[torch.Tensor] = []
        self.module.eval()
        with torch.no_grad(), _qwen3_autocast(self.module.policy, self.compute_dtype):
            for _ in range(self.max_new_tokens):
                output = self.module.policy(
                    input_ids=sequence,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                logits = _logits(output)[:, -1].float() / temperature
                probabilities = torch.softmax(logits, dim=-1)
                token = torch.multinomial(probabilities, 1, generator=generator)
                tokens.append(token.squeeze(0))
                log_probs.append(torch.log_softmax(logits, dim=-1).gather(1, token).squeeze(0))
                sequence = torch.cat((sequence, token), dim=1)
                attention_mask = torch.cat(
                    (attention_mask, attention_mask.new_ones((1, 1))),
                    dim=1,
                )
                if int(token.item()) in self.codec.eos_token_ids:
                    break
        return (
            torch.cat(tokens).to(dtype=torch.int64),
            torch.cat(log_probs).to(dtype=torch.float32),
        )

    def rollout(
        self,
        request: TokenPPORolloutRequest,
        *,
        generator: torch.Generator | None = None,
    ) -> PackedTokenPPOTrajectory:
        if not isinstance(request, TokenPPORolloutRequest):
            raise TypeError("request must be TokenPPORolloutRequest")
        samples = request.conditioning.get("samples")
        if not isinstance(samples, tuple) or len(samples) != request.batch_size:
            raise ValueError("Qwen3 PPO request must contain one sample payload per row")
        token_chunks: list[torch.Tensor] = []
        log_prob_chunks: list[torch.Tensor] = []
        for sample in samples:
            if not isinstance(sample, Mapping):
                raise TypeError("Qwen3 PPO sample payload must be a mapping")
            tokens, log_probs = self._sample_response(
                _messages(sample),
                temperature=request.sampling_temperature,
                generator=generator,
            )
            token_chunks.append(tokens)
            log_prob_chunks.append(log_probs)
        lengths = torch.tensor(
            [int(tokens.numel()) for tokens in token_chunks],
            device=token_chunks[0].device,
            dtype=torch.int64,
        )
        tokens = torch.cat(token_chunks)
        return PackedTokenPPOTrajectory(
            sample_ids=request.sample_ids,
            policy_revision=request.policy_revision,
            tokens=tokens,
            lengths=lengths,
            old_log_probs=torch.cat(log_prob_chunks),
            loss_mask=torch.ones_like(tokens, dtype=torch.bool),
            sampling_temperature=request.sampling_temperature,
            conditioning=request.conditioning,
        )

    def replay(
        self,
        trajectory: PackedTokenPPOTrajectory | PackedTokenPPOReplayBatch,
        *,
        training: bool,
    ) -> TokenPPOReplayResult:
        samples = trajectory.conditioning.get("samples")
        if not isinstance(samples, tuple) or len(samples) != trajectory.batch_size:
            raise ValueError("Qwen3 PPO trajectory lost its sample payloads")
        device = _module_device(self.module)
        offsets = torch.cat(
            (trajectory.lengths.new_zeros(1), trajectory.lengths.cumsum(0)),
        )
        replayed_log_probs: list[torch.Tensor] = []
        replayed_values: list[torch.Tensor] = []
        self.module.eval()
        with torch.set_grad_enabled(training):
            for row, sample in enumerate(samples):
                if not isinstance(sample, Mapping):
                    raise TypeError("Qwen3 PPO sample payload must be a mapping")
                encoded = self.codec.encode_prompt(
                    _messages(sample),
                    conditioning={},
                    device=device,
                )
                start = int(offsets[row].item())
                end = int(offsets[row + 1].item())
                response = trajectory.tokens[start:end].to(device=device, dtype=torch.int64).unsqueeze(0)
                prompt_ids = encoded["input_ids"]
                prompt_mask = encoded["attention_mask"]
                input_ids = torch.cat((prompt_ids, response), dim=1)
                attention_mask = torch.cat(
                    (prompt_mask, prompt_mask.new_ones(response.shape)),
                    dim=1,
                )
                with _qwen3_autocast(self.module.policy, self.compute_dtype):
                    output = self.module.policy(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_hidden_states=True,
                    )
                prompt_length = int(prompt_ids.shape[1])
                positions = slice(prompt_length - 1, prompt_length + int(response.shape[1]) - 1)
                logits = _logits(output)[0, positions].float() / trajectory.sampling_temperature
                hidden = _hidden(output)[0, positions]
                replayed_log_probs.append(
                    torch.log_softmax(logits, dim=-1).gather(1, response.squeeze(0).unsqueeze(1)).squeeze(1)
                )
                replayed_values.append(self.module.values(hidden))
        return TokenPPOReplayResult(
            log_probs=torch.cat(replayed_log_probs),
            values=torch.cat(replayed_values),
            sampling_temperature=trajectory.sampling_temperature,
        )


_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def normalized_qwen3_answer(value: object) -> str:
    text = str(value).strip()
    match = _ANSWER_TAG.search(text)
    if match is not None:
        text = match.group(1)
    return " ".join(text.casefold().split())


class Qwen3TokenPPORewardAdapter:
    """Exact terminal-answer reward over Qwen3 PPO response text."""

    def __init__(
        self,
        tokenizer: object,
        *,
        reward_ids: Sequence[str],
    ) -> None:
        decode = getattr(tokenizer, "decode", None)
        if not callable(decode):
            raise TypeError("Qwen3 tokenizer must expose decode")
        resolved = tuple(str(reward_id) for reward_id in reward_ids)
        if not resolved or any(reward_id not in {"correctness", "outcome"} for reward_id in resolved):
            raise ValueError("local Qwen3 PPO rewards support correctness or outcome")
        self.tokenizer = tokenizer
        self.reward_ids = resolved

    def score(self, trajectory: PackedTokenPPOTrajectory) -> Mapping[str, torch.Tensor]:
        samples = trajectory.conditioning.get("samples")
        if not isinstance(samples, tuple) or len(samples) != trajectory.batch_size:
            raise ValueError("Qwen3 PPO trajectory lost its reward targets")
        offsets = torch.cat(
            (trajectory.lengths.new_zeros(1), trajectory.lengths.cumsum(0)),
        )
        values: list[float] = []
        for row, sample in enumerate(samples):
            if not isinstance(sample, Mapping) or "answer" not in sample:
                raise ValueError("Qwen3 PPO sample requires an answer target")
            start = int(offsets[row].item())
            end = int(offsets[row + 1].item())
            prediction = self.tokenizer.decode(
                trajectory.tokens[start:end].tolist(),
                skip_special_tokens=True,
            )
            values.append(float(normalized_qwen3_answer(prediction) == normalized_qwen3_answer(sample["answer"])))
        tensor = torch.tensor(values, device=trajectory.tokens.device, dtype=torch.float32)
        return {reward_id: tensor for reward_id in self.reward_ids}


__all__ = [
    "Qwen3ActorCritic",
    "Qwen3TokenPPOAdapter",
    "Qwen3TokenPPORewardAdapter",
    "normalized_qwen3_answer",
]
