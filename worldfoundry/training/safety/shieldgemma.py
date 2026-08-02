"""Pinned ShieldGemma prompt filtering for licensed SANA training.

The SANA 600M weight license names ShieldGemma-2B as its required input
filter.  This module follows Google's prompt-only scoring contract and uses
the next-token probabilities of the ``Yes`` and ``No`` tokens.  It stores only
prompt hashes in audit records so reports do not become a second prompt corpus.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.loaders.materialize import (
    NativeCheckpointResolver,
)
from worldfoundry.core.io.integrity import canonical_json, text_sha256

SHIELDGEMMA_PROMPT_AUDIT_SCHEMA = "worldfoundry-shieldgemma-prompt-audit"
SHIELDGEMMA_REPO_ID = "google/shieldgemma-2b"
SHIELDGEMMA_REVISION = "d1dffc9c8c9237a90aab09c61383791e718ef9e8"

SHIELDGEMMA_FILES = (
    "config.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)
SHIELDGEMMA_FILE_SHA256 = MappingProxyType(
    {
        "config.json": "a6ed3351158dadd870db10fa19ec69b897001f37c1339881165bfbd267875b42",
        "model-00001-of-00002.safetensors": "343f39c3f075eda5077dab58476d3e3231432ea3cddeb81e6f5ad106b0b524f5",
        "model-00002-of-00002.safetensors": "ce68ef43056ff896f0fa55bada30885eb192a1a439cc984d44d836329c752f21",
        "model.safetensors.index.json": "974f980f459399b2fda03e556f9917f5d03228f4c07dfbe7a6b146be02d4e91c",
        "special_tokens_map.json": "db82f8bd9b25d14f9c788e6bde64de84d42f1c2538f1c245ba6cb3e872d14b18",
        "tokenizer.json": "3f289bc05132635a8bc7aca7aa21255efd5e18f3710f43e3cdb96bcd41be4922",
        "tokenizer.model": "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
        "tokenizer_config.json": "20dc327cc0ddb4bad2ebc5042f561f89fc980855a87db593ed80ac07ca5efcba",
    }
)
SHIELDGEMMA_FILE_SIZE_BYTES = MappingProxyType(
    {
        "config.json": 842,
        "model-00001-of-00002.safetensors": 4988025760,
        "model-00002-of-00002.safetensors": 240691728,
        "model.safetensors.index.json": 24223,
        "special_tokens_map.json": 555,
        "tokenizer.json": 17525357,
        "tokenizer.model": 4241003,
        "tokenizer_config.json": 47904,
    }
)

SHIELDGEMMA_PROMPT_POLICIES = MappingProxyType(
    {
        "dangerous": (
            '"No Dangerous Content": The prompt shall not contain or seek generation of '
            "content that harming oneself and/or others (e.g., accessing or building firearms "
            "and explosive devices, promotion of terrorism, instructions for suicide)."
        ),
        "harassment": (
            '"No Harassment": The prompt shall not contain or seek generation of content that '
            "is malicious, intimidating, bullying, or abusive content targeting another "
            "individual (e.g., physical threats, denial of tragic events, disparaging victims "
            "of violence)."
        ),
        "hate": (
            '"No Hate Speech": The prompt shall not contain or seek generation of content that '
            "targeting identity and/or protected attributes (e.g., racial slurs, promotion of "
            "discrimination, calls to violence against protected groups) or "
            "dehumanizing/belittling/vilifying on the bases of race, ethnicity, religion, "
            "disability, age, nationality, veteran status, sexual orientation, gender, gender "
            "identity, caste, or any other protected status. We do allow content that includes "
            "a reference to a protected individual or group in a positive manner."
        ),
        "sexually-explicit": (
            '"No Sexually Explicit Information": The prompt shall not contain or seek '
            "generation of content that contains references to sexual acts or other lewd "
            "content (e.g., sexually graphic descriptions, content aimed at causing arousal). "
            "Medical or scientific terms related to human anatomy or sex education are permitted."
        ),
    }
)


def _policy_digest() -> str:
    return text_sha256(canonical_json(dict(SHIELDGEMMA_PROMPT_POLICIES)))


def _format_prompt(prompt: str, policy: str) -> str:
    return (
        "You are a policy expert trying to help determine whether a user prompt is in "
        "violation of the defined safety policies.\n\n"
        f"<start_of_turn>\nHuman Question: {prompt.strip()}\n<end_of_turn>\n\n"
        "Our safety principle is defined in the below:\n\n"
        f"{policy.strip()}\n\n"
        "Does the human question violate the above principle? Your answer must start with "
        "'Yes' or 'No'. And then walk through step by step to be sure we answer correctly.\n"
    )


@dataclass(frozen=True, slots=True)
class PromptSafetyAudit:
    """One content-addressed ShieldGemma decision without raw prompt text."""

    prompt_sha256: str
    unsafe_probabilities: Mapping[str, float]
    threshold: float
    model_revision: str = SHIELDGEMMA_REVISION
    policy_digest: str = _policy_digest()
    schema: str = SHIELDGEMMA_PROMPT_AUDIT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SHIELDGEMMA_PROMPT_AUDIT_SCHEMA:
            raise ValueError(f"unsupported prompt safety audit schema: {self.schema!r}")
        if len(self.prompt_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.prompt_sha256
        ):
            raise ValueError("prompt_sha256 must be lowercase SHA-256")
        revision = str(self.model_revision).strip()
        if not revision:
            raise ValueError("prompt safety model_revision cannot be empty")
        if self.policy_digest != _policy_digest():
            raise ValueError("prompt safety policy digest differs from the pinned policies")
        if set(self.unsafe_probabilities) != set(SHIELDGEMMA_PROMPT_POLICIES):
            raise ValueError("prompt safety audit categories differ from the pinned policies")
        scores = {str(name): float(value) for name, value in self.unsafe_probabilities.items()}
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in scores.values()):
            raise ValueError("prompt safety probabilities must be finite values in [0, 1]")
        threshold = float(self.threshold)
        if not isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError("prompt safety threshold must be finite and in (0, 1)")
        object.__setattr__(self, "unsafe_probabilities", MappingProxyType(scores))
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "model_revision", revision)

    @property
    def safe(self) -> bool:
        return all(value < self.threshold for value in self.unsafe_probabilities.values())

    @property
    def blocked_categories(self) -> tuple[str, ...]:
        return tuple(name for name, value in self.unsafe_probabilities.items() if value >= self.threshold)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "prompt_sha256": self.prompt_sha256,
            "model": {"repository": SHIELDGEMMA_REPO_ID, "revision": self.model_revision},
            "policy_digest": self.policy_digest,
            "threshold": self.threshold,
            "unsafe_probabilities": dict(self.unsafe_probabilities),
            "safe": self.safe,
            "blocked_categories": list(self.blocked_categories),
        }

    @classmethod
    def from_mapping(cls, value: object) -> PromptSafetyAudit:
        """Parse one serialized audit and revalidate every derived field."""

        if not isinstance(value, Mapping):
            raise TypeError("prompt safety audit must be a mapping")
        fields = {str(key): item for key, item in value.items()}
        expected = {
            "schema",
            "prompt_sha256",
            "model",
            "policy_digest",
            "threshold",
            "unsafe_probabilities",
            "safe",
            "blocked_categories",
        }
        if set(fields) != expected:
            raise ValueError(
                "prompt safety audit fields mismatch; "
                f"missing={sorted(expected - set(fields))}, "
                f"unknown={sorted(set(fields) - expected)}"
            )
        model = fields["model"]
        if not isinstance(model, Mapping) or set(model) != {"repository", "revision"}:
            raise ValueError("prompt safety audit model must contain repository and revision")
        if str(model["repository"]) != SHIELDGEMMA_REPO_ID:
            raise ValueError("prompt safety audit repository differs from pinned ShieldGemma")
        probabilities = fields["unsafe_probabilities"]
        if not isinstance(probabilities, Mapping):
            raise TypeError("prompt safety audit unsafe_probabilities must be a mapping")
        audit = cls(
            prompt_sha256=str(fields["prompt_sha256"]),
            unsafe_probabilities={str(key): float(item) for key, item in probabilities.items()},
            threshold=float(fields["threshold"]),
            model_revision=str(model["revision"]),
            policy_digest=str(fields["policy_digest"]),
            schema=str(fields["schema"]),
        )
        blocked = fields["blocked_categories"]
        if not isinstance(blocked, Sequence) or isinstance(blocked, (str, bytes, bytearray)):
            raise TypeError("prompt safety audit blocked_categories must be a sequence")
        if not isinstance(fields["safe"], bool):
            raise TypeError("prompt safety audit safe field must be a bool")
        if fields["safe"] != audit.safe or tuple(str(item) for item in blocked) != audit.blocked_categories:
            raise ValueError("prompt safety audit derived decision fields are inconsistent")
        return audit

    @property
    def digest(self) -> str:
        return text_sha256(canonical_json(self.to_dict()))


class UnsafeTrainingPromptError(ValueError):
    """Raised when any prompt fails the pinned ShieldGemma policy gate."""

    def __init__(self, audits: Sequence[PromptSafetyAudit]) -> None:
        blocked = tuple(audit for audit in audits if not audit.safe)
        self.audits = blocked
        summary = [{"prompt_sha256": audit.prompt_sha256, "categories": audit.blocked_categories} for audit in blocked]
        super().__init__(f"training prompts failed ShieldGemma safety filtering: {summary}")


def shieldgemma_checkpoint_spec() -> CheckpointSpec:
    """Return the pinned and fully content-audited ShieldGemma checkpoint."""

    return CheckpointSpec(
        repo_id=SHIELDGEMMA_REPO_ID,
        revision=SHIELDGEMMA_REVISION,
        files=SHIELDGEMMA_FILES,
        allow_patterns=SHIELDGEMMA_FILES,
        file_sha256=SHIELDGEMMA_FILE_SHA256,
        file_size_bytes=SHIELDGEMMA_FILE_SIZE_BYTES,
        metadata={
            "license": "Gemma Terms of Use",
            "purpose": "SANA licensed input safety filtering",
        },
    )


class ShieldGemmaPromptFilter:
    """Score prompt-only safety policies with a frozen local ShieldGemma model."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: object,
        *,
        threshold: float = 0.5,
        model_revision: str = SHIELDGEMMA_REVISION,
        max_input_tokens: int = 4096,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("ShieldGemma model must be an nn.Module")
        if not callable(tokenizer) or not callable(getattr(tokenizer, "get_vocab", None)):
            raise TypeError("ShieldGemma tokenizer must be callable and expose get_vocab")
        if isinstance(max_input_tokens, bool) or int(max_input_tokens) <= 0:
            raise ValueError("max_input_tokens must be a positive integer")
        resolved_threshold = float(threshold)
        if not isfinite(resolved_threshold) or not 0.0 < resolved_threshold < 1.0:
            raise ValueError("ShieldGemma threshold must be finite and in (0, 1)")
        vocabulary = tokenizer.get_vocab()
        if "Yes" not in vocabulary or "No" not in vocabulary:
            raise ValueError("ShieldGemma tokenizer must contain single Yes and No tokens")
        yes_token = int(vocabulary["Yes"])
        no_token = int(vocabulary["No"])
        if yes_token == no_token:
            raise ValueError("ShieldGemma Yes and No token ids must differ")

        model.requires_grad_(False)
        model.eval()
        tokenizer.padding_side = "left"
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = resolved_threshold
        self.model_revision = str(model_revision)
        self.max_input_tokens = int(max_input_tokens)
        self.yes_no_tokens = (yes_token, no_token)
        reference = next(model.parameters(), None)
        self.device = torch.device("cpu") if reference is None else reference.device

    def audit(self, prompts: Sequence[str]) -> tuple[PromptSafetyAudit, ...]:
        values = tuple(str(prompt).strip() for prompt in prompts)
        if not values or any(not prompt for prompt in values):
            raise ValueError("ShieldGemma prompts must be a non-empty sequence of non-empty text")
        formatted = [
            _format_prompt(prompt, policy) for prompt in values for policy in SHIELDGEMMA_PROMPT_POLICIES.values()
        ]
        encoded = self.tokenizer(
            formatted,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        if int(input_ids.shape[1]) > self.max_input_tokens:
            raise ValueError("ShieldGemma input exceeds max_input_tokens; filtering refuses to truncate content")
        encoded = {
            name: tensor.to(device=self.device) for name, tensor in encoded.items() if isinstance(tensor, torch.Tensor)
        }
        with torch.inference_mode():
            output = self.model(**encoded, use_cache=False)
        logits = getattr(output, "logits", None)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise TypeError("ShieldGemma must return [batch,sequence,vocabulary] logits")
        selected = logits[:, -1, list(self.yes_no_tokens)].float()
        if not bool(torch.isfinite(selected).all()):
            raise FloatingPointError("ShieldGemma returned non-finite Yes/No logits")
        unsafe = torch.softmax(selected, dim=-1)[:, 0].detach().cpu()
        category_count = len(SHIELDGEMMA_PROMPT_POLICIES)
        if int(unsafe.numel()) != len(values) * category_count:
            raise RuntimeError("ShieldGemma score count does not match prompt-policy pairs")
        audits = []
        categories = tuple(SHIELDGEMMA_PROMPT_POLICIES)
        for index, prompt in enumerate(values):
            offset = index * category_count
            audits.append(
                PromptSafetyAudit(
                    prompt_sha256=text_sha256(prompt),
                    unsafe_probabilities={
                        name: float(unsafe[offset + category_index]) for category_index, name in enumerate(categories)
                    },
                    threshold=self.threshold,
                    model_revision=self.model_revision,
                )
            )
        return tuple(audits)

    def require_safe(self, prompts: Sequence[str]) -> tuple[PromptSafetyAudit, ...]:
        audits = self.audit(prompts)
        if any(not audit.safe for audit in audits):
            raise UnsafeTrainingPromptError(audits)
        return audits


def build_shieldgemma_prompt_filter(
    checkpoint: CheckpointSpec | None = None,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    threshold: float = 0.5,
    max_input_tokens: int = 4096,
) -> ShieldGemmaPromptFilter:
    """Load the pinned checkpoint locally after complete content verification."""

    spec = checkpoint or shieldgemma_checkpoint_spec()
    materialized = NativeCheckpointResolver().materialize(spec)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as error:
        raise RuntimeError("ShieldGemma filtering requires the train-core Transformers dependency") from error
    resolved_device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(materialized.root, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        materialized.root,
        local_files_only=True,
        dtype=dtype,
        device_map={"": str(resolved_device)},
        attn_implementation="sdpa",
    )
    return ShieldGemmaPromptFilter(
        model,
        tokenizer,
        threshold=threshold,
        model_revision=spec.revision or "local-explicit",
        max_input_tokens=max_input_tokens,
    )


__all__ = [
    "SHIELDGEMMA_PROMPT_AUDIT_SCHEMA",
    "SHIELDGEMMA_PROMPT_POLICIES",
    "SHIELDGEMMA_REPO_ID",
    "SHIELDGEMMA_REVISION",
    "PromptSafetyAudit",
    "ShieldGemmaPromptFilter",
    "UnsafeTrainingPromptError",
    "build_shieldgemma_prompt_filter",
    "shieldgemma_checkpoint_spec",
]
