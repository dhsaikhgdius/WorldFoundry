"""Fail-closed safety gates for native training inputs.

Scope: prompt-text audits only, enforced by the cache tool chain rather than
the training loop; audit records are integrity-checked but unsigned.  See the
``shieldgemma`` module docstring for the exact trust boundary.
"""

from .shieldgemma import (
    SHIELDGEMMA_PROMPT_AUDIT_SCHEMA,
    SHIELDGEMMA_REPO_ID,
    SHIELDGEMMA_REVISION,
    PromptSafetyAudit,
    ShieldGemmaPromptFilter,
    UnsafeTrainingPromptError,
    build_shieldgemma_prompt_filter,
    shieldgemma_checkpoint_spec,
)

__all__ = [
    "SHIELDGEMMA_PROMPT_AUDIT_SCHEMA",
    "SHIELDGEMMA_REPO_ID",
    "SHIELDGEMMA_REVISION",
    "PromptSafetyAudit",
    "ShieldGemmaPromptFilter",
    "UnsafeTrainingPromptError",
    "build_shieldgemma_prompt_filter",
    "shieldgemma_checkpoint_spec",
]
