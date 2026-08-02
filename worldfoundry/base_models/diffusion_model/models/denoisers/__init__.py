"""Model-family adapters for the shared native denoising contract.

Network modules contain checkpoint-compatible math only. This role owns the
small amount of family-specific state-dict conversion and contract adaptation
needed by the common loader and runners.
"""

