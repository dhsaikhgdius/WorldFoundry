"""Lazy Qwen3 post-training materialization for the CLI."""

from __future__ import annotations

from pathlib import Path


def materialize_qwen3_cli_run(
    recipe: object,
    *,
    base_dir: Path,
    device: str,
    output_dir: Path | None,
    resume_checkpoint: Path | None,
    reward_url: str | None,
    initialization_seed: int,
):
    from worldfoundry.training.post_training.causal_lm.qwen3 import (
        materialize_qwen3_post_training_run,
    )

    return materialize_qwen3_post_training_run(
        recipe,
        base_dir=base_dir,
        device=device,
        output_dir=output_dir,
        resume_checkpoint=resume_checkpoint,
        reward_url=reward_url,
        initialization_seed=initialization_seed,
    )


__all__ = ["materialize_qwen3_cli_run"]
