"""Registration and execution of training prompt safety audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _build_prompt_filter(args: argparse.Namespace):
    import gc
    from dataclasses import replace

    import torch

    from worldfoundry.training.safety import (
        build_shieldgemma_prompt_filter,
        shieldgemma_checkpoint_spec,
    )

    checkpoint = None
    if args.shieldgemma_checkpoint is not None:
        checkpoint = replace(
            shieldgemma_checkpoint_spec(),
            source=args.shieldgemma_checkpoint.expanduser().resolve(),
        )
    prompt_filter = build_shieldgemma_prompt_filter(
        checkpoint,
        device=args.device,
        dtype=torch.bfloat16,
        threshold=args.threshold,
    )
    return prompt_filter, gc, torch


def _handle_train_audit_prompts(args: argparse.Namespace) -> int:
    from worldfoundry.training.data.prompt_audits import audit_training_manifest_prompts

    prompt_filter, gc, torch = _build_prompt_filter(args)
    try:
        result = audit_training_manifest_prompts(
            manifest_path=args.manifest,
            output_manifest_path=args.output_manifest,
            output_audit_path=args.output_audits,
            prompt_filter=prompt_filter,
            batch_size=args.batch_size,
            verify_media_hashes=not args.skip_media_hash_verification,
        )
    finally:
        del prompt_filter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": str(result.manifest_path),
                "prompt_audits": str(result.audit_path),
                "manifest_sha256": result.audit_set.manifest_sha256,
                "dataset_digest": result.audit_set.dataset_digest,
                "sample_count": len(result.audit_set.records),
                "safety_audit_digests": [audit.digest for _, audit in result.audit_set.records],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _handle_train_audit_rollout_prompts(args: argparse.Namespace) -> int:
    from worldfoundry.training.data.rollout_audits import audit_rollout_prompts

    prompt_filter, gc, torch = _build_prompt_filter(args)
    try:
        result = audit_rollout_prompts(
            source_path=args.source,
            output_manifest_path=args.output_manifest,
            prompt_filter=prompt_filter,
            batch_size=args.batch_size,
        )
    finally:
        del prompt_filter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": str(result.manifest_path),
                "manifest_sha256": result.manifest_sha256,
                "prompt_count": len(result.records),
                "safety_audit_digests": [record.safety_audit.digest for record in result.records],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _add_shieldgemma_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--shieldgemma-checkpoint",
        type=Path,
        help="Use a local directory while retaining the pinned file integrity contract.",
    )


def register_train_audit_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "train-audit-prompts",
        help="Create a ShieldGemma-audited training manifest",
        description=(
            "Filter every prompt with the pinned ShieldGemma checkpoint, then write a new "
            "manifest containing content-addressed decisions and a strict audit sidecar."
        ),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-audits", type=Path, required=True)
    _add_shieldgemma_arguments(parser)
    parser.add_argument(
        "--skip-media-hash-verification",
        action="store_true",
        help="Skip source media byte hashing while prompt decisions are created.",
    )
    parser.set_defaults(func=_handle_train_audit_prompts)


def register_train_audit_rollout_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "train-audit-rollout-prompts",
        help="Create a ShieldGemma-audited rollout prompt manifest",
        description=(
            "Filter prompt-only JSONL rows, then emit the strict manifest consumed by "
            "native rollout conditioning caches."
        ),
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="JSONL rows with prompt_id/prompt and optional split/generation fields.",
    )
    parser.add_argument("--output-manifest", type=Path, required=True)
    _add_shieldgemma_arguments(parser)
    parser.set_defaults(func=_handle_train_audit_rollout_prompts)


__all__ = [
    "register_train_audit_rollout_subparser",
    "register_train_audit_subparser",
]
