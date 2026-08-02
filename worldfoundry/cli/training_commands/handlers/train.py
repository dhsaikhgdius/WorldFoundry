"""Registration and execution of the native ``train`` command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..common import checkpoint_overrides, training_base_dir, training_family


def _handle_train(args: argparse.Namespace) -> int:
    from worldfoundry.training.recipes import TrainingRecipe

    recipe_path = args.recipe.expanduser().resolve()
    recipe = TrainingRecipe.from_file(recipe_path)
    base_dir = training_base_dir(args.base_dir)
    resolved_overrides = checkpoint_overrides(
        args.checkpoint_override,
        base_dir=base_dir,
    )
    if args.checkpoint is not None:
        if "dit" in resolved_overrides:
            raise ValueError("--checkpoint and --checkpoint-override dit=... cannot both be set")
        checkpoint_path = args.checkpoint.expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = base_dir / checkpoint_path
        resolved_overrides["dit"] = str(checkpoint_path.resolve())

    common = {
        "base_dir": base_dir,
        "device": args.device,
        "output_dir": args.output_dir,
        "checkpoint_overrides": resolved_overrides,
        "verify_media_hashes": not args.skip_media_hash_verification,
        "audit_cache_on_open": True,
        "verify_cache_on_read": not args.trust_audited_read_only_cache,
        "initialization_seed": args.seed,
    }
    family = training_family(recipe.model.recipe)
    if family == "sana":
        from worldfoundry.training.engine.sana.sft import (
            materialize_sana_cached_training_session,
        )

        session = materialize_sana_cached_training_session(
            recipe,
            disable_xformers=not args.allow_unverified_attention_backend,
            **common,
        )
    else:
        from worldfoundry.training.engine.wan.sft import (
            materialize_wan_cached_training_session,
        )

        session = materialize_wan_cached_training_session(
            recipe,
            force_torch_attention=not args.allow_unverified_attention_backend,
            **common,
        )
    try:
        fixed_batch = args.fixed_batch or args.one_batch_overfit
        fixed_corruption = args.fixed_corruption or args.one_batch_overfit
        ratio = args.maximum_loss_ratio if args.one_batch_overfit else None
        summary = session.run(
            max_steps=args.steps,
            seed=args.seed,
            fixed_batch=fixed_batch,
            fixed_corruption=fixed_corruption,
            maximum_final_to_initial_loss_ratio=ratio,
            resume_checkpoint=args.resume_checkpoint,
        )
        artifact = None
        if args.export_adapter and session.peft_application:
            artifact = session.export_peft()
        payload: dict[str, object] = {
            "run_dir": str(session.output_dir),
            "recipe_digest": recipe.digest,
            "backend": recipe.distributed.backend,
            "rank_count": session.world_size,
            "summary": summary.to_dict(),
        }
        if artifact is not None:
            payload["adapter"] = {
                "status": "exported",
                "path": str(artifact.path),
                "manifest_sha256": artifact.manifest_sha256,
                "file_sha256": dict(artifact.file_digests),
            }
        if session.is_coordinator:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    finally:
        session.close()


def register_train_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "train",
        help="Run WorldFoundry-native training from a strict recipe",
        description=(
            "Execute one canonical recipe with WorldFoundry-owned SANA/Wan engines. "
            "Inputs and environment identity are verified before launch."
        ),
    )
    parser.add_argument("--recipe", type=Path, required=True, help="Training YAML or JSON recipe.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="Resolve relative manifest/cache/output paths here (default: current working directory).",
    )
    parser.add_argument("--output-dir", type=Path, help="Override recipe.run.output_dir.")
    parser.add_argument("--checkpoint", type=Path, help="Override the recipe denoiser checkpoint.")
    parser.add_argument(
        "--checkpoint-override",
        action="append",
        metavar="NAME=PATH",
        help="Override a named local asset; repeat for dit/text-encoder/tokenizer/codec/vae.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device; FSDP2 resolves the local CUDA device from LOCAL_RANK (default: cuda).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        required=True,
        help="Number of optimizer steps to execute in this invocation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help=(
            "Resume exact model/optimizer/data/RNG state from one committed DCP directory; "
            "the output directory must still be new."
        ),
    )
    parser.add_argument("--fixed-batch", action="store_true")
    parser.add_argument("--fixed-corruption", action="store_true")
    parser.add_argument(
        "--one-batch-overfit",
        action="store_true",
        help="Reuse one accumulation window and reset corruption/dropout RNG every step.",
    )
    parser.add_argument(
        "--maximum-loss-ratio",
        type=float,
        default=0.9,
        help="Overfit gate: final loss must be at most this fraction of initial loss.",
    )
    parser.add_argument(
        "--export-adapter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export a digest-audited PEFT adapter after a successful LoRA run.",
    )
    parser.add_argument(
        "--trust-audited-read-only-cache",
        action="store_true",
        help="Hash every object at startup, then skip repeat hashes while reading an immutable cache mount.",
    )
    parser.add_argument(
        "--skip-media-hash-verification",
        action="store_true",
        help="Skip source media byte hashing; manifest/cache identities are still cross-checked.",
    )
    parser.add_argument(
        "--allow-unverified-attention-backend",
        "--allow-unverified-xformers",
        dest="allow_unverified_attention_backend",
        action="store_true",
        help="Allow an optimized attention backend before model-specific gradient parity is established.",
    )
    parser.set_defaults(func=_handle_train, _requires_exclusive_output_dir=True)


__all__ = ["register_train_subparser"]
