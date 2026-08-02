"""Registration and execution of the native ``post-train`` command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..common import checkpoint_overrides, training_base_dir, training_family


def _handle_post_train(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from worldfoundry.training.recipes import (
        DiffusionNFTAlgorithmSpec,
        DMDAlgorithmSpec,
        FlowPolicyAlgorithmSpec,
        PostTrainingRecipe,
        SCMLADDAlgorithmSpec,
        SelfForcingAlgorithmSpec,
        SIDAlgorithmSpec,
    )

    recipe_path = args.recipe.expanduser().resolve()
    recipe = PostTrainingRecipe.from_file(recipe_path)
    base_dir = training_base_dir(args.base_dir)
    family = training_family(recipe.model.recipe)
    if args.checkpoint_override and not isinstance(recipe.algorithm, SIDAlgorithmSpec):
        raise ValueError("post-train checkpoint overrides currently require SANA SiD")
    resume_checkpoint = args.resume_checkpoint
    if resume_checkpoint is not None:
        resume_checkpoint = resume_checkpoint.expanduser()
        if not resume_checkpoint.is_absolute():
            resume_checkpoint = base_dir / resume_checkpoint
        resume_checkpoint = resume_checkpoint.resolve()

    if isinstance(recipe.algorithm, SCMLADDAlgorithmSpec):
        if family != "sana":
            raise ValueError("SCM-LADD materialization requires a SANA recipe")
        from worldfoundry.training.engine.sana.scm_ladd import (
            materialize_sana_scm_ladd_training_run,
        )

        run = materialize_sana_scm_ladd_training_run(
            recipe,
            base_dir=base_dir,
            device=args.device,
            output_dir=args.output_dir,
            resume_checkpoint=resume_checkpoint,
            verify_media_hashes=not args.skip_media_hash_verification,
            audit_cache_on_open=True,
            verify_cache_on_read=not args.trust_audited_read_only_cache,
            initialization_seed=args.seed,
        )
    elif isinstance(recipe.algorithm, SIDAlgorithmSpec):
        if family != "sana":
            raise ValueError("SiD materialization requires a SANA recipe")
        from worldfoundry.training.engine.sana.sid import (
            materialize_sana_sid_training_run,
        )

        run = materialize_sana_sid_training_run(
            recipe,
            base_dir=base_dir,
            device=args.device,
            output_dir=args.output_dir,
            resume_checkpoint=resume_checkpoint,
            local_role_paths=checkpoint_overrides(
                args.checkpoint_override,
                base_dir=base_dir,
            ),
            verify_media_hashes=not args.skip_media_hash_verification,
            audit_cache_on_open=True,
            verify_cache_on_read=not args.trust_audited_read_only_cache,
            initialization_seed=args.seed,
        )
    elif isinstance(recipe.algorithm, DMDAlgorithmSpec):
        if family != "wan":
            raise ValueError("DMD materialization requires a Wan recipe")
        from worldfoundry.training.engine.wan.dmd import (
            materialize_wan_dmd_training_run,
        )

        run = materialize_wan_dmd_training_run(
            recipe,
            base_dir=base_dir,
            device=args.device,
            output_dir=args.output_dir,
            resume_checkpoint=resume_checkpoint,
            verify_media_hashes=not args.skip_media_hash_verification,
            audit_cache_on_open=True,
            verify_cache_on_read=not args.trust_audited_read_only_cache,
            force_torch_attention=not args.allow_unverified_attention_backend,
            initialization_seed=args.seed,
        )
    elif isinstance(recipe.algorithm, SelfForcingAlgorithmSpec):
        if family != "wan":
            raise ValueError("Self-Forcing materialization requires a Wan recipe")
        from worldfoundry.training.engine.wan.self_forcing import (
            materialize_wan_self_forcing_training_run,
        )

        run = materialize_wan_self_forcing_training_run(
            recipe,
            base_dir=base_dir,
            device=args.device,
            output_dir=args.output_dir,
            resume_checkpoint=resume_checkpoint,
            force_torch_attention=not args.allow_unverified_attention_backend,
            initialization_seed=args.seed,
        )
    elif isinstance(recipe.algorithm, DiffusionNFTAlgorithmSpec):
        if family != "wan":
            raise ValueError("Diffusion-NFT materialization requires a Wan recipe")
        from worldfoundry.training.engine.wan.diffusion_nft import (
            materialize_wan_diffusion_nft_training_run,
        )

        run = materialize_wan_diffusion_nft_training_run(
            recipe,
            base_dir=base_dir,
            device=args.device,
            reward_device=args.reward_device,
            output_dir=args.output_dir,
            resume_checkpoint=resume_checkpoint,
            force_torch_attention=not args.allow_unverified_attention_backend,
            videoalign_attention_implementation=args.reward_attention,
            initialization_seed=args.seed,
        )
    elif isinstance(recipe.algorithm, FlowPolicyAlgorithmSpec):
        if family != "wan":
            raise ValueError("flow-policy materialization requires a Wan recipe")
        from worldfoundry.training.engine.wan.flow_policy import (
            materialize_wan_flow_policy_training_run,
        )

        run = materialize_wan_flow_policy_training_run(
            recipe,
            base_dir=base_dir,
            device=args.device,
            reward_device=args.reward_device,
            output_dir=args.output_dir,
            resume_checkpoint=resume_checkpoint,
            force_torch_attention=not args.allow_unverified_attention_backend,
            videoalign_attention_implementation=args.reward_attention,
            initialization_seed=args.seed,
        )
    else:
        raise TypeError(f"unsupported native post-training algorithm: {recipe.algorithm!r}")

    try:
        artifact = None
        artifact_role = None
        if isinstance(
            recipe.algorithm,
            (
                DMDAlgorithmSpec,
                SCMLADDAlgorithmSpec,
                SelfForcingAlgorithmSpec,
                SIDAlgorithmSpec,
            ),
        ):
            summary = run.run(max_steps=args.steps)
            summary_payload = asdict(summary)
            if args.export_artifact:
                artifact = run.export_student()
                artifact_role = "student"
        else:
            summary = run.run(max_iterations=args.steps)
            summary_payload = summary.to_dict()
            if args.export_artifact:
                artifact = run.export_policy()
                artifact_role = "policy"
        payload: dict[str, object] = {
            "run_dir": str(run.output_dir),
            "recipe_digest": recipe.digest,
            "algorithm": recipe.algorithm.type,
            "backend": recipe.distributed.backend,
            "rank_count": run.world_size,
            "summary": summary_payload,
        }
        if artifact is not None:
            file_digests = getattr(artifact, "file_digests", None)
            if file_digests is None:
                file_digests = artifact.file_sha256
            payload["trained_artifact"] = {
                "status": "exported",
                "role": artifact_role,
                "format": recipe.export.format,
                "path": str(artifact.path),
                "manifest_sha256": artifact.manifest_sha256,
                "file_sha256": dict(file_digests),
            }
        if run.is_coordinator:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    finally:
        run.close()


def register_post_train_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "post-train",
        help="Run WorldFoundry-native diffusion post-training",
        description="Materialize model roles and execute native diffusion post-training.",
    )
    parser.add_argument(
        "--recipe",
        type=Path,
        required=True,
        help="Strict post-training YAML or JSON recipe.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="Resolve relative manifest/cache/output paths here (default: current working directory).",
    )
    parser.add_argument("--output-dir", type=Path, help="Override recipe.run.output_dir.")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device; FSDP2 derives the local CUDA device from LOCAL_RANK.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        required=True,
        help="Number of optimizer steps or rollout iterations in this invocation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Restore exact roles, optimizers, data cursor, algorithm state, and RNG.",
    )
    parser.add_argument(
        "--checkpoint-override",
        action="append",
        metavar="ROLE=PATH",
        help=(
            "Override a local SANA SiD role directory; repeat for student, teacher, "
            "and fake_score."
        ),
    )
    parser.add_argument(
        "--export-trained-artifact",
        "--export-adapter",
        "--export-student-adapter",
        dest="export_artifact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Export the configured digest-audited PEFT, Safetensors, or DCP student/policy artifact after success."),
    )
    parser.add_argument(
        "--reward-device",
        help="VAE and reward-model device (default: local policy device).",
    )
    parser.add_argument(
        "--reward-attention",
        choices=("sdpa", "flash_attention_2"),
        default="sdpa",
        help="VideoAlign attention implementation; SDPA is the correctness-first default.",
    )
    parser.add_argument(
        "--trust-audited-read-only-cache",
        action="store_true",
        help="Hash cache objects at startup and skip repeated per-read hashes.",
    )
    parser.add_argument(
        "--skip-media-hash-verification",
        action="store_true",
        help="Skip source media byte hashing while retaining manifest/cache identity checks.",
    )
    parser.add_argument(
        "--allow-unverified-attention-backend",
        action="store_true",
        help="Allow an optimized attention backend instead of the Torch backend.",
    )
    parser.set_defaults(
        func=_handle_post_train,
        _requires_exclusive_output_dir=True,
    )


__all__ = ["register_post_train_subparser"]
