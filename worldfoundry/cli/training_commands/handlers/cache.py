"""Registration and execution of immutable training-cache creation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..common import (
    checkpoint_overrides,
    load_cache_recipe,
    training_base_dir,
    training_family,
)


def _handle_train_cache(args: argparse.Namespace) -> int:
    from worldfoundry.training.recipes import (
        DiffusionNFTAlgorithmSpec,
        FlowPolicyAlgorithmSpec,
        PostTrainingRecipe,
    )

    recipe_path = args.recipe.expanduser().resolve()
    recipe = load_cache_recipe(recipe_path)
    base_dir = training_base_dir(args.base_dir)
    manifest_path = args.manifest or Path(recipe.data.manifest)
    cache_path = args.cache or (None if recipe.data.cache is None else Path(recipe.data.cache))
    if cache_path is None:
        raise ValueError("training cache materialization requires data.cache or --cache")
    if not manifest_path.is_absolute():
        manifest_path = base_dir / manifest_path
    if not cache_path.is_absolute():
        cache_path = base_dir / cache_path

    if isinstance(recipe, PostTrainingRecipe) and isinstance(
        recipe.algorithm,
        (DiffusionNFTAlgorithmSpec, FlowPolicyAlgorithmSpec),
    ):
        if args.prompt_audits is not None:
            raise ValueError(
                "rollout manifests embed one complete safety audit per prompt; "
                "--prompt-audits is only for media training manifests"
            )
        if training_family(recipe.model.recipe) != "wan":
            raise ValueError("native rollout conditioning cache currently supports Wan")
        from worldfoundry.training.data.wan.rollout_cache import (
            materialize_wan_rollout_conditioning_cache,
        )

        result = materialize_wan_rollout_conditioning_cache(
            recipe,
            manifest_path=manifest_path.resolve(),
            cache_dir=cache_path.resolve(),
            device=args.device,
            checkpoint_overrides=checkpoint_overrides(
                args.checkpoint_override,
                base_dir=base_dir,
            ),
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "model_recipe": recipe.model.recipe,
                    "recipe_digest": recipe.digest,
                    "manifest": str(manifest_path.resolve()),
                    "cache": str(cache_path.resolve()),
                    "dataset_digest": result.index.dataset_digest,
                    "index_sha256": result.index.digest,
                    "prompt_count": len(result.entries),
                    "unconditional_conditioning": (
                        None
                        if result.unconditional_conditioning is None
                        else {
                            "identity_sha256": result.unconditional_conditioning.identity_sha256,
                            "object_sha256": result.unconditional_conditioning.object_sha256,
                        }
                    ),
                    "conditioning_objects": [
                        {
                            "prompt_id": entry.prompt_id,
                            "identity_sha256": entry.artifact.identity_sha256,
                            "object_sha256": entry.artifact.object_sha256,
                        }
                        for entry in result.entries
                    ],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    safety_audits = None
    if args.prompt_audits is not None:
        from worldfoundry.training.data.dataset import TrainingManifestDataset
        from worldfoundry.training.data.prompt_audits import PromptAuditSet

        audit_path = args.prompt_audits
        if not audit_path.is_absolute():
            audit_path = base_dir / audit_path
        selected_manifest = TrainingManifestDataset.from_file(
            manifest_path.resolve(),
            split=recipe.data.split,
            verify_files=True,
            verify_hashes=not args.skip_media_hash_verification,
        )
        safety_audits = PromptAuditSet.from_file(audit_path).select_for_manifest(selected_manifest)

    common = {
        "manifest_path": manifest_path.resolve(),
        "cache_dir": cache_path.resolve(),
        "device": args.device,
        "verify_media_hashes": not args.skip_media_hash_verification,
        "safety_batch_size": args.safety_batch_size,
        "safety_audits": safety_audits,
        "checkpoint_overrides": checkpoint_overrides(
            args.checkpoint_override,
            base_dir=base_dir,
        ),
    }
    family = training_family(recipe.model.recipe)
    if family == "sana":
        from worldfoundry.training.data.sana_precompute import materialize_sana_training_cache

        result = materialize_sana_training_cache(recipe, **common)
    else:
        from worldfoundry.training.data.wan.training_cache import (
            materialize_wan_training_cache,
        )

        result = materialize_wan_training_cache(recipe, **common)
    payload: dict[str, object] = {
        "status": "complete",
        "model_recipe": recipe.model.recipe,
        "recipe_digest": recipe.digest,
        "manifest": str(manifest_path.resolve()),
        "cache": str(cache_path.resolve()),
        "dataset_digest": result.index.dataset_digest,
        "index_sha256": result.index.index_sha256,
        "sample_count": len(result.entries),
        "safety_audit_digests": [audit.digest for audit in result.safety_audits],
    }
    unconditional = getattr(result, "unconditional_conditioning", None)
    if unconditional is not None:
        payload["shared_unconditional_conditioning"] = {
            "identity_sha256": unconditional.identity_sha256,
            "object_sha256": unconditional.object_sha256,
            "object_size_bytes": unconditional.object_size_bytes,
            "object_path": unconditional.object_path,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def register_train_cache_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "train-cache",
        help="Build immutable model features for native training",
        description=(
            "Audit prompts and source media, then materialize SANA image features or "
            "Wan UMT5/video-VAE features with content-addressed provenance."
        ),
    )
    parser.add_argument(
        "--recipe",
        type=Path,
        required=True,
        help="Native training or post-training YAML/JSON recipe.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="Resolve relative manifest/cache paths here (default: current working directory).",
    )
    parser.add_argument("--manifest", type=Path, help="Override recipe.data.manifest.")
    parser.add_argument("--cache", type=Path, help="Override recipe.data.cache.")
    parser.add_argument("--device", default="cuda", help="Feature-encoder torch device.")
    parser.add_argument(
        "--checkpoint-override",
        action="append",
        metavar="NAME=PATH",
        help="Override a named local asset; repeat for text-encoder/tokenizer/codec/vae.",
    )
    parser.add_argument("--safety-batch-size", type=int, default=4)
    parser.add_argument(
        "--prompt-audits",
        type=Path,
        help="Use a manifest-bound sidecar from train-audit-prompts and skip reloading ShieldGemma.",
    )
    parser.add_argument(
        "--skip-media-hash-verification",
        action="store_true",
        help="Skip source media byte hashing; cache object identities remain content-bound.",
    )
    parser.set_defaults(func=_handle_train_cache)


__all__ = ["register_train_cache_subparser"]
