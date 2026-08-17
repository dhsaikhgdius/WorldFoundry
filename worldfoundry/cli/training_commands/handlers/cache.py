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
        family = training_family(recipe.model.recipe)
        if family == "wan":
            from worldfoundry.training.data.wan.rollout_cache import (
                materialize_wan_rollout_conditioning_cache as materialize,
            )
        elif family == "wan22":
            from worldfoundry.training.data.wan22.rollout_cache import (
                materialize_wan22_rollout_conditioning_cache as materialize,
            )
        elif family == "ltx":
            from worldfoundry.training.data.ltx.rollout_cache import (
                materialize_ltx_rollout_conditioning_cache as materialize,
            )
        elif family == "hunyuan-video":
            from worldfoundry.training.data.hunyuan_video.rollout_cache import (
                materialize_hunyuan_video_rollout_conditioning_cache as materialize,
            )
        else:
            raise ValueError(f"train-cache has no rollout conditioner for model family {family!r}")

        result = materialize(
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
                    "manifest": str(manifest_path.resolve()),
                    "cache": str(cache_path.resolve()),
                    "index": result.index.to_dict(),
                    "prompt_count": len(result.entries),
                    "unconditional_conditioning": (
                        None
                        if result.unconditional_conditioning is None
                        else result.unconditional_conditioning.to_dict()
                    ),
                    "conditioning_entries": [entry.to_dict() for entry in result.entries],
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
        )
        safety_audits = PromptAuditSet.from_file(audit_path).select_for_manifest(selected_manifest)

    common = {
        "manifest_path": manifest_path.resolve(),
        "cache_dir": cache_path.resolve(),
        "device": args.device,
        "verify_media_files": not args.skip_media_file_verification,
        "safety_batch_size": args.safety_batch_size,
        "safety_audits": safety_audits,
        "checkpoint_overrides": checkpoint_overrides(
            args.checkpoint_override,
            base_dir=base_dir,
        ),
    }
    family = training_family(recipe.model.recipe)
    if family in {"lvdm", "dynamicrafter", "t2v-turbo"}:
        from worldfoundry.training.data.video_tensor_import import (
            materialize_precomputed_video_training_cache,
        )

        source = recipe.data.options.get("precomputed_tensors")
        if source is None:
            raise ValueError(f"{family} train-cache requires data.options.precomputed_tensors")
        source_path = Path(str(source)).expanduser()
        if not source_path.is_absolute():
            source_path = base_dir / source_path
        result = materialize_precomputed_video_training_cache(
            recipe,
            source_dir=source_path.resolve(),
            **common,
        )
    elif family == "sana":
        from worldfoundry.training.data.sana_precompute import materialize_sana_training_cache

        result = materialize_sana_training_cache(recipe, **common)
    elif family == "wan":
        from worldfoundry.training.data.wan.training_cache import (
            materialize_wan_training_cache,
        )

        result = materialize_wan_training_cache(recipe, **common)
    elif family == "ltx":
        from worldfoundry.training.data.ltx.training_cache import (
            materialize_ltx_training_cache,
        )

        result = materialize_ltx_training_cache(recipe, base_dir=base_dir, **common)
    elif family == "cosmos":
        from worldfoundry.training.data.cosmos.training_cache import (
            materialize_cosmos_training_cache,
        )

        result = materialize_cosmos_training_cache(recipe, base_dir=base_dir, **common)
    else:
        raise ValueError(f"train-cache has no native materializer for model family {family!r}")
    payload: dict[str, object] = {
        "status": "complete",
        "model_recipe": recipe.model.recipe,
        "manifest": str(manifest_path.resolve()),
        "cache": str(cache_path.resolve()),
        "index": result.index.to_dict(),
        "sample_count": len(result.entries),
        "safety_audits": [audit.to_dict() for audit in result.safety_audits],
    }
    unconditional = getattr(result, "unconditional_conditioning", None)
    if unconditional is not None:
        payload["shared_unconditional_conditioning"] = unconditional.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def register_train_cache_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "train-cache",
        help="Build immutable model features for native training",
        description=(
            "Audit prompts and source media, then materialize native image/video "
            "conditioning and VAE features for training."
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
        "--skip-media-file-verification",
        action="store_true",
        help="Skip checking referenced source media files before building the cache.",
    )
    parser.set_defaults(func=_handle_train_cache)


__all__ = ["register_train_cache_subparser"]
