"""Registration and execution of the native ``post-train`` command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..common import checkpoint_overrides, training_base_dir, training_family


def _handle_post_train(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from worldfoundry.training.recipes import (
        AnyFlowBidirectionalOnPolicyAlgorithmSpec,
        AnyFlowBidirectionalPretrainAlgorithmSpec,
        AnyFlowFAROnPolicyAlgorithmSpec,
        AnyFlowFARPretrainAlgorithmSpec,
        DiffusionNFTAlgorithmSpec,
        DMD2AlgorithmSpec,
        DMDAlgorithmSpec,
        FlowPolicyAlgorithmSpec,
        PostTrainingRecipe,
        SCMLADDAlgorithmSpec,
        SelfForcingAlgorithmSpec,
        SIDAlgorithmSpec,
        T2VTurboAlgorithmSpec,
    )

    recipe_path = args.recipe.expanduser().resolve()
    recipe = PostTrainingRecipe.from_file(recipe_path)
    base_dir = training_base_dir(args.base_dir)
    family = training_family(recipe.model.recipe)
    video_flow_family = family in {"ltx", "wan22", "hunyuan-video"}
    qwen_grouped_algorithms = {
        "token-cppo",
        "token-dppo",
        "token-drpo",
        "token-grpo",
        "token-gspo",
    }
    qwen_agentic = family == "qwen3" and recipe.algorithm.type in qwen_grouped_algorithms
    supports_checkpoint_overrides = isinstance(
        recipe.algorithm,
        (SIDAlgorithmSpec, T2VTurboAlgorithmSpec),
    ) or (isinstance(recipe.algorithm, FlowPolicyAlgorithmSpec) and video_flow_family)
    if args.checkpoint_override and not supports_checkpoint_overrides:
        raise ValueError(
            "post-train checkpoint overrides require SANA SiD, T2V-Turbo, "
            "or an LTX, Wan2.2, or HunyuanVideo flow-policy recipe"
        )
    if args.reward_url is not None and not (
        (isinstance(recipe.algorithm, FlowPolicyAlgorithmSpec) and video_flow_family) or qwen_agentic
    ):
        raise ValueError(
            "--reward-url requires LTX, Wan2.2, or HunyuanVideo flow-policy or Qwen3 grouped Agentic token policy"
        )
    resume_checkpoint = args.resume_checkpoint
    if resume_checkpoint is not None:
        resume_checkpoint = resume_checkpoint.expanduser()
        if not resume_checkpoint.is_absolute():
            resume_checkpoint = base_dir / resume_checkpoint
        resume_checkpoint = resume_checkpoint.resolve()

    if isinstance(
        recipe.algorithm,
        (
            AnyFlowFARPretrainAlgorithmSpec,
            AnyFlowBidirectionalPretrainAlgorithmSpec,
            AnyFlowFAROnPolicyAlgorithmSpec,
            AnyFlowBidirectionalOnPolicyAlgorithmSpec,
        ),
    ):
        if family != "wan":
            raise ValueError("AnyFlow materialization requires a Wan recipe")
        from worldfoundry.training.engine.anyflow import (
            materialize_anyflow_training_run,
        )

        run = materialize_anyflow_training_run(
            recipe,
            base_dir=base_dir,
            device=args.device,
            output_dir=args.output_dir,
            resume_checkpoint=resume_checkpoint,
            audit_cache_on_open=True,
            force_torch_attention=not args.allow_unverified_attention_backend,
            initialization_seed=args.seed,
        )
    elif isinstance(recipe.algorithm, SCMLADDAlgorithmSpec):
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
            audit_cache_on_open=True,
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
            audit_cache_on_open=True,
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
            audit_cache_on_open=True,
            force_torch_attention=not args.allow_unverified_attention_backend,
            initialization_seed=args.seed,
        )
    elif isinstance(recipe.algorithm, DMD2AlgorithmSpec):
        if family != "cosmos":
            raise ValueError("DMD2 materialization requires a Cosmos Predict2.5 recipe")
        from worldfoundry.training.engine.cosmos.dmd2 import (
            materialize_cosmos_predict25_dmd2_training_run,
        )

        run = materialize_cosmos_predict25_dmd2_training_run(
            recipe,
            base_dir=base_dir,
            device=args.device,
            output_dir=args.output_dir,
            resume_checkpoint=resume_checkpoint,
            audit_cache_on_open=True,
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
        if family == "wan":
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
        elif video_flow_family:
            from worldfoundry.training.engine.video_policy import (
                materialize_video_flow_policy_training_run,
            )

            run = materialize_video_flow_policy_training_run(
                recipe,
                base_dir=base_dir,
                device=args.device,
                reward_device=args.reward_device,
                output_dir=args.output_dir,
                resume_checkpoint=resume_checkpoint,
                checkpoint_overrides=checkpoint_overrides(
                    args.checkpoint_override,
                    base_dir=base_dir,
                ),
                reward_url=args.reward_url,
                reward_attention_implementation=args.reward_attention,
                initialization_seed=args.seed,
            )
        else:
            raise ValueError("flow-policy materialization requires a Wan, LTX, Wan2.2, or HunyuanVideo recipe")
    elif family == "qwen3" and recipe.algorithm.type in qwen_grouped_algorithms | {"token-ppo"}:
        from .qwen import materialize_qwen3_cli_run

        run = materialize_qwen3_cli_run(
            recipe,
            base_dir=base_dir,
            device=args.device,
            output_dir=args.output_dir,
            resume_checkpoint=resume_checkpoint,
            reward_url=args.reward_url,
            initialization_seed=args.seed,
        )
    elif isinstance(recipe.algorithm, T2VTurboAlgorithmSpec):
        if family != "t2v-turbo":
            raise ValueError("T2V-Turbo distillation requires the T2V-Turbo model recipe")
        from worldfoundry.training.post_training.distillation.t2v_turbo import (
            materialize_t2v_turbo_training_run,
        )

        run = materialize_t2v_turbo_training_run(
            recipe,
            base_dir=base_dir,
            device=args.device,
            output_dir=args.output_dir,
            resume_checkpoint=resume_checkpoint,
            checkpoint_overrides=checkpoint_overrides(
                args.checkpoint_override,
                base_dir=base_dir,
            ),
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
                DMD2AlgorithmSpec,
                AnyFlowFARPretrainAlgorithmSpec,
                AnyFlowBidirectionalPretrainAlgorithmSpec,
                AnyFlowFAROnPolicyAlgorithmSpec,
                AnyFlowBidirectionalOnPolicyAlgorithmSpec,
                SCMLADDAlgorithmSpec,
                SelfForcingAlgorithmSpec,
                SIDAlgorithmSpec,
                T2VTurboAlgorithmSpec,
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
                artifact_role = getattr(run, "artifact_role", "policy")
        payload: dict[str, object] = {
            "run_dir": str(run.output_dir),
            "algorithm": recipe.algorithm.type,
            "backend": recipe.distributed.backend,
            "rank_count": run.world_size,
            "summary": summary_payload,
        }
        if artifact is not None:
            payload["trained_artifact"] = {
                "status": "exported",
                "role": artifact_role,
                "format": recipe.export.format,
                "path": str(artifact.path),
                "file_size_bytes": dict(artifact.file_size_bytes),
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
        help="Run WorldFoundry-native post-training",
        description="Materialize model roles and execute native post-training.",
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
        help="Override a named role checkpoint; repeat for student, teacher, and fake_score where applicable.",
    )
    parser.add_argument(
        "--export-trained-artifact",
        "--export-adapter",
        "--export-student-adapter",
        dest="export_artifact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Export the configured PEFT, Safetensors, or DCP student/policy artifact after success."),
    )
    parser.add_argument(
        "--reward-device",
        help="VAE and reward-model device (default: local policy device).",
    )
    parser.add_argument(
        "--reward-url",
        help="HTTP reward service URL for video flow-policy or Qwen3 grouped Agentic training.",
    )
    parser.add_argument(
        "--reward-attention",
        choices=("sdpa", "flash_attention_2"),
        default="sdpa",
        help="VideoAlign attention implementation; SDPA is the correctness-first default.",
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
