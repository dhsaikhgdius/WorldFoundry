#!/usr/bin/env python3
"""Run real Wan cache, training/resume, PEFT, inference, and FSDP gates.

Architecture and asset identities are pinned to the official Wan2.1 source
and Wan-AI/Wan2.1-T2V-1.3B model revisions declared by the native recipe.  The
small lossless input video is generated locally and declared CC0-1.0.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import socket
import time
from dataclasses import replace
from pathlib import Path

import torch

from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler
from worldfoundry.base_models.diffusion_model.components import (
    BuildPurpose,
    ComponentKey,
    ComponentKind,
)
from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
from worldfoundry.base_models.diffusion_model.optimizations import AttentionBackend, RuntimePolicy
from worldfoundry.base_models.diffusion_model.recipes.registry import (
    default_native_diffusion_registry,
)
from worldfoundry.base_models.diffusion_model.runners import ExecutionBuildContext
from worldfoundry.training.data import (
    TRAINING_SAMPLE_SCHEMA,
    VideoCachedDataset,
    collate_video_cached_samples,
    materialize_wan_training_cache,
)
from worldfoundry.training.engine import materialize_wan_cached_training_session
from worldfoundry.training.models import WanTrainAdapter
from worldfoundry.training.recipes import TrainingRecipe
from worldfoundry.training.safety import PromptSafetyAudit
from worldfoundry.training.safety.shieldgemma import SHIELDGEMMA_PROMPT_POLICIES
from worldfoundry.training.tuning import load_peft_adapter, merge_peft_adapter

_MERGE_MAX_ABS_TOLERANCE = 3.0e-3
_MERGE_RELATIVE_RMSE_TOLERANCE = 3.0e-3


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--maximum-loss-ratio", type=float, default=0.98)
    parser.add_argument("--inference-steps", type=int, default=2)
    parser.add_argument(
        "--skip-fsdp-world-one",
        action="store_true",
        help="Skip the real FSDP2 world-size-one gate.",
    )
    return parser.parse_args()


def _release(*values: object) -> None:
    del values
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_video(path: Path, *, frames: int = 5, size: int = 64, fps: int = 8) -> None:
    import av
    import numpy as np

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("ffv1", rate=fps)
        stream.width = size
        stream.height = size
        stream.pix_fmt = "yuv444p"
        y, x = np.mgrid[0:size, 0:size]
        for frame_index in range(frames):
            red = (24 + x * 2 + frame_index * 9) % 256
            green = (48 + y * 2 + frame_index * 5) % 256
            blue = (112 + x + y + frame_index * 13) % 256
            pixels = np.stack((red, green, blue), axis=-1).astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _fixture_audit(prompt: str) -> PromptSafetyAudit:
    """Create a caller-owned safe audit for a fixed non-user pilot prompt."""

    return PromptSafetyAudit(
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        unsafe_probabilities={name: 0.0 for name in SHIELDGEMMA_PROMPT_POLICIES},
        threshold=0.5,
    )


def _write_manifest(
    path: Path,
    *,
    video_path: Path,
    prompt: str,
    audit: PromptSafetyAudit,
) -> None:
    payload = video_path.read_bytes()
    row = {
        "schema": TRAINING_SAMPLE_SCHEMA,
        "sample_id": "synthetic-moving-gradient",
        "task": "t2v",
        "prompt": prompt,
        "media": {
            "uri": video_path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "mime_type": "video/x-matroska",
        },
        "width": 64,
        "height": 64,
        "num_frames": 5,
        "fps": 8,
        "conditions": {},
        "split": "train",
        "license": "CC0-1.0",
        "provenance": {"source": "locally-generated-training-gate"},
        "quality": {"accepted": True, "purpose": "implementation-gate"},
        "safety": {
            "filter": "pre-audited-fixed-fixture",
            "prompt_audit_digest": audit.digest,
        },
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def _recipe(
    *,
    work_dir: Path,
    manifest_path: Path,
    cache_dir: Path,
    learning_rate: float,
    backend: str = "single",
    save_every_steps: int = 1,
) -> TrainingRecipe:
    return TrainingRecipe.from_mapping(
        {
            "schema": "worldfoundry-training",
            "run": {
                "id": f"wan-1p3b-real-{backend}-gate",
                "output_dir": str(work_dir / f"{backend}-unused"),
            },
            "provider": {"name": "native"},
            "model": {
                "recipe": "wan2.1-t2v-1.3b",
                "checkpoint": "default",
                "options": {"vae_tiled": False},
            },
            "tuning": {
                "mode": "lora",
                "preset": "wan-attention",
                "rank": 4,
                "alpha": 4,
                "dropout": 0.0,
            },
            "data": {
                "manifest": str(manifest_path),
                "cache": str(cache_dir),
                "max_latent_tokens_per_microbatch": 128,
                "split": "train",
                "shuffle": False,
                "shuffle_seed": 42,
                "tail_policy": "pad",
                "options": {
                    "video_buckets": [
                        {
                            "num_frames": 5,
                            "height": 64,
                            "width": 64,
                            "conditioning_layout": "umt5-sequence",
                            "tasks": ["t2v"],
                        }
                    ],
                    "bucket_policy": {
                        "allow_spatial_upscale": False,
                        "allow_temporal_padding": False,
                    },
                    "decode": {
                        "frame_sampling": "uniform-full",
                        "interpolation": "bicubic",
                        "value_range": "minus-one-one",
                        "decoder_thread_type": "auto",
                        "verify_media_sha256": True,
                        "verify_manifest_frame_count": True,
                        "verify_manifest_geometry": True,
                        "fps_tolerance": 0.01,
                    },
                    "num_workers": 0,
                    "pin_memory": True,
                    "persistent_workers": False,
                    "snapshot_every_n_steps": 1,
                },
            },
            "objective": {
                "type": "flow_matching",
                "prediction_type": "flow_velocity",
                "timestep_sampler": "logit_normal",
                "conditioning_dropout": 0.0,
                "options": {
                    "num_train_timesteps": 1000,
                    "flow_shift": 1.0,
                    "logit_mean": 0.0,
                    "logit_std": 1.0,
                },
            },
            "optimizer": {
                "type": "adamw",
                "learning_rate": learning_rate,
                "weight_decay": 0.0,
                "betas": [0.9, 0.99],
                "epsilon": 1.0e-8,
                "max_grad_norm": 1.0,
                "gradient_accumulation_steps": 1,
            },
            "runtime": {
                "param_dtype": "bfloat16",
                "reduce_dtype": "float32",
                "activation_checkpoint": "full",
                "compile": False,
            },
            "distributed": {
                "backend": backend,
                "dp_shard": "auto",
                "cp": 1,
                "tp": 1,
            },
            "checkpoint": {
                "save_every_steps": save_every_steps,
                "async": False,
                "export_every_steps": 0,
            },
            "validation": {"every_steps": 0, "fixed_seed": 42},
            "export": {"format": "peft", "merge_adapter": False},
            "metadata": {
                "gate": "real-cache-overfit-resume-export-inference",
                "generated_sample_license": "CC0-1.0",
            },
        }
    )


def _checkpoint_overrides(model_root: Path) -> dict[str, str]:
    root = str(model_root.expanduser().resolve())
    return {
        "dit": root,
        "text-encoder": root,
        "tokenizer": root,
        "vae": root,
    }


def _fresh_denoiser(
    assembler: NativeDiffusionAssembler,
    native_recipe,
    *,
    policy: RuntimePolicy,
    checkpoint_overrides: dict[str, str],
):
    key = ComponentKey(ComponentKind.DENOISER)
    return assembler.build_components(
        native_recipe,
        purpose=BuildPurpose.TRAINING,
        policy=policy,
        checkpoint_overrides=checkpoint_overrides,
        component_options={key: {"weight_dtype": policy.dtype}},
        component_keys=(key,),
    )[key]


def _float_objective_batch(batch):
    def convert(value):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.float()
        return value

    return replace(
        batch,
        model_input=convert(batch.model_input),
        target=convert(batch.target),
        sigmas=convert(batch.sigmas),
        conditioning={name: convert(value) for name, value in batch.conditioning.items()},
        noise=convert(batch.noise),
        loss_mask=convert(batch.loss_mask),
        sample_weights=convert(batch.sample_weights),
    )


def _trainable_state(session) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in session.engine.adapter.trainable_module.named_parameters()
        if parameter.requires_grad
    }


def _checkpoint_after_first_step(session) -> Path:
    manifest = json.loads(session.manifest_path.read_text(encoding="utf-8"))
    checkpoints = manifest.get("checkpoints", [])
    if not checkpoints or checkpoints[0].get("global_step") != 1:
        raise RuntimeError("real Wan gate did not commit its first-step DCP checkpoint")
    return Path(checkpoints[0]["path"])


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _write_inference_video(path: Path, sample: torch.Tensor, *, fps: int = 8) -> None:
    import av

    video = sample.detach().float().cpu().clamp(-1, 1)
    pixels = ((video + 1) * 127.5).round().to(torch.uint8)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("ffv1", rate=fps)
        stream.width = int(pixels.shape[-1])
        stream.height = int(pixels.shape[-2])
        stream.pix_fmt = "yuv444p"
        for frame_pixels in pixels.permute(1, 2, 3, 0).numpy():
            frame = av.VideoFrame.from_ndarray(frame_pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main() -> int:
    args = _arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("the real Wan round-trip gate requires CUDA")
    if args.steps < 2:
        raise ValueError("--steps must be at least two for exact-resume validation")
    work_dir = args.work_dir.expanduser().resolve()
    model_root = args.model_root.expanduser().resolve()
    if work_dir.exists():
        raise FileExistsError(f"gate work directory already exists: {work_dir}")
    if not model_root.is_dir():
        raise FileNotFoundError(f"Wan model root does not exist: {model_root}")
    work_dir.mkdir(parents=True)
    os.environ["WORLDFOUNDRY_ATTENTION_IMPLEMENTATION"] = "torch"
    os.environ["WORLDFOUNDRY_ATTENTION_BACKEND"] = "torch"
    started = time.perf_counter()

    prompt = "A smooth colorful gradient moves gently across a square frame."
    video_path = work_dir / "synthetic-moving-gradient.mkv"
    manifest_path = work_dir / "manifest.jsonl"
    cache_dir = work_dir / "cache"
    _write_video(video_path)
    fixture_audit = _fixture_audit(prompt)
    _write_manifest(
        manifest_path,
        video_path=video_path,
        prompt=prompt,
        audit=fixture_audit,
    )

    overrides = _checkpoint_overrides(model_root)
    resume_recipe = _recipe(
        work_dir=work_dir,
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        learning_rate=args.learning_rate,
    )
    cache_result = materialize_wan_training_cache(
        resume_recipe,
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        device="cuda",
        checkpoint_overrides=overrides,
        safety_audits=(fixture_audit,),
        verify_media_hashes=True,
    )
    _release()

    continuous = materialize_wan_cached_training_session(
        resume_recipe,
        device="cuda",
        output_dir=work_dir / "continuous-resume",
        checkpoint_overrides=overrides,
        verify_media_hashes=True,
        audit_cache_on_open=True,
        verify_cache_on_read=True,
        force_torch_attention=True,
        fused_adamw=False,
        initialization_seed=47,
    )
    continuous_summary = continuous.run(
        max_steps=2,
        seed=53,
    )
    continuous_state = _trainable_state(continuous)
    first_checkpoint = _checkpoint_after_first_step(continuous)
    continuous_artifact = continuous.export_peft()
    continuous.close()
    del continuous
    _release()

    resumed = materialize_wan_cached_training_session(
        resume_recipe,
        device="cuda",
        output_dir=work_dir / "resumed",
        checkpoint_overrides=overrides,
        verify_media_hashes=True,
        audit_cache_on_open=True,
        verify_cache_on_read=True,
        force_torch_attention=True,
        fused_adamw=False,
        initialization_seed=47,
    )
    resumed_summary = resumed.run(
        max_steps=1,
        seed=53,
        resume_checkpoint=first_checkpoint,
    )
    resumed_state = _trainable_state(resumed)
    if set(resumed_state) != set(continuous_state):
        raise AssertionError("resumed Wan LoRA parameter names differ from the continuous run")
    for name, parameter in resumed_state.items():
        torch.testing.assert_close(parameter, continuous_state[name], rtol=0, atol=0)
    resumed_artifact = resumed.export_peft()
    if (
        resumed_artifact.file_digests["adapter_model.safetensors"]
        != continuous_artifact.file_digests["adapter_model.safetensors"]
    ):
        raise AssertionError("resumed Wan adapter bytes differ from the continuous run")
    resumed.close()
    del resumed, resumed_state, continuous_state
    _release()

    overfit_recipe = _recipe(
        work_dir=work_dir,
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        learning_rate=args.learning_rate,
        save_every_steps=0,
    )
    overfit = materialize_wan_cached_training_session(
        overfit_recipe,
        device="cuda",
        output_dir=work_dir / "overfit",
        checkpoint_overrides=overrides,
        verify_media_hashes=True,
        audit_cache_on_open=True,
        verify_cache_on_read=True,
        force_torch_attention=True,
        fused_adamw=False,
        initialization_seed=47,
    )
    training_summary = overfit.run(
        max_steps=args.steps,
        seed=53,
        fixed_batch=True,
        fixed_corruption=True,
        maximum_final_to_initial_loss_ratio=args.maximum_loss_ratio,
    )
    cache_dataset = VideoCachedDataset(cache_dir)
    cached_batch = collate_video_cached_samples([cache_dataset[0]])
    prepared = overfit.engine.adapter.prepare_batch(cached_batch)
    objective_batch = overfit.engine.objective.corrupt(
        prepared,
        generator=torch.Generator(device="cuda").manual_seed(2027),
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        trained_prediction = overfit.engine.adapter.forward_train(objective_batch).float().cpu()
    artifact = overfit.export_peft()
    overfit.close()
    del overfit, prepared, cached_batch, cache_dataset
    _release()

    native_recipe = default_native_diffusion_registry().resolve("wan2.1-t2v-1.3b")
    assembler = NativeDiffusionAssembler()
    policy = RuntimePolicy(
        device="cuda",
        dtype=torch.bfloat16,
        attention=AttentionBackend.TORCH,
    )
    denoiser = _fresh_denoiser(
        assembler,
        native_recipe,
        policy=policy,
        checkpoint_overrides=overrides,
    )
    denoiser.model = load_peft_adapter(denoiser.model, artifact.path)
    reloaded_adapter = WanTrainAdapter(
        denoiser,
        codec=None,
        conditioner=None,
        gradient_checkpointing=False,
        attention_compatibility_mode=True,
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        reloaded_prediction = reloaded_adapter.forward_train(objective_batch).float().cpu()
    reload_max_abs = float((trained_prediction - reloaded_prediction).abs().max())
    torch.testing.assert_close(trained_prediction, reloaded_prediction, rtol=0, atol=0)

    denoiser.model.float()
    denoiser.compute_dtype = torch.float32
    float_batch = _float_objective_batch(objective_batch)
    with torch.no_grad():
        unmerged_prediction = reloaded_adapter.forward_train(float_batch).cpu()
    denoiser.model = merge_peft_adapter(denoiser.model)
    denoiser.model.eval()
    merged_adapter = WanTrainAdapter(
        denoiser,
        codec=None,
        conditioner=None,
        gradient_checkpointing=False,
        attention_compatibility_mode=True,
    )
    with torch.no_grad():
        merged_prediction = merged_adapter.forward_train(float_batch).cpu()
    merge_delta = unmerged_prediction.float() - merged_prediction.float()
    merge_max_abs = float(merge_delta.abs().max())
    merge_rmse = float(merge_delta.square().mean().sqrt())
    merge_reference_rms = float(unmerged_prediction.float().square().mean().sqrt())
    merge_relative_rmse = merge_rmse / max(
        merge_reference_rms,
        torch.finfo(torch.float32).eps,
    )
    if (
        merge_max_abs > _MERGE_MAX_ABS_TOLERANCE
        or merge_relative_rmse > _MERGE_RELATIVE_RMSE_TOLERANCE
    ):
        raise AssertionError(
            "merged Wan LoRA prediction exceeded numerical parity bounds: "
            f"max_abs={merge_max_abs:.9g}, relative_rmse={merge_relative_rmse:.9g}"
        )
    denoiser.model.to(dtype=torch.bfloat16)
    denoiser.compute_dtype = torch.bfloat16
    denoiser.manage_autocast = True

    conditioner_key = ComponentKey(ComponentKind.CONDITIONER)
    initializer_key = ComponentKey(ComponentKind.LATENT_INITIALIZER)
    scheduler_key = ComponentKey(ComponentKind.SCHEDULER)
    codec_key = ComponentKey(ComponentKind.LATENT_ENCODER, "codec")
    inference_components = assembler.build_components(
        native_recipe,
        purpose=BuildPurpose.INFERENCE,
        policy=policy,
        checkpoint_overrides=overrides,
        component_keys=(conditioner_key, initializer_key, scheduler_key, codec_key),
    )
    inference_components[ComponentKey(ComponentKind.DENOISER)] = denoiser
    runner = assembler.strategies.build(
        native_recipe.execution.strategy,
        ExecutionBuildContext(
            recipe=native_recipe,
            components=inference_components,
            policy=policy,
            extensions=(),
        ),
    )
    output = runner.run(
        DiffusionRequest(
            prompt=(prompt,),
            height=64,
            width=64,
            num_frames=5,
            sampling=SamplingConfig(
                num_inference_steps=args.inference_steps,
                guidance_scale=1.0,
                seed=77,
            ),
        )
    )
    expected_shape = (1, 3, 5, 64, 64)
    if tuple(output.sample.shape) != expected_shape or not bool(
        torch.isfinite(output.sample).all()
    ):
        raise RuntimeError(
            f"native Wan inference returned invalid sample {tuple(output.sample.shape)}"
        )
    inference_path = work_dir / "adapter-native-inference.mkv"
    _write_inference_video(inference_path, output.sample[0])
    del runner, inference_components, output, merged_adapter, reloaded_adapter, denoiser
    _release()

    fsdp_report: dict[str, object] | None = None
    if not args.skip_fsdp_world_one:
        fsdp_recipe = _recipe(
            work_dir=work_dir,
            manifest_path=manifest_path,
            cache_dir=cache_dir,
            learning_rate=args.learning_rate,
            backend="fsdp2",
            save_every_steps=0,
        )
        os.environ.update(
            {
                "RANK": "0",
                "WORLD_SIZE": "1",
                "LOCAL_RANK": "0",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(_free_local_port()),
                "NCCL_DEBUG": "WARN",
            }
        )
        fsdp_session = materialize_wan_cached_training_session(
            fsdp_recipe,
            device="cuda",
            output_dir=work_dir / "fsdp2-world-one",
            checkpoint_overrides=overrides,
            verify_media_hashes=True,
            audit_cache_on_open=True,
            verify_cache_on_read=True,
            force_torch_attention=True,
            fused_adamw=False,
            initialization_seed=79,
        )
        try:
            fsdp_summary = fsdp_session.run(max_steps=1, seed=83)
            fsdp_artifact = fsdp_session.export_peft()
            fsdp_report = {
                "world_size": fsdp_session.world_size,
                "summary": fsdp_summary.to_dict(),
                "adapter_manifest_sha256": fsdp_artifact.manifest_sha256,
            }
        finally:
            fsdp_session.close()
            del fsdp_session
            _release()

    report = {
        "schema": "worldfoundry-wan-training-roundtrip-gate",
        "model": {
            "recipe": native_recipe.model_id,
            "model_root": str(model_root),
            "revision": native_recipe.checkpoints["dit"].revision,
            "source_revision": native_recipe.metadata["upstream_source_revision"],
        },
        "cache": {
            "dataset_digest": cache_result.index.dataset_digest,
            "index_sha256": cache_result.index.index_sha256,
            "object_sha256": cache_result.entries[0].object_sha256,
            "latent_shape": list(cache_result.entries[0].tensors["clean_latents"].shape),
            "context_shape": list(cache_result.entries[0].tensors["condition.context"].shape),
            "fixture_audit_digest": fixture_audit.digest,
        },
        "training": training_summary.to_dict(),
        "resume": {
            "checkpoint": str(first_checkpoint),
            "continuous_summary": continuous_summary.to_dict(),
            "summary": resumed_summary.to_dict(),
            "parameter_parity": "exact",
            "adapter_byte_parity": "exact",
        },
        "adapter": {
            "manifest_sha256": artifact.manifest_sha256,
            "file_sha256": dict(artifact.file_digests),
            "reload_max_abs": reload_max_abs,
            "merge_max_abs": merge_max_abs,
            "merge_rmse": merge_rmse,
            "merge_reference_rms": merge_reference_rms,
            "merge_relative_rmse": merge_relative_rmse,
            "merge_max_abs_tolerance": _MERGE_MAX_ABS_TOLERANCE,
            "merge_relative_rmse_tolerance": _MERGE_RELATIVE_RMSE_TOLERANCE,
        },
        "inference": {
            "path": str(inference_path),
            "shape": list(expected_shape),
            "steps": args.inference_steps,
            "finite": True,
        },
        "fsdp2_world_one": fsdp_report,
        "wall_time_seconds": time.perf_counter() - started,
    }
    report_path = work_dir / "gate_result.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
