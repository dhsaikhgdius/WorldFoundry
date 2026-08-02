#!/usr/bin/env python3
"""Run the real SANA cache, overfit, PEFT reload/merge, and inference gates.

Training semantics follow NVlabs/Sana revision
6298508fcb511762a11c42cff45b2fc9fd930325.  The generated pilot image is
created locally by this script and declared CC0-1.0 in its manifest.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
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
from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.optimizations import AttentionBackend, RuntimePolicy
from worldfoundry.base_models.diffusion_model.recipes.registry import (
    default_native_diffusion_registry,
)
from worldfoundry.base_models.diffusion_model.runners import ExecutionBuildContext
from worldfoundry.training.data import (
    TRAINING_SAMPLE_SCHEMA,
    SanaCachedDataset,
    SanaCacheStore,
    SanaFeatureEncoder,
    TrainingManifestDataset,
    checkpoint_asset_digest,
    collate_sana_cached_samples,
    prepare_sana_training_cache_from_audits,
    prompt_enhancement_digest,
)
from worldfoundry.training.engine import materialize_sana_cached_training_session
from worldfoundry.training.models import SanaTrainAdapter
from worldfoundry.training.recipes import TrainingRecipe
from worldfoundry.training.safety import (
    build_shieldgemma_prompt_filter,
    shieldgemma_checkpoint_spec,
)
from worldfoundry.training.tuning import load_peft_adapter, merge_peft_adapter

_MERGE_MAX_ABS_TOLERANCE = 2.0e-3
_MERGE_RELATIVE_RMSE_TOLERANCE = 2.0e-3


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--sana-checkpoint", type=Path, required=True)
    parser.add_argument("--gemma-dir", type=Path, required=True)
    parser.add_argument("--dcae-dir", type=Path, required=True)
    parser.add_argument("--shieldgemma-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--maximum-loss-ratio", type=float, default=0.95)
    parser.add_argument("--inference-steps", type=int, default=2)
    return parser.parse_args()


def _local_checkpoint(spec: CheckpointSpec, root: Path) -> CheckpointSpec:
    return CheckpointSpec(
        source=root,
        repo_id=spec.repo_id,
        revision=spec.revision,
        files=spec.files,
        allow_patterns=spec.allow_patterns,
        metadata=spec.metadata,
        file_sha256=spec.file_sha256,
        file_size_bytes=spec.file_size_bytes,
        resource_sha256=spec.resource_sha256,
        resource_size_bytes=spec.resource_size_bytes,
    )


def _asset_digest(spec: CheckpointSpec) -> str:
    files = {f"file:{name}": digest for name, digest in spec.file_sha256.items()}
    files.update(
        {f"resource:{name}": digest for name, digest in spec.resource_sha256.items()}
    )
    return checkpoint_asset_digest(
        repository=spec.repo_id or "local-explicit",
        revision=spec.revision or "local-explicit",
        file_sha256=files,
    )


def _write_image(path: Path) -> None:
    import numpy as np
    from PIL import Image

    height = width = 128
    y, x = np.mgrid[0:height, 0:width]
    red = 24 + (x * 24 // width)
    green = 72 + (y * 32 // height)
    blue = 156 + ((x + y) * 32 // (height + width))
    image = np.stack((red, green, blue), axis=-1).astype(np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


def _write_manifest(
    path: Path,
    *,
    image_path: Path,
    prompt: str,
    safety_audit_digest: str,
) -> None:
    payload = image_path.read_bytes()
    row = {
        "schema": TRAINING_SAMPLE_SCHEMA,
        "sample_id": "synthetic-blue-cup",
        "task": "text_to_image",
        "prompt": prompt,
        "media": {
            "uri": image_path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "mime_type": "image/png",
        },
        "width": 128,
        "height": 128,
        "num_frames": 1,
        "fps": 1,
        "conditions": {},
        "split": "train",
        "license": "CC0-1.0",
        "provenance": {"source": "locally-generated-training-gate"},
        "quality": {"accepted": True, "purpose": "implementation-gate"},
        "safety": {
            "filter": "ShieldGemma-2B",
            "prompt_audit_digest": safety_audit_digest,
        },
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def _recipe(
    *,
    work_dir: Path,
    manifest_path: Path,
    cache_dir: Path,
    sana_checkpoint: Path,
    learning_rate: float,
) -> TrainingRecipe:
    return TrainingRecipe.from_mapping(
        {
            "run": {
                "id": "sana-600m-real-training-gate",
                "output_dir": str(work_dir / "run"),
            },
            "provider": {"name": "native"},
            "model": {
                "recipe": "sana-600m-512px",
                "checkpoint": str(sana_checkpoint),
            },
            "tuning": {
                "mode": "lora",
                "preset": "sana-attention",
                "rank": 4,
                "alpha": 4,
                "dropout": 0.0,
            },
            "data": {
                "manifest": str(manifest_path),
                "cache": str(cache_dir),
                "max_latent_tokens_per_microbatch": 4096,
                "split": "train",
                "shuffle": False,
                "shuffle_seed": 42,
                "tail_policy": "uneven",
                "options": {
                    "microbatch_size": 1,
                    "num_workers": 0,
                    "pin_memory": True,
                    "snapshot_every_n_steps": 1,
                },
            },
            "objective": {
                "type": "flow_matching",
                "prediction_type": "flow_velocity",
                "timestep_sampler": "logit_normal",
                "conditioning_dropout": 0.1,
                "options": {"num_train_timesteps": 1000, "flow_shift": 3.0},
            },
            "optimizer": {
                "type": "adamw",
                "learning_rate": learning_rate,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "gradient_accumulation_steps": 1,
            },
            "runtime": {
                "param_dtype": "bfloat16",
                "reduce_dtype": "float32",
                "activation_checkpoint": "none",
                "compile": False,
            },
            "distributed": {"backend": "single"},
            "export": {"format": "peft", "merge_adapter": False},
            "metadata": {
                "gate": "real-cache-overfit-export-native-inference",
                "generated_sample_license": "CC0-1.0",
            },
        }
    )


def _fresh_denoiser(
    assembler: NativeDiffusionAssembler,
    native_recipe,
    *,
    policy: RuntimePolicy,
    checkpoint_overrides,
):
    key = ComponentKey(ComponentKind.DENOISER)
    return assembler.build_components(
        native_recipe,
        purpose=BuildPurpose.TRAINING,
        policy=policy,
        checkpoint_overrides=checkpoint_overrides,
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


def _set_execution_dtype(module: torch.nn.Module, dtype: torch.dtype) -> None:
    """Keep Sana's explicit execution dtype aligned with converted parameters."""

    for child in module.modules():
        if hasattr(child, "_worldfoundry_execution_dtype"):
            child._worldfoundry_execution_dtype = dtype


def main() -> int:
    args = _arguments()
    if args.work_dir.exists():
        raise FileExistsError(f"gate work directory already exists: {args.work_dir}")
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True)
    os.environ["DISABLE_XFORMERS"] = "1"
    started = time.perf_counter()
    prompt = "A studio photograph of a blue ceramic cup on a plain neutral table."
    image_path = work_dir / "synthetic-blue-cup.png"
    manifest_path = work_dir / "manifest.jsonl"
    cache_dir = work_dir / "cache"
    _write_image(image_path)

    shield_spec = _local_checkpoint(
        shieldgemma_checkpoint_spec(),
        args.shieldgemma_dir.expanduser().resolve(),
    )
    prompt_filter = build_shieldgemma_prompt_filter(
        shield_spec,
        device="cuda",
        dtype=torch.bfloat16,
    )
    (safety_audit,) = prompt_filter.require_safe((prompt,))
    _write_manifest(
        manifest_path,
        image_path=image_path,
        prompt=prompt,
        safety_audit_digest=safety_audit.digest,
    )
    manifest = TrainingManifestDataset.from_file(
        manifest_path,
        split="train",
        verify_files=True,
        verify_hashes=True,
    )
    del prompt_filter
    gc.collect()
    torch.cuda.empty_cache()

    native_recipe = default_native_diffusion_registry().resolve("sana-600m-512px")
    assembler = NativeDiffusionAssembler()
    policy = RuntimePolicy(
        device="cuda",
        dtype=torch.bfloat16,
        attention=AttentionBackend.TORCH,
    )
    checkpoint_overrides = {
        "dit": str(args.sana_checkpoint.expanduser().resolve()),
        "text-encoder": str(args.gemma_dir.expanduser().resolve()),
        "tokenizer": str(args.gemma_dir.expanduser().resolve()),
        "codec": str(args.dcae_dir.expanduser().resolve()),
    }
    conditioner_key = ComponentKey(ComponentKind.CONDITIONER)
    codec_key = ComponentKey(ComponentKind.LATENT_ENCODER, "codec")
    encoding_components = assembler.build_components(
        native_recipe,
        purpose=BuildPurpose.TRAINING,
        policy=policy,
        checkpoint_overrides=checkpoint_overrides,
        component_keys=(conditioner_key, codec_key),
    )
    feature_encoder = SanaFeatureEncoder(
        encoding_components[codec_key],
        encoding_components[conditioner_key],
    )
    from worldfoundry.base_models.diffusion_model.models.encoders.sana.component import (
        SANA_PROMPT_PREFIX,
    )

    result = prepare_sana_training_cache_from_audits(
        manifest=manifest,
        store=SanaCacheStore(cache_dir),
        feature_encoder=feature_encoder,
        safety_audits=(safety_audit,),
        model_recipe=native_recipe.model_id,
        codec_digest=_asset_digest(native_recipe.checkpoints["codec"]),
        conditioner_digest=_asset_digest(native_recipe.checkpoints["text-encoder"]),
        tokenizer_digest=_asset_digest(native_recipe.checkpoints["tokenizer"]),
        prompt_enhancement_digest_value=prompt_enhancement_digest(
            enabled=bool(getattr(encoding_components[conditioner_key], "enhance_prompt", True)),
            max_text_length=feature_encoder.max_text_length,
            prefix=SANA_PROMPT_PREFIX,
        ),
        spatial_compression=32,
    )
    del feature_encoder, encoding_components
    gc.collect()
    torch.cuda.empty_cache()

    recipe = _recipe(
        work_dir=work_dir,
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        sana_checkpoint=args.sana_checkpoint.expanduser().resolve(),
        learning_rate=args.learning_rate,
    )
    session = materialize_sana_cached_training_session(
        recipe,
        device="cuda",
        checkpoint_overrides=None,
        verify_media_hashes=True,
        audit_cache_on_open=True,
        verify_cache_on_read=True,
        disable_xformers=True,
    )
    training_summary = session.run(
        max_steps=args.steps,
        seed=1234,
        fixed_batch=True,
        fixed_corruption=True,
        maximum_final_to_initial_loss_ratio=args.maximum_loss_ratio,
    )
    cache_dataset = SanaCachedDataset(cache_dir)
    cached_batch = collate_sana_cached_samples([cache_dataset[0]])
    prepared = session.engine.adapter.prepare_batch(cached_batch)
    objective_batch = session.engine.objective.corrupt(
        prepared,
        generator=torch.Generator(device="cuda").manual_seed(2027),
    )
    torch.manual_seed(3031)
    with torch.no_grad():
        trained_prediction = session.engine.adapter.forward_train(objective_batch).float().cpu()
    artifact = session.export_peft()
    del session, prepared, cached_batch, cache_dataset
    gc.collect()
    torch.cuda.empty_cache()

    denoiser = _fresh_denoiser(
        assembler,
        native_recipe,
        policy=policy,
        checkpoint_overrides=checkpoint_overrides,
    )
    loaded_model = load_peft_adapter(denoiser.model, artifact.path)
    denoiser.model = loaded_model
    reloaded_adapter = SanaTrainAdapter(
        denoiser,
        codec=None,
        conditioner=None,
        expected_latent_channels=32,
        spatial_compression=32,
    )
    torch.manual_seed(3031)
    with torch.no_grad():
        reloaded_prediction = reloaded_adapter.forward_train(objective_batch).float().cpu()
    reload_max_abs = float((trained_prediction - reloaded_prediction).abs().max())
    torch.testing.assert_close(trained_prediction, reloaded_prediction, rtol=0, atol=0)

    loaded_model.float()
    _set_execution_dtype(loaded_model, torch.float32)
    float_objective_batch = _float_objective_batch(objective_batch)
    if float_objective_batch.model_input.dtype is not torch.float32:
        raise AssertionError("FP32 merge validation requires an FP32 latent input")
    torch.manual_seed(3031)
    with torch.no_grad():
        unmerged_float_prediction = reloaded_adapter.forward_train(float_objective_batch).cpu()
    merged_model = merge_peft_adapter(loaded_model)
    merged_model.eval()
    denoiser.model = merged_model
    merged_adapter = SanaTrainAdapter(
        denoiser,
        codec=None,
        conditioner=None,
        expected_latent_channels=32,
        spatial_compression=32,
    )
    torch.manual_seed(3031)
    with torch.no_grad():
        merged_prediction = merged_adapter.forward_train(float_objective_batch).cpu()
    merge_delta = unmerged_float_prediction.float() - merged_prediction.float()
    merge_max_abs = float(merge_delta.abs().max())
    merge_rmse = float(merge_delta.square().mean().sqrt())
    merge_reference_rms = float(unmerged_float_prediction.float().square().mean().sqrt())
    merge_relative_rmse = merge_rmse / max(merge_reference_rms, torch.finfo(torch.float32).eps)
    if (
        merge_max_abs > _MERGE_MAX_ABS_TOLERANCE
        or merge_relative_rmse > _MERGE_RELATIVE_RMSE_TOLERANCE
    ):
        raise AssertionError(
            "merged LoRA prediction exceeded the FP32 numerical parity bounds: "
            f"max_abs={merge_max_abs:.9g} (limit {_MERGE_MAX_ABS_TOLERANCE:.9g}), "
            f"relative_rmse={merge_relative_rmse:.9g} "
            f"(limit {_MERGE_RELATIVE_RMSE_TOLERANCE:.9g})"
        )
    merged_model.to(dtype=torch.bfloat16)
    _set_execution_dtype(merged_model, torch.bfloat16)

    inference_keys = (
        conditioner_key,
        ComponentKey(ComponentKind.LATENT_INITIALIZER),
        ComponentKey(ComponentKind.SCHEDULER),
        codec_key,
    )
    inference_components = assembler.build_components(
        native_recipe,
        purpose=BuildPurpose.INFERENCE,
        policy=policy,
        checkpoint_overrides=checkpoint_overrides,
        component_keys=inference_keys,
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
            prompt=prompt,
            height=128,
            width=128,
            sampling=SamplingConfig(
                num_inference_steps=args.inference_steps,
                guidance_scale=1.0,
                seed=77,
            ),
        )
    )
    if output.sample.shape != (1, 3, 128, 128) or not bool(torch.isfinite(output.sample).all()):
        raise RuntimeError(f"native inference returned invalid sample {tuple(output.sample.shape)}")
    from PIL import Image

    sample = output.sample[0].detach().float().cpu().clamp(-1, 1)
    pixels = ((sample + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
    inference_path = work_dir / "adapter-native-inference.png"
    Image.fromarray(pixels, mode="RGB").save(inference_path)

    report = {
        "schema": "worldfoundry-sana-training-roundtrip-gate",
        "cache": {
            "dataset_digest": result.index.dataset_digest,
            "index_sha256": result.index.index_sha256,
            "object_sha256": result.entries[0].object_sha256,
            "safety_audit_digest": safety_audit.digest,
            "unsafe_probabilities": dict(safety_audit.unsafe_probabilities),
        },
        "training": training_summary.to_dict(),
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
            "shape": list(output.sample.shape),
            "steps": args.inference_steps,
            "finite": True,
        },
        "wall_time_seconds": time.perf_counter() - started,
    }
    report_path = work_dir / "gate_result.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
