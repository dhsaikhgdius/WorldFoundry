#!/usr/bin/env python3
"""Run a real-weight WorldFoundry Wan Flow-GRPO update and DCP resume gate.

The gate consumes an already-audited Wan video-training cache only to reuse
its UMT5 conditioning bytes.  It does not execute an external trainer and it
does not load the 11 GB text encoder again.  Wan DiT/VAE and VideoAlign are
still materialized through their native WorldFoundry execution paths.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from collections.abc import Mapping
from pathlib import Path

import torch

from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler
from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.recipes.registry import (
    default_native_diffusion_registry,
)
from worldfoundry.core.io.file_utils import file_sha256
from worldfoundry.core.io.integrity import canonical_json
from worldfoundry.training.data import (
    RolloutPromptDataset,
    RolloutPromptRecord,
    SharedConditioningStore,
    VideoCachedDataset,
    prepare_rollout_conditioning_cache,
)
from worldfoundry.training.data.wan.contracts import (
    wan_cache_contract_digest,
    wan_checkpoint_asset_digest,
)
from worldfoundry.training.engine import materialize_wan_flow_policy_training_run
from worldfoundry.training.recipes import PostTrainingRecipe
from worldfoundry.training.safety import PromptSafetyAudit
from worldfoundry.training.safety.shieldgemma import SHIELDGEMMA_PROMPT_POLICIES


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    return parser.parse_args()


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _stage(name: str, **values: object) -> None:
    print(
        json.dumps({"stage": name, **values}, ensure_ascii=False, sort_keys=True),
        flush=True,
    )


def _read_one_jsonl(path: Path) -> dict[str, object]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid source manifest: {path}") from error
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("the real Flow-GRPO gate requires exactly one source sample")
    return rows[0]


def _prepare_rollout_cache(
    *,
    source_cache: Path,
    source_manifest: Path,
    work_dir: Path,
    source_conditioner_digest: str,
    source_tokenizer_digest: str,
    conditioner_digest: str,
    tokenizer_digest: str,
) -> tuple[Path, Path, dict[str, object]]:
    """Convert one audited cached context into the strict rollout-cache schema."""

    row = _read_one_jsonl(source_manifest)
    sample_id = str(row.get("sample_id", "")).strip()
    prompt = str(row.get("prompt", "")).strip()
    if not sample_id or not prompt:
        raise ValueError("source manifest sample_id and prompt must be non-empty")
    source = VideoCachedDataset(source_cache)
    if len(source) != 1 or source.index.entries[0].sample_id != sample_id:
        raise ValueError("source Wan cache does not contain the selected manifest sample")
    sample = source[0]
    context = sample.tensors.get("condition.context")
    if not isinstance(context, torch.Tensor) or tuple(context.shape) != (512, 4096):
        raise ValueError("source Wan cache lacks one official [512,4096] UMT5 context")
    provenance = sample.entry.provenance
    if (
        provenance.conditioner_digest != source_conditioner_digest
        or provenance.tokenizer_digest != source_tokenizer_digest
    ):
        raise ValueError(
            "source cached context was not produced by the same resolved Wan conditioner/tokenizer"
        )

    audit = PromptSafetyAudit(
        prompt_sha256=provenance.prompt_sha256,
        unsafe_probabilities={name: 0.0 for name in SHIELDGEMMA_PROMPT_POLICIES},
        threshold=0.5,
    )
    if audit.digest != provenance.safety_audit_digest:
        raise ValueError("source fixture safety audit differs from the fixed zero-risk gate audit")
    record = RolloutPromptRecord(
        prompt_id=sample_id,
        prompt=prompt,
        safety_audit=audit,
        split="train",
        generation={"height": 64, "width": 64, "num_frames": 5},
    )
    manifest_path = work_dir / "rollout-prompts.jsonl"
    manifest_path.write_text(canonical_json(record.to_dict()) + "\n", encoding="utf-8")
    prompts = RolloutPromptDataset.from_file(manifest_path)
    cache_path = work_dir / "rollout-conditioning"

    class CachedContextEncoder:
        def encode(
            self,
            *,
            sample_id: str,
            prompt: str,
            frames: int,
            height: int,
            width: int,
        ) -> torch.Tensor:
            if (sample_id, prompt, frames, height, width) != (
                record.prompt_id,
                record.prompt,
                5,
                64,
                64,
            ):
                raise ValueError("rollout prompt differs from its cached conditioning")
            return context.detach().clone()

    prepared = prepare_rollout_conditioning_cache(
        prompts,
        cache_root=cache_path,
        encoder=CachedContextEncoder(),
        model_recipe="wan2.1-t2v-1.3b",
        model_recipe_digest=wan_cache_contract_digest("wan2.1-t2v-1.3b"),
        conditioner_digest=conditioner_digest,
        tokenizer_digest=tokenizer_digest,
    )
    source_unconditional = SharedConditioningStore(source_cache).read("unconditional")
    source_identity = source_unconditional.artifact.identity
    if (
        source_identity.conditioner_digest != source_conditioner_digest
        or source_identity.tokenizer_digest != source_tokenizer_digest
        or source_identity.model_recipe_digest
        != wan_cache_contract_digest("wan2.1-t2v-1.3b")
    ):
        raise ValueError("source unconditional context identity differs from the resolved Wan assets")
    unconditional = SharedConditioningStore(cache_path).write(
        branch="unconditional",
        prompt_sha256=source_unconditional.artifact.identity.prompt_sha256,
        model_recipe_digest=source_unconditional.artifact.identity.model_recipe_digest,
        conditioner_digest=conditioner_digest,
        tokenizer_digest=tokenizer_digest,
        tensors=source_unconditional.tensors,
        layouts={"context": "sequence-features"},
    )
    return (
        manifest_path,
        cache_path,
        {
            "source_cache_index_sha256": source.index.index_sha256,
            "rollout_index_sha256": prepared.index.digest,
            "conditioning_object_sha256": prepared.entries[0].artifact.object_sha256,
            "unconditional_object_sha256": unconditional.object_sha256,
        },
    )


def _recipe(
    *,
    manifest_path: Path,
    cache_path: Path,
    work_dir: Path,
) -> PostTrainingRecipe:
    root = Path(__file__).resolve().parents[2]
    payload = PostTrainingRecipe.from_file(
        root / "configs/post_training/wan_1p3b_flow_grpo.yaml"
    ).to_dict()
    payload["run"] = {
        "id": "wan-flow-grpo-real-roundtrip-gate",
        "output_dir": str(work_dir / "declared-run"),
    }
    payload["tuning"] = {
        "mode": "lora",
        "preset": "wan-attention",
        "rank": 4,
        "alpha": 4,
        "dropout": 0.0,
    }
    payload["data"] = {
        "manifest": str(manifest_path),
        "cache": str(cache_path),
        "split": "train",
        "shuffle": False,
        "shuffle_seed": 42,
        "tail_policy": "pad",
        "options": {
            "prompt_batch_size": 1,
            "rollout_forward_batch_size": 1,
            "replay_microbatch_size": 1,
            "num_workers": 0,
            "pin_memory": True,
            "snapshot_every_n_steps": 1,
            "generation": {"height": 64, "width": 64, "num_frames": 5},
            "vae_tiled": False,
        },
    }
    algorithm = payload["algorithm"]
    if not isinstance(algorithm, dict):
        raise TypeError("Flow-GRPO recipe algorithm must be a mapping")
    algorithm.update(
        {
            "sigmas": [1.0, 0.6, 0.0],
            "sde_step_indices": [0, 1],
            "sde_timestep_fraction": None,
            "num_sde_steps": None,
            "sde_window": None,
            "guidance_scale": 1.0,
            "init_same_noise": False,
            "eta": 0.25,
            "sigma_max": 0.6,
            "updates_per_trajectory": 1,
            "group_size": 2,
            "old_log_prob_source": "replay",
            "reference_kl_weight": 0.0,
            "reference_checkpoint": None,
            "trajectory_dtype": "float32",
            "clip_range": 0.0001,
            "clip_schedule": "constant",
            "clip_schedule_steps": None,
        }
    )
    payload["optimizer"] = {
        "type": "adamw",
        "learning_rate": 1.0e-5,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "epsilon": 1.0e-8,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 1,
    }
    payload["runtime"] = {
        "param_dtype": "bfloat16",
        "reduce_dtype": "float32",
        "activation_checkpoint": "full",
        "compile": False,
    }
    payload["distributed"] = {
        "backend": "single",
        "dp_replicate": 1,
        "dp_shard": "auto",
        "cp": 1,
        "tp": 1,
    }
    payload["checkpoint"] = {
        "save_every_steps": 1,
        "async": False,
        "export_every_steps": 0,
    }
    payload["export"] = {"format": "peft"}
    payload.pop("validation", None)
    payload.pop("metadata", None)
    return PostTrainingRecipe.from_mapping(payload)


def _configure_offline_hub(cache: Path) -> None:
    hub = cache.expanduser().resolve()
    expected = (
        hub
        / "models--Qwen--Qwen2-VL-2B-Instruct"
        / "snapshots"
        / "895c3a49bc3fa70a340399125c650a463535e71c"
        / "config.json",
        hub
        / "models--KlingTeam--VideoReward"
        / "snapshots"
        / "b8e421fe21aec3dde5f61fdd1dc44e1d603b9727"
        / "checkpoint-11352"
        / "model.pth",
    )
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"pinned VideoAlign assets are absent: {missing}")
    os.environ["HF_HUB_CACHE"] = str(hub)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _resolved_overrides(
    model_root: Path,
) -> tuple[dict[str, object], dict[str, object], object, dict[str, str]]:
    native_recipe = default_native_diffusion_registry().resolve("wan2.1-t2v-1.3b")
    assembler = NativeDiffusionAssembler()
    resolved = assembler.resolve_checkpoints(
        native_recipe,
        {name: str(model_root) for name in ("dit", "text-encoder", "tokenizer", "vae")},
    )
    source_digests = {
        "conditioner": wan_checkpoint_asset_digest(resolved["text-encoder"]),
        "tokenizer": wan_checkpoint_asset_digest(resolved["tokenizer"]),
    }
    tokenizer = resolved["tokenizer"]
    if len(tokenizer.sources) != 1:
        raise ValueError("local Wan tokenizer must resolve from one directory")
    tokenizer_root = Path(tokenizer.sources[0]).expanduser().resolve()
    tokenizer_hashes = dict(tokenizer.file_sha256)
    tokenizer_sizes = dict(tokenizer.file_size_bytes)
    for name in tokenizer.files:
        path = tokenizer_root / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"local Wan tokenizer file is absent or a symlink: {path}")
        tokenizer_sizes[name] = path.stat().st_size
        tokenizer_hashes.setdefault(name, file_sha256(path))
    resolved["tokenizer"] = CheckpointSpec(
        source=tokenizer.sources,
        files=tokenizer.files,
        allow_patterns=tokenizer.allow_patterns,
        metadata=tokenizer.metadata,
        file_sha256=tokenizer_hashes,
        file_size_bytes=tokenizer_sizes,
        resource_sha256=tokenizer.resource_sha256,
        resource_size_bytes=tokenizer.resource_size_bytes,
    )
    return (
        {"policy": resolved["dit"]},
        {name: resolved[name] for name in ("text-encoder", "tokenizer", "vae")},
        native_recipe,
        source_digests,
    )


def _trainable_state(run: object) -> dict[str, torch.Tensor]:
    roles = getattr(run, "roles")
    module = roles.policy.trainable_module
    result = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    if not result:
        raise RuntimeError("real Flow-GRPO gate found no trainable policy parameters")
    return result


def _parameter_delta(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    if set(before) != set(after):
        raise RuntimeError("Flow-GRPO trainable parameter inventory changed")
    differences = {name: after[name] - before[name] for name in before}
    return {
        "changed_parameter_tensors": sum(
            int(bool(torch.count_nonzero(value))) for value in differences.values()
        ),
        "parameter_delta_l2": math.sqrt(
            sum(float(value.double().square().sum().item()) for value in differences.values())
        ),
        "parameter_delta_max_abs": max(
            float(value.abs().max().item()) for value in differences.values()
        ),
    }


def _assert_tensor_state(
    expected: Mapping[str, torch.Tensor],
    actual: Mapping[str, torch.Tensor],
) -> int:
    if set(expected) != set(actual):
        raise AssertionError("resumed policy parameter inventory differs")
    for name in expected:
        if not torch.equal(expected[name], actual[name]):
            raise AssertionError(f"resumed policy tensor differs: {name}")
    return len(expected)


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the real Wan Flow-GRPO gate requires CUDA")
    model_root = args.model_root.expanduser().resolve()
    source_cache = args.source_cache.expanduser().resolve()
    source_manifest = args.source_manifest.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    if work_dir.exists():
        raise FileExistsError(f"Flow-GRPO gate output already exists: {work_dir}")
    work_dir.mkdir(parents=True)
    _configure_offline_hub(args.hf_cache)
    started = time.perf_counter()
    torch.manual_seed(101)
    torch.cuda.manual_seed_all(103)
    torch.cuda.reset_peak_memory_stats()

    (
        role_overrides,
        component_overrides,
        native_recipe,
        source_digests,
    ) = _resolved_overrides(model_root)
    _stage("wan-assets-audited")
    manifest_path, cache_path, cache_identity = _prepare_rollout_cache(
        source_cache=source_cache,
        source_manifest=source_manifest,
        work_dir=work_dir,
        source_conditioner_digest=source_digests["conditioner"],
        source_tokenizer_digest=source_digests["tokenizer"],
        conditioner_digest=wan_checkpoint_asset_digest(
            component_overrides["text-encoder"]
        ),
        tokenizer_digest=wan_checkpoint_asset_digest(component_overrides["tokenizer"]),
    )
    _stage("rollout-cache-ready", **cache_identity)
    recipe = _recipe(
        manifest_path=manifest_path,
        cache_path=cache_path,
        work_dir=work_dir,
    )
    _stage("materializing-training-run")
    run = materialize_wan_flow_policy_training_run(
        recipe,
        device="cuda",
        output_dir=work_dir / "run",
        audited_role_overrides=role_overrides,
        audited_component_overrides=component_overrides,
        force_torch_attention=True,
        videoalign_attention_implementation="sdpa",
        fused_adamw=False,
        initialization_seed=107,
    )
    _stage("training-run-ready")
    try:
        before = _trainable_state(run)
        summary = run.run(max_iterations=1)
        trained = _trainable_state(run)
        delta = _parameter_delta(before, trained)
        if delta["changed_parameter_tensors"] <= 0:
            raise AssertionError("real Flow-GRPO did not update any LoRA tensor")
        if not math.isfinite(summary.final_policy_loss):
            raise FloatingPointError("real Flow-GRPO returned a non-finite policy loss")
        checkpoint_path = run.checkpointer.root / f"step-{summary.final_optimizer_step:08d}"
        checkpoint = run.checkpointer.inspect(checkpoint_path)
        exported = run.export_policy_peft()
        saved_engine = dict(run.session.engine.state_dict())
        saved_progress = dict(run.session.progress.state_dict())
        saved_loader = dict(run.dataloader.state_dict())
        saved_generator = run.checkpoint_state.objective_generator.get_state().cpu().clone()
        reward_identity = dict(run.reward_identity)
        _stage(
            "optimizer-update-complete",
            optimizer_step=summary.final_optimizer_step,
            changed_parameter_tensors=delta["changed_parameter_tensors"],
        )
    finally:
        run.close()
    del run
    _release()

    _stage("materializing-resume-run")
    resumed = materialize_wan_flow_policy_training_run(
        recipe,
        device="cuda",
        output_dir=work_dir / "resumed-run",
        resume_checkpoint=checkpoint.path,
        audited_role_overrides=role_overrides,
        audited_component_overrides=component_overrides,
        force_torch_attention=True,
        videoalign_attention_implementation="sdpa",
        fused_adamw=False,
        initialization_seed=107,
    )
    _stage("resume-run-ready")
    try:
        if resumed.resume_artifact is None:
            raise AssertionError("real Flow-GRPO resume did not bind the DCP artifact")
        restored_tensors = _assert_tensor_state(trained, _trainable_state(resumed))
        if dict(resumed.session.engine.state_dict()) != saved_engine:
            raise AssertionError("resumed Flow-GRPO engine state differs")
        if dict(resumed.session.progress.state_dict()) != saved_progress:
            raise AssertionError("resumed Flow-GRPO progress differs")
        if dict(resumed.dataloader.state_dict()) != saved_loader:
            raise AssertionError("resumed Flow-GRPO data cursor differs")
        if not torch.equal(
            resumed.checkpoint_state.objective_generator.get_state().cpu(),
            saved_generator,
        ):
            raise AssertionError("resumed Flow-GRPO generator state differs")
        _stage("resume-audit-complete", restored_parameter_tensors=restored_tensors)
    finally:
        resumed.close()
    del resumed
    _release()

    report = {
        "schema": "worldfoundry-wan-flow-grpo-roundtrip-gate",
        "model": {
            "recipe": native_recipe.model_id,
            "asset_revision": native_recipe.checkpoints["dit"].revision,
        },
        "algorithm": recipe.to_dict()["algorithm"],
        "cache": cache_identity,
        "summary": summary.to_dict(),
        "policy_update": delta,
        "checkpoint": {
            "path": str(checkpoint.path),
            "manifest_sha256": checkpoint.manifest_sha256,
            "identity_digest": checkpoint.identity_digest,
            "restored_parameter_tensors": restored_tensors,
            "engine_state_exact": True,
            "progress_exact": True,
            "data_cursor_exact": True,
            "generator_state_exact": True,
        },
        "export": {
            "path": str(exported.path),
            "manifest_sha256": exported.manifest_sha256,
            "file_sha256": dict(exported.file_digests),
        },
        "reward": reward_identity,
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "cuda_device": torch.cuda.get_device_name(),
        },
    }
    destination = work_dir / "gate_result.json"
    destination.write_text(canonical_json(report) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
