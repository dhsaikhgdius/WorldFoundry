#!/usr/bin/env python3
"""Run a strict real-weight AnyFlow update and exact DCP-resume gate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.optimizations import (
    AttentionBackend,
    RuntimePolicy,
)
from worldfoundry.core.attention.chunk_partition import TemporalChunkPartition
from worldfoundry.training.checkpoint import (
    TrainingCheckpointer,
    TrainingProgress,
    TrainingState,
)
from worldfoundry.training.models.anyflow import (
    ANYFLOW_BIDIRECTIONAL_WAN_SMALL_CHECKPOINT,
    ANYFLOW_FAR_WAN_SMALL_CHECKPOINT,
    NativeAnyFlowModelMaterializer,
)
from worldfoundry.training.post_training import (
    AnyFlowTrainingBatch,
    NativeAnyFlowPretrainingSession,
    build_native_anyflow_pretraining_stack,
)
from worldfoundry.training.recipes import PostTrainingRecipe


class _GateLoader:
    """Deterministic typed batches with the state contract used by training."""

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        frames: int,
        height: int,
        width: int,
        context_tokens: int,
        seed: int,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.frames = int(frames)
        self.height = int(height)
        self.width = int(width)
        self.context_tokens = int(context_tokens)
        self.seed = int(seed)
        self.cursor = 0

    def __iter__(self) -> _GateLoader:
        return self

    def __next__(self) -> AnyFlowTrainingBatch:
        cursor = self.cursor
        generator = torch.Generator(device=self.device).manual_seed(
            self.seed + cursor
        )
        clean = torch.randn(
            (1, 16, self.frames, self.height, self.width),
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        context = torch.randn(
            (1, self.context_tokens, 4096),
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        self.cursor += 1
        return AnyFlowTrainingBatch(
            sample_ids=(f"official-anyflow-{cursor}",),
            clean_latents=clean,
            conditioning={"context": context},
            unconditional_conditioning={"context": torch.zeros_like(context)},
        )

    def state_dict(self) -> dict[str, int]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping) or set(state_dict) != {"cursor"}:
            raise ValueError("AnyFlow gate loader state fields differ")
        cursor = state_dict["cursor"]
        if isinstance(cursor, bool) or int(cursor) < 0:
            raise ValueError("AnyFlow gate loader cursor must be non-negative")
        self.cursor = int(cursor)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional Hugging Face Hub cache containing the pinned snapshot.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--height", type=int, default=2)
    parser.add_argument("--width", type=int, default=2)
    parser.add_argument("--context-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--far-forward-only",
        action="store_true",
        help=(
            "Strictly load the pinned FAR checkpoint and execute its causal "
            "compressed-context forward without running an optimizer roundtrip."
        ),
    )
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _checkpoint_identity() -> str:
    return _audited_checkpoint_identity(
        ANYFLOW_BIDIRECTIONAL_WAN_SMALL_CHECKPOINT
    )


def _far_checkpoint_identity() -> str:
    return _audited_checkpoint_identity(ANYFLOW_FAR_WAN_SMALL_CHECKPOINT)


def _audited_checkpoint_identity(checkpoint: CheckpointSpec) -> str:
    if checkpoint.repo_id is None or checkpoint.revision is None:
        raise RuntimeError("AnyFlow real gate requires an immutable Hub checkpoint")
    weight = "transformer/diffusion_pytorch_model.safetensors"
    if set(checkpoint.file_sha256) != {weight} or set(
        checkpoint.file_size_bytes
    ) != {weight}:
        raise RuntimeError("AnyFlow real gate checkpoint has no strict weight audit")
    if set(checkpoint.resource_sha256) != {"transformer/config.json"} or set(
        checkpoint.resource_size_bytes
    ) != {"transformer/config.json"}:
        raise RuntimeError("AnyFlow real gate checkpoint has no strict config audit")
    return f"{checkpoint.repo_id}@{checkpoint.revision}"


def _recipe(
    *,
    output_dir: Path,
    identity: str,
    dtype: str,
    learning_rate: float,
    seed: int,
) -> PostTrainingRecipe:
    return PostTrainingRecipe.from_mapping(
        {
            "run": {
                "id": "anyflow-bidirectional-real-roundtrip",
                "output_dir": str(output_dir),
            },
            "model": {"recipe": "wan", "checkpoint": identity},
            "tuning": {"mode": "full"},
            "data": {
                "manifest": "synthetic-real-checkpoint-gate",
                "shuffle_seed": seed,
            },
            "algorithm": {
                "type": "anyflow-bidirectional-pretrain",
                "flow_map": {
                    "num_train_timesteps": 1000,
                    "timestep_shift": 5.0,
                    "central_difference_epsilon": 5.0,
                    "diffusion_ratio": 1.0,
                    "consistency_ratio": 0.0,
                    "fused_guidance_scale": 1.0,
                },
                "image_conditioning_probability": 0.0,
                "conditioning_dropout_probability": 0.0,
            },
            "optimizer": {
                "type": "adamw",
                "learning_rate": learning_rate,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "gradient_accumulation_steps": 1,
            },
            "runtime": {
                "param_dtype": dtype,
                "reduce_dtype": "float32",
                "activation_checkpoint": "full",
                "compile": False,
            },
            "distributed": {"backend": "single"},
            "checkpoint": {
                "save_every_steps": 1,
                "async": False,
            },
            "export": {"format": "safetensors"},
        }
    )


def _hash_value(digest: object, value: object) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous().cpu()
        descriptor = {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        digest.update(b"tensor:")
        digest.update(json.dumps(descriptor, sort_keys=True).encode("utf-8"))
        digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping:")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _hash_value(digest, key)
            _hash_value(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        digest.update(f"{type(value).__name__}:".encode("utf-8"))
        for item in value:
            _hash_value(digest, item)
        return
    digest.update(f"scalar:{type(value).__name__}:{value!r}".encode("utf-8"))


def _state_digest(value: object) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


def _runtime_rng_digest(objective_generator: torch.Generator) -> str:
    return _state_digest(
        {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": tuple(torch.cuda.get_rng_state_all()),
            "objective_generator": objective_generator.get_state(),
            "python": random.getstate(),
        }
    )


def _tracked_parameter(model: nn.Module) -> tuple[str, nn.Parameter]:
    for name, parameter in model.named_parameters():
        if name.endswith("proj_out.weight"):
            return name, parameter
    raise RuntimeError("materialized AnyFlow model has no proj_out.weight")


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_far_forward(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> int:
    if args.height % 4 or args.width % 4:
        raise ValueError(
            "AnyFlow FAR latent geometry must be divisible by "
            "compressed_patch_size=(1,4,4)"
        )
    checkpoint = ANYFLOW_FAR_WAN_SMALL_CHECKPOINT
    identity = _far_checkpoint_identity()
    partition = TemporalChunkPartition(
        chunks=(1, 3, 3, 3, 3, 3, 3, 2),
        full_chunk_limit=3,
        patch_size=(1, 2, 2),
        compressed_patch_size=(1, 4, 4),
    )
    sampled_chunk_count = 4
    context_frames, target_frames = partition.context_target_frames(
        sampled_chunk_count
    )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    adapter = NativeAnyFlowModelMaterializer().far_student(
        checkpoint,
        checkpoint_identity=identity,
        partition=partition,
        policy=RuntimePolicy(
            device=device,
            dtype=dtype,
            attention=AttentionBackend.TORCH,
        ),
    )
    _synchronize(device)
    load_seconds = time.perf_counter() - started
    adapter.module.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    shape = (1, 16, target_frames, args.height, args.width)
    noisy = torch.randn(
        shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    clean = torch.randn(
        shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    context = torch.randn(
        (1, 16, context_frames, args.height, args.width),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    text_context = torch.randn(
        (1, args.context_tokens, 4096),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    source = torch.full(
        (1, target_frames),
        800.0,
        device=device,
        dtype=torch.float32,
    )
    destination = torch.full(
        (1, target_frames),
        200.0,
        device=device,
        dtype=torch.float32,
    )
    started = time.perf_counter()
    with torch.no_grad():
        output = adapter.predict_flow_map(
            noisy,
            source,
            destination,
            clean_latents=clean,
            context_latents=context,
            partition=partition,
            sampled_chunk_count=sampled_chunk_count,
            sample_ids=("official-anyflow-far",),
            conditioning={"context": text_context},
            training=False,
        )
    _synchronize(device)
    forward_seconds = time.perf_counter() - started
    if output.shape != noisy.shape or output.dtype != noisy.dtype:
        raise AssertionError("AnyFlow FAR output contract differs from its input")
    if not bool(torch.isfinite(output).all()):
        raise FloatingPointError("AnyFlow FAR checkpoint produced non-finite output")

    weight = "transformer/diffusion_pytorch_model.safetensors"
    summary = {
        "checkpoint_identity": identity,
        "weight_size_bytes": checkpoint.file_size_bytes[weight],
        "weight_sha256": checkpoint.file_sha256[weight],
        "config_size_bytes": checkpoint.resource_size_bytes[
            "transformer/config.json"
        ],
        "config_sha256": checkpoint.resource_sha256[
            "transformer/config.json"
        ],
        "parameter_count": sum(
            parameter.numel() for parameter in adapter.module.parameters()
        ),
        "device": str(device),
        "dtype": str(dtype),
        "sampled_chunk_count": sampled_chunk_count,
        "context_frames": context_frames,
        "target_frames": target_frames,
        "input_shape": list(noisy.shape),
        "output_shape": list(output.shape),
        "output_finite": True,
        "output_mean": float(output.float().mean()),
        "load_seconds": load_seconds,
        "forward_seconds": forward_seconds,
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    del adapter
    _release()
    return 0


def main() -> int:
    args = _arguments()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"AnyFlow gate output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    if args.cache_dir is not None:
        os.environ["HF_HUB_CACHE"] = str(args.cache_dir.expanduser().resolve())
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the real AnyFlow update/resume gate requires CUDA")
    dtype = _torch_dtype(args.dtype)
    for name in ("frames", "height", "width", "context_tokens"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.frames % 1 or args.height % 2 or args.width % 2:
        raise ValueError(
            "AnyFlow gate latent geometry must be divisible by patch_size=(1,2,2)"
        )
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning-rate must be finite and positive")
    if isinstance(args.seed, bool) or args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.far_forward_only:
        return _run_far_forward(
            args=args,
            output_dir=output_dir,
            device=device,
            dtype=dtype,
        )

    started = time.perf_counter()
    timings: dict[str, float] = {}
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    identity = _checkpoint_identity()
    checkpoint = ANYFLOW_BIDIRECTIONAL_WAN_SMALL_CHECKPOINT
    recipe = _recipe(
        output_dir=output_dir,
        identity=identity,
        dtype=args.dtype,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    stage = time.perf_counter()
    adapter = NativeAnyFlowModelMaterializer().bidirectional_student(
        checkpoint,
        checkpoint_identity=identity,
        policy=RuntimePolicy(
            device=device,
            dtype=dtype,
            attention=AttentionBackend.TORCH,
        ),
        gradient_checkpointing=True,
    )
    _synchronize(device)
    timings["load_seconds"] = time.perf_counter() - stage

    stack = build_native_anyflow_pretraining_stack(
        recipe,
        student=adapter,
        fused_adamw=False,
    )
    loader = _GateLoader(
        device=device,
        dtype=dtype,
        frames=args.frames,
        height=args.height,
        width=args.width,
        context_tokens=args.context_tokens,
        seed=args.seed + 1000,
    )
    progress = TrainingProgress()
    objective_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    state = TrainingState(
        model=stack.model,
        optimizer=stack.optimizer,
        engine=stack.engine,
        dataloader=loader,
        objective_generator=objective_generator,
        progress=progress,
        identity={
            "recipe_digest": recipe.digest,
            "checkpoint": identity,
            "weight_sha256": dict(checkpoint.file_sha256),
            "config_sha256": dict(checkpoint.resource_sha256),
            "input": {
                "frames": args.frames,
                "height": args.height,
                "width": args.width,
                "context_tokens": args.context_tokens,
                "loader_seed": args.seed + 1000,
            },
        },
        **stack.checkpoint_state_kwargs(),
    )
    manager = TrainingCheckpointer(output_dir / "checkpoints")
    session = NativeAnyFlowPretrainingSession(
        stack.engine,
        loader,
        progress,
        checkpoint_state=state,
        checkpointer=manager,
        save_every_steps=recipe.checkpoint.save_every_steps,
        asynchronous_checkpoints=recipe.checkpoint.async_save,
    )
    parameter_count = sum(parameter.numel() for parameter in stack.model.parameters())
    tracked_name, tracked_parameter = _tracked_parameter(stack.model)
    tracked_before = tracked_parameter.detach().clone()

    stage = time.perf_counter()
    first_summary = session.run(max_steps=1)
    _synchronize(device)
    timings["first_update_and_dcp_save_seconds"] = time.perf_counter() - stage
    artifact = manager.inspect(output_dir / "checkpoints" / "step-00000001")
    if first_summary.final_step != 1 or loader.cursor != 1:
        raise AssertionError("AnyFlow production session did not commit step one")
    first_delta = tracked_parameter.detach().float() - tracked_before.float()
    changed_elements = int(torch.count_nonzero(first_delta).item())
    if changed_elements <= 0:
        raise AssertionError(
            "AnyFlow optimizer did not change proj_out.weight; increase --learning-rate"
        )
    tracked_gradient = tracked_parameter.grad
    if tracked_gradient is None or not bool(torch.isfinite(tracked_gradient).all()):
        raise AssertionError("AnyFlow production engine produced no finite tracked gradient")
    if not bool(torch.count_nonzero(tracked_gradient)):
        raise AssertionError("AnyFlow production engine produced a zero tracked gradient")

    checkpoint_engine_state = stack.engine.state_dict()
    checkpoint_engine_digest = _state_digest(checkpoint_engine_state)
    checkpoint_progress_state = progress.state_dict()
    checkpoint_model_digest = _state_digest(stack.model.state_dict())
    checkpoint_optimizer_digest = _state_digest(stack.optimizer.state_dict())
    checkpoint_rng_digest = _runtime_rng_digest(objective_generator)

    session.save_every_steps = 0
    stage = time.perf_counter()
    continuous_summary = session.run(max_steps=1)
    _synchronize(device)
    timings["continuous_update_seconds"] = time.perf_counter() - stage
    expected_engine_state = stack.engine.state_dict()
    expected_engine_digest = _state_digest(expected_engine_state)
    expected_progress_state = progress.state_dict()
    expected_model_digest = _state_digest(stack.model.state_dict())
    expected_optimizer_digest = _state_digest(stack.optimizer.state_dict())
    expected_rng_digest = _runtime_rng_digest(objective_generator)

    stage = time.perf_counter()
    loaded = manager.load(state, artifact.path)
    _synchronize(device)
    timings["dcp_load_seconds"] = time.perf_counter() - stage
    if progress.state_dict() != checkpoint_progress_state:
        raise AssertionError("AnyFlow DCP did not restore production progress state")
    if _state_digest(stack.engine.state_dict()) != checkpoint_engine_digest:
        raise AssertionError("AnyFlow DCP did not restore engine decisions and cadence")
    if loader.cursor != 1:
        raise AssertionError("AnyFlow DCP did not restore the data cursor")
    if _state_digest(stack.model.state_dict()) != checkpoint_model_digest:
        raise AssertionError("AnyFlow DCP model bytes differ at the restored boundary")
    if _state_digest(stack.optimizer.state_dict()) != checkpoint_optimizer_digest:
        raise AssertionError(
            "AnyFlow DCP optimizer bytes differ at the restored boundary"
        )
    if _runtime_rng_digest(objective_generator) != checkpoint_rng_digest:
        raise AssertionError("AnyFlow DCP did not restore all rank-local RNG state")

    stage = time.perf_counter()
    resumed_summary = session.run(max_steps=1)
    _synchronize(device)
    timings["resumed_update_seconds"] = time.perf_counter() - stage
    if resumed_summary.final_loss != continuous_summary.final_loss:
        raise AssertionError("AnyFlow resumed continuation loss is not bit-exact")
    if _state_digest(stack.engine.state_dict()) != expected_engine_digest:
        raise AssertionError("AnyFlow resumed engine decisions are not exact")
    if progress.state_dict() != expected_progress_state:
        raise AssertionError("AnyFlow resumed progress counters are not exact")
    resumed_model_digest = _state_digest(stack.model.state_dict())
    resumed_optimizer_digest = _state_digest(stack.optimizer.state_dict())
    if resumed_model_digest != expected_model_digest:
        raise AssertionError("AnyFlow resumed model update is not byte-exact")
    if resumed_optimizer_digest != expected_optimizer_digest:
        raise AssertionError("AnyFlow resumed optimizer update is not byte-exact")
    if _runtime_rng_digest(objective_generator) != expected_rng_digest:
        raise AssertionError("AnyFlow resumed rank-local RNG continuation is not exact")
    if loaded.manifest_sha256 != artifact.manifest_sha256:
        raise AssertionError("AnyFlow resumed a different DCP artifact")

    timings["total_seconds"] = time.perf_counter() - started
    summary = {
        "checkpoint_identity": identity,
        "checkpoint_manifest_sha256": artifact.manifest_sha256,
        "recipe_digest": recipe.digest,
        "parameter_count": parameter_count,
        "device": str(device),
        "dtype": str(dtype),
        "input_shape": [1, 16, args.frames, args.height, args.width],
        "first_loss": first_summary.final_loss,
        "continuous_loss": continuous_summary.final_loss,
        "resumed_loss": resumed_summary.final_loss,
        "changed_parameter": tracked_name,
        "changed_elements": changed_elements,
        "tracked_gradient_l2": float(
            tracked_gradient.detach().double().square().sum().sqrt()
        ),
        "parameter_delta_l2": float(first_delta.double().square().sum().sqrt()),
        "parameter_delta_max_abs": float(first_delta.abs().max()),
        "checkpoint_model_digest": checkpoint_model_digest,
        "checkpoint_optimizer_digest": checkpoint_optimizer_digest,
        "continuation_model_digest": expected_model_digest,
        "continuation_optimizer_digest": expected_optimizer_digest,
        "engine_decision_draw_count": int(
            expected_engine_state["decisions"]["draw_count"]
        ),
        "exact_resume": True,
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "timings": timings,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    del state, stack, adapter
    _release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
