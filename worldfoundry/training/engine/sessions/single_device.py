"""Single-device native training run lifecycle."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType

import torch
import torch.distributed as dist

from worldfoundry.core.time import utc_now_iso
from worldfoundry.training.api.contracts import TrainingBatch, TrainStepResult
from worldfoundry.training.checkpoint.artifacts import TrainingCheckpointArtifact
from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.staging import PendingTrainingCheckpoint
from worldfoundry.training.checkpoint.state import TrainingProgress, TrainingState
from worldfoundry.training.distributed.parallel import DistributedTrainingContext
from worldfoundry.training.recipes.spec import TrainingRecipe
from worldfoundry.training.tuning.peft import PeftAdapterArtifact, PeftLoraApplication

from ..artifacts import create_run_directory, export_peft_application
from ..single_device import SingleDeviceTrainEngine
from .io import (
    TRAINING_METRIC_SCHEMA,
    TRAINING_RUN_SCHEMA,
    MetricWriter,
    NullMetricWriter,
    write_json_atomic,
)
from .statistics import (
    OverfitGateError,
    SingleDeviceRunSummary,
    parameter_delta,
    parameter_snapshot,
)


class SingleDeviceTrainingSession:
    """Own one recipe run directory and its exact optimizer-step lifecycle."""

    def __init__(
        self,
        *,
        recipe: TrainingRecipe,
        engine: SingleDeviceTrainEngine,
        dataloader: Iterable[TrainingBatch],
        output_dir: str | Path | None = None,
        peft_application: PeftLoraApplication | None = None,
        data_identity: Mapping[str, object] | None = None,
        distributed_context: DistributedTrainingContext | None = None,
    ) -> None:
        if not isinstance(recipe, TrainingRecipe):
            raise TypeError("recipe must be TrainingRecipe")
        if not isinstance(engine, SingleDeviceTrainEngine):
            raise TypeError("engine must be SingleDeviceTrainEngine")
        if recipe.execution_owner != "worldfoundry-native":
            raise ValueError("training session requires WorldFoundry execution ownership")
        if distributed_context is None:
            if recipe.distributed.backend != "single":
                raise ValueError("single-device session requires distributed.backend='single'")
            rank = 0
            world_size = 1
        else:
            if not isinstance(distributed_context, DistributedTrainingContext):
                raise TypeError("distributed_context must be DistributedTrainingContext")
            if recipe.distributed.backend != "fsdp2":
                raise ValueError("distributed native session currently requires backend='fsdp2'")
            if engine.device != distributed_context.device:
                raise ValueError("training engine device differs from the distributed context")
            rank = distributed_context.rank
            world_size = distributed_context.world_size
        if not isinstance(dataloader, Iterable):
            raise TypeError("dataloader must be iterable")
        destination = Path(output_dir or recipe.run.output_dir).expanduser().resolve()
        create_run_directory(destination, distributed_context)

        self.recipe = recipe
        self.engine = engine
        self.dataloader = dataloader
        self.output_dir = destination
        self.peft_application = peft_application
        self.distributed_context = distributed_context
        self.rank = rank
        self.world_size = world_size
        self.data_identity = MappingProxyType(dict(data_identity or {}))
        self.metrics_path = destination / "metrics.jsonl"
        self.manifest_path = destination / "run.json"
        self.summary: SingleDeviceRunSummary | None = None
        self._manifest: dict[str, object] | None = None
        self._started = False
        self.objective_generator = torch.Generator(device=self.engine.device)
        self.progress = TrainingProgress(optimizer_steps=self.engine.global_step)
        self._checkpoint_state: TrainingState | None = None

    @property
    def is_coordinator(self) -> bool:
        return self.rank == 0

    def _rank_seed(self, seed: int) -> int:
        return (int(seed) + self.rank) % (2**63 - 1)

    def _maximum_across_ranks(self, value: float) -> float:
        if self.world_size == 1:
            return float(value)
        tensor = torch.tensor(value, device=self.engine.device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return float(tensor)

    def close(self) -> None:
        if self.distributed_context is not None:
            self.distributed_context.close()

    def _next_batch(self, iterator: object) -> tuple[TrainingBatch, object]:
        try:
            batch = next(iterator)  # type: ignore[arg-type]
        except StopIteration:
            iterator = iter(self.dataloader)
            try:
                batch = next(iterator)
            except StopIteration as error:
                raise RuntimeError("training dataloader produced no batches") from error
        if not isinstance(batch, TrainingBatch):
            raise TypeError(f"dataloader returned {type(batch).__name__}, expected TrainingBatch")
        return batch, iterator

    def _resume_identity(
        self,
        *,
        seed: int,
        fixed_batch: bool,
        fixed_corruption: bool,
    ) -> dict[str, object]:
        recipe = self.recipe.to_dict()
        environment: dict[str, object] = {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_type": self.engine.device.type,
        }
        if self.engine.device.type == "cuda":
            properties = torch.cuda.get_device_properties(self.engine.device)
            environment["compute_capability"] = [
                int(properties.major),
                int(properties.minor),
            ]
        return {
            "schema": "worldfoundry-training-resume-identity",
            "recipe_digest": self.recipe.digest,
            "model": recipe["model"],
            "tuning": recipe["tuning"],
            "objective": recipe["objective"],
            "optimizer": recipe["optimizer"],
            "runtime": recipe["runtime"],
            "distributed": recipe["distributed"],
            "data": dict(self.data_identity),
            "run_controls": {
                "seed": seed,
                "rank_seed_derivation": "base-seed-plus-rank",
                "fixed_batch": fixed_batch,
                "fixed_corruption": fixed_corruption,
            },
            "resolved_parallel": (
                None if getattr(self.engine, "application", None) is None else self.engine.application.to_dict()
            ),
            "environment": environment,
        }

    def _build_checkpoint_state(
        self,
        *,
        seed: int,
        fixed_batch: bool,
        fixed_corruption: bool,
    ) -> TrainingState:
        if not callable(getattr(self.dataloader, "state_dict", None)) or not callable(
            getattr(self.dataloader, "load_state_dict", None)
        ):
            raise TypeError("training checkpointing requires a stateful dataloader")
        module = getattr(self.engine.adapter, "trainable_module", None)
        if not isinstance(module, torch.nn.Module):
            raise TypeError("training adapter must expose an nn.Module for checkpointing")
        state = TrainingState(
            model=module,
            optimizer=self.engine.optimizer,
            engine=self.engine,
            dataloader=self.dataloader,
            objective_generator=self.objective_generator,
            progress=self.progress,
            identity=self._resume_identity(
                seed=seed,
                fixed_batch=fixed_batch,
                fixed_corruption=fixed_corruption,
            ),
            ignore_frozen_parameters=self.recipe.tuning.mode == "lora",
        )
        self._checkpoint_state = state
        return state

    def _record_checkpoint(self, artifact: TrainingCheckpointArtifact) -> None:
        assert self._manifest is not None
        checkpoints = list(self._manifest.get("checkpoints", []))
        checkpoints.append(
            {
                "path": str(artifact.path),
                "global_step": artifact.global_step,
                "staging_strategy": artifact.staging_strategy,
                "optional_state_presence": dict(artifact.optional_state_presence),
                "manifest_sha256": artifact.manifest_sha256,
                "identity_digest": artifact.identity_digest,
                "file_sha256": dict(artifact.file_sha256),
                "recorded_at": utc_now_iso(),
            }
        )
        self._manifest["checkpoints"] = checkpoints
        self._write_manifest()

    def _base_manifest(
        self,
        *,
        max_steps: int,
        seed: int,
        fixed_batch: bool,
        fixed_corruption: bool,
        overfit_ratio: float | None,
        initial_global_step: int,
        resumed_from: TrainingCheckpointArtifact | None,
    ) -> dict[str, object]:
        return {
            "schema": TRAINING_RUN_SCHEMA,
            "status": "running",
            "run_id": self.recipe.run.id,
            "recipe_digest": self.recipe.digest,
            "recipe": self.recipe.to_dict(),
            "data": dict(self.data_identity),
            "rank_count": self.world_size,
            "resolved_parallel": (
                None if getattr(self.engine, "application", None) is None else self.engine.application.to_dict()
            ),
            "trainable_parameter_count": sum(parameter.numel() for parameter in self.engine.parameters),
            "trainable_parameter_tensors": len(self.engine.parameters),
            "max_steps": max_steps,
            "initial_global_step": initial_global_step,
            "gradient_accumulation_steps": self.recipe.optimizer.gradient_accumulation_steps,
            "seed": seed,
            "fixed_batch": fixed_batch,
            "fixed_corruption": fixed_corruption,
            "maximum_final_to_initial_loss_ratio": overfit_ratio,
            "started_at": utc_now_iso(),
            "artifacts": {},
            "checkpoints": [],
            "resumed_from": (
                None
                if resumed_from is None
                else {
                    "path": str(resumed_from.path),
                    "global_step": resumed_from.global_step,
                    "manifest_sha256": resumed_from.manifest_sha256,
                    "identity_digest": resumed_from.identity_digest,
                }
            ),
        }

    def _write_manifest(self) -> None:
        assert self._manifest is not None
        if self.is_coordinator:
            write_json_atomic(self.manifest_path, self._manifest)

    def run(
        self,
        *,
        max_steps: int,
        seed: int = 42,
        fixed_batch: bool = False,
        fixed_corruption: bool = False,
        maximum_final_to_initial_loss_ratio: float | None = None,
        resume_checkpoint: str | Path | None = None,
    ) -> SingleDeviceRunSummary:
        """Execute optimizer steps and persist a complete or failed run record."""

        if self._started:
            raise RuntimeError("a training session can only run once")
        if isinstance(max_steps, bool) or int(max_steps) <= 0:
            raise ValueError("max_steps must be a positive integer")
        if isinstance(seed, bool):
            raise TypeError("seed must be an integer, not bool")
        if not isinstance(fixed_batch, bool) or not isinstance(fixed_corruption, bool):
            raise TypeError("fixed_batch and fixed_corruption must be bool values")
        ratio = maximum_final_to_initial_loss_ratio
        if ratio is not None:
            ratio = float(ratio)
            if not math.isfinite(ratio) or not 0 < ratio <= 1:
                raise ValueError("maximum_final_to_initial_loss_ratio must be in (0, 1]")
            if not fixed_batch or not fixed_corruption:
                raise ValueError("an overfit loss gate requires fixed_batch and fixed_corruption")
        checkpoint_interval = self.recipe.checkpoint.save_every_steps
        checkpoint_requested = checkpoint_interval > 0 or resume_checkpoint is not None
        if checkpoint_requested and fixed_batch:
            raise ValueError("exact checkpoint resume for diagnostic fixed_batch runs is not supported")

        self._started = True
        accumulation = self.recipe.optimizer.gradient_accumulation_steps
        checkpoint_state: TrainingState | None = None
        resumed_from: TrainingCheckpointArtifact | None = None
        if checkpoint_requested:
            checkpoint_state = self._build_checkpoint_state(
                seed=int(seed),
                fixed_batch=fixed_batch,
                fixed_corruption=fixed_corruption,
            )
        if resume_checkpoint is not None:
            resume_path = Path(resume_checkpoint).expanduser().resolve()
            if not resume_path.is_dir():
                raise FileNotFoundError(f"resume checkpoint is not a directory: {resume_path}")
            assert checkpoint_state is not None
            resumed_from = TrainingCheckpointer(resume_path.parent).load(
                checkpoint_state,
                resume_path,
            )
        else:
            rank_seed = self._rank_seed(int(seed))
            self.objective_generator.manual_seed(rank_seed)
            random.seed(rank_seed)
            torch.manual_seed(rank_seed)
            if self.engine.device.type == "cuda":
                torch.cuda.manual_seed_all(rank_seed)
        initial_global_step = self.engine.global_step
        self._manifest = self._base_manifest(
            max_steps=int(max_steps),
            seed=int(seed),
            fixed_batch=fixed_batch,
            fixed_corruption=fixed_corruption,
            overfit_ratio=ratio,
            initial_global_step=initial_global_step,
            resumed_from=resumed_from,
        )
        self._write_manifest()
        writer: MetricWriter | NullMetricWriter = (
            MetricWriter(self.metrics_path) if self.is_coordinator else NullMetricWriter()
        )
        before = parameter_snapshot(self.engine.parameters)
        iterator = iter(self.dataloader)
        fixed_microbatches: tuple[TrainingBatch, ...] | None = None
        completed_steps = 0
        total_microbatches = 0
        total_samples = 0
        total_tokens = 0
        losses: list[float] = []
        started = time.perf_counter()
        checkpointer = TrainingCheckpointer(self.output_dir / "checkpoints") if checkpoint_interval > 0 else None
        pending_checkpoint: PendingTrainingCheckpoint | None = None

        def finish_pending_checkpoint() -> None:
            nonlocal pending_checkpoint
            current = pending_checkpoint
            pending_checkpoint = None
            if current is not None:
                self._record_checkpoint(current.wait())

        if self.engine.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.engine.device)
        try:
            for step_index in range(int(max_steps)):
                if fixed_microbatches is None:
                    current: list[TrainingBatch] = []
                    for _ in range(accumulation):
                        batch, iterator = self._next_batch(iterator)
                        current.append(batch)
                    microbatches = tuple(current)
                    if fixed_batch:
                        fixed_microbatches = microbatches
                else:
                    microbatches = fixed_microbatches

                if fixed_corruption:
                    rank_seed = self._rank_seed(int(seed))
                    self.objective_generator.manual_seed(rank_seed)
                    torch.manual_seed(rank_seed)
                    if self.engine.device.type == "cuda":
                        torch.cuda.manual_seed_all(rank_seed)

                step_started = time.perf_counter()
                result = self.engine.train_accumulation(
                    microbatches,
                    generator=self.objective_generator,
                )
                if self.engine.device.type == "cuda":
                    torch.cuda.synchronize(self.engine.device)
                step_seconds = self._maximum_across_ranks(time.perf_counter() - step_started)
                loss = float(result.loss.detach())
                losses.append(loss)
                completed_steps += 1
                total_microbatches += len(microbatches)
                total_samples += result.sample_count
                total_tokens += result.latent_token_count
                self.progress.record_step(
                    microbatches=len(microbatches),
                    samples=result.sample_count,
                    latent_tokens=result.latent_token_count,
                )
                if self.progress.optimizer_steps != self.engine.global_step:
                    raise RuntimeError("session progress differs from the engine global step")
                metric = self._metric_record(
                    step_index=step_index,
                    result=result,
                    step_seconds=step_seconds,
                    microbatch_count=len(microbatches),
                )
                writer.write(metric)
                if checkpointer is not None and self.engine.global_step % checkpoint_interval == 0:
                    assert checkpoint_state is not None
                    finish_pending_checkpoint()
                    saved = checkpointer.save(
                        checkpoint_state,
                        asynchronous=self.recipe.checkpoint.async_save,
                    )
                    if isinstance(saved, PendingTrainingCheckpoint):
                        pending_checkpoint = saved
                    else:
                        self._record_checkpoint(saved)
            finish_pending_checkpoint()
        except Exception as error:
            checkpoint_error: Exception | None = None
            if pending_checkpoint is not None:
                try:
                    finish_pending_checkpoint()
                except Exception as pending_error:  # noqa: BLE001 - preserve the training failure.
                    checkpoint_error = pending_error
            elapsed = time.perf_counter() - started
            self._manifest.update(
                {
                    "status": "failed",
                    "finished_at": utc_now_iso(),
                    "completed_steps": completed_steps,
                    "cumulative_progress": self.progress.state_dict(),
                    "wall_time_seconds": elapsed,
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            )
            if checkpoint_error is not None:
                self._manifest["checkpoint_error"] = {
                    "type": type(checkpoint_error).__name__,
                    "message": str(checkpoint_error),
                }
            self._write_manifest()
            raise
        finally:
            writer.close()

        elapsed = self._maximum_across_ranks(time.perf_counter() - started)
        changed, delta_l2, delta_max = parameter_delta(
            self.engine.parameters,
            before,
            device=self.engine.device,
            distributed=self.world_size > 1,
        )
        initial_loss = losses[0]
        final_loss = losses[-1]
        reduction = 0.0 if initial_loss == 0 else (initial_loss - final_loss) / initial_loss
        overfit_passed = None
        if ratio is not None:
            overfit_passed = final_loss <= initial_loss * ratio and changed > 0
        peak_allocated = None
        peak_reserved = None
        if self.engine.device.type == "cuda":
            peak_allocated = int(torch.cuda.max_memory_allocated(self.engine.device))
            peak_reserved = int(torch.cuda.max_memory_reserved(self.engine.device))
            if self.world_size > 1:
                peak_allocated = int(self._maximum_across_ranks(float(peak_allocated)))
                peak_reserved = int(self._maximum_across_ranks(float(peak_reserved)))
        summary = SingleDeviceRunSummary(
            optimizer_steps=completed_steps,
            microbatches=total_microbatches,
            sample_count=total_samples,
            latent_token_count=total_tokens,
            initial_loss=initial_loss,
            final_loss=final_loss,
            best_loss=min(losses),
            loss_reduction_fraction=reduction,
            wall_time_seconds=elapsed,
            samples_per_second=total_samples / elapsed,
            latent_tokens_per_second=total_tokens / elapsed,
            changed_parameter_tensors=changed,
            parameter_delta_l2=delta_l2,
            parameter_delta_max_abs=delta_max,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            overfit_gate_passed=overfit_passed,
        )
        self.summary = summary
        self._manifest.update(
            {
                "status": "complete" if overfit_passed is not False else "gate-failed",
                "finished_at": utc_now_iso(),
                "summary": summary.to_dict(),
                "cumulative_progress": self.progress.state_dict(),
            }
        )
        self._write_manifest()
        if overfit_passed is False:
            raise OverfitGateError(summary)
        return summary

    def _metric_record(
        self,
        *,
        step_index: int,
        result: TrainStepResult,
        step_seconds: float,
        microbatch_count: int,
    ) -> dict[str, object]:
        record: dict[str, object] = {
            "schema": TRAINING_METRIC_SCHEMA,
            "run_id": self.recipe.run.id,
            "recipe_digest": self.recipe.digest,
            "optimizer_step": self.engine.global_step,
            "run_step_index": step_index,
            "microbatch_count": microbatch_count,
            "sample_count": result.sample_count,
            "latent_token_count": result.latent_token_count,
            "loss": result.loss,
            "losses": result.losses,
            "metrics": result.metrics,
            "learning_rates": [float(group["lr"]) for group in self.engine.optimizer.param_groups],
            "step_time_seconds": step_seconds,
            "samples_per_second": result.sample_count / step_seconds,
            "latent_tokens_per_second": result.latent_token_count / step_seconds,
            "recorded_at": utc_now_iso(),
        }
        if self.engine.device.type == "cuda":
            record["cuda_memory"] = {
                "allocated_bytes": int(torch.cuda.memory_allocated(self.engine.device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(self.engine.device)),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.engine.device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.engine.device)),
            }
        return record

    def export_peft(self, output_dir: str | Path | None = None) -> PeftAdapterArtifact:
        """Export and attach a digest-audited PEFT artifact to the run manifest."""

        if self.summary is None or self._manifest is None:
            raise RuntimeError("training must complete before PEFT export")
        if self.peft_application is None:
            raise RuntimeError("this training session has no PEFT application to export")
        destination = Path(output_dir or (self.output_dir / "adapter"))
        metadata = {
            "run_id": self.recipe.run.id,
            "recipe_digest": self.recipe.digest,
            "data": dict(self.data_identity),
            "training_summary": self.summary.to_dict(),
        }
        artifact = export_peft_application(
            self.peft_application,
            destination,
            metadata=metadata,
            distributed_context=self.distributed_context,
            role="training PEFT adapter",
        )
        artifacts = dict(self._manifest.get("artifacts", {}))
        artifacts["peft_adapter"] = {
            "path": str(artifact.path),
            "manifest_sha256": artifact.manifest_sha256,
            "file_sha256": dict(artifact.file_digests),
        }
        self._manifest["artifacts"] = artifacts
        self._write_manifest()
        return artifact


__all__ = ["SingleDeviceTrainingSession"]
