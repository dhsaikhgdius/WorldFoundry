"""FSDP2 optimizer stepping with globally weighted loss reduction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from worldfoundry.core.gradient import clip_grad_norm_
from worldfoundry.training.api.contracts import (
    TrainingBatch,
    TrainingObjective,
    TrainModelAdapter,
    TrainStepResult,
)
from worldfoundry.training.distributed.fsdp import FSDP2Application

from .single_device import SingleDeviceTrainEngine

FSDP2_ENGINE_STATE_SCHEMA = "worldfoundry-training-engine-fsdp2"


def _reduced(value: torch.Tensor, op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM) -> torch.Tensor:
    result = value.detach().clone()
    dist.all_reduce(result, op=op)
    return result


def _local_scalar(value: torch.Tensor) -> torch.Tensor:
    local = value.to_local() if isinstance(value, DTensor) else value
    if local.numel() != 1:
        raise RuntimeError("expected a scalar distributed metric")
    return local.detach()


class FSDP2TrainEngine(SingleDeviceTrainEngine):
    """Run globally token-weighted FSDP2 optimizer steps.

    FSDP2 averages reduced gradients across the data-parallel mesh. Each local
    numerator is therefore scaled by ``data_parallel_size/global_denominator``
    before backward, yielding the gradient of the true global weighted loss.
    """

    def __init__(
        self,
        adapter: TrainModelAdapter,
        objective: TrainingObjective,
        optimizer: torch.optim.Optimizer,
        *,
        application: FSDP2Application,
        max_grad_norm: float = 1.0,
        autocast_dtype: torch.dtype | None = None,
    ) -> None:
        if not isinstance(application, FSDP2Application):
            raise TypeError("application must be an FSDP2Application")
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("FSDP2TrainEngine requires an initialized process group")
        if dist.get_world_size() != application.parallel_plan.world_size:
            raise RuntimeError("active process-group world size differs from FSDP2 application")
        module = getattr(adapter, "trainable_module", None)
        if not isinstance(module, FSDPModule):
            raise TypeError("adapter.trainable_module must have FSDP2 applied before engine creation")
        if tuple(name for name, _ in module.named_parameters()) != application.parameter_names:
            raise ValueError("FSDP2 application parameter identity differs from the active module")
        if not all(isinstance(parameter, DTensor) for parameter in module.parameters()):
            raise TypeError("FSDP2 engine parameters must be DTensor values")
        super().__init__(
            adapter,
            objective,
            optimizer,
            max_grad_norm=max_grad_norm,
            autocast_dtype=autocast_dtype,
        )
        self.application = application
        self.data_parallel_size = application.parallel_plan.data_parallel_size
        if self.data_parallel_size != dist.get_world_size():
            raise NotImplementedError("FSDP2 engine requires cp=tp=1 so the world group is the data-parallel group")

    def train_step(
        self,
        batch: TrainingBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> TrainStepResult:
        return self.train_accumulation((batch,), generator=generator)

    def train_accumulation(
        self,
        batches: Sequence[TrainingBatch],
        *,
        generator: torch.Generator | None = None,
    ) -> TrainStepResult:
        """Apply one globally token-weighted step over local microbatches."""

        self._ensure_ready()
        microbatches = tuple(batches)
        if not microbatches:
            raise ValueError("gradient accumulation requires at least one microbatch")
        if not all(isinstance(batch, TrainingBatch) for batch in microbatches):
            raise TypeError("gradient accumulation batches must be TrainingBatch values")
        denominator_fn = getattr(self.objective, "prepared_loss_denominator", None)
        if not callable(denominator_fn):
            raise TypeError("FSDP2 training requires objective.prepared_loss_denominator")

        root = self.adapter.trainable_module
        assert isinstance(root, FSDPModule)
        accumulating_without_sync = len(microbatches) > 1
        sync_restore_failed = False
        self._begin_training_step()
        try:
            self.optimizer.zero_grad(set_to_none=True)
            try:
                prepared_batches = tuple(self.adapter.prepare_batch(batch) for batch in microbatches)
                denominators = tuple(denominator_fn(batch).detach().float() for batch in prepared_batches)
                local_denominator = sum(
                    denominators,
                    torch.zeros((), device=denominators[0].device, dtype=torch.float32),
                )
                global_denominator = _reduced(local_denominator)
                if not bool(torch.isfinite(global_denominator)) or not bool(global_denominator > 0):
                    raise FloatingPointError("global FSDP2 loss denominator must be finite and positive")

                local_numerator = torch.zeros_like(local_denominator)
                component_numerators: dict[str, torch.Tensor] = {}
                local_sigma_sum = torch.zeros_like(local_denominator)
                local_sigma_min: torch.Tensor | None = None
                local_sigma_max: torch.Tensor | None = None
                local_sample_count = 0
                local_latent_token_count = 0
                first_diagnostics: Mapping[str, object] | None = None

                for index, (prepared, denominator) in enumerate(zip(prepared_batches, denominators)):
                    final_microbatch = index + 1 == len(prepared_batches)
                    if accumulating_without_sync:
                        root.set_requires_gradient_sync(final_microbatch)
                        root.set_reshard_after_backward(final_microbatch)
                    objective_batch = self.objective.corrupt(prepared, generator=generator)
                    with self._autocast():
                        prediction = self.adapter.forward_train(objective_batch)
                    result = self.objective.compute_loss(prediction, objective_batch)
                    if result.skipped:
                        raise RuntimeError("FSDP2 accumulation cannot include a skipped result")
                    if not isinstance(result.loss, torch.Tensor) or result.loss.numel() != 1:
                        raise TypeError("training objective must return one scalar tensor loss")
                    if not bool(torch.isfinite(result.loss.detach()).all()):
                        raise FloatingPointError("non-finite FSDP2 training loss")
                    reported_denominator = result.metrics.get("loss_denominator")
                    if not isinstance(reported_denominator, torch.Tensor):
                        raise TypeError("FSDP2 objective must report tensor loss_denominator")
                    if not torch.equal(reported_denominator.detach().float(), denominator):
                        raise RuntimeError("objective denominator changed between preparation and loss reduction")

                    gradient_weight = denominator / global_denominator * self.data_parallel_size
                    (result.loss * gradient_weight).backward()
                    numerator = result.metrics.get("loss_numerator")
                    if not isinstance(numerator, torch.Tensor):
                        raise TypeError("FSDP2 objective must report tensor loss_numerator")
                    local_numerator += numerator.detach().float()
                    for name, value in result.losses.items():
                        if isinstance(value, torch.Tensor):
                            component_numerators[name] = (
                                component_numerators.get(
                                    name,
                                    torch.zeros_like(local_denominator),
                                )
                                + value.detach().float() * denominator
                            )
                    sigma_mean = result.metrics.get("sigma_mean")
                    current_min = result.metrics.get("sigma_min")
                    current_max = result.metrics.get("sigma_max")
                    if all(isinstance(value, torch.Tensor) for value in (sigma_mean, current_min, current_max)):
                        local_sigma_sum += sigma_mean.detach().float() * result.sample_count
                        local_sigma_min = (
                            current_min.detach().float()
                            if local_sigma_min is None
                            else torch.minimum(local_sigma_min, current_min.detach().float())
                        )
                        local_sigma_max = (
                            current_max.detach().float()
                            if local_sigma_max is None
                            else torch.maximum(local_sigma_max, current_max.detach().float())
                        )
                    local_sample_count += result.sample_count
                    local_latent_token_count += result.latent_token_count
                    if first_diagnostics is None:
                        first_diagnostics = result.diagnostics

                grad_norm = clip_grad_norm_(
                    self.parameters,
                    self.max_grad_norm,
                    error_if_nonfinite=True,
                )
                self._mark_optimizer_step_started()
                self.optimizer.step()
                self.global_step += 1
            finally:
                if accumulating_without_sync:
                    try:
                        root.set_requires_gradient_sync(True)
                    except BaseException:
                        sync_restore_failed = True
                        raise
                    finally:
                        try:
                            root.set_reshard_after_backward(True)
                        except BaseException:
                            sync_restore_failed = True
                            raise

            global_numerator = _reduced(local_numerator)
            global_component_numerators = {name: _reduced(value) for name, value in component_numerators.items()}
            counts = torch.tensor(
                [local_sample_count, local_latent_token_count],
                device=global_denominator.device,
                dtype=torch.int64,
            )
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            sample_count, latent_token_count = (int(value) for value in counts.tolist())

            loss = global_numerator / global_denominator
            losses = {name: numerator / global_denominator for name, numerator in global_component_numerators.items()}
            metrics: dict[str, object] = {
                "loss_numerator": global_numerator,
                "loss_denominator": global_denominator,
                "grad_norm": _local_scalar(grad_norm),
                "global_step": torch.tensor(self.global_step, device=loss.device, dtype=torch.int64),
                "microbatch_count_per_rank": torch.tensor(
                    len(microbatches),
                    device=loss.device,
                    dtype=torch.int64,
                ),
                "data_parallel_size": torch.tensor(
                    self.data_parallel_size,
                    device=loss.device,
                    dtype=torch.int64,
                ),
            }
            if local_sample_count and local_sigma_min is not None and local_sigma_max is not None:
                global_sigma_sum = _reduced(local_sigma_sum)
                global_sigma_min = _reduced(local_sigma_min, dist.ReduceOp.MIN)
                global_sigma_max = _reduced(local_sigma_max, dist.ReduceOp.MAX)
                metrics.update(
                    {
                        "sigma_mean": global_sigma_sum / sample_count,
                        "sigma_min": global_sigma_min,
                        "sigma_max": global_sigma_max,
                    }
                )
            diagnostics = dict(first_diagnostics or {})
            diagnostics.update(
                {
                    "engine": "fsdp2",
                    "device": str(self.device),
                    "autocast_dtype": None if self.autocast_dtype is None else str(self.autocast_dtype),
                    "max_grad_norm": self.max_grad_norm,
                    "gradient_accumulation": "globally-token-weighted",
                    "parallel_plan": self.application.parallel_plan.to_dict(),
                    "fsdp2_application_digest": self.application.digest,
                }
            )
            completed = TrainStepResult(
                loss=loss,
                losses=losses,
                metrics=metrics,
                sample_count=sample_count,
                latent_token_count=latent_token_count,
                diagnostics=diagnostics,
            )
            self._complete_training_step()
            return completed
        except BaseException:
            self._abort_training_step(force_poison=sync_restore_failed)
            raise

    def state_dict(self) -> dict[str, object]:
        self._ensure_ready()
        return {
            "schema": FSDP2_ENGINE_STATE_SCHEMA,
            "global_step": self.global_step,
            "fsdp2_application_digest": self.application.digest,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        self._ensure_ready()
        if not isinstance(state_dict, Mapping):
            raise TypeError("engine state_dict must be a mapping")
        expected = {"schema", "global_step", "fsdp2_application_digest"}
        if set(state_dict) != expected:
            raise ValueError(f"FSDP2 engine state fields must be {sorted(expected)}; got {sorted(state_dict)}")
        if state_dict["schema"] != FSDP2_ENGINE_STATE_SCHEMA:
            raise ValueError(f"unsupported FSDP2 engine state schema: {state_dict['schema']!r}")
        if state_dict["fsdp2_application_digest"] != self.application.digest:
            raise ValueError("FSDP2 engine state application digest differs from the active model")
        step = state_dict["global_step"]
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("FSDP2 engine global_step must be an integer, not bool")
        if step < 0:
            raise ValueError("FSDP2 engine global_step must be a non-negative integer")
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step = step


__all__ = [
    "FSDP2_ENGINE_STATE_SCHEMA",
    "FSDP2TrainEngine",
]
