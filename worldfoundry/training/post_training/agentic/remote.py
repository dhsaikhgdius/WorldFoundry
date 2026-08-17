"""Ray-backed sibling rollout workers for agentic policy training."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
from torch import nn

from worldfoundry.training.distributed.ray_runtime import (
    DeviceLease,
    RayWorkerContext,
    RayWorkerGroup,
)
from worldfoundry.training.distributed.rollout_runtime import RayPostTrainingRuntime
from worldfoundry.training.distributed.weight_sync import (
    ModuleWeightReceiver,
    NativeWeightSynchronizer,
    WeightKind,
    WeightSyncReport,
)
from worldfoundry.training.post_training.rl.algorithms.token_policy.contracts import (
    PackedTokenTrajectory,
    TokenRolloutRequest,
)

from .contracts import (
    AgenticAssistantTurn,
    AgenticRolloutRequest,
    AgenticSampleRequest,
    AgenticSampleTrajectory,
    AgenticTrajectory,
    AgenticTurn,
)
from .rollout import AgenticTurnModelAdapter, NativeAgenticRolloutAdapter
from .tools import AgentToolExecutor

RAY_AGENTIC_ROLLOUT_STATE_SCHEMA = "worldfoundry-ray-agentic-rollout"


@dataclass(frozen=True, slots=True)
class RayAgenticSampleRequest:
    """One sibling trajectory submitted independently to a rollout actor."""

    position: int
    sample: AgenticSampleRequest
    policy_revision: str
    sampling_temperature: float
    max_turns: int
    rollout_index: int
    generator_seed: int | None


@dataclass(frozen=True, slots=True)
class RayAgenticSampleResult:
    """Completed sibling or an isolated rollout failure."""

    position: int
    trajectory: AgenticSampleTrajectory | None
    error: str | None = None


def _module_device(module: nn.Module) -> torch.device:
    tensor = next(iter(module.parameters()), None)
    if tensor is None:
        tensor = next(iter(module.buffers()), None)
    return torch.device("cpu") if tensor is None else tensor.device


def _cpu_value(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu")
    if isinstance(value, Mapping):
        return {str(key): _cpu_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_value(item) for item in value)
    if isinstance(value, list):
        return [_cpu_value(item) for item in value]
    return value


def _cpu_sample_trajectory(
    trajectory: AgenticSampleTrajectory,
) -> AgenticSampleTrajectory:
    request = AgenticSampleRequest(
        sample_id=trajectory.request.sample_id,
        group_id=trajectory.request.group_id,
        messages=trajectory.request.messages,
        conditioning=_cpu_value(trajectory.request.conditioning),  # type: ignore[arg-type]
    )
    turns = tuple(
        AgenticTurn(
            assistant=AgenticAssistantTurn(
                message=turn.assistant.message,
                token_ids=turn.assistant.token_ids.detach().to(device="cpu"),
                old_log_probs=turn.assistant.old_log_probs.detach().to(device="cpu"),
                finish_reason=turn.assistant.finish_reason,
            ),
            tool_results=turn.tool_results,
        )
        for turn in trajectory.turns
    )
    return AgenticSampleTrajectory(
        request=request,
        turns=turns,
        terminal_reason=trajectory.terminal_reason,
    )


class RayAgenticRolloutWorker:
    """Actor-local policy/environment pair with revision-gated generation."""

    def __init__(
        self,
        *,
        context: RayWorkerContext,
        policy_factory: Callable[..., AgenticTurnModelAdapter],
        tool_executor_factory: Callable[..., AgentToolExecutor],
        policy_factory_kwargs: Mapping[str, object] | None = None,
        tool_executor_factory_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        policy = policy_factory(context=context, **dict(policy_factory_kwargs or {}))
        tool_executor = tool_executor_factory(
            context=context,
            **dict(tool_executor_factory_kwargs or {}),
        )
        if not isinstance(policy, AgenticTurnModelAdapter):
            raise TypeError("policy_factory must return an AgenticTurnModelAdapter")
        module = getattr(policy, "module", None)
        if not isinstance(module, nn.Module):
            raise TypeError("remote Agentic policy must expose its nn.Module")
        if not isinstance(tool_executor, AgentToolExecutor):
            raise TypeError("tool_executor_factory must return an AgentToolExecutor")
        self.context = context
        self.rollout_module = module
        self.weight_receiver = ModuleWeightReceiver(module)
        self.device = _module_device(module)
        self.rollout_adapter = NativeAgenticRolloutAdapter(policy, tool_executor)
        self.active_policy_revision: str | None = None

    def activate_policy_revision(
        self,
        policy_revision: str,
        weight_revision: int,
    ) -> None:
        if self.weight_receiver.last_revision != int(weight_revision):
            raise ValueError("actor weights do not match the activated policy revision")
        self.active_policy_revision = str(policy_revision)

    def rollout_sample(
        self,
        request: RayAgenticSampleRequest,
    ) -> RayAgenticSampleResult:
        if request.policy_revision != self.active_policy_revision:
            raise ValueError("Agentic request differs from the actor policy revision")
        generator = None
        if request.generator_seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(request.generator_seed)
        agentic_request = AgenticRolloutRequest(
            samples=(request.sample,),
            policy_revision=request.policy_revision,
            sampling_temperature=request.sampling_temperature,
            max_turns=request.max_turns,
        )
        try:
            trajectory = self.rollout_adapter.rollout_sample(
                request.sample,
                agentic_request,
                rollout_index=request.rollout_index,
                generator=generator,
            )
        except Exception as error:
            return RayAgenticSampleResult(
                position=request.position,
                trajectory=None,
                error=f"{type(error).__name__}: {error}",
            )
        return RayAgenticSampleResult(
            position=request.position,
            trajectory=_cpu_sample_trajectory(trajectory),
        )


class ActorTrainerRolloutRuntime:
    """Actor-local sync client over rollout actors created by the controller."""

    def __init__(
        self,
        actors: tuple[object, ...],
        lease: DeviceLease,
        *,
        weight_bucket_bytes: int,
    ) -> None:
        if not actors:
            raise ValueError("actor-hosted trainer requires rollout actors")
        import ray

        self.rollout_group = RayWorkerGroup(ray, lease, actors)
        self.synchronizer = NativeWeightSynchronizer(max_bucket_bytes=weight_bucket_bytes)

    def sync_rollout_weights(
        self,
        module: nn.Module,
        *,
        revision: int,
        kind: WeightKind | str,
    ) -> WeightSyncReport:
        report = self.synchronizer.sync(
            module,
            (self.rollout_group,),
            revision=revision,
            kind=kind,
        )
        return WeightSyncReport(
            revision=report.revision,
            kind=report.kind,
            tensor_count=report.tensor_count,
            byte_count=report.byte_count,
            bucket_count=report.bucket_count,
            receiver_count=len(self.rollout_group.actors),
            transmitted=report.transmitted,
        )


def _summarize_rollout_errors(
    error_counts: Mapping[str, int],
    *,
    limit: int = 5,
) -> str:
    """Render worker failure strings as ``count x message`` diagnostics."""

    ranked = Counter(dict(error_counts)).most_common()
    parts = [f"{count}x {message}" for message, count in ranked[:limit]]
    remaining = len(ranked) - len(parts)
    if remaining > 0:
        parts.append(f"... {remaining} more distinct errors")
    return "; ".join(parts)


def _generator_seeds(
    generator: torch.Generator | None,
    count: int,
) -> tuple[int | None, ...]:
    if generator is None:
        return (None,) * count
    device = torch.device(generator.device)
    return tuple(
        int(
            torch.randint(
                0,
                2**63 - 1,
                (),
                device=device,
                generator=generator,
                dtype=torch.int64,
            ).item()
        )
        for _ in range(count)
    )


class RayAgenticRolloutAdapter:
    """Trainer-side proxy that synchronizes, dispatches, and assembles groups."""

    def __init__(
        self,
        runtime: RayPostTrainingRuntime,
        source_module: nn.Module,
        *,
        weight_kind: WeightKind | str = WeightKind.FULL,
    ) -> None:
        if runtime.rollout_group is None:
            raise RuntimeError("RayPostTrainingRuntime must have rollout workers")
        if not isinstance(source_module, nn.Module):
            raise TypeError("source_module must be an nn.Module")
        self.runtime = runtime
        self.module = source_module
        self.weight_kind = WeightKind(str(weight_kind).strip().lower())
        self.completed_rollouts = 0
        self.weight_revision = -1
        self.active_policy_revision: str | None = None
        self.last_sync_report: WeightSyncReport | None = None
        # Worker error strings from the most recent rollout, keyed by message
        # with occurrence counts; empty when every sibling succeeded.
        self.last_rollout_error_counts: dict[str, int] = {}

    def _synchronize(self, policy_revision: str) -> None:
        next_revision = self.weight_revision + 1
        self.last_sync_report = self.runtime.sync_rollout_weights(
            self.module,
            revision=next_revision,
            kind=self.weight_kind,
        )
        assert self.runtime.rollout_group is not None
        self.runtime.rollout_group.broadcast(
            "activate_policy_revision",
            policy_revision,
            next_revision,
        )
        self.weight_revision = next_revision
        self.active_policy_revision = policy_revision

    def rollout_agentic(
        self,
        request: AgenticRolloutRequest,
        *,
        generator: torch.Generator | None = None,
    ) -> AgenticTrajectory:
        self._synchronize(request.policy_revision)
        rollout_index = self.completed_rollouts
        seeds = _generator_seeds(generator, len(request.samples))
        remote_requests = tuple(
            RayAgenticSampleRequest(
                position=position,
                sample=sample,
                policy_revision=request.policy_revision,
                sampling_temperature=request.sampling_temperature,
                max_turns=request.max_turns,
                rollout_index=rollout_index,
                generator_seed=seed,
            )
            for position, (sample, seed) in enumerate(zip(request.samples, seeds, strict=True))
        )
        assert self.runtime.rollout_group is not None
        raw_results = self.runtime.rollout_group.map(
            "rollout_sample",
            remote_requests,
        )
        if not all(isinstance(result, RayAgenticSampleResult) for result in raw_results):
            raise TypeError("Ray Agentic worker returned an invalid result")
        results = tuple(sorted(raw_results, key=lambda result: result.position))
        if tuple(result.position for result in results) != tuple(range(len(request.samples))):
            raise ValueError("Ray Agentic results do not cover every submitted sibling")

        error_counts = Counter(result.error for result in results if result.error is not None)
        self.last_rollout_error_counts = dict(error_counts)

        successful_counts = Counter(
            result.trajectory.request.group_id for result in results if result.trajectory is not None
        )
        trajectories = tuple(
            result.trajectory
            for result in results
            if result.trajectory is not None and successful_counts[result.trajectory.request.group_id] >= 2
        )
        selected_ids = {trajectory.request.sample_id for trajectory in trajectories}
        failed_sample_ids = tuple(
            sample.sample_id for sample in request.samples if sample.sample_id not in selected_ids
        )
        if not trajectories:
            message = "Ray Agentic rollout produced no trainable sibling group"
            if error_counts:
                message = f"{message}; sibling failures: {_summarize_rollout_errors(error_counts)}"
            raise RuntimeError(message)
        self.completed_rollouts += 1
        return AgenticTrajectory(
            samples=trajectories,
            policy_revision=request.policy_revision,
            sampling_temperature=request.sampling_temperature,
            rollout_index=rollout_index,
            failed_sample_ids=failed_sample_ids,
        )

    def rollout(
        self,
        request: TokenRolloutRequest,
        *,
        generator: torch.Generator | None = None,
    ) -> PackedTokenTrajectory:
        agentic_request = request.conditioning.get("agentic_request")
        if not isinstance(agentic_request, AgenticRolloutRequest):
            raise ValueError("token rollout request does not contain an AgenticRolloutRequest")
        expected = agentic_request.to_token_request()
        if (
            request.sample_ids != expected.sample_ids
            or request.group_ids != expected.group_ids
            or request.policy_revision != expected.policy_revision
            or request.sampling_temperature != expected.sampling_temperature
        ):
            raise ValueError("token request differs from its agentic request")
        return self.rollout_agentic(
            agentic_request,
            generator=generator,
        ).to_packed_token_trajectory()

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": RAY_AGENTIC_ROLLOUT_STATE_SCHEMA,
            "completed_rollouts": self.completed_rollouts,
            "weight_revision": self.weight_revision,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if not isinstance(state_dict, Mapping) or set(state_dict) != {
            "schema",
            "completed_rollouts",
            "weight_revision",
        }:
            raise ValueError("Ray Agentic rollout state fields differ")
        if state_dict["schema"] != RAY_AGENTIC_ROLLOUT_STATE_SCHEMA:
            raise ValueError("unsupported Ray Agentic rollout state schema")
        completed = int(state_dict["completed_rollouts"])
        weight_revision = int(state_dict["weight_revision"])
        if completed < 0 or weight_revision < -1:
            raise ValueError("saved Ray Agentic rollout position is invalid")
        self.completed_rollouts = completed
        self.weight_revision = weight_revision
        self.active_policy_revision = None
        self.last_sync_report = None


def setup_ray_agentic_rollout(
    runtime: RayPostTrainingRuntime,
    source_module: nn.Module,
    *,
    rollout_policy_factory: Callable[..., AgenticTurnModelAdapter],
    tool_executor_factory: Callable[..., AgentToolExecutor],
    rollout_policy_factory_kwargs: Mapping[str, object] | None = None,
    tool_executor_factory_kwargs: Mapping[str, object] | None = None,
    weight_kind: WeightKind | str = WeightKind.FULL,
    reward_factory: Callable[..., object] | None = None,
    reward_factory_kwargs: Mapping[str, object] | None = None,
) -> RayAgenticRolloutAdapter:
    """Start rollout actors and return a weight-synchronized trainer proxy."""

    try:
        runtime.setup(
            RayAgenticRolloutWorker,
            rollout_factory_kwargs={
                "policy_factory": rollout_policy_factory,
                "policy_factory_kwargs": dict(rollout_policy_factory_kwargs or {}),
                "tool_executor_factory": tool_executor_factory,
                "tool_executor_factory_kwargs": dict(tool_executor_factory_kwargs or {}),
            },
            reward_factory=reward_factory,
            reward_factory_kwargs=reward_factory_kwargs,
        )
        return RayAgenticRolloutAdapter(
            runtime,
            source_module,
            weight_kind=weight_kind,
        )
    except Exception:
        runtime.shutdown()
        raise


__all__ = [
    "ActorTrainerRolloutRuntime",
    "RAY_AGENTIC_ROLLOUT_STATE_SCHEMA",
    "RayAgenticRolloutAdapter",
    "RayAgenticRolloutWorker",
    "RayAgenticSampleRequest",
    "RayAgenticSampleResult",
    "setup_ray_agentic_rollout",
]
