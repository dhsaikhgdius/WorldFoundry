"""Qwen3 materialization for native grouped Agentic policy learning and token PPO."""

from __future__ import annotations

import ast
import copy
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from worldfoundry.training.distributed.ray_runtime import (
    DeviceLease,
    RayWorkerContext,
)
from worldfoundry.training.distributed.rollout_runtime import (
    RayPostTrainingRuntime,
    TrainerBinding,
    ray_runtime_config_from_rollout_spec,
)
from worldfoundry.training.post_training.agentic import (
    ActorTrainerRolloutRuntime,
    AgenticPrompt,
    AgenticRewardComponent,
    AgenticSampleTrajectory,
    AgentToolExecutor,
    CausalLMAgenticPolicyAdapter,
    CausalLMGenerationConfig,
    HTTPAgenticRewardAdapter,
    LocalAgentTool,
    LocalToolExecutor,
    NativeAgenticTrainingRun,
    RayAgenticRolloutAdapter,
    RayAgenticRolloutWorker,
    load_agentic_prompts,
    materialize_agentic_training_run,
    setup_ray_agentic_rollout,
)
from worldfoundry.training.post_training.rewards.http.client import (
    HTTPRewardEvaluator,
)
from worldfoundry.training.post_training.rl.algorithms.token_policy.contracts import (
    TokenTrajectoryRewardAdapter,
)
from worldfoundry.training.post_training.rl.algorithms.token_ppo import (
    NativeTokenPPOTrainingRun,
    TokenPPOSample,
    materialize_token_ppo_training_run,
)
from worldfoundry.training.post_training.rl.algorithms.token_ppo.contracts import (
    TokenPPOTerminalRewardAdapter,
)
from worldfoundry.training.post_training.shared.building import resolve_tensor_dtype
from worldfoundry.training.recipes.post_training.algorithms.token_policy import (
    TokenPolicyAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.algorithms.token_ppo import (
    TokenPPOAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.recipes.post_training.rollout import (
    LocalRolloutSpec,
    RayRolloutSpec,
)

from .codec import (
    QWEN3_CALCULATOR_TOOL_SCHEMA,
    Qwen3ChatCodec,
)
from .models import (
    Qwen3ActorCritic,
    Qwen3TokenPPOAdapter,
    Qwen3TokenPPORewardAdapter,
    normalized_qwen3_answer,
)

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _arithmetic(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _arithmetic(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        return _BINARY_OPERATORS[type(node.op)](
            _arithmetic(node.left),
            _arithmetic(node.right),
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_arithmetic(node.operand))
    raise ValueError("calculator supports numeric arithmetic expressions")


def _calculator(arguments: Mapping[str, object]) -> int | float:
    expression = str(arguments.get("expression", "")).strip()
    if not expression:
        raise ValueError("calculator requires expression")
    return _arithmetic(ast.parse(expression, mode="eval"))


def _terminal_answer(sample: AgenticSampleTrajectory) -> str:
    return sample.turns[-1].assistant.message.content


def _correctness(sample: AgenticSampleTrajectory) -> float:
    answer = sample.request.conditioning.get("answer")
    if answer is None:
        raise ValueError("agentic correctness reward requires conditioning.answer")
    return float(normalized_qwen3_answer(_terminal_answer(sample)) == normalized_qwen3_answer(answer))


def _tool_success(sample: AgenticSampleTrajectory) -> float:
    results = tuple(result for turn in sample.turns for result in turn.tool_results)
    return float(bool(results) and all(not result.tool_failed for result in results))


def _local_agentic_rewards(reward_ids: Sequence[str]) -> tuple[AgenticRewardComponent, ...]:
    evaluators = {
        "correctness": _correctness,
        "tool-success": _tool_success,
    }
    resolved = tuple(str(reward_id) for reward_id in reward_ids)
    unknown = set(resolved) - set(evaluators)
    if unknown:
        raise ValueError(f"local Qwen3 Agentic rewards do not implement {sorted(unknown)}")
    return tuple(AgenticRewardComponent(reward_id, evaluators[reward_id]) for reward_id in resolved)


def _calculator_tool_executor_factory(*, context: object) -> LocalToolExecutor:
    del context
    return LocalToolExecutor((LocalAgentTool("calculator", _calculator),))


def _checkpoint_source(checkpoint: str, base_dir: Path) -> str:
    candidate = Path(checkpoint).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    local = base_dir / candidate
    return str(local.resolve()) if local.exists() else checkpoint


def _model_options(recipe: PostTrainingRecipe) -> dict[str, object]:
    options = dict(recipe.model.options)
    unknown = set(options) - {
        "attention_implementation",
        "enable_thinking",
        "max_new_tokens",
    }
    if unknown:
        raise ValueError(f"unknown Qwen3 model options: {sorted(unknown)}")
    return options


def _generation_limit(
    recipe: PostTrainingRecipe,
    options: Mapping[str, object],
) -> int:
    if "max_new_tokens" in recipe.data.options:
        raise ValueError("Qwen3 max_new_tokens belongs in model.options")
    limit = options.get("max_new_tokens")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("Qwen3 model.options.max_new_tokens must be a positive integer")
    horizon = getattr(recipe.algorithm, "horizon", None)
    if horizon is not None and int(horizon) != limit:
        raise ValueError("Qwen3 algorithm.horizon must equal model.options.max_new_tokens")
    return limit


def _ray_worker_device_type(recipe: PostTrainingRecipe) -> str:
    rollout = recipe.rollout
    if not isinstance(rollout, RayRolloutSpec):
        raise TypeError("Ray Qwen3 device resolution requires a Ray rollout recipe")
    accelerator = rollout.pool.accelerator_resource.upper()
    if accelerator == "GPU":
        return "cuda"
    if accelerator == "CPU":
        return "cpu"
    raise ValueError("native Qwen3 Ray roles support CPU or GPU workers")


def _load_qwen3_tokenizer(source: str) -> object:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_qwen3_policy(
    source: str,
    *,
    attention_implementation: str | None,
) -> nn.Module:
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, object] = {"dtype": torch.float32}
    if attention_implementation is not None:
        kwargs["attn_implementation"] = attention_implementation
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError("Transformers did not return a Qwen3 nn.Module")
    return model


def _enable_training_features(model: nn.Module, recipe: PostTrainingRecipe) -> None:
    if recipe.runtime.activation_checkpoint == "full":
        enable = getattr(model, "gradient_checkpointing_enable", None)
        if not callable(enable):
            raise TypeError("Qwen3 model does not expose gradient_checkpointing_enable")
        enable()
    elif recipe.runtime.activation_checkpoint != "none":
        raise ValueError("Qwen3 activation_checkpoint must be none or full")
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False


def _qwen3_roles(
    recipe: PostTrainingRecipe,
    *,
    base_dir: Path,
    policy_module: nn.Module | None,
    tokenizer: object | None,
) -> tuple[nn.Module, object, dict[str, object]]:
    if not recipe.model.recipe.startswith("qwen3-"):
        raise ValueError("Qwen3 materialization requires a qwen3 model recipe")
    options = _model_options(recipe)
    source = _checkpoint_source(recipe.model.checkpoint, base_dir)
    resolved_tokenizer = tokenizer or _load_qwen3_tokenizer(source)
    policy = policy_module or _load_qwen3_policy(
        source,
        attention_implementation=(
            None if options.get("attention_implementation") is None else str(options["attention_implementation"])
        ),
    )
    policy.float()
    _enable_training_features(policy, recipe)
    return policy, resolved_tokenizer, options


@dataclass(frozen=True, slots=True)
class _Qwen3RayPolicyFactory:
    """Materialize an independent inference role inside each Ray worker."""

    checkpoint_source: str
    attention_implementation: str | None
    enable_thinking: bool
    max_new_tokens: int
    compute_dtype: torch.dtype
    device_type: str
    policy_template: nn.Module | None = None
    tokenizer_template: object | None = None

    def __call__(self, *, context: object):
        del context
        tokenizer = (
            _load_qwen3_tokenizer(self.checkpoint_source)
            if self.tokenizer_template is None
            else copy.deepcopy(self.tokenizer_template)
        )
        policy = (
            _load_qwen3_policy(
                self.checkpoint_source,
                attention_implementation=self.attention_implementation,
            )
            if self.policy_template is None
            else copy.deepcopy(self.policy_template)
        )
        device = torch.device(self.device_type)
        policy.float().to(device)
        config = getattr(policy, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False
        return CausalLMAgenticPolicyAdapter(
            policy,
            Qwen3ChatCodec(tokenizer, enable_thinking=self.enable_thinking),
            generation=CausalLMGenerationConfig(max_new_tokens=self.max_new_tokens),
            compute_dtype=self.compute_dtype,
        )


@dataclass(frozen=True, slots=True)
class _Qwen3PolicyModuleFactory:
    checkpoint_source: str
    attention_implementation: str | None

    def __call__(self, *, context: RayWorkerContext) -> nn.Module:
        del context
        return _load_qwen3_policy(
            self.checkpoint_source,
            attention_implementation=self.attention_implementation,
        )


@dataclass(frozen=True, slots=True)
class _Qwen3TokenizerFactory:
    checkpoint_source: str

    def __call__(self, *, context: RayWorkerContext) -> object:
        del context
        return _load_qwen3_tokenizer(self.checkpoint_source)


class _Qwen3ActorTrainerRole:
    """One replicated trainer hosted in the trainer slot of a Ray DevicePool."""

    def __init__(
        self,
        *,
        context: RayWorkerContext,
        recipe_payload: Mapping[str, object],
        base_dir: str,
        output_dir: str | None,
        resume_checkpoint: str | None,
        device: str | None,
        initialization_seed: int | None,
        prompts: Sequence[AgenticPrompt] | None,
        reward_url: str | None,
        policy_factory: Callable[..., nn.Module],
        tokenizer_factory: Callable[..., object],
        policy_factory_kwargs: Mapping[str, object] | None,
        tokenizer_factory_kwargs: Mapping[str, object] | None,
        fused_adamw: bool | Literal["auto"],
    ) -> None:
        if context.world_size != 1:
            raise ValueError("actor-hosted Qwen3 currently supports exactly one replicated trainer")
        recipe = PostTrainingRecipe.from_mapping(recipe_payload)
        if not isinstance(recipe.rollout, RayRolloutSpec) or recipe.rollout.trainer_binding != "actor":
            raise ValueError("actor trainer requires a Ray actor-bound rollout recipe")
        self.context = context
        self.recipe = recipe
        self.base_dir = Path(base_dir)
        self.output_dir = output_dir
        self.resume_checkpoint = resume_checkpoint
        self.device = device or _ray_worker_device_type(recipe)
        self.initialization_seed = initialization_seed
        self.prompts = None if prompts is None else tuple(prompts)
        self.reward_url = reward_url
        self.fused_adamw = fused_adamw
        self.policy = policy_factory(context=context, **dict(policy_factory_kwargs or {}))
        if not isinstance(self.policy, nn.Module):
            raise TypeError("Ray trainer policy factory must return nn.Module")
        self.tokenizer = tokenizer_factory(context=context, **dict(tokenizer_factory_kwargs or {}))
        self.policy.float()
        _enable_training_features(self.policy, recipe)
        self.training_run: NativeAgenticTrainingRun | None = None
        self.rollout_lease: DeviceLease | None = None

    def attach_rollout(self, lease: DeviceLease, actors: tuple[object, ...]) -> None:
        if self.training_run is not None:
            raise RuntimeError("actor trainer rollout is already attached")
        self.rollout_lease = lease
        runtime = ActorTrainerRolloutRuntime(
            actors,
            lease,
            weight_bucket_bytes=self.recipe.rollout.weight_bucket_bytes,
        )
        rollout = RayAgenticRolloutAdapter(
            runtime,  # type: ignore[arg-type]
            self.policy,
            weight_kind=self.recipe.rollout.weight_kind,
        )
        options = _model_options(self.recipe)
        generation = CausalLMGenerationConfig(
            max_new_tokens=_generation_limit(self.recipe, options),
        )
        reward_adapter = None
        closeables: tuple[object, ...] = ()
        if self.reward_url is not None:
            evaluator = HTTPRewardEvaluator(self.reward_url)
            reward_adapter = HTTPAgenticRewardAdapter(
                evaluator,
                reward_ids=tuple(self.recipe.algorithm.reward_weights),
            )
            closeables = (evaluator,)
        reward_components = (
            () if reward_adapter is not None else _local_agentic_rewards(tuple(self.recipe.algorithm.reward_weights))
        )
        self.training_run = materialize_agentic_training_run(
            self.recipe,
            policy_module=self.policy,
            codec=Qwen3ChatCodec(
                self.tokenizer,
                enable_thinking=bool(options.get("enable_thinking", False)),
            ),
            rollout_adapter=rollout,
            reward_adapter=reward_adapter,
            reward_components=reward_components,
            closeables=closeables,
            prompts=self.prompts,
            generation=generation,
            base_dir=self.base_dir,
            output_dir=self.output_dir,
            resume_checkpoint=self.resume_checkpoint,
            device=self.device,
            initialization_seed=self.initialization_seed,
            fused_adamw=self.fused_adamw,
        )

    def _run(self) -> NativeAgenticTrainingRun:
        if self.training_run is None:
            raise RuntimeError("actor trainer has not attached its rollout workers")
        return self.training_run

    def run_iterations(self, max_iterations: int):
        return self._run().run(max_iterations=max_iterations)

    def export_policy(self):
        return self._run().export_policy()

    def policy_state(self) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().to(device="cpu", copy=True) for name, tensor in self.policy.state_dict().items()}

    def rollout_state(self) -> dict[str, object]:
        state = self._run().rollout_adapter.state_dict()
        return dict(state)

    def placement(self) -> dict[str, object]:
        if self.rollout_lease is None:
            raise RuntimeError("actor trainer has not attached its rollout workers")
        return {
            "trainer_device": self.context.device_id,
            "trainer_slot": self.context.slot,
            "rollout_devices": self.rollout_lease.device_ids,
            "rollout_slot": self.rollout_lease.slot,
            "rollout_placement": self.rollout_lease.placement.value,
        }

    def close(self) -> None:
        if self.training_run is not None:
            self.training_run.close()


class Qwen3ActorHostedTrainingRun:
    """Controller proxy for a complete Agentic lifecycle inside one Ray actor."""

    world_size = 1
    is_coordinator = True
    artifact_role = "policy"

    def __init__(
        self,
        runtime: RayPostTrainingRuntime,
        *,
        output_dir: str | Path,
    ) -> None:
        if runtime.trainer_group is None:
            raise RuntimeError("actor-hosted Qwen3 runtime has no trainer group")
        self.runtime = runtime
        self.output_dir = Path(output_dir).expanduser().resolve()
        self._closed = False

    @property
    def trainer_group(self):
        group = self.runtime.trainer_group
        if group is None:
            raise RuntimeError("actor-hosted Qwen3 runtime is closed")
        return group

    def run(self, *, max_iterations: int):
        return self.trainer_group.broadcast("run_iterations", max_iterations)[0]

    def export_policy(self):
        return self.trainer_group.broadcast("export_policy")[0]

    def policy_state(self) -> dict[str, torch.Tensor]:
        return self.trainer_group.broadcast("policy_state")[0]

    def rollout_state(self) -> dict[str, object]:
        return self.trainer_group.broadcast("rollout_state")[0]

    def placement(self) -> dict[str, object]:
        return self.trainer_group.broadcast("placement")[0]

    def close(self) -> None:
        if not self._closed:
            self.runtime.shutdown()
            self._closed = True


class Qwen3PostTrainingRun:
    """Uniform CLI surface over the native Agentic and actor-critic runs."""

    world_size = 1
    is_coordinator = True

    def __init__(
        self,
        run: NativeAgenticTrainingRun | NativeTokenPPOTrainingRun | Qwen3ActorHostedTrainingRun,
    ) -> None:
        self.native_run = run
        self.output_dir = run.output_dir
        self.artifact_role = "actor-critic" if isinstance(run, NativeTokenPPOTrainingRun) else "policy"

    def __getattr__(self, name: str) -> object:
        return getattr(self.native_run, name)

    def run(self, *, max_iterations: int):
        return self.native_run.run(max_iterations=max_iterations)

    def export_policy(self):
        if not isinstance(self.native_run, NativeTokenPPOTrainingRun):
            return self.native_run.export_policy()
        return self.native_run.export_actor_critic()

    def close(self) -> None:
        close = getattr(self.native_run, "close", None)
        if callable(close):
            close()


def materialize_qwen3_agentic_training_run(
    recipe: PostTrainingRecipe,
    *,
    base_dir: str | Path = ".",
    output_dir: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    device: str | torch.device | None = None,
    initialization_seed: int | None = None,
    reward_url: str | None = None,
    policy_module: nn.Module | None = None,
    tokenizer: object | None = None,
    tool_executor: AgentToolExecutor | None = None,
    reward_adapter: TokenTrajectoryRewardAdapter | None = None,
    prompts: Sequence[AgenticPrompt] | None = None,
    ray_rollout_policy_factory: Callable[..., object] | None = None,
    ray_rollout_policy_factory_kwargs: Mapping[str, object] | None = None,
    ray_tool_executor_factory: Callable[..., AgentToolExecutor] | None = None,
    ray_tool_executor_factory_kwargs: Mapping[str, object] | None = None,
    ray_trainer_policy_factory: Callable[..., nn.Module] | None = None,
    ray_trainer_policy_factory_kwargs: Mapping[str, object] | None = None,
    ray_tokenizer_factory: Callable[..., object] | None = None,
    ray_tokenizer_factory_kwargs: Mapping[str, object] | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> Qwen3PostTrainingRun:
    """Materialize Qwen3 multi-turn Hermes rollouts into a grouped token learner."""

    if not isinstance(recipe.algorithm, TokenPolicyAlgorithmSpec):
        raise TypeError("Qwen3 Agentic training requires a grouped token-policy algorithm")
    generation_limit = _generation_limit(recipe, _model_options(recipe))
    root = Path(base_dir).expanduser().resolve()
    if isinstance(recipe.rollout, RayRolloutSpec) and recipe.rollout.trainer_binding == "actor":
        if recipe.rollout.trainer_devices != 1:
            raise ValueError("actor-hosted Qwen3 currently supports trainer_devices=1")
        if recipe.distributed.backend != "single":
            raise ValueError("actor-hosted Qwen3 does not support multi-rank sharded trainers")
        if recipe.tuning.mode != "full":
            raise ValueError("actor-hosted Qwen3 currently supports full tuning")
        if (
            policy_module is not None
            or tokenizer is not None
            or tool_executor is not None
            or reward_adapter is not None
        ):
            raise ValueError("actor-hosted Qwen3 roles must be supplied by Ray worker factories")
        source = _checkpoint_source(recipe.model.checkpoint, root)
        options = _model_options(recipe)
        attention = (
            None if options.get("attention_implementation") is None else str(options["attention_implementation"])
        )
        trainer_policy_factory = ray_trainer_policy_factory or _Qwen3PolicyModuleFactory(
            source,
            attention,
        )
        tokenizer_factory = ray_tokenizer_factory or _Qwen3TokenizerFactory(source)
        rollout_policy_factory = ray_rollout_policy_factory or _Qwen3RayPolicyFactory(
            checkpoint_source=source,
            attention_implementation=attention,
            enable_thinking=bool(options.get("enable_thinking", False)),
            max_new_tokens=generation_limit,
            compute_dtype=resolve_tensor_dtype(recipe.runtime.param_dtype),
            device_type=_ray_worker_device_type(recipe),
        )
        destination = Path(output_dir or recipe.run.output_dir).expanduser()
        if not destination.is_absolute():
            destination = root / destination
        destination = destination.resolve()
        runtime = RayPostTrainingRuntime(ray_runtime_config_from_rollout_spec(recipe.rollout))
        try:
            runtime.setup(
                RayAgenticRolloutWorker,
                rollout_factory_kwargs={
                    "policy_factory": rollout_policy_factory,
                    "policy_factory_kwargs": dict(ray_rollout_policy_factory_kwargs or {}),
                    "tool_executor_factory": ray_tool_executor_factory or _calculator_tool_executor_factory,
                    "tool_executor_factory_kwargs": dict(ray_tool_executor_factory_kwargs or {}),
                },
                trainer_factory=_Qwen3ActorTrainerRole,
                trainer_factory_kwargs={
                    "recipe_payload": recipe.to_dict(),
                    "base_dir": str(root),
                    "output_dir": str(destination),
                    "resume_checkpoint": (None if resume_checkpoint is None else str(resume_checkpoint)),
                    "device": None if device is None else str(device),
                    "initialization_seed": initialization_seed,
                    "prompts": None if prompts is None else tuple(prompts),
                    "reward_url": reward_url,
                    "policy_factory": trainer_policy_factory,
                    "tokenizer_factory": tokenizer_factory,
                    "policy_factory_kwargs": dict(ray_trainer_policy_factory_kwargs or {}),
                    "tokenizer_factory_kwargs": dict(ray_tokenizer_factory_kwargs or {}),
                    "fused_adamw": fused_adamw,
                },
            )
            assert runtime.trainer_group is not None
            assert runtime.rollout_group is not None
            runtime.trainer_group.broadcast(
                "attach_rollout",
                runtime.rollout_group.lease,
                runtime.rollout_group.actors,
            )
        except Exception:
            runtime.shutdown()
            raise
        return Qwen3PostTrainingRun(Qwen3ActorHostedTrainingRun(runtime, output_dir=destination))
    policy, resolved_tokenizer, options = _qwen3_roles(
        recipe,
        base_dir=root,
        policy_module=policy_module,
        tokenizer=tokenizer,
    )
    codec = Qwen3ChatCodec(
        resolved_tokenizer,
        enable_thinking=bool(options.get("enable_thinking", False)),
    )
    closeables: list[object] = []
    resolved_reward = reward_adapter
    if resolved_reward is None and reward_url is not None:
        evaluator = HTTPRewardEvaluator(reward_url)
        resolved_reward = HTTPAgenticRewardAdapter(
            evaluator,
            reward_ids=tuple(recipe.algorithm.reward_weights),
        )
        closeables.append(evaluator)
    reward_components = ()
    if resolved_reward is None:
        reward_components = _local_agentic_rewards(tuple(recipe.algorithm.reward_weights))
    generation = CausalLMGenerationConfig(
        max_new_tokens=generation_limit,
    )
    resolved_rollout = None
    resolved_tools = tool_executor
    if isinstance(recipe.rollout, LocalRolloutSpec):
        if any(
            factory is not None
            for factory in (
                ray_rollout_policy_factory,
                ray_tool_executor_factory,
                ray_trainer_policy_factory,
                ray_tokenizer_factory,
            )
        ):
            raise ValueError("Ray rollout factories require rollout.backend=ray")
        resolved_tools = resolved_tools or LocalToolExecutor(
            (LocalAgentTool("calculator", _calculator),),
        )
    elif isinstance(recipe.rollout, RayRolloutSpec):
        if recipe.rollout.trainer_binding != TrainerBinding.EXTERNAL:
            raise ValueError("actor-hosted Qwen3 Agentic training must use the actor lifecycle materializer")
        if tool_executor is not None:
            raise ValueError("Ray Agentic rollout requires a worker-side tool executor factory")
        runtime = RayPostTrainingRuntime(ray_runtime_config_from_rollout_spec(recipe.rollout))
        policy_factory = ray_rollout_policy_factory
        if policy_factory is None:
            source = _checkpoint_source(recipe.model.checkpoint, root)
            injected_roles = policy_module is not None or tokenizer is not None
            policy_factory = _Qwen3RayPolicyFactory(
                checkpoint_source=source,
                attention_implementation=(
                    None
                    if options.get("attention_implementation") is None
                    else str(options["attention_implementation"])
                ),
                enable_thinking=bool(options.get("enable_thinking", False)),
                max_new_tokens=generation_limit,
                compute_dtype=resolve_tensor_dtype(recipe.runtime.param_dtype),
                device_type=_ray_worker_device_type(recipe),
                policy_template=(policy if injected_roles else None),
                tokenizer_template=(resolved_tokenizer if injected_roles else None),
            )
        try:
            resolved_rollout = setup_ray_agentic_rollout(
                runtime,
                policy,
                rollout_policy_factory=policy_factory,
                rollout_policy_factory_kwargs=ray_rollout_policy_factory_kwargs,
                tool_executor_factory=ray_tool_executor_factory or _calculator_tool_executor_factory,
                tool_executor_factory_kwargs=ray_tool_executor_factory_kwargs,
                weight_kind=recipe.rollout.weight_kind,
            )
        except Exception:
            for resource in reversed(closeables):
                resource.close()
            raise
        closeables.append(runtime)
    else:
        raise TypeError("unsupported rollout recipe")
    try:
        run = materialize_agentic_training_run(
            recipe,
            policy_module=policy,
            codec=codec,
            tool_executor=resolved_tools,
            reward_components=reward_components,
            reward_adapter=resolved_reward,
            rollout_adapter=resolved_rollout,
            closeables=closeables,
            prompts=prompts,
            generation=generation,
            base_dir=root,
            output_dir=output_dir,
            resume_checkpoint=resume_checkpoint,
            device=device,
            initialization_seed=initialization_seed,
            fused_adamw=fused_adamw,
        )
    except Exception:
        for resource in reversed(closeables):
            close = getattr(resource, "close", None)
            if callable(close):
                close()
            else:
                resource.shutdown()
        raise
    return Qwen3PostTrainingRun(run)


def _ppo_samples(
    recipe: PostTrainingRecipe,
    *,
    base_dir: Path,
) -> tuple[TokenPPOSample, ...]:
    manifest = Path(recipe.data.manifest).expanduser()
    if not manifest.is_absolute():
        manifest = base_dir / manifest
    prompts = load_agentic_prompts(manifest, split=recipe.data.split)
    return tuple(
        TokenPPOSample(
            sample_id=prompt.prompt_id,
            conditioning={
                "messages": prompt.messages,
                **dict(prompt.conditioning),
            },
        )
        for prompt in prompts
    )


def materialize_qwen3_token_ppo_training_run(
    recipe: PostTrainingRecipe,
    *,
    base_dir: str | Path = ".",
    output_dir: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    device: str | torch.device | None = None,
    initialization_seed: int | None = None,
    policy_module: nn.Module | None = None,
    actor_critic: Qwen3ActorCritic | None = None,
    tokenizer: object | None = None,
    reward_adapter: TokenPPOTerminalRewardAdapter | None = None,
    samples: Sequence[TokenPPOSample] | None = None,
    fused_adamw: bool | Literal["auto"] = "auto",
) -> Qwen3PostTrainingRun:
    """Materialize Qwen3 policy plus value head into native token PPO."""

    if not isinstance(recipe.algorithm, TokenPPOAlgorithmSpec):
        raise TypeError("Qwen3 actor-critic training requires token-ppo")
    generation_limit = _generation_limit(recipe, _model_options(recipe))
    root = Path(base_dir).expanduser().resolve()
    if actor_critic is None:
        policy, resolved_tokenizer, options = _qwen3_roles(
            recipe,
            base_dir=root,
            policy_module=policy_module,
            tokenizer=tokenizer,
        )
        actor_critic = Qwen3ActorCritic(policy)
    else:
        if policy_module is not None:
            raise ValueError("actor_critic and policy_module cannot both be provided")
        source = _checkpoint_source(recipe.model.checkpoint, root)
        resolved_tokenizer = tokenizer or _load_qwen3_tokenizer(source)
        options = _model_options(recipe)
        actor_critic.float()
        _enable_training_features(actor_critic.policy, recipe)
    codec = Qwen3ChatCodec(
        resolved_tokenizer,
        tool_schemas=(),
        enable_thinking=bool(options.get("enable_thinking", True)),
    )
    adapter = Qwen3TokenPPOAdapter(
        actor_critic,
        codec,
        max_new_tokens=generation_limit,
        compute_dtype=resolve_tensor_dtype(recipe.runtime.param_dtype),
    )
    resolved_reward = reward_adapter or Qwen3TokenPPORewardAdapter(
        resolved_tokenizer,
        reward_ids=tuple(recipe.algorithm.reward_weights),
    )
    resolved_samples = tuple(samples) if samples is not None else _ppo_samples(recipe, base_dir=root)
    run = materialize_token_ppo_training_run(
        recipe,
        rollout_adapter=adapter,
        replay_adapter=adapter,
        reward_adapter=resolved_reward,
        samples=resolved_samples,
        base_dir=root,
        output_dir=output_dir,
        resume_checkpoint=resume_checkpoint,
        device=device,
        initialization_seed=initialization_seed,
        fused_adamw=fused_adamw,
    )
    return Qwen3PostTrainingRun(run)


def materialize_qwen3_post_training_run(
    recipe: PostTrainingRecipe,
    *,
    reward_url: str | None = None,
    **kwargs: object,
) -> Qwen3PostTrainingRun:
    """Dispatch one strict Qwen3 recipe to its native learner."""

    if isinstance(recipe.algorithm, TokenPolicyAlgorithmSpec):
        return materialize_qwen3_agentic_training_run(
            recipe,
            reward_url=reward_url,
            **kwargs,
        )
    if isinstance(recipe.algorithm, TokenPPOAlgorithmSpec):
        if reward_url is not None:
            raise ValueError("reward_url is only supported by Qwen3 grouped Agentic token policy")
        return materialize_qwen3_token_ppo_training_run(recipe, **kwargs)
    raise TypeError("Qwen3 post-training supports grouped token policy and token-ppo")


__all__ = [
    "QWEN3_CALCULATOR_TOOL_SCHEMA",
    "Qwen3PostTrainingRun",
    "Qwen3ActorHostedTrainingRun",
    "materialize_qwen3_agentic_training_run",
    "materialize_qwen3_post_training_run",
    "materialize_qwen3_token_ppo_training_run",
]
