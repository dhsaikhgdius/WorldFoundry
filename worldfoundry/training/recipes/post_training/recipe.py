"""Root parser and serialization for native post-training recipes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry.core.io.integrity import canonical_json, canonical_sha256

from ..spec import (
    NATIVE_EXECUTION_OWNER,
    DatasetSpec,
    DistributedSpec,
    ExportSpec,
    ModelSpec,
    OptimizerSpec,
    PostTrainingCheckpointSpec,
    RunSpec,
    TrainingRuntimeSpec,
    TuningSpec,
)
from .algorithms import (
    AdaptiveVideoAlgorithmSpec,
    AdversarialDiffusionAlgorithmSpec,
    AnyFlowBidirectionalOnPolicyAlgorithmSpec,
    AnyFlowBidirectionalPretrainAlgorithmSpec,
    AnyFlowFAROnPolicyAlgorithmSpec,
    AnyFlowFARPretrainAlgorithmSpec,
    BagelFlowUniGRPOAlgorithmSpec,
    CausalRCMAlgorithmSpec,
    DanceGRPOAlgorithmSpec,
    DDRLAlgorithmSpec,
    DFDAlgorithmSpec,
    DiagonalAlgorithmSpec,
    DiffusionNFTAlgorithmSpec,
    DMD2AlgorithmSpec,
    DMDAlgorithmSpec,
    FlowDPPOAlgorithmSpec,
    FlowGRPOAlgorithmSpec,
    FlowPolicyAlgorithmSpec,
    FlowSDEWindowSpec,
    GRPOGuardAlgorithmSpec,
    LatentConsistencyAlgorithmSpec,
    MixGRPOAlgorithmSpec,
    PostTrainingAlgorithmSpec,
    ProgressiveDistillationAlgorithmSpec,
    RCMAlgorithmSpec,
    RewardForcingAlgorithmSpec,
    ScaleWiseAlgorithmSpec,
    SCMLADDAlgorithmSpec,
    SelfForcingAlgorithmSpec,
    SelfGradientForcingAlgorithmSpec,
    SenseFlowAlgorithmSpec,
    SGMDAlgorithmSpec,
    SIDAlgorithmSpec,
    TokenCPPOAlgorithmSpec,
    TokenDPPOAlgorithmSpec,
    TokenDRPOAlgorithmSpec,
    TokenGRPOAlgorithmSpec,
    TokenGSPOAlgorithmSpec,
    TokenPolicyAlgorithmSpec,
)
from .algorithms.adaptive_video import parse_adaptive_video_algorithm
from .algorithms.adversarial_diffusion import parse_adversarial_diffusion_algorithm
from .algorithms.anyflow import (
    parse_anyflow_bidirectional_on_policy_algorithm,
    parse_anyflow_bidirectional_pretrain_algorithm,
    parse_anyflow_far_on_policy_algorithm,
    parse_anyflow_far_pretrain_algorithm,
)
from .algorithms.bagel_flow_unigrpo import parse_bagel_flow_unigrpo_algorithm
from .algorithms.causal_consistency import parse_causal_consistency_algorithm
from .algorithms.causal_ode import parse_causal_ode_algorithm
from .algorithms.dance_grpo import parse_dance_grpo_algorithm
from .algorithms.ddrl import parse_ddrl_algorithm
from .algorithms.dfd import parse_dfd_algorithm
from .algorithms.diagonal import parse_diagonal_algorithm
from .algorithms.diffusion_dpo import parse_diffusion_dpo_algorithm
from .algorithms.diffusion_nft import parse_diffusion_nft_algorithm
from .algorithms.dmd import parse_dmd_algorithm
from .algorithms.dmd2 import parse_dmd2_algorithm
from .algorithms.flow_dppo import parse_flow_dppo_algorithm
from .algorithms.flow_grpo import parse_flow_grpo_algorithm
from .algorithms.grpo_guard import parse_grpo_guard_algorithm
from .algorithms.latent_consistency import parse_latent_consistency_algorithm
from .algorithms.mix_grpo import parse_mix_grpo_algorithm
from .algorithms.progressive import parse_progressive_distillation_algorithm
from .algorithms.rcm import parse_causal_rcm_algorithm, parse_rcm_algorithm
from .algorithms.reward_forcing import parse_reward_forcing_algorithm
from .algorithms.scale_wise import parse_scale_wise_algorithm
from .algorithms.scm_ladd import parse_scm_ladd_algorithm
from .algorithms.self_forcing import parse_self_forcing_algorithm
from .algorithms.self_gradient_forcing import (
    parse_self_gradient_forcing_algorithm,
)
from .algorithms.senseflow import parse_senseflow_algorithm
from .algorithms.sgmd import parse_sgmd_algorithm
from .algorithms.sid import parse_sid_algorithm
from .algorithms.token_policy import parse_token_policy_algorithm
from .common import mapping, plain_data, strict_mapping
from .rewards import (
    VIDEOALIGN_BASE_MODEL_REPOSITORY,
    VIDEOALIGN_BASE_MODEL_REVISION,
    VIDEOALIGN_CALIBRATION_MEAN,
    VIDEOALIGN_CALIBRATION_STD,
    VIDEOALIGN_CHECKPOINT_FILE,
    VIDEOALIGN_CHECKPOINT_REPOSITORY,
    VIDEOALIGN_CHECKPOINT_REVISION,
    VIDEOALIGN_CHECKPOINT_SHA256,
    VIDEOALIGN_CHECKPOINT_SIZE_BYTES,
    VIDEOALIGN_REWARD_IDS,
    VideoAlignRewardSpec,
)

POST_TRAINING_RECIPE_SCHEMA = "worldfoundry-post-training"

_ROOT_FIELDS = {
    "schema",
    "run",
    "model",
    "tuning",
    "data",
    "algorithm",
    "optimizer",
    "fake_score_optimizer",
    "guidance_optimizer",
    "discriminator_optimizer",
    "execution_owner",
    "runtime",
    "distributed",
    "checkpoint",
    "export",
}
_REQUIRED_ROOT_FIELDS = {"run", "model", "tuning", "data", "algorithm", "optimizer"}
_OPTIMIZER_FIELDS = {
    "type",
    "learning_rate",
    "weight_decay",
    "betas",
    "epsilon",
    "update_clip_threshold",
    "max_grad_norm",
    "gradient_accumulation_steps",
}


@dataclass(frozen=True, slots=True)
class PostTrainingRecipe:
    """A native-only run request; it has no external-provider execution field."""

    run: RunSpec
    model: ModelSpec
    tuning: TuningSpec
    data: DatasetSpec
    algorithm: PostTrainingAlgorithmSpec
    optimizer: OptimizerSpec
    fake_score_optimizer: OptimizerSpec | None = None
    guidance_optimizer: OptimizerSpec | None = None
    discriminator_optimizer: OptimizerSpec | None = None
    execution_owner: str = NATIVE_EXECUTION_OWNER
    runtime: TrainingRuntimeSpec = TrainingRuntimeSpec()
    distributed: DistributedSpec = DistributedSpec()
    checkpoint: PostTrainingCheckpointSpec = PostTrainingCheckpointSpec()
    export: ExportSpec = ExportSpec()
    schema: str = POST_TRAINING_RECIPE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != POST_TRAINING_RECIPE_SCHEMA:
            raise ValueError(f"unsupported post-training recipe schema: {self.schema!r}")
        owner = str(self.execution_owner).strip().lower().replace("_", "-")
        if owner != NATIVE_EXECUTION_OWNER:
            raise ValueError(
                f"execution_owner must be {NATIVE_EXECUTION_OWNER!r}; external training loops are unsupported"
            )
        if isinstance(
            self.algorithm,
            (
                AnyFlowFAROnPolicyAlgorithmSpec,
                AnyFlowBidirectionalOnPolicyAlgorithmSpec,
            ),
        ):
            if self.fake_score_optimizer is None:
                raise ValueError("AnyFlow on-policy training requires fake_score_optimizer")
            if self.guidance_optimizer is not None or self.discriminator_optimizer is not None:
                raise ValueError("AnyFlow on-policy training only accepts fake_score_optimizer")
        elif isinstance(
            self.algorithm,
            (
                AnyFlowFARPretrainAlgorithmSpec,
                AnyFlowBidirectionalPretrainAlgorithmSpec,
            ),
        ):
            if any(
                value is not None
                for value in (
                    self.fake_score_optimizer,
                    self.guidance_optimizer,
                    self.discriminator_optimizer,
                )
            ):
                raise ValueError("AnyFlow pretraining accepts only the primary optimizer")
        elif isinstance(self.algorithm, AdversarialDiffusionAlgorithmSpec):
            if self.discriminator_optimizer is None:
                raise ValueError("adversarial diffusion distillation requires discriminator_optimizer")
            if self.fake_score_optimizer is not None or self.guidance_optimizer is not None:
                raise ValueError(
                    "adversarial diffusion distillation only accepts discriminator_optimizer"
                )
        elif isinstance(self.algorithm, RewardForcingAlgorithmSpec):
            if self.fake_score_optimizer is None:
                raise ValueError("Reward-Forcing requires fake_score_optimizer")
            if self.guidance_optimizer is not None or self.discriminator_optimizer is not None:
                raise ValueError("Reward-Forcing only accepts fake_score_optimizer")
        elif isinstance(self.algorithm, SenseFlowAlgorithmSpec):
            if self.fake_score_optimizer is None:
                raise ValueError("SenseFlow requires fake_score_optimizer")
            if self.discriminator_optimizer is None:
                raise ValueError("SenseFlow requires discriminator_optimizer")
            if self.guidance_optimizer is not None:
                raise ValueError("SenseFlow does not accept guidance_optimizer")
        elif isinstance(self.algorithm, (AdaptiveVideoAlgorithmSpec, DMDAlgorithmSpec)):
            if self.fake_score_optimizer is None:
                algorithm_name = (
                    "DMD" if isinstance(self.algorithm, DMDAlgorithmSpec) else "adaptive video distillation"
                )
                raise ValueError(f"{algorithm_name} requires fake_score_optimizer")
            if self.guidance_optimizer is not None or self.discriminator_optimizer is not None:
                raise ValueError(f"{self.algorithm.type} only accepts fake_score_optimizer")
        elif isinstance(self.algorithm, (CausalRCMAlgorithmSpec, RCMAlgorithmSpec)):
            dmd_enabled = self.algorithm.dmd_loss_scale > 0
            if dmd_enabled and self.fake_score_optimizer is None:
                raise ValueError("rCM DMD requires fake_score_optimizer")
            if not dmd_enabled and self.fake_score_optimizer is not None:
                raise ValueError("rCM without DMD cannot configure fake_score_optimizer")
            if self.guidance_optimizer is not None or self.discriminator_optimizer is not None:
                raise ValueError("rCM only accepts fake_score_optimizer when DMD is enabled")
        elif isinstance(self.algorithm, DMD2AlgorithmSpec):
            if self.guidance_optimizer is None:
                raise ValueError("DMD2 requires guidance_optimizer")
            if self.fake_score_optimizer is not None or self.discriminator_optimizer is not None:
                raise ValueError("DMD2 only accepts guidance_optimizer")
        elif isinstance(self.algorithm, DFDAlgorithmSpec):
            if self.fake_score_optimizer is None:
                raise ValueError("DFD requires fake_score_optimizer")
            if self.algorithm.adversarial_enabled:
                if self.discriminator_optimizer is None:
                    raise ValueError("DFD GAN loss requires discriminator_optimizer")
            elif self.discriminator_optimizer is not None:
                raise ValueError("DFD without GAN loss cannot configure discriminator_optimizer")
            if self.guidance_optimizer is not None:
                raise ValueError("DFD does not accept guidance_optimizer")
        elif isinstance(self.algorithm, DiagonalAlgorithmSpec):
            if self.fake_score_optimizer is None:
                raise ValueError("diagonal distillation requires fake_score_optimizer")
            if self.guidance_optimizer is not None or self.discriminator_optimizer is not None:
                raise ValueError("diagonal distillation only accepts fake_score_optimizer")
        elif isinstance(self.algorithm, SCMLADDAlgorithmSpec):
            if self.discriminator_optimizer is None:
                raise ValueError("SCM-LADD requires discriminator_optimizer")
            if self.fake_score_optimizer is not None or self.guidance_optimizer is not None:
                raise ValueError("SCM-LADD only accepts discriminator_optimizer")
        elif isinstance(self.algorithm, ScaleWiseAlgorithmSpec):
            if self.algorithm.dmd_enabled:
                if self.fake_score_optimizer is None:
                    raise ValueError("scale-wise DMD requires fake_score_optimizer")
            elif self.fake_score_optimizer is not None:
                raise ValueError("MMD-only scale-wise training cannot configure fake_score_optimizer")
            if self.guidance_optimizer is not None or self.discriminator_optimizer is not None:
                raise ValueError("scale-wise distillation only accepts fake_score_optimizer")
        elif isinstance(
            self.algorithm,
            (SelfForcingAlgorithmSpec, SelfGradientForcingAlgorithmSpec),
        ):
            if self.fake_score_optimizer is None:
                raise ValueError(f"{self.algorithm.type} DMD requires fake_score_optimizer")
            if self.guidance_optimizer is not None or self.discriminator_optimizer is not None:
                raise ValueError(f"{self.algorithm.type} only accepts fake_score_optimizer")
        elif isinstance(self.algorithm, SIDAlgorithmSpec):
            if self.fake_score_optimizer is None:
                raise ValueError("SiD requires fake_score_optimizer")
            if self.guidance_optimizer is not None or self.discriminator_optimizer is not None:
                raise ValueError("SiD only accepts fake_score_optimizer")
        elif isinstance(self.algorithm, SGMDAlgorithmSpec):
            if self.fake_score_optimizer is None:
                raise ValueError("SGMD requires fake_score_optimizer")
            if self.guidance_optimizer is not None or self.discriminator_optimizer is not None:
                raise ValueError("SGMD only accepts fake_score_optimizer")
        elif isinstance(self.algorithm, DDRLAlgorithmSpec):
            if self.fake_score_optimizer is not None:
                raise ValueError("DDRL cannot configure fake_score_optimizer")
            if self.guidance_optimizer is not None:
                raise ValueError("DDRL cannot configure guidance_optimizer")
            if self.discriminator_optimizer is not None:
                raise ValueError("DDRL cannot configure discriminator_optimizer")
        elif self.fake_score_optimizer is not None:
            raise ValueError("this algorithm cannot configure fake_score_optimizer")
        elif self.guidance_optimizer is not None:
            raise ValueError("this algorithm cannot configure guidance_optimizer")
        elif self.discriminator_optimizer is not None:
            raise ValueError("this algorithm cannot configure discriminator_optimizer")
        if self.tuning.mode == "lora":
            if self.export.format != "peft":
                raise ValueError("LoRA post-training export.format must be 'peft'")
            if self.export.options:
                raise ValueError("PEFT post-training export.options must be empty")
        elif self.tuning.mode == "full":
            if self.export.format not in {"safetensors", "distributed-checkpoint"}:
                raise ValueError("full post-training export.format must be safetensors or distributed-checkpoint")
            allowed_export_options = {"max_shard_size_bytes"} if self.export.format == "safetensors" else set()
            unknown_export_options = set(self.export.options) - allowed_export_options
            if unknown_export_options:
                raise ValueError(f"unknown full post-training export options: {sorted(unknown_export_options)}")
            max_shard_size = self.export.options.get("max_shard_size_bytes")
            if max_shard_size is not None and (isinstance(max_shard_size, bool) or int(max_shard_size) <= 0):
                raise ValueError("export.options.max_shard_size_bytes must be positive")
        object.__setattr__(self, "execution_owner", owner)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PostTrainingRecipe:
        root = strict_mapping(
            value,
            field_name="post-training recipe",
            allowed=_ROOT_FIELDS,
        )
        missing = sorted(_REQUIRED_ROOT_FIELDS - set(root))
        if missing:
            raise ValueError(f"post-training recipe is missing required sections: {missing}")

        def section(name: str, allowed: set[str]) -> dict[str, Any]:
            return strict_mapping(root.get(name, {}), field_name=name, allowed=allowed)

        algorithm_payload = mapping(root["algorithm"], field_name="algorithm")
        algorithm_type = str(algorithm_payload.get("type", "")).lower().replace("_", "-")
        algorithm_parsers = {
            "adaptive-video-distillation": parse_adaptive_video_algorithm,
            "adversarial-diffusion-distillation": parse_adversarial_diffusion_algorithm,
            "anyflow-bidirectional-on-policy": parse_anyflow_bidirectional_on_policy_algorithm,
            "anyflow-bidirectional-pretrain": parse_anyflow_bidirectional_pretrain_algorithm,
            "anyflow-far-on-policy": parse_anyflow_far_on_policy_algorithm,
            "anyflow-far-pretrain": parse_anyflow_far_pretrain_algorithm,
            "bagel-flow-unigrpo": parse_bagel_flow_unigrpo_algorithm,
            "causal-rcm": parse_causal_rcm_algorithm,
            "causal-consistency": parse_causal_consistency_algorithm,
            "causal-ode": parse_causal_ode_algorithm,
            "dmd": parse_dmd_algorithm,
            "dmd2": parse_dmd2_algorithm,
            "ddrl": parse_ddrl_algorithm,
            "dfd": parse_dfd_algorithm,
            "diagonal-distillation": parse_diagonal_algorithm,
            "dance-grpo": parse_dance_grpo_algorithm,
            "diffusion-dpo": parse_diffusion_dpo_algorithm,
            "diffusion-nft": parse_diffusion_nft_algorithm,
            "flow-dppo": parse_flow_dppo_algorithm,
            "flow-grpo": parse_flow_grpo_algorithm,
            "grpo-guard": parse_grpo_guard_algorithm,
            "latent-consistency": parse_latent_consistency_algorithm,
            "mix-grpo": parse_mix_grpo_algorithm,
            "progressive-distillation": parse_progressive_distillation_algorithm,
            "rcm": parse_rcm_algorithm,
            "reward-forcing": parse_reward_forcing_algorithm,
            "scm-ladd": parse_scm_ladd_algorithm,
            "scale-wise-distillation": parse_scale_wise_algorithm,
            "sid": parse_sid_algorithm,
            "sgmd": parse_sgmd_algorithm,
            "self-forcing": parse_self_forcing_algorithm,
            "self-gradient-forcing": parse_self_gradient_forcing_algorithm,
            "senseflow": parse_senseflow_algorithm,
            "token-cppo": parse_token_policy_algorithm,
            "token-dppo": parse_token_policy_algorithm,
            "token-drpo": parse_token_policy_algorithm,
            "token-grpo": parse_token_policy_algorithm,
            "token-gspo": parse_token_policy_algorithm,
        }
        parser = algorithm_parsers.get(algorithm_type)
        if parser is None:
            raise ValueError(f"unsupported native post-training algorithm: {algorithm_type!r}")
        algorithm: PostTrainingAlgorithmSpec = parser(algorithm_payload)

        fake_optimizer = root.get("fake_score_optimizer")
        guidance_optimizer = root.get("guidance_optimizer")
        discriminator_optimizer = root.get("discriminator_optimizer")
        checkpoint_payload = section(
            "checkpoint",
            {"save_every_steps", "async", "export_every_steps"},
        )
        if "async" in checkpoint_payload:
            checkpoint_payload["async_save"] = checkpoint_payload.pop("async")
        return cls(
            schema=str(root.get("schema", POST_TRAINING_RECIPE_SCHEMA)),
            run=RunSpec(**section("run", {"id", "output_dir"})),
            model=ModelSpec(**section("model", {"recipe", "checkpoint", "options"})),
            tuning=TuningSpec(
                **section(
                    "tuning",
                    {"mode", "preset", "rank", "alpha", "dropout", "modules_to_save"},
                )
            ),
            data=DatasetSpec(
                **section(
                    "data",
                    {
                        "manifest",
                        "cache",
                        "max_latent_tokens_per_microbatch",
                        "split",
                        "shuffle",
                        "shuffle_seed",
                        "tail_policy",
                        "options",
                    },
                )
            ),
            algorithm=algorithm,
            optimizer=OptimizerSpec(**section("optimizer", _OPTIMIZER_FIELDS)),
            execution_owner=str(root.get("execution_owner", NATIVE_EXECUTION_OWNER)),
            fake_score_optimizer=(
                None
                if fake_optimizer is None
                else OptimizerSpec(
                    **strict_mapping(
                        fake_optimizer,
                        field_name="fake_score_optimizer",
                        allowed=_OPTIMIZER_FIELDS,
                    )
                )
            ),
            guidance_optimizer=(
                None
                if guidance_optimizer is None
                else OptimizerSpec(
                    **strict_mapping(
                        guidance_optimizer,
                        field_name="guidance_optimizer",
                        allowed=_OPTIMIZER_FIELDS,
                    )
                )
            ),
            discriminator_optimizer=(
                None
                if discriminator_optimizer is None
                else OptimizerSpec(
                    **strict_mapping(
                        discriminator_optimizer,
                        field_name="discriminator_optimizer",
                        allowed=_OPTIMIZER_FIELDS,
                    )
                )
            ),
            runtime=TrainingRuntimeSpec(
                **section(
                    "runtime",
                    {"param_dtype", "reduce_dtype", "activation_checkpoint", "compile"},
                )
            ),
            distributed=DistributedSpec(
                **section(
                    "distributed",
                    {"backend", "dp_replicate", "dp_shard", "cp", "tp"},
                )
            ),
            checkpoint=PostTrainingCheckpointSpec(**checkpoint_payload),
            export=ExportSpec(**section("export", {"format", "options"})),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> PostTrainingRecipe:
        source = Path(path)
        if source.suffix.lower() == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
        elif source.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ModuleNotFoundError as error:
                raise RuntimeError("loading YAML post-training recipes requires pyyaml") from error
            payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        else:
            raise ValueError(f"post-training recipe must be .json, .yaml, or .yml: {source}")
        return cls.from_mapping(mapping(payload, field_name=str(source)))

    def to_dict(self) -> dict[str, object]:
        result = plain_data(self)
        assert isinstance(result, dict)
        # Keep legacy recipe identities stable.  The DMD2-only optimizer is
        # serialized when active and is otherwise absent, not a dead null key.
        if self.guidance_optimizer is None:
            result.pop("guidance_optimizer", None)
        if self.discriminator_optimizer is None:
            result.pop("discriminator_optimizer", None)
        return result

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


__all__ = [
    "AdaptiveVideoAlgorithmSpec",
    "AdversarialDiffusionAlgorithmSpec",
    "BagelFlowUniGRPOAlgorithmSpec",
    "CausalRCMAlgorithmSpec",
    "DanceGRPOAlgorithmSpec",
    "DDRLAlgorithmSpec",
    "DiagonalAlgorithmSpec",
    "DMDAlgorithmSpec",
    "DMD2AlgorithmSpec",
    "DiffusionNFTAlgorithmSpec",
    "FlowDPPOAlgorithmSpec",
    "FlowGRPOAlgorithmSpec",
    "FlowPolicyAlgorithmSpec",
    "FlowSDEWindowSpec",
    "GRPOGuardAlgorithmSpec",
    "LatentConsistencyAlgorithmSpec",
    "MixGRPOAlgorithmSpec",
    "POST_TRAINING_RECIPE_SCHEMA",
    "PostTrainingCheckpointSpec",
    "PostTrainingAlgorithmSpec",
    "PostTrainingRecipe",
    "ProgressiveDistillationAlgorithmSpec",
    "RCMAlgorithmSpec",
    "RewardForcingAlgorithmSpec",
    "ScaleWiseAlgorithmSpec",
    "SIDAlgorithmSpec",
    "SelfForcingAlgorithmSpec",
    "SelfGradientForcingAlgorithmSpec",
    "SenseFlowAlgorithmSpec",
    "SGMDAlgorithmSpec",
    "TokenCPPOAlgorithmSpec",
    "TokenDPPOAlgorithmSpec",
    "TokenDRPOAlgorithmSpec",
    "TokenGRPOAlgorithmSpec",
    "TokenGSPOAlgorithmSpec",
    "TokenPolicyAlgorithmSpec",
    "VIDEOALIGN_BASE_MODEL_REPOSITORY",
    "VIDEOALIGN_BASE_MODEL_REVISION",
    "VIDEOALIGN_CALIBRATION_MEAN",
    "VIDEOALIGN_CALIBRATION_STD",
    "VIDEOALIGN_CHECKPOINT_FILE",
    "VIDEOALIGN_CHECKPOINT_REPOSITORY",
    "VIDEOALIGN_CHECKPOINT_REVISION",
    "VIDEOALIGN_CHECKPOINT_SHA256",
    "VIDEOALIGN_CHECKPOINT_SIZE_BYTES",
    "VIDEOALIGN_REWARD_IDS",
    "VideoAlignRewardSpec",
]
