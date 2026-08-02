"""Algorithm-specific post-training recipe contracts."""

from .adaptive_video import AdaptiveVideoAlgorithmSpec
from .adversarial_diffusion import AdversarialDiffusionAlgorithmSpec
from .anyflow import (
    AnyFlowAlgorithmSpec,
    AnyFlowBidirectionalOnPolicyAlgorithmSpec,
    AnyFlowBidirectionalPretrainAlgorithmSpec,
    AnyFlowFAROnPolicyAlgorithmSpec,
    AnyFlowFARPretrainAlgorithmSpec,
    AnyFlowFARSpec,
    AnyFlowMapSpec,
)
from .bagel_flow_unigrpo import BagelFlowUniGRPOAlgorithmSpec
from .causal_consistency import CausalConsistencyAlgorithmSpec
from .causal_ode import CausalODEAlgorithmSpec
from .dance_grpo import DanceGRPOAlgorithmSpec
from .ddrl import DDRLAlgorithmSpec
from .dfd import DFDAlgorithmSpec
from .diagonal import DiagonalAlgorithmSpec
from .diffusion_dpo import DiffusionDPOAlgorithmSpec
from .diffusion_nft import (
    DiffusionNFTAlgorithmSpec,
    DiffusionNFTOldPolicyRefreshSpec,
    DiffusionNFTTerminalLatentCollectionSpec,
)
from .dmd import DMDAlgorithmSpec
from .dmd2 import DMD2AlgorithmSpec
from .flow_dppo import FlowDPPOAlgorithmSpec
from .flow_grpo import FlowGRPOAlgorithmSpec
from .flow_policy import FlowPolicyAlgorithmSpec, FlowSDEWindowSpec
from .grpo_guard import GRPOGuardAlgorithmSpec
from .latent_consistency import LatentConsistencyAlgorithmSpec
from .mix_grpo import MixGRPOAlgorithmSpec
from .progressive import ProgressiveDistillationAlgorithmSpec
from .rcm import CausalRCMAlgorithmSpec, RCMAlgorithmSpec
from .reward_forcing import RewardForcingAlgorithmSpec
from .scale_wise import ScaleWiseAlgorithmSpec
from .scm_ladd import SCMLADDAlgorithmSpec
from .self_forcing import SelfForcingAlgorithmSpec
from .self_gradient_forcing import SelfGradientForcingAlgorithmSpec
from .senseflow import SenseFlowAlgorithmSpec, SenseFlowScheduleSpec
from .sgmd import SGMDAlgorithmSpec
from .sid import SIDAlgorithmSpec
from .token_policy import (
    TokenCPPOAlgorithmSpec,
    TokenDPPOAlgorithmSpec,
    TokenDRPOAlgorithmSpec,
    TokenGRPOAlgorithmSpec,
    TokenGSPOAlgorithmSpec,
    TokenPolicyAlgorithmSpec,
)

PostTrainingAlgorithmSpec = (
    AdaptiveVideoAlgorithmSpec
    | AdversarialDiffusionAlgorithmSpec
    | AnyFlowAlgorithmSpec
    | DMDAlgorithmSpec
    | DMD2AlgorithmSpec
    | DDRLAlgorithmSpec
    | DFDAlgorithmSpec
    | DiagonalAlgorithmSpec
    | DiffusionDPOAlgorithmSpec
    | DiffusionNFTAlgorithmSpec
    | FlowPolicyAlgorithmSpec
    | CausalConsistencyAlgorithmSpec
    | CausalODEAlgorithmSpec
    | LatentConsistencyAlgorithmSpec
    | ProgressiveDistillationAlgorithmSpec
    | RCMAlgorithmSpec
    | CausalRCMAlgorithmSpec
    | RewardForcingAlgorithmSpec
    | SCMLADDAlgorithmSpec
    | ScaleWiseAlgorithmSpec
    | SelfForcingAlgorithmSpec
    | SelfGradientForcingAlgorithmSpec
    | SenseFlowAlgorithmSpec
    | SIDAlgorithmSpec
    | SGMDAlgorithmSpec
    | TokenPolicyAlgorithmSpec
)

__all__ = [
    "AdaptiveVideoAlgorithmSpec",
    "AdversarialDiffusionAlgorithmSpec",
    "AnyFlowAlgorithmSpec",
    "AnyFlowBidirectionalOnPolicyAlgorithmSpec",
    "AnyFlowBidirectionalPretrainAlgorithmSpec",
    "AnyFlowFAROnPolicyAlgorithmSpec",
    "AnyFlowFARPretrainAlgorithmSpec",
    "AnyFlowFARSpec",
    "AnyFlowMapSpec",
    "BagelFlowUniGRPOAlgorithmSpec",
    "CausalRCMAlgorithmSpec",
    "CausalConsistencyAlgorithmSpec",
    "CausalODEAlgorithmSpec",
    "DanceGRPOAlgorithmSpec",
    "DMDAlgorithmSpec",
    "DMD2AlgorithmSpec",
    "DDRLAlgorithmSpec",
    "DFDAlgorithmSpec",
    "DiagonalAlgorithmSpec",
    "DiffusionDPOAlgorithmSpec",
    "DiffusionNFTAlgorithmSpec",
    "DiffusionNFTOldPolicyRefreshSpec",
    "DiffusionNFTTerminalLatentCollectionSpec",
    "FlowDPPOAlgorithmSpec",
    "FlowGRPOAlgorithmSpec",
    "FlowPolicyAlgorithmSpec",
    "FlowSDEWindowSpec",
    "GRPOGuardAlgorithmSpec",
    "LatentConsistencyAlgorithmSpec",
    "MixGRPOAlgorithmSpec",
    "PostTrainingAlgorithmSpec",
    "ProgressiveDistillationAlgorithmSpec",
    "RCMAlgorithmSpec",
    "RewardForcingAlgorithmSpec",
    "SCMLADDAlgorithmSpec",
    "ScaleWiseAlgorithmSpec",
    "SelfForcingAlgorithmSpec",
    "SelfGradientForcingAlgorithmSpec",
    "SenseFlowAlgorithmSpec",
    "SenseFlowScheduleSpec",
    "SIDAlgorithmSpec",
    "SGMDAlgorithmSpec",
    "TokenCPPOAlgorithmSpec",
    "TokenDPPOAlgorithmSpec",
    "TokenDRPOAlgorithmSpec",
    "TokenGRPOAlgorithmSpec",
    "TokenGSPOAlgorithmSpec",
    "TokenPolicyAlgorithmSpec",
]
