"""Composable, in-tree diffusion acceleration techniques.

These utilities operate above individual kernels. They intentionally expose
the approximation policy so callers can validate quality for each model.
"""

from worldfoundry.core.acceleration.cache import AdaptiveResidualCache, FixedStepCache
from worldfoundry.core.acceleration.cuda_graph_dispatch import (
    CUDAGraphDispatch,
    cuda_graph_capture_ar_index,
)
from worldfoundry.core.acceleration.encoder_lifecycle import (
    collect_and_release_cuda_memory,
    ensure_one_shot_encoder,
    move_tensors_to_cpu,
    offload_module_to_cpu,
    release_one_shot_encoder_references,
    run_one_shot_encoder_stage,
    setup_one_shot_encoder,
)
from worldfoundry.core.acceleration.frame_prefetch import (
    CudaHostPrefetch,
    LazyCudaFrame,
    prefetch_to_numpy,
)
from worldfoundry.core.acceleration.nvfp4 import (
    NVFP4Linear,
    dequantize_nvfp4,
    quantize_nvfp4,
    replace_linear_with_nvfp4,
)
from worldfoundry.core.acceleration.overlap import (
    CudaStreamOverlap,
    HostThreadOverlap,
    SynchronousOverlap,
)
from worldfoundry.core.acceleration.prewarm import (
    PrewarmDeadline,
    PrewarmSequenceTiming,
    PrewarmTimeoutError,
    PrewarmTiming,
    cuda_graph_prewarm_steps,
    run_async_prewarm_sequence,
    run_prewarm_sequence,
    run_timed_prewarm,
)
from worldfoundry.core.acceleration.quantization import (
    Float8Linear,
    replace_linear_with_float8,
    set_low_precision_enabled,
)
from worldfoundry.core.acceleration.technology import (
    AccelerationTechnology,
    acceleration_technology_report,
)
from worldfoundry.core.acceleration.token_pruning import (
    TokenPruner,
    TokenPruneState,
    prune_tokens,
    restore_tokens,
    select_token_indices,
)

__all__ = [
    "AdaptiveResidualCache",
    "AccelerationTechnology",
    "CUDAGraphDispatch",
    "CudaHostPrefetch",
    "CudaStreamOverlap",
    "FixedStepCache",
    "Float8Linear",
    "HostThreadOverlap",
    "LazyCudaFrame",
    "NVFP4Linear",
    "PrewarmDeadline",
    "PrewarmSequenceTiming",
    "PrewarmTimeoutError",
    "PrewarmTiming",
    "SynchronousOverlap",
    "TokenPruneState",
    "TokenPruner",
    "collect_and_release_cuda_memory",
    "cuda_graph_capture_ar_index",
    "cuda_graph_prewarm_steps",
    "ensure_one_shot_encoder",
    "move_tensors_to_cpu",
    "offload_module_to_cpu",
    "prefetch_to_numpy",
    "prune_tokens",
    "dequantize_nvfp4",
    "quantize_nvfp4",
    "replace_linear_with_float8",
    "replace_linear_with_nvfp4",
    "release_one_shot_encoder_references",
    "restore_tokens",
    "run_async_prewarm_sequence",
    "run_one_shot_encoder_stage",
    "run_prewarm_sequence",
    "run_timed_prewarm",
    "select_token_indices",
    "set_low_precision_enabled",
    "setup_one_shot_encoder",
    "acceleration_technology_report",
]
