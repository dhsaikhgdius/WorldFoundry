# Third-party notices — rCM / Causal-rCM runtime

## NVlabs/rCM

Everything under this directory except `io_compat.py` is the official
NVIDIA NVlabs rCM inference stack, vendored unmodified apart from the import
rewiring listed below. Upstream SPDX headers are preserved in every file.

Source: https://github.com/NVlabs/rcm

Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Licensed under the Apache License, Version 2.0.

Papers:

- rCM — *Large Scale Diffusion Distillation via Score-Regularized Continuous-Time
  Consistency*, Zheng et al., ICLR 2026 (arXiv:2510.08431)
- Causal-rCM — *A Unified Teacher-Forcing and Self-Forcing Open Recipe for
  Autoregressive Diffusion Distillation in Streaming Video Generation and
  Interactive World Models*, Zheng et al. (arXiv:2606.25473)

### Vendored files

| Upstream path | Vendored path |
| --- | --- |
| `rcm/datasets/utils.py` | `datasets/utils.py` |
| `rcm/inference/wan2pt1_t2v_causal_infer.py` | `inference/wan2pt1_t2v_causal_infer.py` |
| `rcm/networks/wan2pt1.py` | `networks/wan2pt1.py` |
| `rcm/tokenizers/interface.py` | `tokenizers/interface.py` |
| `rcm/tokenizers/wan2pt1.py` | `tokenizers/wan2pt1.py` |
| `rcm/utils/a2a_cp.py` | `utils/a2a_cp.py` |
| `rcm/utils/attention.py` | `utils/attention.py` |
| `rcm/utils/blockmask.py` | `utils/blockmask.py` |
| `rcm/utils/context_parallel.py` | `utils/context_parallel.py` |
| `rcm/utils/kv_cache.py` | `utils/kv_cache.py` |
| `rcm/utils/magimask.py` | `utils/magimask.py` |
| `rcm/utils/model_utils.py` | `utils/model_utils.py` |
| `rcm/utils/rope.py` | `utils/rope.py` |
| `rcm/utils/umt5.py` | `utils/umt5.py` |

Pinned upstream revision: `ed3cb14dd936f92cdc9f9381af7369991509b41f`.

### Modifications

The upstream `imaginaire` framework package is **not** vendored. WorldFoundry
already ships API-compatible equivalents, so the vendored files import those
instead:

| Upstream import | WorldFoundry replacement |
| --- | --- |
| `imaginaire.lazy_config` | `worldfoundry.core.configuration.lazy_config` |
| `imaginaire.utils.log` | `worldfoundry.core.distributed.logging.log` |
| `imaginaire.utils.distributed` | `worldfoundry.core.distributed.torch_process_group` |
| `imaginaire.utils.easy_io.easy_io` | `worldfoundry.core.io.easy_io.easy_io` |
| `imaginaire.utils.io.save_image_or_video` | `io_compat.save_image_or_video` (this directory) |
| `rcm.utils.selective_activation_checkpoint` | `worldfoundry.core.nn.activation_checkpointing` |

`rcm/utils/selective_activation_checkpoint.py` is not vendored: WorldFoundry's
`core/nn/activation_checkpointing.py` already implements the same
`CheckpointMode` / `SACConfig` policy.

`BlockPattern`, `AttnMaskSpec`, and the block-causal / teacher-forcing mask
predicates were lifted out of `utils/blockmask.py` into
`worldfoundry/core/attention/block_pattern.py` so causal chunk schedules are a
single shared primitive rather than a per-network copy. The vendored
`utils/blockmask.py` re-exports them, so upstream call sites are unchanged.

Three further edits make the cache's horizon policy pluggable without altering
upstream behaviour. Each is behaviour-preserving on its own, and
`tests/synthesis/test_policy_kv_cache.py` proves the default path stays
bit-identical to the unmodified cache — in metadata, buffers, and attention
outputs.

| File | Edit | Why |
| --- | --- | --- |
| `utils/kv_cache.py` | added a `committed_blocks` property returning `len(self._cum_ends)` | Pure refactor. Separates "blocks the caller has committed" from "chunks still resident" so an evicting subclass can keep the runtime's global block cursor meaningful. |
| `utils/a2a_cp.py` | `assert block_range == len(cache._cum_ends)` became `assert block_range == cache.committed_blocks` | Identical for the plain cache; lets an evicting cache satisfy the same invariant. |
| `networks/wan2pt1.py` | `allocate_kv_caches` gained an optional `cache_factory` argument | Defaults to `None`, which preserves the upstream allocation exactly. |
| `inference/wan2pt1_t2v_causal_infer.py` | added `keep_all` / `sliding_window` cache-plan flags and passes the resulting factory to both T2V and I2V cache allocation paths | Keeps the upstream `keep_all` path byte-for-byte equivalent while a sliding window allocates only its retained prefix plus one transient in-flight chunk. |
| `utils/umt5.py` | dropped the unused `misc` name from the rewritten `imaginaire.utils` import | `misc` is imported upstream but never referenced in this module. |

Every other vendored file is byte-identical to upstream once the import rewiring
above is undone.  The cache policy implementation itself is WorldFoundry-authored
and lives outside this directory, at `../policy_kv_cache.py`.

Only `keep_all` and `sliding_window` are exposed through the causal entrypoint:
it stores post-RoPE keys, so policies that remap temporal positions must use the
separate pre-RoPE extrapolation path rather than silently produce incorrect
attention geometry.
