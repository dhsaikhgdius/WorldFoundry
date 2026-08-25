# Physical AI Bench (PAI-Bench) in-tree runtime provenance

`physical_ai_bench/` is a WorldFoundry in-tree reimplementation of the SHI-Labs PAI-Bench generation and conditional-generation tracks that reuses shared base-model runtimes (VBench metric stack, DOVER, LPIPS, Video-Depth-Anything, Grounding-DINO, SAM2, Qwen judging) instead of duplicating metric code.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/SHI-Labs/physical-ai-bench |
| Pinned revision | `2f3b687410029b98397fbc51fa4de36bfd45627d` |
| Paper | https://arxiv.org/abs/2512.01989 |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/**/physical-ai-bench.yaml` |
| Task manifest | `worldfoundry/data/benchmarks/tasks/external/physical-ai-bench.yaml` |

## Modifications

Code in this tree is adapted only as needed for in-tree execution (import paths, path resolution, and the WorldFoundry runner contract). All benchmark-defining logic follows the upstream reference above; refresh against the pinned revision when updating.

## Official boundary

- `--run-official` evaluates generated videos locally; model-backed metrics need the shared checkpoints (`--depth-checkpoint`, `--grounding-checkpoint`, `--sam2-checkpoint`, `--dover-checkpoint`) and a GPU.
- Generation-track VQA needs a Qwen-compatible judge: local Qwen weights or an OpenAI-compatible endpoint (`--judge-backend`), so that sub-metric can involve hosted inference.
- Result normalization from an existing official results file is pure-local.
- Full-dataset numerical parity on the ~273 GB conditional dataset has not been audited; scores are not official leaderboard submissions.

## Not vendored

- Official Hugging Face datasets (`shi-labs/physical-ai-bench-*`).
- All metric/judge checkpoints (shared base-model registry).

## License status

Upstream evaluator is MIT at commit `2f3b687410029b98397fbc51fa4de36bfd45627d` (recorded in the catalog).
