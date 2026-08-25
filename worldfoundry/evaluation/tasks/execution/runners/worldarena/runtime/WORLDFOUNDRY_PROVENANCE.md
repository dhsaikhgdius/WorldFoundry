# WorldArena in-tree runtime provenance

`video_quality/` vendors the WorldArena (tsinghua-fib-lab) evaluation code for the wired dimensions (action following, VLM judging glue, quality metrics), adapted minimally for in-tree execution.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/tsinghua-fib-lab/WorldArena |
| Pinned revision | `329f6c7ce6c8019715aa8a2fff7b71112e4001d4` |
| Paper | https://arxiv.org/abs/2602.08971 |
| Project page | https://world-arena.ai/ |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/**/worldarena.yaml` |
| Task manifest | `worldfoundry/data/benchmarks/tasks/external/worldarena.yaml` |

## Modifications

Code in this tree is adapted only as needed for in-tree execution (import paths, path resolution, and the WorldFoundry runner contract). All benchmark-defining logic follows the upstream reference above; refresh against the pinned revision when updating.

## Official boundary

- Per-dimension `--run-official` executes the vendored code locally; model-backed dimensions need CLIP-family weights and a GPU.
- VLM-judge dimensions require hosted or locally served judge models and are not purely local.
- Result normalization from existing outputs is pure-local.

## Not vendored

- WorldArena datasets and per-dimension configuration assets.
- Judge model weights or hosted API access.

## License status

No LICENSE file is recorded for the upstream repository in the catalog; pinned revision `329f6c7ce6c8019715aa8a2fff7b71112e4001d4`.
