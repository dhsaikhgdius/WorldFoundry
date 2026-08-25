# 4DWorldBench in-tree runtime provenance

`four_d_worldbench/` is a vendored copy of the official 4DWorldBench evaluation code (yeppp27/4dworldbench_code) so per-dimension official evaluation can run in tree.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/yeppp27/4dworldbench_code.git |
| Paper | https://arxiv.org/pdf/2511.19836 |
| Project page | https://yeppp27.github.io/4DWorldBench.github.io/ |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/**/4dworldbench.yaml` |
| Task manifest | `worldfoundry/data/benchmarks/tasks/external/4dworldbench.yaml` |

## Modifications

Code in this tree is adapted only as needed for in-tree execution (import paths, path resolution, and the WorldFoundry runner contract). All benchmark-defining logic follows the upstream reference above; refresh against the pinned revision when updating.

## Official boundary

- Per-dimension `--run-official` executes the vendored code locally.
- Several alignment/QA dimensions call hosted VLM judges (OpenAI-compatible; `OPENAI_API_KEY`) and are therefore not purely local.
- Geometry/camera dimensions need Keye-VL (`WORLDFOUNDRY_4DWORLDBENCH_KEYE_MODEL`) and DROID-SLAM (`WORLDFOUNDRY_4DWORLDBENCH_DROID_CKPT`) checkpoints plus a GPU.
- Result normalization from existing official outputs is pure-local.

## Not vendored

- Keye-VL and DROID-SLAM checkpoints.
- 4DWorldBench dataset JSON and media.

## License status

No LICENSE file is recorded for the upstream repository in the catalog; verify upstream terms before redistribution.
