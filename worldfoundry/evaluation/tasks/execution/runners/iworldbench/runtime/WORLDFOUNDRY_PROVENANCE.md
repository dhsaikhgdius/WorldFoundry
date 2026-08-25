# iWorld-Bench in-tree runtime provenance

`iworldbench/` vendors the iWorld-Bench (EmbodiedCity) metric implementations, including the trajectory metrics and a ViPE subprocess worker, adapted minimally for in-tree execution.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/EmbodiedCity/iWorld-Bench |
| Paper | unknown_pending_review |
| Project page | https://iworld-bench.com/ |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/**/iworld-bench.yaml` |
| Task manifest | `worldfoundry/data/benchmarks/tasks/external/iworld-bench.yaml` |

## Modifications

Code in this tree is adapted only as needed for in-tree execution (import paths, path resolution, and the WorldFoundry runner contract). All benchmark-defining logic follows the upstream reference above; refresh against the pinned revision when updating.

## Official boundary

- The memory metric has a bounded official run wired in tree (`--run-official --metric memory`).
- Trajectory metrics require ViPE pose estimation (GPU) and the official dataset (`WORLDFOUNDRY_IWORLD_BENCH_DATASET_ROOT`).
- Result normalization from official outputs is pure-local.

## Not vendored

- iWorld-Bench dataset.
- ViPE checkout and pose-estimation weights.

## License status

Upstream repository is Apache-2.0 (recorded in the catalog).
