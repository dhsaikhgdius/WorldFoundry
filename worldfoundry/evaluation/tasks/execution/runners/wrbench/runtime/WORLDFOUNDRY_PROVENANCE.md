# WRBench in-tree runtime provenance

`wrbench/` implements the WRBench D1-D6 evaluation protocol in tree (camera-action grammar, adapters, and scorers), derived from the official JinPLu/WRBench release at the pinned revision.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/JinPLu/WRBench |
| Pinned revision | `629595dc60ec08a29711af0377280c4ac9dd40bc` |
| Paper | https://arxiv.org/abs/2606.20545 |
| Project page | https://jinplu.github.io/WRBench/ |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/**/wrbench.yaml` |
| Task manifest | `worldfoundry/data/benchmarks/tasks/external/wrbench.yaml` |

## Modifications

Code in this tree is adapted only as needed for in-tree execution (import paths, path resolution, and the WorldFoundry runner contract). All benchmark-defining logic follows the upstream reference above; refresh against the pinned revision when updating.

## Official boundary

- D1-D6 scoring runs locally, but the heavy metrics need VGGT-Omega, DINOv2, Qwen3.5, and Qwen3-VL checkpoints plus GPUs; without those assets only the lightweight paths run.
- Official-validation normalizes precomputed WRBench results and is pure-local.
- Leaderboard parity additionally requires complete Natural-25 generation.

## Not vendored

- VGGT-Omega, DINOv2, Qwen3.5, and Qwen3-VL checkpoints.
- Natural-25 assets beyond the bundled manifests.

## License status

No LICENSE file is recorded for the upstream repository in the catalog; pinned revision `629595dc60ec08a29711af0377280c4ac9dd40bc`.
