# World-in-World in-tree runtime provenance

`official/` is a vendored copy of the World-In-World downstream evaluation code at the pinned revision, used for result import and provenance.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/World-In-World/world-in-world |
| Pinned revision | `6ac81ef12451c29d22cdec9ac96e3fe46b22ac2a` |
| Paper | https://arxiv.org/abs/2510.18135 |
| Project page | https://world-in-world.github.io/ |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/**/world-in-world.yaml` |
| Task manifest | `worldfoundry/data/benchmarks/tasks/external/world-in-world.yaml` |

## Modifications

Code in this tree is adapted only as needed for in-tree execution (import paths, path resolution, and the WorldFoundry runner contract). All benchmark-defining logic follows the upstream reference above; refresh against the pinned revision when updating.

## Official boundary

- Result import/normalization of official outputs is pure-local.
- The full World-In-World protocol runs closed-loop agents in simulated/embodied environments upstream; that pipeline (simulators, policies, world models) is not vendored and not claimed as recomputable in tree.

## Not vendored

- Simulation environments, agent policies, and world-model checkpoints used by the official closed-loop evaluation.

## License status

No LICENSE file is recorded for the upstream repository in the catalog; pinned revision `6ac81ef12451c29d22cdec9ac96e3fe46b22ac2a`.
