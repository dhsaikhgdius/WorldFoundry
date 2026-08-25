# PhyGround in-tree runtime provenance

`phyground/` carries the PhyGround evaluation-protocol code (typed VLM-eval structures and physics criteria) in tree for result import and provenance; official judging runs upstream.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/NU-World-Model-Embodied-AI/PhyGround |
| Pinned revision | `99c511b5bddbe2d4d5a3aa66a0282a0fbbf2292d` |
| Paper | https://arxiv.org/abs/2605.10806 |
| Project page | https://phyground.github.io/ |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/**/phyground.yaml` |
| Task manifest | `worldfoundry/data/benchmarks/tasks/external/phyground.yaml` |

## Modifications

Code in this tree is adapted only as needed for in-tree execution (import paths, path resolution, and the WorldFoundry runner contract). All benchmark-defining logic follows the upstream reference above; refresh against the pinned revision when updating.

## Official boundary

- Result import (`--benchmark-data-root` + `--result-model-id`) is pure-local.
- Official judging uses VLM APIs upstream; no in-tree official-run is claimed.

## Not vendored

- PhyGround benchmark media and per-model outputs.
- VLM judge access.

## License status

No LICENSE file is recorded for the upstream repository in the catalog; verify upstream terms before redistribution.
