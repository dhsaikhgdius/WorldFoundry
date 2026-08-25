# RBench in-tree runtime provenance

This package aggregates and normalizes official RBench (DAGroup-PKU/ReVidgen) judge outputs; the scoring/aggregation modules follow the official protocol. RBench judging itself happens upstream.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/DAGroup-PKU/ReVidgen |
| Paper | https://arxiv.org/abs/2601.15282 |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/**/rbench.yaml` |
| Task manifest | `worldfoundry/data/benchmarks/tasks/external/rbench.yaml` |

## Modifications

Code in this tree is adapted only as needed for in-tree execution (import paths, path resolution, and the WorldFoundry runner contract). All benchmark-defining logic follows the upstream reference above; refresh against the pinned revision when updating.

## Official boundary

- Aggregation of per-judge outputs (`gpt` or `qwen`, never mixed) is pure-local.
- The VLM judging step runs upstream; there is no in-tree official-run and scores are not marked recomputable.

## Not vendored

- Upstream judging pipeline and VLM access.
- RBench prompt/video data (CC-BY-4.0 datasets recorded in the catalog).

## License status

Dataset releases are CC-BY-4.0; the upstream code repository publishes no license file recorded in the catalog — verify before redistribution.
