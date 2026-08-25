# WorldModelBench in-tree runtime provenance

`evaluation.py` is the official WorldModelBench evaluation script carried in tree at the pinned revision so result scoring can run without an external checkout.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/WorldModelBench-Team/WorldModelBench |
| Pinned revision | `00b7aa17a05f9fd1ab5c8f66bcf476d04c9c33bf` |
| Paper | https://arxiv.org/abs/2502.20694 |
| Project page | https://worldmodelbench-team.github.io/ |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/**/worldmodelbench.yaml` |
| Task manifest | `worldfoundry/data/benchmarks/tasks/external/worldmodelbench.yaml` |

## Modifications

Code in this tree is adapted only as needed for in-tree execution (import paths, path resolution, and the WorldFoundry runner contract). All benchmark-defining logic follows the upstream reference above; refresh against the pinned revision when updating.

## Official boundary

- Scoring from existing judge outputs and result normalization are pure-local.
- The instruction/physics judging stage uses a VLM judge with its own weights (GPU or hosted serving); that stage is not purely local.

## Not vendored

- VLM judge weights and the WorldModelBench validation dataset.

## License status

Upstream license is `unknown_pending_review` in the catalog; treat redistribution cautiously and re-check upstream before wider distribution.
