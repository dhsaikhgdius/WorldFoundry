# PhyGenBench in-tree runtime provenance

`phygenbench/PhyGenEval/` is a vendored subset of the official OpenGVLab PhyGenBench evaluation code used for provenance and the bounded overall aggregation; the runner primarily imports official per-dimension results.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/OpenGVLab/PhyGenBench |
| Pinned revision | `f8642cb796f3bcb01f0b7c1b2ec53b75d357c739` |
| Paper | https://arxiv.org/abs/2410.05363 |
| Project page | https://phygenbench123.github.io/ |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/**/phygenbench.yaml` |
| Task manifest | `worldfoundry/data/benchmarks/tasks/external/phygenbench.yaml` |

## Modifications

Code in this tree is adapted only as needed for in-tree execution (import paths, path resolution, and the WorldFoundry runner contract). All benchmark-defining logic follows the upstream reference above; refresh against the pinned revision when updating.

## Official boundary

- Result import and the bounded overall aggregation are pure-local.
- Full official PhyGenEval scoring requires hosted judges (GPT-family APIs) and model-backed scorers (e.g. InternVideo) with GPUs; that path is not claimed as recomputable in tree.

## Not vendored

- Hosted judge access (OpenAI-compatible API keys).
- InternVideo/VLM scorer weights.

## License status

No LICENSE file is recorded for the upstream repository in the catalog; verify upstream terms before redistribution.
