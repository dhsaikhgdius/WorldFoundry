# EWMBench in-tree runtime provenance

`ewmbench/EWMBench/` is a vendored copy of the official AgibotTech EWMBench evaluator so the embodied world-model metrics can run in tree without a separate checkout.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/AgibotTech/EWMBench |
| Pinned revision | `3a5531c0e6b6be51431251ed5617b9e48c852098` |
| Paper | https://arxiv.org/abs/2505.09694 |
| Project page | https://huggingface.co/datasets/agibot-world/EWMBench |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/**/ewmbench.yaml` |
| Task manifest | `worldfoundry/data/benchmarks/tasks/external/ewmbench.yaml` |

## Modifications

Code in this tree is adapted only as needed for in-tree execution (import paths, path resolution, and the WorldFoundry runner contract). All benchmark-defining logic follows the upstream reference above; refresh against the pinned revision when updating.

## Official boundary

- Official-run executes the vendored evaluator locally and requires the EWMBench dataset layout, metric model weights, and a GPU.
- Official-validation normalizes a precomputed `official_results.json` and is pure-local.

## Not vendored

- EWMBench dataset (upstream Hugging Face release is CC-BY-NC-SA-4.0).
- Perception model weights the evaluator downloads or expects locally.

## License status

The upstream dataset is CC-BY-NC-SA-4.0; the upstream code repository publishes no license file recorded in the catalog — verify before redistribution.
