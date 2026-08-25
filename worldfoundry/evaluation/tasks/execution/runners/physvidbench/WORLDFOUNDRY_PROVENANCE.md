# PhysVidBench in-tree runtime provenance

This package imports official PhysVidBench results and provides a bounded caption-QA path; `run_multi_ask_with_api_key.py` follows the upstream PhysVidBenchCode multi-ask judging flow.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/ensanli/PhysVidBenchCode |
| Pinned revision | `b1d20f121ed3ebe5d18337d884fdecfad065096b` |
| Paper | https://arxiv.org/abs/2507.15824 |
| Project page | https://cyberiada.github.io/PhysVidBench/ |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/**/physvidbench.yaml` |
| Task manifest | `worldfoundry/data/benchmarks/tasks/external/physvidbench.yaml` |

## Modifications

Code in this tree is adapted only as needed for in-tree execution (import paths, path resolution, and the WorldFoundry runner contract). All benchmark-defining logic follows the upstream reference above; refresh against the pinned revision when updating.

## Official boundary

- Result import from the official `output.csv` is pure-local.
- The bounded caption-QA path calls hosted LLM APIs (API key required) and is therefore not purely local.
- The full official pipeline (captioning + multi-ask judging) runs upstream and is not claimed as recomputable in tree.

## Not vendored

- Captioner model weights and hosted judge access.
- PhysVidBench prompt/question data beyond the bundled manifests.

## License status

Upstream materials are CC-BY-4.0 (recorded in the catalog); pinned revision `b1d20f121ed3ebe5d18337d884fdecfad065096b`.
