# DEVIL Dynamics in-tree runtime provenance

`official/` contains the DEVIL official evaluator code (metric utilities and launcher glue) carried in tree so `run_devil_dynamics_official_runner.py --run-official` can execute the official dynamics metrics without an external checkout. `devil_official_runtime.py` is a WorldFoundry-authored launcher around the vendored metric modules.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/MingXiangL/DEVIL |
| Pinned revision | `48556c329e6a6f92a587021bc97022494faec69c` |
| Paper | https://arxiv.org/abs/2407.01094 |
| Project page | https://huggingface.co/papers/2407.01094 |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/**/devil-dynamics.yaml` |
| Task manifest | `worldfoundry/data/benchmarks/tasks/external/devil-dynamics.yaml` |

## Modifications

Code in this tree is adapted only as needed for in-tree execution (import paths, path resolution, and the WorldFoundry runner contract). All benchmark-defining logic follows the upstream reference above; refresh against the pinned revision when updating.

## Official boundary

- Official-run executes the vendored DEVIL metric stack locally; model-backed metrics require their Python dependencies and a GPU for realistic runtimes.
- Official-validation (`--official-results-path`) normalizes precomputed DEVIL results and is pure-local.

## Not vendored

- DEVIL benchmark videos and annotation data (obtained per upstream instructions).
- Metric model weights downloaded on demand by the metric dependencies.

## License status

No LICENSE file is recorded for the upstream repository in the catalog; verify upstream terms before redistribution.
