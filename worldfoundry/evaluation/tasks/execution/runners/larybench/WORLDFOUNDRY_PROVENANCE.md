# LARYBench in-tree runtime provenance

The stage modules in this package (`extract.py`, `classification*.py`, `regression*.py`, `cli.py`, `models.py`, `registry.py`) are the LARYBench latent-action runtime adapted from meituan-longcat/LARYBench for in-tree execution; `run_larybench_official_runner.py` wraps them behind the WorldFoundry benchmark contract.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/meituan-longcat/LARYBench |
| Pinned revision | `e7feaf1b72921ee2c34e489adb0f45faf356ecee` |
| Paper | https://arxiv.org/abs/2604.11689 |
| Project page | https://meituan-longcat.github.io/LARYBench/ |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/**/larybench.yaml` |
| Task manifest | `worldfoundry/data/benchmarks/tasks/external/larybench.yaml` |

## Modifications

Code in this tree is adapted only as needed for in-tree execution (import paths, path resolution, and the WorldFoundry runner contract). All benchmark-defining logic follows the upstream reference above; refresh against the pinned revision when updating.

## Official boundary

- `--run-official --stage {extract,classify,regress}` runs latent-action extraction, probing classification, or regression locally; extraction and training stages require GPUs and the LARYBench datasets (`DATA_DIR`).
- Official-validation normalizes a stage result (`classification_summary.json`, `best_result.json`, or an extraction CSV) and is pure-local.

## Not vendored

- LARYBench datasets and pretrained latent-action model checkpoints (`MODEL_DIR`).

## License status

Upstream repository is MIT (recorded in the catalog); pinned revision `e7feaf1b72921ee2c34e489adb0f45faf356ecee`.
