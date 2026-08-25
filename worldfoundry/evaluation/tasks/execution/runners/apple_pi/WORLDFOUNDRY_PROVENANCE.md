# Apple-PI in-tree runtime provenance

`apple_pi_runtime.py` is a WorldFoundry-native, model-backed implementation of the Apple-PI evaluation protocol. No upstream source tree is vendored; the protocol and metric definitions follow the official repository at the reference below.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/21yrm/Apple-PI |
| Paper | https://arxiv.org/abs/2607.16401 |
| Project page | https://21yrm.github.io/Apple-PI-homepage/ |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/**/apple-pi.yaml` |
| Task manifest | `worldfoundry/data/benchmarks/tasks/external/apple-pi.yaml` |

## Modifications

Not applicable: no upstream code is vendored here. `apple_pi_runtime.py` is WorldFoundry-authored; metric and protocol definitions follow the official Apple-PI repository, and the runner is validated by `tests/evaluation/test_apple_pi_in_tree_runtime.py`.

## Official boundary

- `--run-official` executes the in-tree model-backed evaluator locally against caller-supplied ground-truth (`--gt-dir` / `WORLDFOUNDRY_APPLE_PI_GT_DIR`) and prediction (`--pred-dir` / `WORLDFOUNDRY_APPLE_PI_PREDICTION_DIR`) roots.
- Model-backed stages need SAM3 and MoGe-2 checkpoints (`WORLDFOUNDRY_APPLE_PI_SAM3_CHECKPOINT`, `WORLDFOUNDRY_APPLE_PI_MOGE_CHECKPOINT`) and a GPU for real runs; the `mock` judge backend allows offline plumbing validation only.
- `--official-results-path` (official-validation mode) normalizes a precomputed Apple-PI result JSON without any model execution.

## Not vendored

- SAM3 and MoGe-2 metric checkpoints (Hugging Face; see `checkpoint_refs` in the catalog entry).
- Apple-PI ground-truth data; availability follows the upstream release.

## License status

Upstream repository is Apache-2.0 (recorded in the catalog); no upstream code is redistributed here.
