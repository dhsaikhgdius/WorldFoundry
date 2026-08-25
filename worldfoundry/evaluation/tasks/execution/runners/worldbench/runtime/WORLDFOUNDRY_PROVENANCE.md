# WorldBench in-tree runtime provenance

`worldbench/` is a WorldFoundry clean-room reimplementation of the WorldBench IntuitivePhysics evaluation protocol (the upstream `worldbench_eval` repository publishes no license, so its code is deliberately not vendored).

| Field | Value |
| --- | --- |
| Paper | https://arxiv.org/abs/2601.21282 |
| Project page | https://world-bench.github.io/ |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/**/worldbench.yaml` |
| Task manifest | `worldfoundry/data/benchmarks/tasks/external/worldbench.yaml` |

## Modifications

Not applicable: no upstream code is vendored here. This tree is WorldFoundry-authored; the evaluation protocol follows the official WorldBench paper and the `worldbench_eval` reference implementation, and behavior is validated by `tests/evaluation/test_worldbench_runtime.py`.

## Official boundary

- `--run-official` evaluates generated artifacts locally; segmentation-backed metrics need the SAM2.1 checkpoint (`WORLDFOUNDRY_WORLDBENCH_SAM2_CKPT`) and a GPU.
- The official dataset (`worldbenchmark/IntuitivePhysics`) is supplied through `WORLDFOUNDRY_WORLDBENCH_DATASET_ROOT`.
- Text-track evaluation and result normalization are pure-local.

## Not vendored

- Upstream `worldbench_eval` code (license not provided upstream).
- SAM2.1 checkpoint and the official dataset.

## License status

Upstream evaluator license is `not_provided`; this directory contains only WorldFoundry-authored code.
