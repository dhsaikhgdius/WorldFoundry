# MIND bundled fixture

`fixtures/mind_result_fixture.json` is a **synthetic** MIND result document used to
exercise the WorldFoundry normalizer without GPUs, ViPE, model weights, or the
official dataset.

It contains **no upstream MIND records**: every sample id, frame count, and score
was invented for this repository. The numbers are only chosen to sit inside the
plausible range of each metric so that normalization, aggregation, and the
scorecard invariants can be asserted. They must never be reported, compared, or
published as MIND results.

The file reproduces the schema written by the official
`src/process.py` entry point (see
`worldfoundry/evaluation/tasks/execution/runners/mind/runtime/mind/README.md`):

- `video_max_time` — the `--video_max_time` value of the run.
- `data[*]` — one record per test directory, keyed by `perspective`
  (`1st_data` / `3rd_data`) and `test_type` (`mem_test` / `action_space_test` /
  `mirror_test`).
- `mem_test` / `action_space_test` records carry `lcm`, `visual_quality`,
  `dino`, and `action` blocks.
- `mirror_test` records carry `video_results[*].gsc`.

Run it with:

```bash
PYTHONPATH=. python worldfoundry/evaluation/tasks/execution/runners/mind/run_mind_official_runner.py \
  --run-fixture --output-dir tmp/mind/fixture --json
```
