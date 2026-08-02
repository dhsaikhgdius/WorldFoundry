# WorldOlympiad in-tree fixture

Synthetic judge outputs used by
`run_worldolympiad_official_runner.py --run-fixture` to validate the result
normalizer without GPUs, model weights, or an upstream checkout.

These files contain **no upstream benchmark records**: the videos, prompts,
physics questions, and scores are invented and mirror only the public judge
JSON *shape* written by `scripts/score_video_physical_3d.py`. They are not
comparable to any WorldOlympiad result and must never be reported as scores.

```text
cases/
  general/case_fixture_001/wf_fixture_judge_case_fixture_001.json   # all three tracks
  gaming/case_fixture_002/wf_fixture_judge_case_fixture_002.json    # all three tracks
  embodied/case_fixture_003/wf_fixture_judge_case_fixture_003.json  # geometry track skipped
```

Real evaluation requires `--worldolympiad-root` (or
`WORLDFOUNDRY_WORLDOLYMPIAD_ROOT`) plus the DA3, SAM3, and QwenVL services
described in the upstream README.
