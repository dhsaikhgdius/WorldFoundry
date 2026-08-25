# STEVO-Bench bundled assets

`sample_run/` is a synthetic fixture that mirrors the artifact layout written by
the official STEVO-Bench evaluator (`eval/eval_cli.py` plus
`eval/summarize_results.py` at the pinned vendored revision):

```
sample_run/
|-- summary.json                 # run-level summary with per-task judge verdicts
`-- per_task/
    `-- <task_id>/
        `-- se_report__<provider>__<model>.json
```

The fixture exists only to validate the WorldFoundry result normalizer and the
`--run-fixture` path of `run_stevo_bench_official_runner.py`. It contains **no
upstream benchmark tasks, prompts, videos, or real judge outputs**; every task
id and verdict is invented. The real 225-task suite is a separate Hugging Face
dataset (`JhanLiufu/StEvo-Bench`) and real verdicts come from hosted VLM judge
APIs during `--run-official`.

`per_task/candle_burn_01/` deliberately carries a verdict that is missing from
`summary.json` so the normalizer's report-merge path stays covered.
