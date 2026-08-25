# STEVO-Bench in-tree runtime provenance

`stevo_bench/` is a vendored copy of the official STEVO-Bench evaluator so that
`run_stevo_bench_official_runner.py --run-official` executes the official judge
pipeline without requiring a separate checkout.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/jhanliufu-personal/STEVO-Bench |
| Revision | `680fb6ee2733894ebc8e5584c08146f4bf7e6415` |
| Paper | https://arxiv.org/abs/2603.13215 |
| Project page | https://glab-caltech.github.io/STEVOBench/ |
| Task dataset | https://huggingface.co/datasets/JhanLiufu/StEvo-Bench |
| Upstream license | **MIT License**, `Copyright (c) 2026 Mengzhan 'Jhan' Liufu` (`stevo_bench/LICENSE`) |

## Modifications

None. WorldFoundry adapts the runtime purely through CLI arguments, environment
variables, `PYTHONPATH`, and the process working directory: the runner invokes
`python -m eval.eval_cli` with `cwd` set to this tree and writes every mutable
artifact (staged videos, upstream `runs/` directories, logs) under the caller's
`--output-dir`. The vendored copy can therefore be refreshed by re-running the
same export.

Known upstream quirks preserved as-is:

- `run_eval.sh` passes `--artifact` and `--coherence`, which `eval/eval_cli.py`
  does not define at this revision; those two criteria are not executable
  upstream, so the WorldFoundry runner exposes only `control`, `physics`, and
  `state`.
- `generation/world_models/generate_lingbot_poses.py` contains a literal
  placeholder `sys.path.insert(0, '<path_to_your_HY-WorldPlay_repo>/hyvideo')`.
  The generation package is never imported by the evaluation path; it is kept
  only for completeness of the vendored tree.

## Exclusions

- `figures/` — README/paper images with no role in evaluation.
- `.git/` — upstream VCS metadata.

## Official boundary

- **official-validation** (result normalization) is pure-local: it parses
  upstream `summary.json` / `per_task/*_report*.json` trees with the in-tree
  normalizer and needs no API keys, GPUs, or downloads.
- **official-run** executes this vendored evaluator, but every judge verdict is
  produced by a **hosted VLM API** (Gemini by default, model
  `gemini-3.1-pro-preview`; OpenAI supported). `GOOGLE_API_KEY` or
  `OPENAI_API_KEY` plus outbound network access are required. There are no
  local judge weights and no simulator dependency.

## Assets that are deliberately *not* vendored

- **Benchmark task YAMLs** — the 225-task suite is a Hugging Face dataset
  (`JhanLiufu/StEvo-Bench`) supplied through `--task-root` /
  `WORLDFOUNDRY_STEVO_BENCH_TASK_ROOT`.
- **Generated videos and output maps** — caller-supplied through
  `--generated-artifact-dir` (plus optional `--output-map`).
- **Judge models** — remote services reached through the upstream
  `eval/judge_client.py`; nothing is downloaded into this tree.
