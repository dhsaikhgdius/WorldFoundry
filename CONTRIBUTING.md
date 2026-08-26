# Contributing

Thanks for helping improve WorldFoundry. Keep changes scoped, document public
behavior, and prefer manifests, task YAML, runtime profiles, and public CLI
commands over private scripts.

The detailed maintainer guide lives in
[`docs/fumadocs/content/docs/maintainers/contributing.mdx`](docs/fumadocs/content/docs/maintainers/contributing.mdx).

## Before Opening A Pull Request

- Keep the change scoped to one pipeline, benchmark path, CLI surface, or docs area.
- Do not commit checkpoints, generated videos, local caches, API keys, tokens, or credentials.
- State any GPU, API quota, checkpoint, simulator, official repo, or gated-data requirement.
- Run the public validation commands relevant to the changed surface.
- Run `bash scripts/docs/build.sh --skip-bootstrap` when docs or README links change.
- Include scorecard, preflight, or validation evidence for any readiness claim.


## Local Validation (CPU)

These commands are expected to run on a CPU-only machine (no CUDA checkpoints):

```bash
make lint                 # ruff + shell syntax + catalog checks (+ docker-smoke when present)
make cli-check            # evaluation runner hello-world scorecard
make test-eval-core PYTEST_ARGS='-m "not gpu and not network"'
make test-training PYTEST_ARGS='-m "not gpu and not network"'
```

Notes:

- On `main`, the full `test/eval_core` suite still has known catalog/contract assertion failures. Prefer the layered CI pattern from `infra-ci-modernize` (hard gate: `test_cuda_tiers.py`; full suite `continue-on-error`) until those contracts are repaired.
- Do not require GPU inference or downloaded checkpoints for PR validation unless the change is specifically about those paths.
- Studio UI: keep `gradio<6` / Starlette `<1` if installing `[ui]` until the pin lands on `main` (see open infra PRs).

## Change Checklists

### Add Or Update A Model

- Update `data/models/catalog/` and, when needed, `data/models/runtime_profiles/`.
- Declare aliases, source status, integration status, auth, checkpoint refs, license, and blockers.
- Add or reuse a public adapter target and a minimal validation command before claiming `integrated`.
- Keep API/GPU/checkpoint requirements out of quickstart defaults.
- Verify with `worldfoundry-eval zoo models --json` and the smallest documented model smoke run.

### Add Or Update A Benchmark

- Update `data/benchmarks/catalog/` and task YAML under `data/benchmarks/tasks/external/` when applicable.
- Declare dataset refs, auth, unsafe/gated data, official repo, simulator, metrics, and blockers.
- Add a contract adapter or normalizer before claiming `contract_ready`.
- Add official runtime evidence before claiming leaderboard validity.
- Verify with `worldfoundry-eval zoo benchmarks --json` and the documented readiness checks.

### Add Or Update A Metric

- Add deterministic validation examples for metric correctness.
- Document whether higher is better, output units, primary metric status, and leaderboard mapping.
- Ensure scorecards expose the metric under stable keys.

### Add Or Update Docs

- Keep first-run docs CPU-only, no-download, no-token, and no-GPU.
- Link complex environment, official runtime, and API paths to reference docs.
- Run the docs build and docs checks.

### Add Or Update A CLI Argument

- Update `docs/fumadocs/content/docs/reference/cli.mdx` when public behavior changes.
- Preserve JSON output compatibility or document the breaking change.
- Run `make docs-check` and document help, error handling, and no-heavy-import behavior.

## Synthesis Is Infer-Only

`worldfoundry/synthesis/**` packages upstream **inference** runtimes only. Do not add:

- training launchers, dataset builders, finetune scripts, or benchmark-only tooling
- demo media, generated outputs, checkpoints, or downloaded datasets
- notebooks, README files, or other documentation inside the synthesis tree

Put reusable demo assets under `worldfoundry/data/test_cases/`.

Before release or large runtime imports, run the public quality gates:

```bash
make lint
make docs-check
```
