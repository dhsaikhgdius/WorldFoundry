# AGENTS.md

## Cursor Cloud specific instructions

WorldFoundry is a CPU-friendly, pip-installable world-model infrastructure repo. The cloud VM has **no GPU**, so only catalog/CLI/UI/evaluation surfaces that don't require CUDA or downloaded checkpoints can be exercised end to end. Actual model inference (video/world generation) needs GPUs + checkpoints and cannot run here.

The startup update script already installs everything below (editable core, `.[ui]`, CPU torch, and the extra CPU deps the `test/eval_core` suite imports). You normally do NOT need to reinstall.

### Surfaces and how to run them (CPU-only)

| Surface | Command | Notes |
| --- | --- | --- |
| CLI / evaluation | `worldfoundry-eval --help`, `worldfoundry-eval zoo models --json`, `make cli-check` | Console scripts install to `~/.local/bin` (on PATH). `make cli-check` runs the evaluation runner end-to-end and writes a scorecard — a good CPU hello-world. |
| Lint / quality gates | `make lint`, `make compile-eval`, `make docs-check` | `make lint` runs ruff + compileall + shell + zoo catalog + runtime-registry checks. Prefer the smallest gate that covers the change. |
| Tests | `make test-eval-core` (release-gate CPU suite), `make test-training` | See "Tests" caveat below. Fresh machines: install CPU torch from the PyTorch CPU index, then the deps the suite imports (or `.[eval_core]` when that extra is available). |
| Studio workspace (browser app) | `bash scripts/workspace/run_workspace.sh --host 127.0.0.1 --port 7870 --max-jobs 2` | FastAPI app; serves at `http://127.0.0.1:7870/`. Catalog, Create Job form, Visualizers all work on CPU. Do not submit heavy inference jobs (no GPU). |
| Docs site (Next.js / Fumadocs) | `cd docs/fumadocs && npm run dev -- --port 8014` | Serves at `http://127.0.0.1:8014/docs`. `npm ci` deps are pre-installed. `predev`/`prebuild` shell out to Python for API/catalog codegen (`WF_DOCS_PYTHON` / `PYTHON` / `python3` / `python` when the portable launcher is present). |

### Non-obvious caveats

- **`python` symlink**: the `Makefile` and workspace scripts invoke `python` (not `python3`). The VM only ships `python3`, so a `/usr/local/bin/python -> /usr/bin/python3` symlink was created during setup and persists in the snapshot. If `python` is ever missing, either recreate it (`sudo ln -sf /usr/bin/python3 /usr/local/bin/python`) or pass `make PYTHON=python3 ...`.
- **Pinned dependency versions (do not "upgrade" blindly)**:
  - `ruff==0.12.7` — matches `.pre-commit-config.yaml`; newer ruff changes import-sort output and reports spurious lint diffs.
  - `gradio<6` (5.50) + `starlette<1.0` — the Studio `workspace_app` uses `app.add_event_handler`, which was removed in Starlette 1.0. The `[ui]` extra is unpinned on some branches, so a bare install can pull Gradio 6 / Starlette ≥1.0 and the Studio server fails on startup with `'FastAPI' object has no attribute 'add_event_handler'`. Keep Gradio pinned below 6.
  - `torch` is the **CPU** build (`--index-url https://download.pytorch.org/whl/cpu`). Several `test/eval_core` modules `import torch` at module top level (not guarded).
- **`make lint` / ruff**: with `ruff==0.12.7` on the current `RUFF_SOURCES` set, `make ruff-check` is expected to pass. Do not "upgrade" ruff to clear import-sort noise.
- **`make test-eval-core` runs (≈974 pass) but has many pre-existing assertion failures** on `main`. These are release-gate contract assertions about the catalog/runtime state (e.g. benchmark dataset refs, runner-kind mismatches, layer-boundary import checks) — not caused by the environment. Do not treat a fully green eval-core suite as a precondition.
- Generated run artifacts land under `tmp/` (gitignored). Studio checkpoint/data roots default to `~/.cache/worldfoundry/...` and are empty here (no downloads).
