# WorldOlympiad in-tree runtime provenance

`worldolympiad/` is a vendored copy of the official WorldOlympiad evaluator so
that `run_worldolympiad_official_runner.py --run-official` executes the official
metric code without requiring a separate checkout.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/alibaba-damo-academy/WorldOlympiad |
| Revision | `bb2e240918d15d822250a33b3504672be545d10d` |
| Vendored on | 2026-07-26 |
| Paper | https://arxiv.org/abs/2606.11129 |
| Project page | https://alibaba-damo-academy.github.io/WorldOlympiad |
| Upstream license | **No LICENSE file is published at this revision.** |

## Modifications

None. The tree is byte-identical to upstream except for the exclusions below.
WorldFoundry adapts the runtime through CLI arguments and environment variables
only — including PYTHONPATH entries that reuse the Depth Anything 3 and CLIP
code from `worldfoundry/base_models` and env vars that point at registered
weights — so the vendored copy can be refreshed by re-running the same export.

## Exclusions

- `figure/` — 38 MB of paper and README images with no role in evaluation.
- `.git/`, `.gitignore` — upstream VCS metadata.

## License status

The upstream repository publishes no LICENSE file at the pinned revision, so no
redistribution terms are stated. This is recorded as
`license.status: unlicensed_upstream` in
`worldfoundry/data/benchmarks/catalog/video/worldolympiad.yaml`. Confirm terms
with the upstream authors before redistributing this directory.

## Assets that are deliberately *not* vendored

These are separate projects with their own licenses and are resolved at runtime:

- **Depth Anything 3** source tree — by default the runner exposes the vendored
  `depth_anything_v3` package from `worldfoundry/base_models` as `depth_anything_3`
  on PYTHONPATH (a symlink shim built under the run's output directory), so the
  geometry track imports `depth_anything_3` with no external checkout. Point
  `WORLDFOUNDRY_WORLDOLYMPIAD_DA3_SRC` (or `--da3-src`) at an external `src`
  directory to override it.
- **Model weights** — DA3, SAM3, QwenVL, and CLIP weights resolve from
  `worldfoundry/base_models` by default; `--weights-dir`
  (`WORLDFOUNDRY_WORLDOLYMPIAD_WEIGHTS_DIR`) overrides them with a caller-supplied
  weights root. Weights never live inside this source tree.
- **CLIP code** — the runtime's `import clip` resolves to the vendored
  `openai_clip_runtime` package in `worldfoundry/base_models` via PYTHONPATH, so a
  separate pip `clip` install is not required; ViT-B/32 auto-downloads into the
  managed CLIP cache on first use.
- **Benchmark videos and prompts** — the 1,000-video split is a Hugging Face
  dataset (`ziplab/WorldOlympiad`).
