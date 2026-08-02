# MIND in-tree runtime provenance

`mind/` is a vendored copy of the official MIND evaluator so that
`run_mind_official_runner.py --run-official` executes the official metric code
without requiring a separate checkout.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/CSU-JPG/MIND |
| Revision | `219f458bbfbc3204e848bb6dd1f45363d4e34730` |
| Vendored on | 2026-07-26 |
| Paper | https://arxiv.org/abs/2602.08025 |
| Project page | https://csu-jpg.github.io/MIND.github.io/ |
| Dataset | https://huggingface.co/datasets/CSU-JPG/MIND |
| Upstream license | **MIT License**, `Copyright (c) 2026 fgloris` (`mind/LICENSE`) |

## Modifications

None. Every vendored file is byte-identical to upstream at the pinned revision
except for the exclusions below. WorldFoundry adapts the runtime purely through
CLI arguments, environment variables, and the process working directory, so the
vendored copy can be refreshed by re-running the same export.

Two adaptations are implemented in the WorldFoundry runner rather than in this
tree, precisely so the vendored source stays untouched:

- `MIND_CACHE_DIR` is pointed at `<output-dir>/mind_work/cache`, because
  `src/utils/utils.py` otherwise caches cropped videos and downloaded metric
  weights under `~/.cache/mind`.
- `src/utils/vipe_utils.py` calls `vipe_to_colmap(out_dir, Path("vipe"))`, which
  resolves `vipe` relative to the process working directory. The runner runs the
  entry point with `cwd=<output-dir>/mind_work` and symlinks a caller-supplied
  `--vipe-repo` to `<work-dir>/vipe`.

## Exclusions

- `assets/` — 1.3 MB of README/paper figures (`Logo.png`, `Overview.jpg`,
  `Dataset.jpg`) with no role in evaluation.
- `.gitmodules` — upstream submodule pointer for
  `https://github.com/nv-tlabs/vipe.git`. The submodule content is not vendored,
  so keeping the pointer would be misleading; ViPE is supplied at runtime
  through `--vipe-repo` / `WORLDFOUNDRY_MIND_VIPE_REPO`.
- `.git/` — upstream VCS metadata.

`README.md`, `LICENSE`, `envs/base.yml`, `envs/requirements.txt`, and the whole
`src/` tree (`process.py`, `metrics/`, `utils/`) are vendored verbatim.

## License status

The upstream repository publishes an MIT `LICENSE` at the pinned revision, so
redistribution of this directory is permitted with the copyright notice
retained. `mind/LICENSE` is kept in place unchanged and is recorded as
`license.status: mit` in
`worldfoundry/data/benchmarks/catalog/video/mind.yaml`.

## Assets that are deliberately *not* vendored

These are separate projects or large artifacts with their own licenses and are
resolved at runtime:

- **ViPE** (`https://github.com/nv-tlabs/vipe`) — required by the `action`
  metric for pose estimation and by `scripts/vipe_to_colmap.py`. Install the
  `vipe` CLI and pass the checkout with `--vipe-repo`.
- **DINOv3 weights** (`dinov3_vitb16`) — required by the `dino` metric. Supplied
  through `--dino-path`, or `<weights-dir>/dinov3_vitb16` where `--weights-dir`
  defaults to `checkpoint_root_path("mind")`. Never inside this source tree.
- **MUSIQ-SPAQ, CLIP ViT-L/14, and LAION aesthetic predictor weights** —
  `src/utils/utils.py` downloads these on demand into `MIND_CACHE_DIR`, which
  the runner pins under `--output-dir`.
- **MIND-Data** — the 250-clip 1080p/24 FPS benchmark split plus training data
  is a Hugging Face dataset (`CSU-JPG/MIND`) passed with `--gt-root`.
- **MIND-World (1.3B) baseline** — not released at the pinned revision.
