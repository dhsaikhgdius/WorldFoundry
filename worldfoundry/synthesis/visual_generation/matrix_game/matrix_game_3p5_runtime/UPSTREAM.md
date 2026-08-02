# Matrix-Game 3.5 upstream provenance

This directory is an infer-only, in-tree adaptation of
[`Riemann-Dynamics/Matrix-Game-3.5`](https://github.com/Riemann-Dynamics/Matrix-Game-3.5)
at revision `6c94dd787659aa19fac22581cd8bea54e65d813f`. It retains the
single-process inference entrypoint, Mosaic rollout logic, DA3 index
construction, frustum retrieval, and subject-reference memory needed by the
released base checkpoints.

The copied runtime is distributed under the upstream Apache-2.0 license in
`LICENSE`. Local changes preserve that license and are documented below.

## External checkpoints

No checkpoint is vendored. The high-level runtime resolves staged local assets
before launching its offline inference subprocess; the shared native resolver
can materialize missing Hub files for direct recipe construction. Revisions are pinned:

- `RiemannDynamics/Matrix-Game-3.5-Base` at
  `40c172355efa32d4e2a44076569e310807788f8a` (Apache-2.0);
- `Wan-AI/Wan2.2-TI2V-5B` at
  `921dbaf3f1674a56f47e83fb80a34bac8a8f203e` (Apache-2.0);
- `depth-anything/DA3NESTED-GIANT-LARGE-1.1` at
  `b2359bdf726fb44ef62acca04d629dcf158053e7` (CC-BY-NC-4.0).

The DA3 dependency therefore has a separate non-commercial checkpoint license.
Users must satisfy every dependency's upstream terms.

## Shared code boundaries

Matrix uses WorldFoundry's canonical native diffusion contracts, recipes,
checkpoint resolver, core-backed model loader, standard runner, Wan2.2 VAE,
UMT5 encoder, and numerical scheduler. Matrix-specific PRoPE, DiT, and Mosaic
tensor computation live under
`worldfoundry/base_models/diffusion_model/models/networks/matrix_game_3p5`.
There is no Matrix-owned pipeline, loader, scheduler, or runner.

Depth Anything 3 is reused from
`worldfoundry/base_models/three_dimensions/depth/depth_anything/depth_anything_v3`.
The former upstream `official/diffsynth`, its copied validation driver, and the
bundled `third_party/depth-anything-3` trees are intentionally excluded.
Training loops, optimizers, SVI buffers,
web assets, samples, checkpoints, and Git metadata are also excluded.

Inference recipes are package data, not runtime source. The first- and
third-person profiles extend
`worldfoundry/data/models/runtime/configs/matrix_game_3p5/infer_common.yaml`.

## Local integration changes

- Imports use stable `worldfoundry.*` paths.
- First- and third-person checkpoints are separate immutable WorldFoundry model
  IDs instead of a mutable public person-mode switch.
- The runner launches as a single Python subprocess without a nested
  `accelerate launch` process.
- Local Wan shards, tokenizer, Matrix DiT, and DA3 paths are resolved before
  execution; the child process is forced offline.
- Transient workspace/cache data is redirected through
  `WORLDFOUNDRY_MATRIX_GAME_3P5_CACHE_DIR`.
- The upstream training facade and its loading compatibility helpers were removed;
  the runtime module is a plain `torch.nn.Module` assembled by the native recipe.
- The upstream validation driver, GT/candidate panels, temporary section
  videos, latent snapshots, online dynamic-object filtering, neighbor-window
  experiment, and alternate VGGT registration branch were removed;
  `mosaic/rollout.py` contains only the inference rollout path.
- Runtime profiles use inference-native option names; no `validation_*` or
  held-out-dataset argument is translated by the entrypoint.
- Dataset and module factories accept explicit subclasses for memory research;
  the supported seams are described in `RESEARCH.md`.
