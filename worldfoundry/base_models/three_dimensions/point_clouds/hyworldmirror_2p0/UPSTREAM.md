# Upstream Provenance — WorldMirror 2.0 (copy 2 of 2 in this tree)

- upstream_url: unknown exact source — needs backfill. Two official candidates
  are recorded in the model catalog; which repository/revision this snapshot
  was taken from was not recorded at vendor time:
  - `https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror` (the WorldMirror
    model repo; `worldfoundry/data/models/catalog/world_models/hunyuanworld-mirror.yaml` line 30)
  - `https://github.com/Tencent-Hunyuan/HY-World-2.0` (the HY-World 2.0 stack
    whose "WorldMirror 2.0 reconstruction component" this copy serves;
    `worldfoundry/data/models/catalog/world_models/hy-world-2.0.yaml` lines 16-17, 34, 43)
- local_path: `worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0`
- license: `other:tencent-hunyuanworld-community` (Tencent HunyuanWorld
  community license; both catalog entries above record this, at
  `hunyuanworld-mirror.yaml:19` and `hy-world-2.0.yaml:22`)
- source_commit: unknown — needs backfill. Catalog github confirmations are
  `git ls-remote` checks at catalog-validation time, not snapshot commits.
- fork_status: vendored inference-only model code; local modification status
  unrecorded at import time.

## Two HunyuanWorld-Mirror copies coexist — cross-reference

This directory and the sibling `../hunyuan_mirror/` are two versions of the
same model family kept side by side (review
`plan/code_review/11_vendored_integration.md` [VI-11]: 17 shared relative
paths, 14 differ after docstring normalization — a real version difference).

- **This copy (`hyworldmirror_2p0/`, class `WorldMirror`)** is consumed by the
  HY-World 2.0 runtime stack:
  `worldfoundry/representations/point_clouds_generation/hunyuan_world/hy_world_2p0/worldmirror_runtime.py`
  and `worldfoundry/synthesis/visual_generation/neoverse/worldfoundry_runtime.py`.
- **Sibling copy `../hunyuan_mirror/`** is the one bound to the catalog model
  id `hunyuanworld-mirror` (HunyuanMirrorPipeline) — see its `UPSTREAM.md`.

## Dependency on the sibling copy

`models/models/worldmirror.py` in this tree imports
`...point_clouds.hunyuan_mirror.models.utils.camera_utils` — i.e. this copy
depends on the sibling copy at runtime. Neither directory can be removed or
"deduplicated" in isolation. The review recommends converging both into a
single multi-version package ([VI-11] recommendation).
