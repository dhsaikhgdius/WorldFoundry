# Upstream Provenance — HunyuanWorld-Mirror (copy 1 of 2 in this tree)

- upstream_url: `https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror`
- local_path: `worldfoundry/base_models/three_dimensions/point_clouds/hunyuan_mirror`
- license: `other:tencent-hunyuanworld-community` (Tencent HunyuanWorld
  community license; recorded in
  `worldfoundry/data/models/catalog/world_models/hunyuanworld-mirror.yaml` line 19)
- source_commit: unknown — needs backfill. The catalog entry confirms the
  upstream repo only via `git ls-remote --heads` at catalog-validation time
  (same yaml, lines 28-29); that is a remote-HEAD check, not a vendored
  snapshot commit, so no SHA can be honestly recorded here.
- fork_status: vendored inference-only model code; local modification status
  unrecorded at import time.

## Two HunyuanWorld-Mirror copies coexist — cross-reference

This repository carries TWO versions of the WorldMirror model code side by
side (review `plan/code_review/11_vendored_integration.md` [VI-11]: 17 shared
relative paths, 14 differ after docstring normalization — a real version
difference, not formatting noise):

- **This copy (`hunyuan_mirror/`)** is the one wired to the catalog model id
  `hunyuanworld-mirror`: it is consumed by
  `worldfoundry/pipelines/hunyuan_world/pipeline_hunyuan_mirror.py`
  (HunyuanMirrorPipeline, the catalog `pipeline_binding`),
  `worldfoundry/operators/hunyuan_mirror_operator.py`, and
  `worldfoundry/representations/point_clouds_generation/hunyuan_world/hunyuan_world_mirror_representation.py`.
- **Sibling copy `../hyworldmirror_2p0/`** is the "WorldMirror 2.0"
  reconstruction component of the HY-World 2.0 stack — see its own
  `UPSTREAM.md`.

Which upstream tag/commit each copy corresponds to was not recorded at vendor
time (unknown — needs backfill).

## Deletion/merge hazard

`../hyworldmirror_2p0/models/models/worldmirror.py` imports utilities FROM
this tree (`...point_clouds.hunyuan_mirror.models.utils.camera_utils`), so the
two copies are not independent: removing or "merging" this directory breaks
the 2.0 copy. The review recommends converging the two into a single
multi-version package, or at minimum keeping both UPSTREAM files accurate
([VI-11] recommendation).
