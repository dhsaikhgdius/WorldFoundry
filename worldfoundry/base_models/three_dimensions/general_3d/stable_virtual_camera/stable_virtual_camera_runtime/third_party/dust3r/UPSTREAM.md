# Upstream Provenance — DUSt3R (older snapshot embedded in Stable Virtual Camera)

- upstream_url: `https://github.com/naver/dust3r`
- local_path: `worldfoundry/base_models/three_dimensions/general_3d/stable_virtual_camera/stable_virtual_camera_runtime/third_party/dust3r`
- license: CC-BY-NC-SA-4.0 (see `LICENSE` in this directory; source headers in
  this tree state "Licensed under CC BY-NC-SA 4.0", and the DUSt3R catalog
  entry `worldfoundry/data/models/catalog/three_d_four_d/dust3r.yaml` line 18
  records the same). Note this license is separate from the Stable Virtual
  Camera runtime that embeds it (SVC itself is "Stability AI Non-Commercial
  License Agreement" per
  `worldfoundry/data/models/catalog/three_d_four_d/stable-virtual-camera.yaml` line 21).
- vendored_via: Stability AI's `https://github.com/Stability-AI/stable-virtual-camera`
  ships this copy in its `third_party/`; WorldFoundry inherited it when
  vendoring the SVC runtime.
- source_commit: unknown — needs backfill. The dust3r catalog entry carries no
  SHA, and catalog `head_sha` fields elsewhere are `git ls-remote` HEAD values
  at catalog-validation time (`worldfoundry/data/models/catalog/video/framepack.yaml:52`),
  not snapshot commits.
- fork_status: **different (older) upstream version** than the canonical tree.

## This is NOT the same version as canonical `general_3d/dust3r`

The code review (`plan/code_review/11_vendored_integration.md` [VI-8])
compared this copy (25 py) against the canonical
`worldfoundry/base_models/three_dimensions/general_3d/dust3r/dust3r` (37 py):
of 24 shared relative paths, **only 2 files are identical** after docstring
normalization — this is another upstream release, not a reformatted duplicate.
The embedded `croco/` here is likewise a different version from canonical
`general_3d/dust3r/croco`. Do not "upgrade" or deduplicate this copy against
the canonical tree without validating the SVC pipeline; SVC depends on this
older API surface.

## Top-level name collision hazard

This tree exposes the top-level module names `dust3r` and `croco` (via
`dust3r/utils/path_to_croco.py`-style sys.path insertion), which collide with
the canonical `general_3d/dust3r` copy and the MonST3R embedded copy (review
[VI-2]/[VI-14]). In one process, whichever wrapper runs its path hack last —
combined with `sys.modules` first-import caching — decides which version a
bare `import dust3r` resolves to. If both SVC and canonical DUSt3R paths are
active in one process, the wrong (older/newer) version may be silently loaded.
The review recommends renaming the exposed top-level package (e.g.
`dust3r_svc`) if this copy must be kept ([VI-8] recommendation).
