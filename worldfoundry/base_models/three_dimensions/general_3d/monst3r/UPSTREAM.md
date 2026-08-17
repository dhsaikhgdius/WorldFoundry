# Upstream Provenance — MonST3R

- upstream_url: `https://github.com/Junyi42/monst3r`
- local_path: `worldfoundry/base_models/three_dimensions/general_3d/monst3r`
- license: CC-BY-NC-SA-4.0 (see `LICENSE` in this directory; recorded in
  `worldfoundry/data/models/catalog/three_d_four_d/monst3r.yaml` lines 7 and 15)
- source_commit: unknown — needs backfill. The model catalog has no SHA for
  monst3r; catalog `head_sha` fields elsewhere are `git ls-remote` HEAD values
  captured at catalog-validation time (see
  `worldfoundry/data/models/catalog/video/framepack.yaml:52`), not vendored
  snapshot commits, so none can be honestly recorded here.
- fork_status: **modified relative to its own embedded dependencies** — see below.

## Embedded `dust3r/` is a FORK of DUSt3R — do not deduplicate blindly

MonST3R upstream itself vendors a modified copy of DUSt3R. The code review
(`plan/code_review/11_vendored_integration.md` [VI-8]) compared this copy
against the canonical tree at
`worldfoundry/base_models/three_dimensions/general_3d/dust3r/dust3r` (37 py;
this copy has 35 py, 32 shared relative paths) and confirmed that after
normalizing the machine-injected docstrings, 27 shared files are identical and
**exactly 5 files carry real MonST3R fork changes**:

1. `dust3r/cloud_opt/base_opt.py`
2. `dust3r/cloud_opt/optimizer.py`
3. `dust3r/model.py`
4. `dust3r/utils/misc.py`
5. `dust3r/utils/vo_eval.py`

These changes carry no patch markers. **Do not blindly deduplicate this tree
against canonical `general_3d/dust3r`** — replacing these 5 files with the
canonical versions silently breaks MonST3R's motion-aware optimization.
Long-term, the review recommends turning the 5-file delta into a variant
overlay on the canonical tree (see [VI-8] recommendation).

## Other embedded third-party code

- `croco/`: copy of Naver CroCo (CC-BY-NC-SA-4.0 per source headers). Per
  review [VI-8], its 9 shared files are byte-identical to canonical
  `general_3d/dust3r/croco` after docstring normalization (pure duplicate).
- `third_party/RAFT/`: RAFT optical flow, vendored by MonST3R upstream.

## Namespace hazard

This tree exposes the top-level module names `dust3r`/`croco` via sys.path
manipulation, colliding with the canonical copies and the older copy under
`stable_virtual_camera` (review [VI-2]/[VI-14]). Which copy wins depends on
import order.
