# Upstream Provenance — MegaSaM

- upstream_url: `https://github.com/mega-sam/mega-sam`
- local_path: `worldfoundry/base_models/three_dimensions/slam/mega_sam_runtime`
- license: Apache-2.0 — the upstream `LICENSE` file is already retained in this
  directory (this was the only 1 of 20 audited vendored trees that carried one;
  do not add a second copy).
- source_commit: unknown — needs backfill. mega-sam has no model-catalog entry,
  and catalog `head_sha` fields elsewhere are `git ls-remote` HEAD values
  captured at catalog-validation time (see
  `worldfoundry/data/models/catalog/video/framepack.yaml:52`), not vendored
  snapshot commits.
- fork_status: vendored as-is from upstream, including upstream's own embedded
  forks of three other projects (see below).

## Embedded forked copies — do NOT merge with the canonical trees

MegaSaM upstream itself vendors modified copies of three projects. The code
review (`plan/code_review/11_vendored_integration.md` [VI-10]) diffed them
against the canonical WorldFoundry copies (after normalizing machine-injected
docstrings) and confirmed the differences are real, i.e. these are upstream
mega-sam's own modifications:

| Embedded copy | Canonical tree | Divergence |
|---|---|---|
| `base/droid_slam/` (27 py) | `three_dimensions/slam/droid_slam` (23 py) | 17 of 19 shared files differ |
| `UniDepth/unidepth/` (40 py) | `three_dimensions/depth/unidepth` (14 py) | all 10 shared files differ |
| `Depth-Anything/depth_anything/` (3 py) | `three_dimensions/depth/depth_anything/depth_anything_v1` (8 py) | all 3 shared files differ |

**Do not "deduplicate" these embedded copies against the canonical trees** —
the deltas are mega-sam's functional modifications and merging would break its
SLAM pipeline. Conversely, bug/security fixes applied to the canonical
DROID-SLAM / UniDepth / Depth-Anything trees do not automatically reach these
copies; they must be evaluated here separately.

Embedded components' upstream identities and licenses (as recorded elsewhere
in this repository):

- DROID-SLAM: `https://github.com/princeton-vl/DROID-SLAM`, BSD-3-Clause
  (see `worldfoundry/base_models/three_dimensions/general_3d/vipe/THIRD_PARTY_LICENSES.md` lines 3-8)
- UniDepth: `https://github.com/lpiccinelli-eth/UniDepth`, CC-BY-NC-4.0
  (same file, lines 522-527)
- Depth-Anything (V1): `https://github.com/LiheYoung/Depth-Anything`, Apache-2.0
  (see `worldfoundry/data/models/catalog/three_d_four_d/depth-anything-v1.yaml` lines 10-11)

Long-term, the review recommends extracting mega-sam's modifications as
patches/variants over the canonical trees instead of keeping full forked
copies ([VI-10] recommendation).
