# LICENSE MISSING — Gaussian-Splatting License (Inria / MPII)

This directory vendors training/rendering code from the official 3D Gaussian
Splatting implementation (`arguments/`, `gaussian_renderer/`, `scene/`, `utils/`).

- upstream_url: `https://github.com/graphdeco-inria/gaussian-splatting`
- copyright: Inria (GRAPHDECO research group) and Max-Planck-Institut fuer Informatik (MPII)
- license: "Gaussian-Splatting License" — non-commercial research/evaluation use only
- source_commit: unknown — needs backfill (no snapshot commit was recorded at vendor time)

In-repo evidence for the license identification:

- Source headers in this tree, e.g. `scene/gaussian_model.py` lines 1-10:
  "Copyright (C) 2023, Inria / GRAPHDECO research group ... This software is
  free for non-commercial, research and evaluation use under the terms of the
  LICENSE.md file."
- `thirdparty/THIRD_PARTY_LICENSES.md` lines 10-12 records the same license for
  the related rasterization forks: "Inria non-commercial research license
  (Gaussian-Splatting License) ... commercial use requires explicit consent
  from Inria (stip-sophia.transfert@inria.fr)."

## Why this file exists instead of the license text

The verbatim Gaussian-Splatting License text is NOT currently present anywhere
in this repository (verified 2026-08-14: `rg "Gaussian-Splatting License"` only
matches summaries, and the distinctive full-text phrases of the upstream
LICENSE.md match nothing). Legal text must not be approximated or retyped from
memory. The upstream license requires its text to accompany the code, so the
full text must be copied VERBATIM from the official source and saved as
`LICENSE.md` in this directory:

    https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md

Until that file is added, treat this tree as: research and evaluation use only;
commercial use requires prior written consent from Inria.

Related review finding: `plan/code_review/11_vendored_integration.md` [VI-19].
