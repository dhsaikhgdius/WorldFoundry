# MiraBench (MiraData) in-tree runtime provenance

`mirabench/` vendors the official MiraBench (MiraData) evaluation code so that `run_mirabench_official_runner.py` can execute the official metric code in-tree without requiring a separate upstream checkout.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/mira-space/MiraData |
| Revision | `7ad05795dd74acdc6b364222a9fa885a94b0c1a2` |
| Verified on | 2026-08-25 (git blob-hash comparison, see below) |
| Paper | https://arxiv.org/abs/2407.06358 |
| Project page | https://mira-space.github.io/ |
| Upstream license | **GNU General Public License v3.0** (`mirabench/LICENSE`) |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/video/mirabench.yaml` |

## Verification method

Every vendored file was classified by comparing its git blob hash (`git hash-object`) against the complete upstream tree at the revision above (`git ls-tree -r`). A blob-hash match proves byte-identity; files whose upstream path exists but whose content differs carry WorldFoundry adaptations. Spot-checked diffs show the adaptations replace hard-coded checkpoint/dataset paths and network download side effects with WorldFoundry environment-variable and checkpoint-root resolution while keeping the official metric formulas; the exact per-file diffs are reproducible with:

```bash
git clone https://github.com/mira-space/MiraData.git upstream && git -C upstream checkout 7ad05795dd74acdc6b364222a9fa885a94b0c1a2
diff -ru upstream/<upstream path> mirabench/<vendored path>
```

## File-level provenance

### Byte-identical to upstream (same path)

- `LICENSE`
- `evaluation/consistency_3D.py`
- `evaluation/fid.py`
- `evaluation/inception.py`
- `evaluation/pytorch_i3d.py`

### Adapted from upstream (same path)

- `calculate_score.py`
- `evaluation/__init__.py`
- `evaluation/aesthetic_quality.py`
- `evaluation/dynamic_degree.py`
- `evaluation/fvd.py`
- `evaluation/imaging_quality.py`
- `evaluation/motion_smoothness.py`
- `evaluation/temporal_clip_consistency.py`
- `evaluation/temporal_dino_consistency.py`
- `evaluation/text_video_consistency.py`

## Assets that are deliberately *not* vendored

Model weights, judge checkpoints, and benchmark datasets are never stored in this tree; they are resolved at runtime through the environment variables and checkpoint roots documented in the catalog entry and the runner CLI.

## License status

The upstream repository publishes a GNU General Public License v3.0 at the pinned revision, so redistribution of this directory is permitted with the license text retained. It is kept unchanged at `mirabench/LICENSE`.
