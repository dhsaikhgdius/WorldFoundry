# WBench in-tree runtime provenance

`wbench/` vendors the official WBench evaluation code so that `run_wbench_official_runner.py` can execute the official metric code in-tree without requiring a separate upstream checkout.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/meituan-longcat/WBench |
| Revision | `f07c4fa4a1ea7873b97293e7a4092e6f6dd356b6` |
| Verified on | 2026-08-25 (git blob-hash comparison, see below) |
| Paper | https://arxiv.org/abs/2605.25874 |
| Project page | https://meituan-longcat.github.io/WBench/ |
| Upstream license | **MIT License** (`wbench/LICENSE`) |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/video/wbench.yaml` |

> The catalog entry does not pin an upstream revision; the verification below was performed against the upstream default-branch HEAD on the verification date.

## Verification method

Every vendored file was classified by comparing its git blob hash (`git hash-object`) against the complete upstream tree at the revision above (`git ls-tree -r`). A blob-hash match proves byte-identity; files whose upstream path exists but whose content differs carry WorldFoundry adaptations. Spot-checked diffs show the adaptations replace hard-coded checkpoint/dataset paths and network download side effects with WorldFoundry environment-variable and checkpoint-root resolution while keeping the official metric formulas; the exact per-file diffs are reproducible with:

```bash
git clone https://github.com/meituan-longcat/WBench.git upstream && git -C upstream checkout f07c4fa4a1ea7873b97293e7a4092e6f6dd356b6
diff -ru upstream/<upstream path> wbench/<vendored path>
```

## File-level provenance

### Byte-identical to upstream (same path)

- `LICENSE`
- `src/__init__.py`
- `src/config.yaml`
- `src/metrics/__init__.py`
- `src/metrics/base.py`
- `src/metrics/consistency/__init__.py`
- `src/metrics/consistency/perspective_consistency.py`
- `src/metrics/consistency/reconstruction_consistency.py`
- `src/metrics/consistency/spatial_consistency.py`
- `src/metrics/interaction/__init__.py`
- `src/metrics/interaction/vlm_interaction.py`
- `src/metrics/physical/__init__.py`
- `src/metrics/physical/causal_fidelity.py`
- `src/metrics/setting_adherence/__init__.py`
- `src/metrics/setting_adherence/scene_adherence.py`
- `src/metrics/setting_adherence/subject_adherence.py`
- `src/metrics/video_quality/__init__.py`
- `src/metrics/video_quality/temporal_flickering.py`
- `src/metrics/vlm/__init__.py`
- `src/utils/__init__.py`
- `src/utils/case_loader.py`
- `src/utils/turn_splitter.py`
- `src/utils/video_utils.py`

### Adapted from upstream (same path)

- `main.py`
- `src/compat.py`
- `src/evaluate.py`
- `src/metrics/consistency/background_consistency.py`
- `src/metrics/consistency/segment_continuity.py`
- `src/metrics/consistency/subject_consistency.py`
- `src/metrics/interaction/navigation_trajectory.py`
- `src/metrics/physical/visual_plausibility.py`
- `src/metrics/video_quality/aesthetic_quality.py`
- `src/metrics/video_quality/dynamic_degree.py`
- `src/metrics/video_quality/evaluator.py`
- `src/metrics/video_quality/hpsv3_quality.py`
- `src/metrics/video_quality/imaging_quality.py`
- `src/metrics/video_quality/motion_smoothness.py`
- `src/metrics/vlm/vlm_evaluator.py`
- `src/metrics/weight_utils.py`
- `tools/run_da3_depth.py`
- `tools/run_megasam.py`
- `tools/run_sam2_track.py`
- `tools/run_visual_plausibility.py`

## Assets that are deliberately *not* vendored

Model weights, judge checkpoints, and benchmark datasets are never stored in this tree; they are resolved at runtime through the environment variables and checkpoint roots documented in the catalog entry and the runner CLI.

## License status

The upstream repository publishes a MIT License at the pinned revision, so redistribution of this directory is permitted with the license text retained. It is kept unchanged at `wbench/LICENSE`.
