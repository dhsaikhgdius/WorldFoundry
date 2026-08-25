# VBench++ in-tree runtime provenance

`vbench2_beta_i2v / vbench2_beta_long / vbench2_beta_trustworthiness/` vendors the official VBench++ evaluation code so that `run_vbench_plus_plus_official_runner.py` can execute the official metric code in-tree without requiring a separate upstream checkout.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/Vchitect/VBench |
| Revision | `45e79ec14e69a2187202c675d2dbce1a71843d53` |
| Verified on | 2026-08-25 (git blob-hash comparison, see below) |
| Paper | https://arxiv.org/abs/2411.13503 |
| Project page | https://vchitect.github.io/VBench-project/ |
| Upstream license | **Apache License 2.0** (`LICENSE`) |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/video/vbench-plus-plus.yaml` |

## Verification method

Every vendored file was classified by comparing its git blob hash (`git hash-object`) against the complete upstream tree at the revision above (`git ls-tree -r`). A blob-hash match proves byte-identity; files whose upstream path exists but whose content differs carry WorldFoundry adaptations. Spot-checked diffs show the adaptations replace hard-coded checkpoint/dataset paths and network download side effects with WorldFoundry environment-variable and checkpoint-root resolution while keeping the official metric formulas; the exact per-file diffs are reproducible with:

```bash
git clone https://github.com/Vchitect/VBench.git upstream && git -C upstream checkout 45e79ec14e69a2187202c675d2dbce1a71843d53
diff -ru upstream/<upstream path> vbench2_beta_i2v/<vendored path>
```

## File-level provenance

### Byte-identical to upstream (same path)

- `LICENSE`
- `vbench2_beta_i2v/crop_to_diff_ratio.py`
- `vbench2_beta_i2v/i2v_background.py`
- `vbench2_beta_i2v/i2v_subject.py`
- `vbench2_beta_long/aesthetic_quality.py`
- `vbench2_beta_long/appearance_style.py`
- `vbench2_beta_long/color.py`
- `vbench2_beta_long/configs/background_mapping_table.yaml`
- `vbench2_beta_long/configs/clip_length_0.5.yaml`
- `vbench2_beta_long/configs/clip_length_1.0.yaml`
- `vbench2_beta_long/configs/clip_length_mix.yaml`
- `vbench2_beta_long/configs/clip_length_short.yaml`
- `vbench2_beta_long/configs/slow_fast_params.yaml`
- `vbench2_beta_long/configs/subject_mapping_table.yaml`
- `vbench2_beta_long/dynamic_degree.py`
- `vbench2_beta_long/human_action.py`
- `vbench2_beta_long/imaging_quality.py`
- `vbench2_beta_long/motion_smoothness.py`
- `vbench2_beta_long/multiple_objects.py`
- `vbench2_beta_long/object_class.py`
- `vbench2_beta_long/overall_consistency.py`
- `vbench2_beta_long/scene.py`
- `vbench2_beta_long/spatial_relationship.py`
- `vbench2_beta_long/temporal_style.py`

### Adapted from upstream (same path)

- `vbench2_beta_i2v/__init__.py`
- `vbench2_beta_i2v/camera_motion.py`
- `vbench2_beta_i2v/utils.py`
- `vbench2_beta_long/__init__.py`
- `vbench2_beta_long/background_consistency.py`
- `vbench2_beta_long/eval_long.py`
- `vbench2_beta_long/static_filter.py`
- `vbench2_beta_long/subject_consistency.py`
- `vbench2_beta_long/temporal_flickering.py`
- `vbench2_beta_long/utils.py`
- `vbench2_beta_trustworthiness/__init__.py`
- `vbench2_beta_trustworthiness/culture_fairness.py`
- `vbench2_beta_trustworthiness/gender_bias.py`
- `vbench2_beta_trustworthiness/safety.py`
- `vbench2_beta_trustworthiness/skin_bias.py`
- `vbench2_beta_trustworthiness/utils.py`

### WorldFoundry-authored (no upstream counterpart)

- `__init__.py`
- `entrypoints/__init__.py`
- `entrypoints/i2v.py`
- `entrypoints/long.py`
- `entrypoints/trustworthiness.py`

## Assets that are deliberately *not* vendored

Model weights, judge checkpoints, and benchmark datasets are never stored in this tree; they are resolved at runtime through the environment variables and checkpoint roots documented in the catalog entry and the runner CLI.

## License status

The upstream repository publishes an Apache License 2.0 at the pinned revision, so redistribution of this directory is permitted with the license text retained. It is kept unchanged at `LICENSE`.
