# VBench in-tree runtime provenance

`vbench/` vendors the official VBench evaluation code so that `run_vbench_official_runner.py` can execute the official metric code in-tree without requiring a separate upstream checkout.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/Vchitect/VBench |
| Revision | `45e79ec14e69a2187202c675d2dbce1a71843d53` |
| Verified on | 2026-08-25 (git blob-hash comparison, see below) |
| Paper | https://arxiv.org/abs/2311.17982 |
| Project page | https://vchitect.github.io/VBench-project/ |
| Upstream license | **Apache License 2.0** (`vbench/LICENSE`) |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/video/vbench.yaml` |

## Verification method

Every vendored file was classified by comparing its git blob hash (`git hash-object`) against the complete upstream tree at the revision above (`git ls-tree -r`). A blob-hash match proves byte-identity; files whose upstream path exists but whose content differs carry WorldFoundry adaptations. Spot-checked diffs show the adaptations replace hard-coded checkpoint/dataset paths and network download side effects with WorldFoundry environment-variable and checkpoint-root resolution while keeping the official metric formulas; the exact per-file diffs are reproducible with:

```bash
git clone https://github.com/Vchitect/VBench.git upstream && git -C upstream checkout 45e79ec14e69a2187202c675d2dbce1a71843d53
diff -ru upstream/<upstream path> vbench/<vendored path>
```

## File-level provenance

### Byte-identical to upstream (relocated)

- `vbench/LICENSE` (upstream `LICENSE`)

### Adapted from upstream (same path)

- `vbench/__init__.py`
- `vbench/aesthetic_quality.py`
- `vbench/appearance_style.py`
- `vbench/background_consistency.py`
- `vbench/color.py`
- `vbench/dynamic_degree.py`
- `vbench/human_action.py`
- `vbench/imaging_quality.py`
- `vbench/motion_smoothness.py`
- `vbench/multiple_objects.py`
- `vbench/object_class.py`
- `vbench/overall_consistency.py`
- `vbench/scene.py`
- `vbench/spatial_relationship.py`
- `vbench/subject_consistency.py`
- `vbench/temporal_flickering.py`
- `vbench/temporal_style.py`
- `vbench/utils.py`

### WorldFoundry-authored (no upstream counterpart)

- `__init__.py`
- `entrypoints/__init__.py`
- `entrypoints/base.py`

## Assets that are deliberately *not* vendored

Model weights, judge checkpoints, and benchmark datasets are never stored in this tree; they are resolved at runtime through the environment variables and checkpoint roots documented in the catalog entry and the runner CLI.

## License status

The upstream repository publishes an Apache License 2.0 at the pinned revision, so redistribution of this directory is permitted with the license text retained. It is kept unchanged at `vbench/LICENSE`.
