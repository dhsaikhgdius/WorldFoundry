# VBench-2.0 in-tree runtime provenance

`vbench2/` vendors the official VBench-2.0 evaluation code so that `run_vbench_2_0_official_runner.py` can execute the official metric code in-tree without requiring a separate upstream checkout.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/Vchitect/VBench (subdirectory `VBench-2.0/vbench2`) |
| Revision | `45e79ec14e69a2187202c675d2dbce1a71843d53` |
| Verified on | 2026-08-25 (git blob-hash comparison, see below) |
| Paper | https://arxiv.org/abs/2503.21755 |
| Project page | https://vchitect.github.io/VBench-2.0-project/ |
| Upstream license | **Apache License 2.0** (`vbench2/LICENSE`) |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/video/vbench-2.0.yaml` |

## Verification method

Every vendored file was classified by comparing its git blob hash (`git hash-object`) against the complete upstream tree at the revision above (`git ls-tree -r`). A blob-hash match proves byte-identity; files whose upstream path exists but whose content differs carry WorldFoundry adaptations. Spot-checked diffs show the adaptations replace hard-coded checkpoint/dataset paths and network download side effects with WorldFoundry environment-variable and checkpoint-root resolution while keeping the official metric formulas; the exact per-file diffs are reproducible with:

```bash
git clone https://github.com/Vchitect/VBench.git upstream && git -C upstream checkout 45e79ec14e69a2187202c675d2dbce1a71843d53
diff -ru upstream/<upstream path> vbench2/<vendored path>
```

## File-level provenance

### Byte-identical to upstream (same path)

- `vbench2/hack_registry.py`

### Byte-identical to upstream (relocated)

- `vbench2/LICENSE` (upstream `LICENSE`)

### Adapted from upstream (same path)

- `vbench2/__init__.py`
- `vbench2/camera_motion.py`
- `vbench2/complex_landscape.py`
- `vbench2/complex_plot.py`
- `vbench2/composition.py`
- `vbench2/diversity.py`
- `vbench2/dynamic_attribute.py`
- `vbench2/dynamic_spatial_relationship.py`
- `vbench2/human_anatomy.py`
- `vbench2/human_clothes.py`
- `vbench2/human_identity.py`
- `vbench2/human_interaction.py`
- `vbench2/instance_preservation.py`
- `vbench2/material.py`
- `vbench2/mechanics.py`
- `vbench2/motion_order_understanding.py`
- `vbench2/motion_rationality.py`
- `vbench2/multi_view_consistency.py`
- `vbench2/thermotics.py`
- `vbench2/utils.py`

### Adapted from upstream (relocated)

- `entrypoints/vbench2.py` (upstream `VBench-2.0/vbench2/cli/vbench2.py`)
- `vbench2/dense_match/core.py` (upstream `VBench-2.0/vbench2/third_party/Instance_detector/swift/llm/dataset/preprocessor/core.py`)

### WorldFoundry-authored (no upstream counterpart)

- `__init__.py`
- `entrypoints/__init__.py`
- `vbench2/dense_match/__init__.py`
- `vbench2/dense_match/patch_auto_evaluate.py`

## Assets that are deliberately *not* vendored

Model weights, judge checkpoints, and benchmark datasets are never stored in this tree; they are resolved at runtime through the environment variables and checkpoint roots documented in the catalog entry and the runner CLI.

## License status

The upstream repository publishes an Apache License 2.0 at the pinned revision, so redistribution of this directory is permitted with the license text retained. It is kept unchanged at `vbench2/LICENSE`.
