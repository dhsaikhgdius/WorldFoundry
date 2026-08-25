# FETV in-tree runtime provenance

`fetv_eval/` vendors the official FETV evaluation code so that `run_fetv_official_runner.py` can execute the official metric code in-tree without requiring a separate upstream checkout.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/llyx97/FETV-EVAL |
| Revision | `1385218bc92899e85aae0941888d8ddcf94bd65d` |
| Verified on | 2026-08-25 (git blob-hash comparison, see below) |
| Paper | https://arxiv.org/abs/2311.01813 |
| Project page | https://github.com/llyx97/FETV |
| Upstream license | No LICENSE file published at the pinned revision |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/video/fetv.yaml` |

## Verification method

Every vendored file was classified by comparing its git blob hash (`git hash-object`) against the complete upstream tree at the revision above (`git ls-tree -r`). A blob-hash match proves byte-identity; files whose upstream path exists but whose content differs carry WorldFoundry adaptations. Spot-checked diffs show the adaptations replace hard-coded checkpoint/dataset paths and network download side effects with WorldFoundry environment-variable and checkpoint-root resolution while keeping the official metric formulas; the exact per-file diffs are reproducible with:

```bash
git clone https://github.com/llyx97/FETV-EVAL.git upstream && git -C upstream checkout 1385218bc92899e85aae0941888d8ddcf94bd65d
diff -ru upstream/<upstream path> fetv_eval/<vendored path>
```

## File-level provenance

### Byte-identical to upstream (same path)

- `metrics/__init__.py`
- `metrics/clips.py`
- `video_dataset.py`

### Adapted from upstream (same path)

- `auto_eval.py`
- `compute_fid.py`

## Assets that are deliberately *not* vendored

Model weights, judge checkpoints, and benchmark datasets are never stored in this tree; they are resolved at runtime through the environment variables and checkpoint roots documented in the catalog entry and the runner CLI.

## License status

The FETV-EVAL code repository publishes no LICENSE file at the pinned revision, so redistribution rights for the code are not explicit; the FETV dataset itself is released under CC-BY-4.0 (recorded in the catalog entry). The vendored subset is retained for evaluation reproducibility with full attribution.
