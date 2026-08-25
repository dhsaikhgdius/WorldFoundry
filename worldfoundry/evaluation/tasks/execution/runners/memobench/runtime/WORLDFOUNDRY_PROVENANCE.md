# MemoBench in-tree runtime provenance

`memobench/` vendors the official MemoBench evaluation code so that `run_memobench_official_runner.py` can execute the official metric code in-tree without requiring a separate upstream checkout.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/MemoBench-Team/MemoBench |
| Revision | `8822e2713b01db90820a2c75115a37814ff992a0` |
| Verified on | 2026-08-25 (git blob-hash comparison, see below) |
| Upstream license | **MIT License** (`memobench/LICENSE.md`) |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/video/memobench.yaml` |

## Verification method

Every vendored file was classified by comparing its git blob hash (`git hash-object`) against the complete upstream tree at the revision above (`git ls-tree -r`). A blob-hash match proves byte-identity; files whose upstream path exists but whose content differs carry WorldFoundry adaptations. Spot-checked diffs show the adaptations replace hard-coded checkpoint/dataset paths and network download side effects with WorldFoundry environment-variable and checkpoint-root resolution while keeping the official metric formulas; the exact per-file diffs are reproducible with:

```bash
git clone https://github.com/MemoBench-Team/MemoBench.git upstream && git -C upstream checkout 8822e2713b01db90820a2c75115a37814ff992a0
diff -ru upstream/<upstream path> memobench/<vendored path>
```

## File-level provenance

### Byte-identical to upstream (same path)

- `LICENSE.md`
- `evaluation/automated/__init__.py`
- `evaluation/automated/io/__init__.py`
- `evaluation/automated/io/frames.py`
- `evaluation/automated/io/metadata.py`
- `evaluation/automated/metrics/__init__.py`
- `evaluation/automated/metrics/camera_controllability.py`
- `evaluation/automated/metrics/clip_metrics.py`
- `evaluation/automated/metrics/geometry.py`
- `evaluation/automated/metrics/reference_fidelity.py`
- `evaluation/automated/metrics/temporal.py`
- `evaluation/automated/metrics/visual_quality.py`
- `evaluation/extract_real_poses_mapanything.py`
- `evaluation/run_eval.py`
- `evaluation/vqa/llm-judger.py`
- `evaluation/vqa/scoring.py`
- `leaderboard/leaderboard.py`

### Adapted from upstream (same path)

- `evaluation/compute_ors.py`
- `evaluation/vqa/llm-vqa.py`

## Assets that are deliberately *not* vendored

Model weights, judge checkpoints, and benchmark datasets are never stored in this tree; they are resolved at runtime through the environment variables and checkpoint roots documented in the catalog entry and the runner CLI.

## License status

The upstream repository publishes a MIT License at the pinned revision, so redistribution of this directory is permitted with the license text retained. It is kept unchanged at `memobench/LICENSE.md`.
