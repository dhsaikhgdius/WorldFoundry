# VideoScience-Bench in-tree runtime provenance

`videoscience_bench/` vendors the official VideoScience-Bench evaluation code so that `run_videoscience_bench_official_runner.py` can execute the official metric code in-tree without requiring a separate upstream checkout.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/hao-ai-lab/VideoScience |
| Revision | `0b3bb2da74388991bf842bca1096a6a25d3c9311` |
| Verified on | 2026-08-25 (git blob-hash comparison, see below) |
| Paper | https://arxiv.org/abs/2512.02942 |
| Project page | https://huggingface.co/spaces/lmgame/videoscience-bench |
| Upstream license | **MIT License** (`videoscience_bench/LICENSE`) |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/video/videoscience-bench.yaml` |

## Verification method

Every vendored file was classified by comparing its git blob hash (`git hash-object`) against the complete upstream tree at the revision above (`git ls-tree -r`). A blob-hash match proves byte-identity; files whose upstream path exists but whose content differs carry WorldFoundry adaptations. Spot-checked diffs show the adaptations replace hard-coded checkpoint/dataset paths and network download side effects with WorldFoundry environment-variable and checkpoint-root resolution while keeping the official metric formulas; the exact per-file diffs are reproducible with:

```bash
git clone https://github.com/hao-ai-lab/VideoScience.git upstream && git -C upstream checkout 0b3bb2da74388991bf842bca1096a6a25d3c9311
diff -ru upstream/<upstream path> videoscience_bench/<vendored path>
```

## File-level provenance

### Byte-identical to upstream (same path)

- `LICENSE`
- `judge/extract_exp_info.py`

### Adapted from upstream (same path)

- `judge/api_manager.py`
- `judge/api_providers.py`
- `judge/vlm_as_a_judge.py`

### WorldFoundry-authored (no upstream counterpart)

- `__init__.py`
- `judge/__init__.py`
- `videoscience_batch.py`

## Assets that are deliberately *not* vendored

Model weights, judge checkpoints, and benchmark datasets are never stored in this tree; they are resolved at runtime through the environment variables and checkpoint roots documented in the catalog entry and the runner CLI.

## License status

The upstream repository publishes a MIT License at the pinned revision, so redistribution of this directory is permitted with the license text retained. It is kept unchanged at `videoscience_bench/LICENSE`.
