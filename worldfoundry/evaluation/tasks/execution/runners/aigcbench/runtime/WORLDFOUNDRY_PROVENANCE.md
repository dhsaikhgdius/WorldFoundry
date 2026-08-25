# AIGCBench in-tree runtime provenance

`aigcbench/` vendors the official AIGCBench evaluation code so that `run_aigcbench_official_runner.py` can execute the official metric code in-tree without requiring a separate upstream checkout.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/BenchCouncil/AIGCBench |
| Revision | `cc230c8474fdd1af8a7a0749981aa1d09198eaf3` |
| Verified on | 2026-08-25 (git blob-hash comparison, see below) |
| Paper | https://arxiv.org/abs/2401.01651 |
| Project page | https://huggingface.co/papers/2401.01651 |
| Upstream license | **Apache License 2.0** (`aigcbench/LICENSE`) |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/video/aigcbench.yaml` |

## Verification method

Every vendored file was classified by comparing its git blob hash (`git hash-object`) against the complete upstream tree at the revision above (`git ls-tree -r`). A blob-hash match proves byte-identity; files whose upstream path exists but whose content differs carry WorldFoundry adaptations. Spot-checked diffs show the adaptations replace hard-coded checkpoint/dataset paths and network download side effects with WorldFoundry environment-variable and checkpoint-root resolution while keeping the official metric formulas; the exact per-file diffs are reproducible with:

```bash
git clone https://github.com/BenchCouncil/AIGCBench.git upstream && git -C upstream checkout cc230c8474fdd1af8a7a0749981aa1d09198eaf3
diff -ru upstream/<upstream path> aigcbench/<vendored path>
```

## File-level provenance

### Byte-identical to upstream (same path)

- `LICENSE`
- `utils.py`

### Adapted from upstream (same path)

- `eval.py`
- `metrics/clip_score.py`

## Assets that are deliberately *not* vendored

Model weights, judge checkpoints, and benchmark datasets are never stored in this tree; they are resolved at runtime through the environment variables and checkpoint roots documented in the catalog entry and the runner CLI.

## License status

The upstream repository publishes an Apache License 2.0 at the pinned revision, so redistribution of this directory is permitted with the license text retained. It is kept unchanged at `aigcbench/LICENSE`.
