# T2VSafetyBench in-tree runtime provenance

`t2v_safety_bench/` vendors the official T2VSafetyBench evaluation code so that `run_t2v_safety_bench_official_runner.py` can execute the official metric code in-tree without requiring a separate upstream checkout.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/yibo-miao/T2VSafetyBench |
| Revision | `97a841699fc25bc61662e6453bd6a2d9d187a9db` |
| Verified on | 2026-08-25 (git blob-hash comparison, see below) |
| Paper | https://arxiv.org/abs/2407.05965 |
| Project page | https://github.com/yibo-miao/T2VSafetyBench |
| Upstream license | No LICENSE file published at the pinned revision |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/video/t2v-safety-bench.yaml` |

## Verification method

Every vendored file was classified by comparing its git blob hash (`git hash-object`) against the complete upstream tree at the revision above (`git ls-tree -r`). A blob-hash match proves byte-identity; files whose upstream path exists but whose content differs carry WorldFoundry adaptations. Spot-checked diffs show the adaptations replace hard-coded checkpoint/dataset paths and network download side effects with WorldFoundry environment-variable and checkpoint-root resolution while keeping the official metric formulas; the exact per-file diffs are reproducible with:

```bash
git clone https://github.com/yibo-miao/T2VSafetyBench.git upstream && git -C upstream checkout 97a841699fc25bc61662e6453bd6a2d9d187a9db
diff -ru upstream/<upstream path> t2v_safety_bench/<vendored path>
```

## File-level provenance

### Byte-identical to upstream (same path)

- `utils.py`

### Adapted from upstream (same path)

- `gpt4.py`
- `main.py`

## Assets that are deliberately *not* vendored

Model weights, judge checkpoints, and benchmark datasets are never stored in this tree; they are resolved at runtime through the environment variables and checkpoint roots documented in the catalog entry and the runner CLI.

## License status

The upstream repository publishes no LICENSE file at the pinned revision, so redistribution rights are not explicit. The vendored subset is retained for evaluation reproducibility with full attribution.
