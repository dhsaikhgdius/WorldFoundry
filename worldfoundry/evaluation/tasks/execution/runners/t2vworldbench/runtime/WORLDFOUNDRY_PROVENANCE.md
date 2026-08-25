# T2VWorldBench in-tree runtime provenance

`t2vworldbench/` vendors the official T2VWorldBench evaluation code so that `run_t2vworldbench_official_runner.py` can execute the official metric code in-tree without requiring a separate upstream checkout.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/magiclinux/world_knowledge |
| Revision | `4c0f18095d1c8053c11dbcfc8382ef9e4d9b768a` |
| Verified on | 2026-08-25 (git blob-hash comparison, see below) |
| Paper | https://arxiv.org/abs/2507.18107 |
| Upstream license | No LICENSE file published at the pinned revision |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/video/t2vworldbench.yaml` |

## Verification method

Every vendored file was classified by comparing its git blob hash (`git hash-object`) against the complete upstream tree at the revision above (`git ls-tree -r`). A blob-hash match proves byte-identity; files whose upstream path exists but whose content differs carry WorldFoundry adaptations. Spot-checked diffs show the adaptations replace hard-coded checkpoint/dataset paths and network download side effects with WorldFoundry environment-variable and checkpoint-root resolution while keeping the official metric formulas; the exact per-file diffs are reproducible with:

```bash
git clone https://github.com/magiclinux/world_knowledge.git upstream && git -C upstream checkout 4c0f18095d1c8053c11dbcfc8382ef9e4d9b768a
diff -ru upstream/<upstream path> t2vworldbench/<vendored path>
```

## File-level provenance

### Adapted from upstream (same path)

- `eval.py`

## Assets that are deliberately *not* vendored

Model weights, judge checkpoints, and benchmark datasets are never stored in this tree; they are resolved at runtime through the environment variables and checkpoint roots documented in the catalog entry and the runner CLI.

## License status

The upstream repository publishes no LICENSE file at the pinned revision (the catalog records unknown_pending_review). The vendored subset is retained for evaluation reproducibility with full attribution.
