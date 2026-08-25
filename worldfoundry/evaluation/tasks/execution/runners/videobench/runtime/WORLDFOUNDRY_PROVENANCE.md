# Video-Bench in-tree runtime provenance

`videobench/` vendors the official Video-Bench evaluation code so that `run_videobench_official_runner.py` can execute the official metric code in-tree without requiring a separate upstream checkout.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/Video-Bench/Video-Bench |
| Revision | `d40b917c4238011aea27d24f85b1e3bf996715bd` |
| Verified on | 2026-08-25 (git blob-hash comparison, see below) |
| Paper | https://arxiv.org/abs/2504.04907 |
| Project page | https://github.com/Video-Bench/Video-Bench |
| Upstream license | **Apache License 2.0** (`videobench/LICENSE`) |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/video/video-bench.yaml` |

## Verification method

Every vendored file was classified by comparing its git blob hash (`git hash-object`) against the complete upstream tree at the revision above (`git ls-tree -r`). A blob-hash match proves byte-identity; files whose upstream path exists but whose content differs carry WorldFoundry adaptations. Spot-checked diffs show the adaptations replace hard-coded checkpoint/dataset paths and network download side effects with WorldFoundry environment-variable and checkpoint-root resolution while keeping the official metric formulas; the exact per-file diffs are reproducible with:

```bash
git clone https://github.com/Video-Bench/Video-Bench.git upstream && git -C upstream checkout d40b917c4238011aea27d24f85b1e3bf996715bd
diff -ru upstream/<upstream path> videobench/<vendored path>
```

## File-level provenance

### Byte-identical to upstream (same path)

- `videobench/VideoTextAlignment.py`
- `videobench/__init__.py`
- `videobench/dynamicquality.py`
- `videobench/dynamicquality_gridview_customized.py`
- `videobench/prompt/PromptTemplate4GPTeval.py`
- `videobench/prompt/action.py`
- `videobench/prompt/color.py`
- `videobench/prompt/object_class.py`
- `videobench/prompt/overall_consistency.py`
- `videobench/prompt/scene.py`
- `videobench/prompt_dict.py`
- `videobench/staticquality.py`
- `videobench/staticquality_customized.py`
- `videobench/utils.py`

### Byte-identical to upstream (relocated)

- `videobench/LICENSE` (upstream `LICENSE`)

### Adapted from upstream (same path)

- `evaluate.py`

## Assets that are deliberately *not* vendored

Model weights, judge checkpoints, and benchmark datasets are never stored in this tree; they are resolved at runtime through the environment variables and checkpoint roots documented in the catalog entry and the runner CLI.

## License status

The upstream repository publishes an Apache License 2.0 at the pinned revision, so redistribution of this directory is permitted with the license text retained. It is kept unchanged at `videobench/LICENSE`.
