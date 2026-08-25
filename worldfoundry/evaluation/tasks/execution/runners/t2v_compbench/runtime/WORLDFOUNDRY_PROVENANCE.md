# T2V-CompBench in-tree runtime provenance

`t2v_compbench/` vendors the official T2V-CompBench evaluation code so that `run_t2v_compbench_official_runner.py` can execute the official metric code in-tree without requiring a separate upstream checkout.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/KaiyueSun98/T2V-CompBench |
| Revision | `dd5eff7b93af0550b9efa2bdabbb21b3b017ceda` |
| Verified on | 2026-08-25 (git blob-hash comparison, see below) |
| Paper | https://arxiv.org/abs/2407.14505 |
| Project page | https://t2v-compbench-2025.github.io/ |
| Upstream license | No LICENSE file published at the pinned revision |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/video/t2v-compbench.yaml` |

## Verification method

Every vendored file was classified by comparing its git blob hash (`git hash-object`) against the complete upstream tree at the revision above (`git ls-tree -r`). A blob-hash match proves byte-identity; files whose upstream path exists but whose content differs carry WorldFoundry adaptations. Spot-checked diffs show the adaptations replace hard-coded checkpoint/dataset paths and network download side effects with WorldFoundry environment-variable and checkpoint-root resolution while keeping the official metric formulas; the exact per-file diffs are reproducible with:

```bash
git clone https://github.com/KaiyueSun98/T2V-CompBench.git upstream && git -C upstream checkout dd5eff7b93af0550b9efa2bdabbb21b3b017ceda
diff -ru upstream/<upstream path> t2v_compbench/<vendored path>
```

> Upstream distributes its metric scripts inside bundled dependency checkouts (`LLaVA/llava/eval/`, `Grounded-Segment-Anything/`, `Depth-Anything/`, `dot/`); the vendored copies are regrouped by dependency stack under `mllm_metrics/`, `grounded_sam_metrics/`, and `dot/`. The upstream path for every relocated file is listed below.

## File-level provenance

### Byte-identical to upstream (same path)

- `dot/configs/bootstapir.json`
- `dot/configs/cotracker2_patch_4_wind_8.json`
- `dot/configs/cotracker3_wind_60.json`
- `dot/configs/cotracker_patch_4_wind_8.json`
- `dot/configs/raft_patch_4_alpha.json`
- `dot/configs/raft_patch_8.json`
- `dot/configs/tapir.json`

### Adapted from upstream (same path)

- `dot/compbench_eval_motion_binding.py`

### Adapted from upstream (relocated)

- `grounded_sam_metrics/compbench_eval_numeracy.py` (upstream `Grounded-Segment-Anything/GroundingDINO/demo/compbench_eval_numeracy.py`)
- `grounded_sam_metrics/compbench_eval_spatial_relationships.py` (upstream `Grounded-Segment-Anything/compbench_eval_spatial_relationships.py`)
- `grounded_sam_metrics/compbench_motion_binding_seg.py` (upstream `Grounded-Segment-Anything/compbench_motion_binding_seg.py`)
- `grounded_sam_metrics/compbench_run_depth.py` (upstream `Depth-Anything/compbench_run_depth.py`)
- `mllm_metrics/compbench_eval_action_binding.py` (upstream `LLaVA/llava/eval/compbench_eval_action_binding.py`)
- `mllm_metrics/compbench_eval_consistent_attr.py` (upstream `LLaVA/llava/eval/compbench_eval_consistent_attr.py`)
- `mllm_metrics/compbench_eval_dynamic_attr.py` (upstream `LLaVA/llava/eval/compbench_eval_dynamic_attr.py`)
- `mllm_metrics/compbench_eval_interaction.py` (upstream `LLaVA/llava/eval/compbench_eval_interaction.py`)

### WorldFoundry-authored (no upstream counterpart)

- `asset_paths.py`
- `grounded_sam_metrics/__init__.py`
- `grounded_sam_metrics/video_io.py`
- `mllm_metrics/__init__.py`

## Assets that are deliberately *not* vendored

Model weights, judge checkpoints, and benchmark datasets are never stored in this tree; they are resolved at runtime through the environment variables and checkpoint roots documented in the catalog entry and the runner CLI.

## License status

The upstream repository publishes no LICENSE file at the pinned revision (the catalog records openrail/unknown_pending_review). The vendored subset is retained for evaluation reproducibility with full attribution.
