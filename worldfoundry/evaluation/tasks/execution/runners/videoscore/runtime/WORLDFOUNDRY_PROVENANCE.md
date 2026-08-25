# VideoScore in-tree runtime provenance

`videoscore/` vendors the official VideoScore evaluation code so that `run_videoscore_official_runner.py` can execute the official metric code in-tree without requiring a separate upstream checkout.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/TIGER-AI-Lab/VideoScore |
| Revision | `f87faf4647e637066bcb74721670cd579e0f4349` |
| Verified on | 2026-08-25 (git blob-hash comparison, see below) |
| Paper | https://arxiv.org/abs/2406.15252 |
| Project page | https://tiger-ai-lab.github.io/VideoScore/ |
| Upstream license | **MIT License** (`videoscore/LICENSE`) |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/video/videoscore.yaml` |

## Verification method

Every vendored file was classified by comparing its git blob hash (`git hash-object`) against the complete upstream tree at the revision above (`git ls-tree -r`). A blob-hash match proves byte-identity; files whose upstream path exists but whose content differs carry WorldFoundry adaptations. Spot-checked diffs show the adaptations replace hard-coded checkpoint/dataset paths and network download side effects with WorldFoundry environment-variable and checkpoint-root resolution while keeping the official metric formulas; the exact per-file diffs are reproducible with:

```bash
git clone https://github.com/TIGER-AI-Lab/VideoScore.git upstream && git -C upstream checkout f87faf4647e637066bcb74721670cd579e0f4349
diff -ru upstream/<upstream path> videoscore/<vendored path>
```

## File-level provenance

### Byte-identical to upstream (same path)

- `LICENSE`
- `benchmark/eval_feature_metric.py`
- `benchmark/eval_gemini.py`
- `benchmark/eval_gpt4o.py`
- `benchmark/eval_videoscore.py`
- `benchmark/feature_metric_tools/dynamic_eval.py`
- `benchmark/feature_metric_tools/t2v_align_eval.py`
- `benchmark/feature_metric_tools/temporal_eval.py`
- `benchmark/feature_metric_tools/visual_eval.py`
- `benchmark/get_genaibench_pairwise_acc.py`
- `benchmark/get_spearman_corr.py`
- `benchmark/get_vbench_pairwise_acc.py`
- `benchmark/mllm_tools/__init__.py`
- `benchmark/mllm_tools/mllm_utils.py`
- `benchmark/utils_conv.py`
- `benchmark/utils_gpt4o.py`
- `benchmark/utils_tools.py`

### Adapted from upstream (same path)

- `benchmark/eval_other_mllm.py`
- `benchmark/mllm_tools/blip_flant5_eval.py`
- `benchmark/mllm_tools/cogvlm_eval.py`
- `benchmark/mllm_tools/emu2_eval.py`
- `benchmark/mllm_tools/fuyu_eval.py`
- `benchmark/mllm_tools/gpt4v_eval.py`
- `benchmark/mllm_tools/idefics1_eval.py`
- `benchmark/mllm_tools/idefics2_eval.py`
- `benchmark/mllm_tools/instructblip_eval.py`
- `benchmark/mllm_tools/kosmos2_eval.py`
- `benchmark/mllm_tools/llava_eval.py`
- `benchmark/mllm_tools/llava_next_eval.py`
- `benchmark/mllm_tools/mfuyu_eval.py`
- `benchmark/mllm_tools/mllava_eval.py`
- `benchmark/mllm_tools/openflamingo_eval.py`
- `benchmark/mllm_tools/otterhd_eval.py`
- `benchmark/mllm_tools/otterimage_eval.py`
- `benchmark/mllm_tools/ottervideo_eval.py`
- `benchmark/mllm_tools/qwenVL_eval.py`
- `benchmark/mllm_tools/videollava_eval.py`
- `benchmark/mllm_tools/vila_eval.py`

## Assets that are deliberately *not* vendored

Model weights, judge checkpoints, and benchmark datasets are never stored in this tree; they are resolved at runtime through the environment variables and checkpoint roots documented in the catalog entry and the runner CLI.

## License status

The upstream repository publishes a MIT License at the pinned revision, so redistribution of this directory is permitted with the license text retained. It is kept unchanged at `videoscore/LICENSE`.
