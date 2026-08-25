# PhyFPS-Bench-Gen (Visual Chronometer) in-tree runtime provenance

`visual_chronometer/` vendors the official PhyFPS-Bench-Gen (Visual Chronometer) evaluation code so that `run_phyfps_bench_gen_official_runner.py / run_visual_chronometer_official_runner.py` can execute the official metric code in-tree without requiring a separate upstream checkout.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/taco-group/Visual_Chronometer |
| Revision | `0dc1fd68641f476f18e76c9d1d0931278c02a386` |
| Verified on | 2026-08-25 (git blob-hash comparison, see below) |
| Upstream license | No LICENSE file published at the pinned revision |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/video/phyfps-bench-gen.yaml` |

## Verification method

Every vendored file was classified by comparing its git blob hash (`git hash-object`) against the complete upstream tree at the revision above (`git ls-tree -r`). A blob-hash match proves byte-identity; files whose upstream path exists but whose content differs carry WorldFoundry adaptations. Spot-checked diffs show the adaptations replace hard-coded checkpoint/dataset paths and network download side effects with WorldFoundry environment-variable and checkpoint-root resolution while keeping the official metric formulas; the exact per-file diffs are reproducible with:

```bash
git clone https://github.com/taco-group/Visual_Chronometer.git upstream && git -C upstream checkout 0dc1fd68641f476f18e76c9d1d0931278c02a386
diff -ru upstream/<upstream path> visual_chronometer/<vendored path>
```

## File-level provenance

### Byte-identical to upstream (same path)

- `inference/src/__init__.py`
- `inference/src/distributions.py`
- `inference/src/models/__init__.py`
- `inference/src/models/autoencoder_temporal.py`
- `inference/src/modules/t5.py`
- `inference/src/modules/utils.py`
- `inference/utils/__init__.py`
- `inference/utils/common_utils.py`

### Adapted from upstream (same path)

- `inference/configs/config_fps.yaml`
- `inference/predict.py`
- `inference/src/models/autoencoder.py`
- `inference/src/models/autoencoder2plus1d_1dcnn.py`
- `inference/src/models/fps_predictor.py`
- `inference/src/modules/__init__.py`
- `inference/src/modules/ae_modules.py`
- `inference/src/modules/attention_temporal_videoae.py`

## Assets that are deliberately *not* vendored

Model weights, judge checkpoints, and benchmark datasets are never stored in this tree; they are resolved at runtime through the environment variables and checkpoint roots documented in the catalog entry and the runner CLI.

## License status

The upstream repository publishes no LICENSE file at the pinned revision, so redistribution rights are not explicit. The vendored subset is retained for evaluation reproducibility with full attribution, and the catalog entry records the pending license status.
