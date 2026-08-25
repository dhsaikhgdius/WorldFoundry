# ChronoMagic-Bench in-tree runtime provenance

`chronomagic_bench/` vendors the official ChronoMagic-Bench evaluation code so that `run_chronomagic_bench_official_runner.py` can execute the official metric code in-tree without requiring a separate upstream checkout.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/PKU-YuanGroup/ChronoMagic-Bench |
| Revision | `e6b75ffc53c20dfea9c1704024d5167845534b63` |
| Verified on | 2026-08-25 (git blob-hash comparison, see below) |
| Paper | https://arxiv.org/abs/2406.18522 |
| Project page | https://pku-yuangroup.github.io/ChronoMagic-Bench/ |
| Upstream license | **Apache License 2.0** (`chronomagic_bench/LICENSE`) |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/video/chronomagic-bench.yaml` |

## Verification method

Every vendored file was classified by comparing its git blob hash (`git hash-object`) against the complete upstream tree at the revision above (`git ls-tree -r`). A blob-hash match proves byte-identity; files whose upstream path exists but whose content differs carry WorldFoundry adaptations. Spot-checked diffs show the adaptations replace hard-coded checkpoint/dataset paths and network download side effects with WorldFoundry environment-variable and checkpoint-root resolution while keeping the official metric formulas; the exact per-file diffs are reproducible with:

```bash
git clone https://github.com/PKU-YuanGroup/ChronoMagic-Bench.git upstream && git -C upstream checkout e6b75ffc53c20dfea9c1704024d5167845534b63
diff -ru upstream/<upstream path> chronomagic_bench/<vendored path>
```

## File-level provenance

### Byte-identical to upstream (same path)

- `CHScore/step1-get_merged_CHScore.py`
- `GPT4o_MTScore/step0-extract_video_frames.py`
- `GPT4o_MTScore/step1-get_temp_results.py`
- `GPT4o_MTScore/step2-get_GPT4o-MTScore.py`
- `GPT4o_MTScore/step3-get_merged_GPT4o-MTScore.py`
- `LICENSE`
- `MTScore/configs/config_bert.json`
- `MTScore/configs/config_bert_large.json`
- `MTScore/configs/data.py`
- `MTScore/configs/easydict.py`
- `MTScore/configs/med_config.json`
- `MTScore/configs/med_config_fusion.json`
- `MTScore/configs/med_large_config.json`
- `MTScore/configs/model.py`
- `MTScore/step1-get_merged_MTScore.py`
- `get_uploaded_json.py`

### Adapted from upstream (same path)

- `CHScore/step0-get_CHScore.py`
- `MTScore/configs/config.py`
- `MTScore/configs/internvideo2_stage2_config.py`
- `MTScore/configs/utils.py`
- `MTScore/step0-get_MTScore.py`

## Assets that are deliberately *not* vendored

Model weights, judge checkpoints, and benchmark datasets are never stored in this tree; they are resolved at runtime through the environment variables and checkpoint roots documented in the catalog entry and the runner CLI.

## License status

The upstream repository publishes an Apache License 2.0 at the pinned revision, so redistribution of this directory is permitted with the license text retained. It is kept unchanged at `chronomagic_bench/LICENSE`.
