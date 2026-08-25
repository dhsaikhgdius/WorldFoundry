# VMBench in-tree runtime provenance

`official/` vendors the official VMBench evaluation code so that `run_vmbench_official_runner.py` can execute the official metric code in-tree without requiring a separate upstream checkout.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/AMAP-ML/VMBench |
| Revision | `9772be365784bc3f641d64fca018b1debb749116` |
| Verified on | 2026-08-25 (git blob-hash comparison, see below) |
| Project page | https://amap-ml.github.io/VMBench-Website/ |
| Upstream license | **Apache License 2.0** (`official/LICENSE`) |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/video/vmbench.yaml` |

## Verification method

Every vendored file was classified by comparing its git blob hash (`git hash-object`) against the complete upstream tree at the revision above (`git ls-tree -r`). A blob-hash match proves byte-identity; files whose upstream path exists but whose content differs carry WorldFoundry adaptations. Spot-checked diffs show the adaptations replace hard-coded checkpoint/dataset paths and network download side effects with WorldFoundry environment-variable and checkpoint-root resolution while keeping the official metric formulas; the exact per-file diffs are reproducible with:

```bash
git clone https://github.com/AMAP-ML/VMBench.git upstream && git -C upstream checkout 9772be365784bc3f641d64fca018b1debb749116
diff -ru upstream/<upstream path> official/<vendored path>
```

## File-level provenance

### Byte-identical to upstream (same path)

- `LICENSE`
- `VideoMAEv2/dataset/__init__.py`
- `VideoMAEv2/dataset/build.py`
- `VideoMAEv2/dataset/datasets.py`
- `VideoMAEv2/dataset/functional.py`
- `VideoMAEv2/dataset/loader.py`
- `VideoMAEv2/dataset/masking_generator.py`
- `VideoMAEv2/dataset/pretrain_datasets.py`
- `VideoMAEv2/dataset/rand_augment.py`
- `VideoMAEv2/dataset/random_erasing.py`
- `VideoMAEv2/dataset/transforms.py`
- `VideoMAEv2/dataset/video_transforms.py`
- `VideoMAEv2/dataset/volume_transforms.py`
- `VideoMAEv2/engine_for_finetuning.py`
- `VideoMAEv2/misc/k710_identical_label_merge.json`
- `VideoMAEv2/misc/label_710to400.json`
- `VideoMAEv2/misc/label_710to600.json`
- `VideoMAEv2/misc/label_710to700.json`
- `VideoMAEv2/misc/label_map_k400.txt`
- `VideoMAEv2/misc/label_map_k600.txt`
- `VideoMAEv2/misc/label_map_k700.txt`
- `VideoMAEv2/misc/label_map_k710.txt`
- `VideoMAEv2/models/__init__.py`
- `VideoMAEv2/models/modeling_finetune.py`
- `VideoMAEv2/models/modeling_pretrain.py`
- `VideoMAEv2/optim_factory.py`
- `VideoMAEv2/run_class_finetuning.py`
- `VideoMAEv2/utils.py`
- `bench_utils/__init__.py`
- `bench_utils/calculate_score.py`
- `bench_utils/create_meta_info.py`
- `bench_utils/pose_utils.py`
- `mmpose/configs/_base_/datasets/coco.py`
- `mmpose/configs/_base_/default_runtime.py`
- `mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py`
- `mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py`

### Byte-identical to upstream (relocated)

- `grounded_sam2_utils/common_utils.py` (upstream `Grounded-SAM-2/utils/common_utils.py`)
- `grounded_sam2_utils/mask_dictionary_model.py` (upstream `Grounded-SAM-2/utils/mask_dictionary_model.py`)
- `grounded_sam2_utils/track_utils.py` (upstream `Grounded-SAM-2/utils/track_utils.py`)
- `grounded_sam2_utils/video_utils.py` (upstream `Grounded-SAM-2/utils/video_utils.py`)
- `q_align/__init__.py` (upstream `Q-Align/q_align/__init__.py`)
- `q_align/constants.py` (upstream `Q-Align/q_align/constants.py`)
- `q_align/conversation.py` (upstream `Q-Align/q_align/conversation.py`)
- `q_align/load_video.py` (upstream `Q-Align/q_align/load_video.py`)
- `q_align/mm_utils.py` (upstream `Q-Align/q_align/mm_utils.py`)
- `q_align/model/__init__.py` (upstream `Q-Align/q_align/model/__init__.py`)
- `q_align/model/builder.py` (upstream `Q-Align/q_align/model/builder.py`)
- `q_align/model/configuration_mplug_owl2.py` (upstream `Q-Align/q_align/model/configuration_mplug_owl2.py`)
- `q_align/model/modeling_attn_mask_utils.py` (upstream `Q-Align/q_align/model/modeling_attn_mask_utils.py`)
- `q_align/model/utils.py` (upstream `Q-Align/q_align/model/utils.py`)
- `q_align/model/visual_encoder.py` (upstream `Q-Align/q_align/model/visual_encoder.py`)
- `q_align/utils.py` (upstream `Q-Align/q_align/utils.py`)

### Adapted from upstream (same path)

- `bench_utils/cas_utils.py`
- `bench_utils/tcs_utils.py`
- `commonsense_adherence_score.py`
- `motion_smoothness_score.py`
- `object_integrity_score.py`
- `perceptible_amplitude_score.py`
- `temporal_coherence_score.py`

### Adapted from upstream (relocated)

- `q_align/model/modeling_llama2.py` (upstream `Q-Align/q_align/model/modeling_llama2.py`)
- `q_align/model/modeling_mplug_owl2.py` (upstream `Q-Align/q_align/model/modeling_mplug_owl2.py`)

### WorldFoundry-authored (no upstream counterpart)

- `__init__.py`
- `grounded_sam2_utils/__init__.py`

## Assets that are deliberately *not* vendored

Model weights, judge checkpoints, and benchmark datasets are never stored in this tree; they are resolved at runtime through the environment variables and checkpoint roots documented in the catalog entry and the runner CLI.

## License status

The upstream repository publishes an Apache License 2.0 at the pinned revision, so redistribution of this directory is permitted with the license text retained. It is kept unchanged at `official/LICENSE`.
