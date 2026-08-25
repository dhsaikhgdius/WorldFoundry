# CameraBench in-tree runtime provenance

`camerabench/` is a vendored copy of the official CameraBench evaluation
scripts from the `camerabench/` subdirectory of `linzhiqiu/t2v_metrics`, so
that `run_camerabench_official_runner.py --run-official` executes the official
metric aggregation code without requiring a separate checkout.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/linzhiqiu/t2v_metrics (subdirectory `camerabench/`) |
| Revision | `0bd9bfc68032ce4f9d5da80d646fa5ceb3b9bb1b` |
| Vendored on | 2026-08-25 |
| Paper | https://arxiv.org/abs/2504.15376 |
| Project page | https://linzhiqiu.github.io/papers/camerabench/ |
| Dataset | https://huggingface.co/datasets/syCen/CameraBench |
| Upstream license | **Apache License 2.0** (`camerabench/LICENSE`, copied from the t2v_metrics repository root) |

## Modifications

None. Every vendored `.py` and `.md` file is byte-identical to upstream at the
pinned revision. WorldFoundry adapts the runtime purely through CLI arguments
and subprocess environment variables:

- `run_camerabench_official_runner.py --run-official` invokes the vendored
  evaluators as subprocesses with `--score_dir`/`--output_file` arguments and
  sets `MPLBACKEND=Agg` so the optional matplotlib plotting import never needs
  a display.
- The vendored scripts are executed with `cwd=<runtime root>`; nothing in this
  directory is imported into the WorldFoundry process.

## Vendored files

- `binary_classification_evaluation.py` — official binary-classification
  aggregator (mAP / ROC-AUC over `classification_scores_*.json`).
- `binary_classification_vlm_scores.py` — official VLM score generation for the
  binary task (requires `t2v_metrics` models; reference only, not executed by
  the WorldFoundry runner).
- `vqa_and_retrieval_evaluation.py` — official VQA and retrieval aggregator
  over `vqa_retrieval_scores_*.json`.
- `vqa_and_retrieval_vlm_scores.py` — official VLM score generation for the
  VQA/retrieval task (reference only).
- `caption_evaluation.py` — official caption metric aggregator (SPICE, CIDEr,
  BLEU-2, ROUGE-L, METEOR, optional GPT-4o judge).
- `caption_generation.py` — official caption generation stage (reference
  only).
- `data_download.py` — official helper that downloads the CameraBench testset
  from Hugging Face.
- `README.md` — upstream documentation for the above scripts.
- `LICENSE` — Apache-2.0 license from the `t2v_metrics` repository root, which
  covers the `camerabench/` subdirectory.

## Exclusions

- `data/` — 5.1 MB of official label/metadata JSONL splits
  (`binary_classification/*.jsonl`, `vqa_and_retrieval/*`,
  `caption_data.json`). The same annotations ship with the official
  Hugging Face dataset `syCen/CameraBench`, which the catalog entry already
  tracks and which `data_download.py` fetches; keeping a second copy in-tree
  would duplicate benchmark data. Supply the dataset root at runtime through
  `--benchmark-data-root` / `WORLDFOUNDRY_CAMERABENCH_DATA_ROOT`.
- The remainder of the `t2v_metrics` repository (the `t2v_metrics/` package,
  GenAI-Bench code, sample images/videos). Only the `camerabench/`
  subdirectory is the official CameraBench evaluation runtime; score
  generation with VQAScore models requires installing `t2v_metrics`
  separately.

## Assets that are deliberately *not* vendored

- **VQAScore / VLM model weights** — the score-generation scripts load models
  through the `t2v_metrics` package at runtime; weights are never stored in
  this tree.
- **CameraBench testset videos** — distributed through the
  `syCen/CameraBench` Hugging Face dataset.

## License status

The upstream repository publishes an Apache-2.0 `LICENSE` at the pinned
revision, so redistribution of this directory is permitted with the license
text retained. It is kept at `camerabench/LICENSE` unchanged.
