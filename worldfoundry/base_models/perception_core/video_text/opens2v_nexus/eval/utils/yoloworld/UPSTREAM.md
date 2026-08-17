# Upstream Provenance — YOLO-World (copy embedded in OpenS2V-Nexus eval)

- upstream_url: `https://github.com/AILab-CVC/YOLO-World` (official YOLO-World
  repository), vendored here as part of the OpenS2V-Nexus evaluation suite
  (`https://github.com/PKU-YuanGroup/OpenS2V-Nexus`, referenced by the metric
  docs at `docs/fumadocs/mdx/partials/metrics-usage.mdx` line 228: the
  NexusScore metric is "YOLO-World + GME" from OpenS2V-Nexus).
- local_path: `worldfoundry/base_models/perception_core/video_text/opens2v_nexus/eval/utils/yoloworld`
- license: GPL-3.0 (per the upstream YOLO-World repository's LICENSE; the
  source files here carry "Copyright (c) Tencent Inc." headers without license
  text, and no LICENSE file was vendored with this tree).
- source_commit: unknown — needs backfill.
- fork_status: same upstream snapshot as the canonical copy, reformatted (see
  below); no semantic local changes identified.

## Duplicate copy — cross-reference

This is the SECOND copy of the YOLO-World snapshot in this repository. The
canonical copy lives at
`worldfoundry/base_models/perception_core/detection/yolo_world/` (which also
carries the vendored `mmyolo` dependency and its own `UPSTREAM.md`).

Per review `plan/code_review/11_vendored_integration.md` [VI-9], the two
copies share all 36 relative paths under `yolo_world/`; 34 files differ ONLY
by formatter noise (quote style, line wrapping, trailing commas — e.g. the 89
diff lines in `datasets/mm_dataset.py` are all style reshuffles). Text-level
diff is therefore useless between the copies; they are the same upstream
snapshot, one reformatted.

Until deduplication, semantic fixes (including any CVE fixes) must be applied
to BOTH copies. Deduplication — deleting this embedded copy in favor of
importing the canonical `detection/yolo_world` — is planned for the second fix
round ([VI-9] recommendation).
