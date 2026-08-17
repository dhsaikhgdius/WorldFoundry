# Upstream Provenance — YOLO-World (canonical copy)

- upstream_url: `https://github.com/AILab-CVC/YOLO-World` (official YOLO-World
  repository; note: this URL is not independently recorded elsewhere in this
  repository — the model catalog has no yolo-world entry — so it is stated
  here from the upstream project identity and should be confirmed when the
  snapshot commit is backfilled)
- local_path: `worldfoundry/base_models/perception_core/detection/yolo_world`
- license: GPL-3.0 (per the upstream repository's LICENSE). In-tree evidence:
  the bundled `mmyolo/setup.py` line 181 self-declares `license='GPL License 3.0'`;
  the `yolo_world/` source files carry "Copyright (c) Tencent Inc." headers
  without license text. No LICENSE file was vendored with this tree.
- source_commit: unknown — needs backfill.
- fork_status: believed unmodified upstream snapshot (plus repo-wide
  machine-generated docstring churn, review [VI-12]); no patch markers found.

## Contents

- `yolo_world/` (36 py): the YOLO-World model/dataset/engine code.
- `mmyolo/` (428 py): full vendored mmyolo dependency, including pure-config
  directories (`configs/yolov5` etc.). The review recommends trimming the
  configs/projects directories from the vendor scope ([VI-9]).
- `data/`, `paths.py`, top-level config py: WorldFoundry integration glue.

## Duplicate copy — cross-reference

A second copy of the same YOLO-World snapshot lives at
`worldfoundry/base_models/perception_core/video_text/opens2v_nexus/eval/utils/yoloworld/`
(embedded in the OpenS2V-Nexus evaluation suite). Per review
`plan/code_review/11_vendored_integration.md` [VI-9], the two copies share all
36 relative paths; 34 files differ ONLY by formatter noise (quote style, line
wrapping, trailing commas) — the same upstream snapshot, one copy reformatted.
Semantic changes must be applied to BOTH copies until they are deduplicated.
Deduplication (deleting the embedded copy and importing this canonical one) is
planned for the second fix round.
