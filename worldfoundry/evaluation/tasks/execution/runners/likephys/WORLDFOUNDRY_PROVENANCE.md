# LikePhys in-tree runtime provenance

`likephys_official_scoring.py` and companions implement the official LikePhys plausibility-preference scoring in tree; no upstream source tree is vendored. The probe (video-model likelihood) stage runs from the official LikePhys checkout supplied by the caller.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/YuanJianhao508/LikePhys |
| Paper | https://arxiv.org/abs/2510.11512 |
| Project page | https://yuanjianhao508.github.io/LikePhys/ |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/**/likephys.yaml` |
| Task manifest | `worldfoundry/data/benchmarks/tasks/external/likephys.yaml` |

## Modifications

Not applicable: no upstream code is vendored here. The in-tree scoring modules are WorldFoundry-authored and reproduce the official LikePhys aggregation; the probe stage always runs from the caller-supplied official checkout.

## Official boundary

- Scoring/aggregation of probe outputs is pure-local and in tree.
- The probe stage requires the external official checkout (`WORLDFOUNDRY_LIKEPHYS_EVALUATOR_ROOT` / `WORLDFOUNDRY_LIKEPHYS_ROOT`), video-model weights, and a GPU; it is not vendored.
- Official-validation normalizes an existing LikePhys results root and is pure-local.

## Not vendored

- The official LikePhys probe code (external checkout).
- Video diffusion model weights used by the probes.

## License status

Upstream dataset is Apache-2.0 (recorded in the catalog); the probe checkout retains its own upstream terms.
