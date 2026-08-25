# WorldReasonBench in-tree runtime provenance

This package contains only WorldFoundry-authored code: a result normalizer for official WorldReasonBench outputs plus in-tree QA-protocol metrics. The upstream repository's terms are restricted (no LICENSE file; README prohibits redistribution/modification without approval), so no upstream code is vendored.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/UniX-AI-Lab/WorldReasonBench |
| Paper | https://arxiv.org/abs/2605.10434 |
| Project page | https://unix-ai-lab.github.io/WorldReasonBench/ |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/**/worldreasonbench.yaml` |
| Task manifest | `worldfoundry/data/benchmarks/tasks/external/worldreasonbench.yaml` |

## Modifications

Not applicable: no upstream code is vendored here (the upstream license is restricted). This package is WorldFoundry-authored; the normalization schema and QA-protocol metrics follow the official WorldReasonBench release documentation.

## Official boundary

- Normalization of official results directories is pure-local.
- The full official protocol (reward-model scoring, hosted judges) runs upstream and is not claimed as recomputable in tree.

## Not vendored

- All upstream WorldReasonBench code (restricted license).
- Judge/reward model weights.

## License status

Upstream is `restricted_noncommercial` (no LICENSE at pinned revision); this directory deliberately contains no upstream code.
