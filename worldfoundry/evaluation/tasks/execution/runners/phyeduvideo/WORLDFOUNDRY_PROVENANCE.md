# PhyEduVideo in-tree runtime provenance

This package is an in-tree importer/normalizer for official PhyEduVideo results plus the bundled prompt suite; the official judging pipeline itself is not vendored.

| Field | Value |
| --- | --- |
| Upstream | https://github.com/meghamariamkm/PhyEduVideo |
| Pinned revision | `5a36a13818552095c9ed12c6562f236d5b5011bd` |
| Paper | https://arxiv.org/abs/2601.00943 |
| Project page | https://meghamariamkm.github.io/phyeduvideo26/ |
| Catalog entry | `worldfoundry/data/benchmarks/catalog/**/phyeduvideo.yaml` |
| Task manifest | `worldfoundry/data/benchmarks/tasks/external/phyeduvideo.yaml` |

## Modifications

Not applicable: no upstream code is vendored here. The importer, prompt bundle, and normalization schema are WorldFoundry-authored and follow the official PhyEduVideo result layout at the pinned revision.

## Official boundary

- Only result import/normalization runs locally (`--official-results-path`).
- Official judging (VLM-based physics-education scoring) happens upstream; there is no in-tree official-run and the catalog does not mark scores as recomputable.

## Not vendored

- The upstream PhyEduVideo judging pipeline (pinned revision `5a36a13818552095c9ed12c6562f236d5b5011bd` for provenance only).

## License status

No LICENSE file is recorded for the upstream repository in the catalog; verify upstream terms before redistribution.
