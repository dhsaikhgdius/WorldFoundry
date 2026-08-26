# SY-05: ``*_runtime`` shipping contract

WorldFoundry vendors dozens of ``*_runtime`` trees. Only an explicit subset
should ship in the wheel/sdist.

| Class | Meaning | Source of truth |
| --- | --- | --- |
| `packaged` | Listed in `runtime_source_manifest.yaml` **and** mirrored in `package-data` + `MANIFEST.in` | Must stay self-consistent |
| `checkout_only` | Default for every other discovered ``*_runtime`` | Editable checkout / SCM only |

Contract tests live in `test/eval_core/test_runtime_source_manifest.py`. Promoting a
tree to `packaged` requires updating the YAML plus both packaging surfaces in
the same PR.
