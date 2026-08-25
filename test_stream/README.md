# Streaming inference demos (manual)

This directory contains **manual smoke scripts** for streaming / incremental inference.
They are **not** collected by pytest (`pytest.ini` lists `test_stream` under
`norecursedirs`).

## How to run

From the repository root:

```bash
PYTHONPATH=. python test_stream/test_wan2p2_stream.py
```

Each script targets one model integration. Expect GPU memory, local checkpoints,
and sometimes network access depending on the model.

## Relationship to other test trees

| Directory | Purpose |
|-----------|---------|
| `test/eval_core/` | Evaluation-framework release-gate tests (CPU) |
| `test/` | Mixed pytest tests plus manual demos (`collect_ignore` in `test/conftest.py`) |
| `tests/` | Source-mirroring unit tests |
| `test_stream/` | Manual streaming inference demos only |
