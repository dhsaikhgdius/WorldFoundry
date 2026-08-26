#!/usr/bin/env python3
"""Evaluate a WorldFoundry realtime timing trace against a JSON manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from worldfoundry.core.io.paths import project_root



REPO_ROOT = project_root(__file__)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worldfoundry.runtime.realtime_regression import (  # noqa: E402
    RealtimeRegressionManifest,
    evaluate_realtime_manifest,
    read_realtime_trace,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Realtime regression manifest JSON.")
    parser.add_argument("--trace", required=True, help="Studio realtime timing JSONL.")
    parser.add_argument("--output", help="Optional path for the machine-readable result JSON.")
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = RealtimeRegressionManifest.read_json(args.manifest)
    run = evaluate_realtime_manifest(manifest, read_realtime_trace(args.trace))
    payload = json.dumps(run.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    return 0 if run.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
