#!/usr/bin/env python3
"""Parallel layered driver for the C scanner, with resume support.

Layers by max speed m: layer file <outdir>/m_XXX.csv is complete iff its last
line is DONE; incomplete layers are re-run from scratch (cheap, single layer).
Workers = min(procs, 8) to respect the 8-core budget.

usage: driver_scan.py k B theta_num theta_den outdir [procs]
"""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCAN = HERE / "scan"


def layer_done(path: Path) -> bool:
    if not path.exists():
        return False
    lines = path.read_text().strip().splitlines()
    return bool(lines) and lines[-1] == "DONE"


def run_layer(k, m, thn, thd, outdir: Path):
    path = outdir / f"m_{m:03d}.csv"
    if layer_done(path):
        return m, "skip"
    r = subprocess.run([str(SCAN), str(k), str(m), str(m), str(thn), str(thd), str(path)])
    if r.returncode == 2:
        print(f"!!! COUNTEREXAMPLE reported in layer m={m} !!!", flush=True)
        return m, "COUNTEREXAMPLE"
    if r.returncode != 0:
        raise RuntimeError(f"layer m={m} failed rc={r.returncode}")
    return m, "done"


def main():
    k = int(sys.argv[1]); B = int(sys.argv[2])
    thn = int(sys.argv[3]); thd = int(sys.argv[4])
    outdir = Path(sys.argv[5]); outdir.mkdir(parents=True, exist_ok=True)
    procs = min(int(sys.argv[6]) if len(sys.argv) > 6 else 8, 8)
    # ascending: maximizes the complete contiguous prefix [k, B'] at any moment,
    # so an interrupted run still yields a clean "verified up to B'" statement
    layers = sorted(range(k, B + 1))
    with ThreadPoolExecutor(max_workers=procs) as ex:
        for m, status in ex.map(lambda m: run_layer(k, m, thn, thd, outdir), layers):
            if status != "skip":
                print(f"layer m={m}: {status}", flush=True)
    print(f"ALL LAYERS COMPLETE k={k} B={B}", flush=True)


if __name__ == "__main__":
    main()
