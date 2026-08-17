#!/usr/bin/env python3
"""Aggregate scan outputs, re-verify every reported instance with the
independent Python verifier, classify spectrum values, and emit reports.

usage: analyze.py k resultsdir [reportdir]
"""

import csv
import json
import sys
from fractions import Fraction
from pathlib import Path

from lrc_verifier import ml_exact


def spectrum_offset(ml: Fraction, k: int):
    """If ml == s/(ks+j) for integers s>=1, 0<j, return (s, j); else None.
    Given ml = p/q in lowest terms, s/(ks+j) = p/q with gcd(s, ks+j)=gcd(s,j):
    take s = p*m, ks+j = q*m for the multiplier m>=1; j = q*m - k*p*m = m(q-kp).
    Lowest-terms solution is m=1: s=p, j=q-k*p (need j>=1)."""
    p, q = ml.numerator, ml.denominator
    j = q - k * p
    if j >= 1:
        return p, j
    return None


def load_instances(k: int, resultsdir: Path):
    inst = []
    stats = {"combos": 0, "hard": 0, "printed": 0, "secs": 0.0, "layers": 0}
    counterexample_lines = []
    for f in sorted(resultsdir.glob("m_*.csv")):
        lines = f.read_text().strip().splitlines()
        assert lines and lines[-1] == "DONE", f"incomplete layer file {f}"
        for line in lines:
            if line == "DONE":
                continue
            if line.startswith("COUNTEREXAMPLE"):
                counterexample_lines.append((f.name, line))
                continue
            if line.startswith("STATS"):
                parts = dict(p.split("=") for p in line.split(",")[1:])
                stats["combos"] += int(parts["combos"])
                stats["hard"] += int(parts["hard"])
                stats["printed"] += int(parts["printed"])
                stats["secs"] += float(parts["secs"])
                stats["layers"] += 1
                continue
            xs = [int(x) for x in line.split(",")]
            m, v, num, den = xs[0], tuple(xs[1 : 1 + k]), xs[-2], xs[-1]
            inst.append({"m": m, "v": v, "ml": Fraction(num, den)})
    return inst, stats, counterexample_lines


def main():
    k = int(sys.argv[1])
    resultsdir = Path(sys.argv[2])
    reportdir = Path(sys.argv[3]) if len(sys.argv) > 3 else resultsdir
    inst, stats, cex = load_instances(k, resultsdir)
    bound = Fraction(1, k + 1)

    # 1. independent exact re-verification of every reported instance
    bad = []
    for it in inst:
        ml2, t = ml_exact(it["v"])
        if ml2 != it["ml"]:
            bad.append((it, ml2))
        it["witness_t"] = t
    assert not bad, f"RE-VERIFICATION FAILED: {bad[:5]}"

    # 2. classification
    tight = [it for it in inst if it["ml"] == bound]
    below = [it for it in inst if it["ml"] < bound]      # would be counterexamples
    near = [it for it in inst if it["ml"] > bound]
    offsets = {}
    unclassified = []
    for it in inst:
        so = spectrum_offset(it["ml"], k)
        it["spec"] = so
        if so is None:
            unclassified.append(it)
        else:
            offsets.setdefault(so[1], set()).add(it["ml"])

    inst.sort(key=lambda it: (it["ml"], it["v"]))

    reportdir.mkdir(parents=True, exist_ok=True)
    with open(reportdir / f"instances_k{k}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ml", "ml_float", "s", "offset_j", "v", "witness_t"])
        for it in inst:
            s, j = it["spec"] if it["spec"] else ("", "")
            w.writerow([str(it["ml"]), f"{float(it['ml']):.8f}", s, j,
                        " ".join(map(str, it["v"])), str(it["witness_t"])])

    summary = {
        "k": k,
        "layers_complete": stats["layers"],
        "combos_scanned": stats["combos"],
        "hard_instances": stats["hard"],
        "reported_instances": len(inst),
        "cpu_seconds_scan": round(stats["secs"], 1),
        "counterexamples_below_bound": len(below) + len(cex),
        "tight_count": len(tight),
        "tight_instances": [" ".join(map(str, it["v"])) for it in tight],
        "observed_offsets_j": {str(j): [str(x) for x in sorted(vals)] for j, vals in sorted(offsets.items())},
        "non_spectrum_values": [str(it["ml"]) for it in unclassified],
        "reverified_with_python_fraction": True,
    }
    (reportdir / f"summary_k{k}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if below or cex:
        print("*** ALERT: ML < 1/(k+1) PRESENT — POTENTIAL COUNTEREXAMPLE ***")


if __name__ == "__main__":
    main()
