#!/usr/bin/env python3
"""Structural analysis of near-tight instances: witness denominators,
congruence skeletons, additive structure. Reads instances_k*.csv."""

import csv
import sys
from fractions import Fraction
from math import gcd
from pathlib import Path


def analyze(k: int, resultsdir: Path):
    path = resultsdir / f"instances_k{k}.csv"
    rows = list(csv.DictReader(open(path)))
    print(f"=== k={k}: {len(rows)} near-tight instances ===")
    for row in rows:
        v = [int(x) for x in row["v"].split()]
        ml = Fraction(row["ml"])
        t = Fraction(row["witness_t"])
        q = ml.denominator
        # congruence skeleton: speeds mod q and the witness numerator
        resid = sorted(set(x % q for x in v))
        # additive closure pairs: v_c = v_a + v_b
        sums = {(a, b): a + b for i, a in enumerate(v) for b in v[i + 1:] if a + b in v}
        # arithmetic progressions of length >= 3
        vs = set(v)
        aps = []
        for i, a in enumerate(v):
            for b in v[i + 1:]:
                d = b - a
                run = [a, b]
                while run[-1] + d in vs:
                    run.append(run[-1] + d)
                if len(run) >= 3:
                    aps.append(tuple(run))
        aps = {max((ap for ap in aps if set(ap) <= set(ap2)), key=len) for ap2 in aps for ap in [ap2]}
        print(f"  v={v}  ML={ml}  t={t} (den {t.denominator})")
        print(f"      speeds mod {q}: {resid}   additive v_c=v_a+v_b count: {len(sums)}   APs(len>=3): {sorted(aps) if aps else '-'}")


if __name__ == "__main__":
    rd = Path(__file__).resolve().parent.parent / "results"
    for k in ([int(sys.argv[1])] if len(sys.argv) > 1 else [4, 5, 6, 7]):
        analyze(k, rd)
