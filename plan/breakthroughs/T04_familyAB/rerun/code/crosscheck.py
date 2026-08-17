#!/usr/bin/env python3
"""Cross-validate the C scanner against the independent Python verifier.

For small (k, B) domains, enumerate ALL gcd-normalized speed sets in Python,
compute exact ML with lrc_verifier.ml_exact (Fraction arithmetic), and demand
that the C scanner's output (instances with ML < theta, plus their exact ML
values) matches the Python ground truth EXACTLY - both the set of reported
instances and every reported fraction.
"""

import itertools
import subprocess
import sys
import tempfile
from fractions import Fraction
from math import gcd
from pathlib import Path

from lrc_verifier import ml_exact

HERE = Path(__file__).resolve().parent
SCAN = HERE / "scan"


def python_ground_truth(k, B, theta):
    res = {}
    for v in itertools.combinations(range(1, B + 1), k):
        g = 0
        for x in v:
            g = gcd(g, x)
        if g != 1:
            continue
        ml, _ = ml_exact(v)
        if ml < theta:
            res[v] = ml
    return res


def c_scan(k, B, theta):
    with tempfile.NamedTemporaryFile(mode="r", suffix=".csv", delete=False) as f:
        outpath = f.name
    subprocess.run(
        [str(SCAN), str(k), str(k), str(B), str(theta.numerator), str(theta.denominator), outpath],
        check=True,
    )
    res = {}
    for line in Path(outpath).read_text().splitlines():
        if line.startswith(("STATS", "DONE")):
            continue
        parts = [int(x) for x in line.split(",")]
        m, v, num, den = parts[0], tuple(parts[1 : 1 + k]), parts[-2], parts[-1]
        assert v[-1] == m
        res[v] = Fraction(num, den)
    return res


def check(k, B, theta):
    py = python_ground_truth(k, B, theta)
    cc = c_scan(k, B, theta)
    assert set(py) == set(cc), (
        f"instance sets differ (k={k},B={B}): "
        f"py-only={set(py)-set(cc)} c-only={set(cc)-set(py)}"
    )
    for v in py:
        assert py[v] == cc[v], f"ML mismatch on {v}: py={py[v]} c={cc[v]}"
    print(f"[ok] k={k} B={B} theta={theta}: {len(py)} near-tight instances, C == Python exactly")


if __name__ == "__main__":
    check(3, 16, Fraction(302, 1000))
    check(4, 14, Fraction(232, 1000))
    check(5, 12, Fraction(188, 1000))
    check(2, 30, Fraction(43, 100))
    print("CROSSCHECK PASSED: C scanner agrees with independent Python verifier")
