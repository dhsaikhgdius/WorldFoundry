#!/usr/bin/env python3
"""Rigorous exact verifier for the Lonely Runner Conjecture (thread T04).

Definitions
-----------
For a set of distinct positive integers v_1..v_k let

    f(t)  = min_i || t v_i ||        (||x|| = distance to nearest integer)
    ML(v) = sup_t f(t)               ("maximum loneliness")

LRC for k runners (plus the stationary observer) states ML(v) >= 1/(k+1).

Critical-time theorem (used by ml_exact; proof)
-----------------------------------------------
f is continuous, piecewise linear, periodic with period 1, and every linear
piece of f is a piece of some ||t v_i||, hence has slope +v_i or -v_i, never 0.
Therefore the maximum of f is attained at a local maximum t*, where some
ascending piece (slope +v_i, i.e. t v_i - a = f(t) for an integer a) meets
some descending piece (slope -v_j, i.e. b - t v_j = f(t) for an integer b).

* If i = j the two pieces belong to the same tent ||t v_i|| and t* is its
  apex:  t* v_i = a + 1/2, so t* = (2a+1) / (2 v_i).
* If i != j, continuity gives t* v_i - a = b - t* v_j, so
  t* = (a+b) / (v_i + v_j).

Hence ML(v) = max over the FINITE candidate set
    T(v) = { c/d : d in D(v), 0 <= c <= d },
    D(v) = { 2 v_i } union { v_i + v_j : i < j },
and evaluating f on T(v) with exact integer arithmetic gives the exact sup:
    f(c/d) = min_i min(c*v_i mod d, d - c*v_i mod d) / d.

Everything below is exact integer / Fraction arithmetic. No floating point.

A second, independent implementation (ml_sweep) uses the *superset* of all
breakpoints of f (tent apexes and ALL pairwise crossings, including the
difference-denominator ones t = m/|v_i - v_j|) and serves as cross-check.
"""

from fractions import Fraction
from math import gcd
import sys


def _f_num(c: int, d: int, v) -> int:
    """Return d * f(c/d) = min_i min(c*v_i mod d, d - (c*v_i mod d)) (exact int)."""
    m = d
    for vi in v:
        r = (c * vi) % d
        if d - r < r:
            r = d - r
        if r < m:
            m = r
            if m == 0:
                return 0
    return m


def candidate_denominators(v):
    """D(v) = {2 v_i} union {v_i + v_j}."""
    dset = set()
    k = len(v)
    for i in range(k):
        dset.add(2 * v[i])
        for j in range(i + 1, k):
            dset.add(v[i] + v[j])
    return sorted(dset)


def ml_exact(v):
    """Exact ML(v) and a witness time, via the critical-time theorem.

    Returns (ML, t) as Fractions with f(t) = ML = sup_s f(s), exactly.
    """
    v = sorted(set(int(x) for x in v))
    assert all(x > 0 for x in v), "speeds must be positive integers"
    best_num, best_den = 0, 1          # current max as fraction best_num/best_den
    best_t = Fraction(0)
    for d in candidate_denominators(v):
        for c in range(1, d // 2 + 1):     # f(c/d) = f((d-c)/d): symmetry
            m = _f_num(c, d, v)
            if m * best_den > best_num * d:     # m/d > best  (exact cross-multiply)
                best_num, best_den = m, d
                best_t = Fraction(c, d)
    return Fraction(best_num, best_den), best_t


def ml_sweep(v):
    """Independent second implementation: evaluate f at ALL breakpoints.

    Breakpoints of f are contained in { apexes a/(2 v_i) } union
    { crossings m/(v_i+v_j) } union { crossings m/|v_i-v_j| }.
    Since linear pieces have nonzero slope, max f is attained at a breakpoint.
    Uses Fractions throughout (slower; for cross-validation only).
    """
    v = sorted(set(int(x) for x in v))
    dens = set()
    k = len(v)
    for i in range(k):
        dens.add(2 * v[i])
        for j in range(i + 1, k):
            dens.add(v[i] + v[j])
            if v[j] != v[i]:
                dens.add(v[j] - v[i])
    best = Fraction(0)
    best_t = Fraction(0)
    for d in sorted(dens):
        for c in range(1, d + 1):
            t = Fraction(c, d)
            val = min(min((t * vi) % 1, 1 - (t * vi) % 1) for vi in v)
            if val > best:
                best, best_t = val, t
    return best, best_t


def has_witness(v, num: int, den: int):
    """Exact decision: does there exist t with f(t) >= num/den ?
    Equivalent to ML(v) >= num/den since the sup is attained.
    Returns a witness Fraction or None."""
    v = sorted(set(int(x) for x in v))
    for d in candidate_denominators(v):
        for c in range(1, d // 2 + 1):
            if den * _f_num(c, d, v) >= num * d:
                return Fraction(c, d)
    return None


def normalize(v):
    """Sort, dedupe, divide by overall gcd (ML-invariant reductions)."""
    v = sorted(set(int(x) for x in v))
    g = 0
    for x in v:
        g = gcd(g, x)
    return [x // g for x in v]


if __name__ == "__main__":
    speeds = [int(a) for a in sys.argv[1:]]
    if not speeds:
        print("usage: lrc_verifier.py v1 v2 ... vk")
        sys.exit(1)
    k = len(set(speeds))
    ml, t = ml_exact(speeds)
    bound = Fraction(1, k + 1)
    print(f"v = {sorted(set(speeds))}  (k={k})")
    print(f"ML(v) = {ml} = {float(ml):.6f}   attained at t = {t}")
    print(f"1/(k+1) = {bound} = {float(bound):.6f}")
    if ml < bound:
        print("*** COUNTEREXAMPLE TO LRC ***")
    elif ml == bound:
        print("TIGHT instance (ML = 1/(k+1))")
    else:
        print("LR property holds strictly")
