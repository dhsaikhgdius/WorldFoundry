#!/usr/bin/env python3
"""Directed search for k=7 spectrum values with offset j=3, i.e. ML = s/(7s+3).

Framework (validated on all in-domain j>=2 instances): if ML(v) = s/q with
q = 7s+j then the optimum is attained on the q-grid and depends only on the
+/- folded residues W of v mod q. For j >= 2 the known instances have a pair
v_a + v_b = q (folded coincidence), so |W| = 6 effective residues.

Search: for q = 7s+3, enumerate 6-subsets W of [1..(q-1)/2] with
grid-ML(W, q) = s, lift to 7 distinct speeds {W} u {q - w} for each w in W,
and verify ML exactly with the rigorous verifier. Small s only (cheap).
"""

import itertools
from fractions import Fraction

from lrc_verifier import ml_exact

k = 7
J = 3


def grid_ml(W, q):
    best = 0
    for c in range(1, q // 2 + 1):
        mn = q
        for w in W:
            r = (c * w) % q
            if q - r < r:
                r = q - r
            if r < mn:
                mn = r
                if mn <= best:
                    break
        if mn > best:
            best = mn
    return best


def main():
    found = []
    for s in range(2, 6):
        q = k * s + J
        half = (q - 1) // 2
        target = Fraction(s, q)
        hits = 0
        for W in itertools.combinations(range(1, half + 1), 6):
            if grid_ml(W, q) != s:
                continue
            hits += 1
            for w in W:
                v = sorted(set(W) | {q - w})
                if len(v) != 7:
                    continue
                ml, t = ml_exact(v)
                if ml == target:
                    found.append((s, q, v))
                    print(f"*** j=3 HIT: s={s} q={q} v={v} ML={ml} t={t}")
        print(f"s={s} (q={q}): grid-solutions={hits}, exact j=3 hits so far={len(found)}")
    if not found:
        print("no j=3 instance found in this (small-s, coincidence-lift) search space")


if __name__ == "__main__":
    main()
