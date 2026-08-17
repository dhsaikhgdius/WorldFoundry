#!/usr/bin/env python3
"""Search for higher j=2 spectrum points of k=7: ML = s/(7s+2) for s = 7, 9, 11.

Uses the folded-lift framework validated in Cycle 1/2: candidate instances are
lifts of 6-element folded residue sets W mod q (q = 7s+2) that contain a
"denominator pair" lift v + v' = q. Every candidate is then verified with the
rigorous exact verifier; a hit requires ML == s/q exactly (in lowest terms,
gcd(s,q)=1 checked).
"""

import itertools
from fractions import Fraction
from math import gcd

from lrc_verifier import ml_exact

K = 7
J = 2


def grid_ml_ge(W, q, s):
    """return True iff max_c min_w fold(c w mod q) >= s (early-exit search)"""
    for c in range(1, q // 2 + 1):
        ok = True
        for w in W:
            r = (c * w) % q
            if r > q - r:
                r = q - r
            if r < s:
                ok = False
                break
        if ok:
            return True
    return False


def grid_ml_exact_is(W, q, s):
    best = 0
    for c in range(1, q // 2 + 1):
        mn = q
        for w in W:
            r = (c * w) % q
            if r > q - r:
                r = q - r
            if r < mn:
                mn = r
                if mn <= best:
                    break
        if mn > best:
            best = mn
            if best > s:
                return False
    return best == s


def main():
    import sys
    for s in ([int(x) for x in sys.argv[1:]] or (7, 9, 11)):
        q = K * s + J
        assert gcd(s, q) == 1
        target = Fraction(s, q)
        half = (q - 1) // 2
        # heuristic prune (documented): all known instances have W <= 2s+4
        wmax = min(half, 2 * s + 4)
        hits = []
        n_grid = 0
        # W must contain only residues with fold >= s at c=1?? no - c=1 need not
        # be the witness. But every w in W must have SOME c good; keep it general.
        for W in itertools.combinations(range(1, wmax + 1), 6):
            if not grid_ml_exact_is(W, q, s):
                continue
            n_grid += 1
            # lift: one element w0 of W becomes the pair (w0, q-w0)
            for w0 in W:
                v = sorted(set(W) | {q - w0})
                if len(v) != 7 or gcd(*v[:2]) and False:
                    continue
                g = 0
                for x in v:
                    g = gcd(g, x)
                if g != 1:
                    continue
                ml, t = ml_exact(v)
                if ml == target:
                    hits.append(v)
                    print(f"*** s={s} q={q}: ML={ml} v={v} witness={t}")
        print(f"s={s} (q={q}, wmax={wmax}): grid-sets={n_grid}, exact hits={len(hits)}", flush=True)


if __name__ == "__main__":
    main()
