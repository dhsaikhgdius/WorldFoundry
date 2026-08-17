#!/usr/bin/env python3
"""Machine-verified symbolic LOWER-bound prover for the two families.

Claim: at t(r) = (T1*r+T0)/(D*r+E), every runner v in V u {w(r)} satisfies
       || v * t(r) || >= alpha(r)   for all r >= 1,
hence ML >= alpha(r).  For each speed we determine the nearest integer
n(r) = n1*r + n0 (affine; fitted from samples, then verified symbolically),
and certify BOTH
     v*t - n >= -alpha      and      v*t - n <= alpha        -- if |v*t-n| is
the distance, that is ||v t|| = |v t - n| >= alpha becomes either
     v*t - n >= alpha   (if v*t >= n on r>=1)   or   n - v*t >= alpha.
We also certify that n(r) IS the nearest integer via |v t - n| <= 1/2.
All checks reduce to integer-polynomial nonnegativity on r >= 1 via the
substitution r = s+1 (exact arithmetic, sufficient criterion; sound).
"""

from fractions import Fraction
import sys

from cover_prover import (RF, pmul, padd, psub, nonneg_on_r_ge, peval,
                          FAMILY_A, FAMILY_B)


def rf_geq_zero(f: RF, r0=1):
    return nonneg_on_r_ge(f.num, r0) if all(c >= 0 for c in f.den) else None


def rf_sub(f: RF, g: RF) -> RF:
    return RF(psub(pmul(f.num, g.den), pmul(g.num, f.den)), pmul(f.den, g.den))


def prove_lower(fam, t_num, t_den, r0=1, samples=(1, 5, 29, 137)):
    """t(r) = t_num/t_den polys; verify min_v ||v t|| >= alpha for r >= r0."""
    t = RF(t_num, t_den)
    alpha = fam.alpha()
    speeds = [( [v], f"v={v}") for v in fam.V] + [(fam.w, f"w={fam.w[1]}r+{fam.w[0]}")]
    ok = True
    achievers = []
    for vpoly, label in speeds:
        vt = RF(pmul(vpoly, t.num), t.den)
        # fit nearest integer n(r) = n1 r + n0 from two samples, verify on all
        ns = []
        for rs in samples:
            val = vt.eval(Fraction(rs))
            n = val.__round__()
            ns.append(n)
        n1 = Fraction(ns[1] - ns[0], samples[1] - samples[0])
        n0 = Fraction(ns[0]) - n1 * samples[0]
        if n1.denominator != 1 or n0.denominator != 1 or \
           any(int(n1) * rs + int(n0) != n for rs, n in zip(samples, ns)):
            print(f"  [{label}] nearest integer not affine: {ns}")
            ok = False
            continue
        npoly = [int(n0), int(n1)]
        diff = rf_sub(vt, RF(npoly, [1]))          # v t - n
        # determine sign at a sample
        sgn = 1 if diff.eval(Fraction(samples[-1])) >= 0 else -1
        dist = diff if sgn > 0 else RF([-c for c in diff.num], diff.den)
        # (a) same sign for all r >= r0  (dist >= 0)
        c1 = nonneg_on_r_ge(dist.num, r0)
        # (b) dist >= alpha
        gap = rf_sub(dist, alpha)
        c2 = nonneg_on_r_ge(pmul(gap.num, [1]), r0) if all(True for _ in [0]) else False
        c2 = nonneg_on_r_ge(gap.num, r0)
        # (c) n is nearest: dist <= 1/2  <=>  1/2 - dist >= 0
        half_minus = rf_sub(RF([1], [2]), dist)
        c3 = nonneg_on_r_ge(half_minus.num, r0)
        # exact equality detection (achiever): gap == 0 identically?
        is_achiever = all(c == 0 for c in gap.num)
        if not (c1 and c2 and c3):
            print(f"  [{label}] FAIL (sign={c1}, >=alpha={c2}, nearest={c3})")
            ok = False
        else:
            achievers.append((label, "EQUALITY" if is_achiever else "strict"))
    if ok:
        print(f"[{fam.name}]")
        print(f"  LOWER BOUND PROVED for all r >= {r0}: f(t(r)) >= alpha(r), with")
        for label, kind in achievers:
            print(f"    {label}: ||v t|| >= alpha  ({kind})")
    return ok


if __name__ == "__main__":
    # Family A: t = (r+3)/(6r+17)
    okA = prove_lower(FAMILY_A, [3, 1], [17, 6])
    print()
    # Family B: t = (r+5)/(6r+31)
    okB = prove_lower(FAMILY_B, [5, 1], [31, 6])
    sys.exit(0 if (okA and okB) else 1)
