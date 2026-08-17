#!/usr/bin/env python3
"""V3 independent re-verification of Lemma 1 (lower bound) of Theorems A and B.

Claim: with t*(r) = (r+3)/(6r+17) [A] resp. (r+5)/(6r+31) [B], every speed
v in V u {w(r)} satisfies ||v * t*(r)|| >= alpha(r) for all integers r >= 1,
where alpha = (r+2)/(6r+17) [A] resp. (r+4)/(6r+31) [B].

NOTE: these witness inequalities are NOT contained in certificates_AB.txt
(the file holds only the upper-bound covering certificates, despite the
docstring of export_certificates.py), so this script re-derives and verifies
them from scratch with sympy: for each speed we find the nearest-integer
polynomial n(r) ourselves, then prove |v t - n| <= 1/2 and |v t - n| >= alpha
for ALL REAL r >= 1 by exact real-root isolation (complete criterion, not the
shifted-coefficient sufficient test used by the thread).  Equality achievers
(the speeds with ||v t*|| == alpha identically) are identified exactly.

Exit 0 iff both families verify.
"""

import sys
from fractions import Fraction

import sympy as sp

r = sp.Symbol("r", real=True)


def nonneg_on_ray(expr_poly, a) -> bool:
    """Exact: polynomial >= 0 for all real x >= a (complete decision)."""
    P = sp.Poly(sp.expand(expr_poly), r)
    if P.is_zero:
        return True
    if P.degree() == 0:
        return P.coeffs()[0] >= 0
    if P.LC() < 0 or P.eval(sp.Rational(a)) < 0:
        return False
    for root, mult in sp.roots(P.as_expr(), r).items():
        if root.is_real and sp.simplify(root - a).is_positive and mult % 2 == 1:
            return False
    return True


def verify_family(name, V, w_expr, t_expr, alpha_expr, r0=1):
    print(f"=== {name}: t* = {t_expr}, alpha = {alpha_expr}, prove for all real r >= {r0} ===")
    q = sp.denom(alpha_expr)                       # 6r+17 / 6r+31, positive on r>=1
    ok_all = True
    achievers = []
    speeds = [(sp.Integer(v), str(v)) for v in V] + [(w_expr, f"w={w_expr}")]
    for v_expr, label in speeds:
        x = sp.cancel(v_expr * t_expr)             # v * t as rational function
        # find candidate nearest integer n(r): affine fit from integer samples
        samples = [1, 2, 3, 50, 1000]
        ns = []
        for rv in samples:
            val = sp.Rational(sp.nsimplify(x.subs(r, rv)))
            fr = Fraction(int(val.p), int(val.q))
            n = round(fr)                          # exact rounding of a Fraction
            ns.append(n)
        n1 = ns[1] - ns[0]
        n0 = ns[0] - n1 * samples[0]
        affine_ok = all(n1 * rv + n0 == n for rv, n in zip(samples, ns))
        n_expr = n1 * r + n0
        diff = sp.cancel(x - n_expr)               # (v t - n) as num/den
        num, den = sp.fraction(sp.together(diff))
        if den.subs(r, 10**6) < 0:                 # normalize denominator positive
            num, den = -num, -den
        den_pos = nonneg_on_ray(den - sp.Rational(1, 10**9), r0)        # den > 0
        # orientation: sign of diff at a large sample (then PROVED constant below)
        sgn = 1 if num.subs(r, 10**6) >= 0 else -1
        num_s = sp.expand(sgn * num)
        c_sign = nonneg_on_ray(num_s, r0)                          # |diff| == sgn*diff
        # dist >= alpha  <=>  sgn*num/den - alpha >= 0
        gap = sp.together(sgn * num / den - alpha_expr)
        gnum, gden = sp.fraction(gap)
        gden_pos = nonneg_on_ray(gden - 1, r0) or nonneg_on_ray(gden, r0)
        c_alpha = nonneg_on_ray(sp.expand(gnum), r0)
        is_eq = sp.expand(gnum) == 0
        # dist <= 1/2  <=>  1/2 - sgn*num/den >= 0
        half = sp.together(sp.Rational(1, 2) - sgn * num / den)
        hnum, hden = sp.fraction(half)
        hden_pos = nonneg_on_ray(hden - 1, r0) or nonneg_on_ray(hden, r0)
        c_half = nonneg_on_ray(sp.expand(hnum), r0)
        ok = affine_ok and den_pos and c_sign and c_alpha and c_half and gden_pos and hden_pos
        ok_all &= ok
        if is_eq:
            achievers.append(label)
        print(f"  [{'OK' if ok else 'FAIL'}] v={label:12s} n(r)={n_expr}; "
              f"affine={affine_ok}, sign-const={c_sign}, >=alpha={c_alpha}"
              f"{' (EQUALITY)' if is_eq else ''}, <=1/2={c_half}")
    print(f"  equality achievers: {achievers}")
    # exact numeric spot-check of f(t*) == alpha at many integer r (Fractions)
    bad = []
    for rv in list(range(1, 200)) + [10**3, 10**6]:
        tv = Fraction(int(sp.numer(t_expr).subs(r, rv)), int(sp.denom(t_expr).subs(r, rv)))
        av = Fraction(int(sp.numer(alpha_expr).subs(r, rv)), int(sp.denom(alpha_expr).subs(r, rv)))
        speeds_num = [int(v) for v in V] + [int(w_expr.subs(r, rv))]
        fval = min(min((tv * v) % 1, 1 - (tv * v) % 1) for v in speeds_num)
        if fval != av:
            bad.append(rv)
    print(f"  exact f(t*(r)) == alpha(r) at r in [1,199] u {{1e3,1e6}}: "
          f"{'all equal' if not bad else f'MISMATCH at {bad[:5]}'}")
    ok_all &= not bad
    return ok_all


okA = verify_family("Family A (1,2,3,4,5,7,6r+12)", [1, 2, 3, 4, 5, 7],
                    6 * r + 12, (r + 3) / (6 * r + 17), (r + 2) / (6 * r + 17))
print()
okB = verify_family("Family B (1,3,4,5,7,11,6r+24)", [1, 3, 4, 5, 7, 11],
                    6 * r + 24, (r + 5) / (6 * r + 31), (r + 4) / (6 * r + 31))
print(f"\nV3 WITNESS (LOWER BOUND) AUDIT: {'ALL CHECKS PASSED' if okA and okB else 'FAILURES FOUND'}")
sys.exit(0 if (okA and okB) else 1)
