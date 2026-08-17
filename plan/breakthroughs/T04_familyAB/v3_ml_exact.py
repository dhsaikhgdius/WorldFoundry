#!/usr/bin/env python3
"""V3 from-scratch exact ML verifier (independent implementation).

Mathematical basis, re-derived independently for this audit:
  f(t) = min_i ||t v_i|| is continuous, 1-periodic, piecewise linear with every
  piece of slope +-v_i != 0.  At a global max t*, an ascending active piece
  (t v_i - a) meets a descending one (b - t v_j), so t* = (a+b)/(v_i+v_j)
  (i = j covers tent apexes, a+b odd).  Hence
      ML(v) = max{ f(c/d) : d = v_i + v_j (i <= j), 0 <= c <= floor(d/2) }
  using f(c/d) = f((d-c)/d) symmetry, computed in exact integer arithmetic.

This file re-verifies, with this independent implementation:
  * a battery of literature values (Goddyn-Wong tight instances, Fan-Sun 5/22, 8/51),
  * Theorem A equality ML(1,2,3,4,5,7,6r+12) = (r+2)/(6r+17) for r = 1..25
    (thread only verified r <= 12 by enumeration; >= 13 is covered by their
    symbolic certificates, so the overlap 13..25 cross-checks the certificates),
  * Theorem B equality ML(1,3,4,5,7,11,6r+24) = (r+4)/(6r+31) for r = 1..25,
  * the new spectrum point ML(2,7,9,11,13,20,54) = 9/65 (witness 11/65),
  * the k=6 anchor sets ML(1,2,3,4,5,7) = ML(1,3,4,5,7,11) = 1/6,
  * the j=2 instances 3/23 (x2) and 5/37 quoted in findings.

Exit 0 iff everything matches.
"""

import sys
from fractions import Fraction
from math import gcd


def ml_v3(speeds):
    """Exact ML and witness, independent implementation (see module docstring)."""
    v = sorted(set(map(int, speeds)))
    assert all(x >= 1 for x in v)
    dens = sorted({v[i] + v[j] for i in range(len(v)) for j in range(i, len(v))})
    best_n, best_d, best_t = 0, 1, Fraction(0)
    for d in dens:
        for c in range(0, d // 2 + 1):
            g = gcd(c, d)
            cc, dd = c // g, d // g
            m = dd  # dd * f(cc/dd) upper init
            for vi in v:
                x = (cc * vi) % dd
                x = min(x, dd - x)
                if x < m:
                    m = x
                    if m * best_d <= best_n * dd:
                        break
            if m * best_d > best_n * dd:
                best_n, best_d, best_t = m, dd, Fraction(cc, dd)
    g = gcd(best_n, best_d)
    return Fraction(best_n, best_d), best_t


def expect(tag, speeds, want, want_t=None):
    ml, t = ml_v3(speeds)
    ok = (ml == want) and (want_t is None or t == want_t)
    print(f"  [{'OK' if ok else 'FAIL'}] {tag}: ML{tuple(speeds)} = {ml} "
          f"(want {want}){f', witness {t}' if want_t else ''}")
    return ok


ok = True
print("battery: literature values")
ok &= expect("tight k=2", [1, 2], Fraction(1, 3))
ok &= expect("tight k=3", [1, 2, 3], Fraction(1, 4))
ok &= expect("tight k=4", [1, 2, 3, 4], Fraction(1, 5))
ok &= expect("GW k=4", [1, 3, 4, 7], Fraction(1, 5))
ok &= expect("GW k=5", [1, 3, 4, 5, 9], Fraction(1, 6))
ok &= expect("tight k=7", [1, 2, 3, 4, 5, 6, 7], Fraction(1, 8))
ok &= expect("GW k=7 a", [1, 2, 3, 4, 5, 7, 12], Fraction(1, 8))
ok &= expect("GW k=7 b", [1, 4, 5, 6, 7, 11, 13], Fraction(1, 8))
ok &= expect("Fan-Sun k=4", [1, 7, 8, 15], Fraction(5, 22))
ok &= expect("Fan-Sun k=6", [5, 6, 11, 17, 23, 28], Fraction(8, 51))

print("k=6 anchor sets (claimed ML = 1/6 exactly)")
ok &= expect("anchor A", [1, 2, 3, 4, 5, 7], Fraction(1, 6))
ok &= expect("anchor B", [1, 3, 4, 5, 7, 11], Fraction(1, 6))

print("j=2 instances from findings")
ok &= expect("3/23 #1", [1, 2, 3, 4, 5, 7, 18], Fraction(3, 23))
ok &= expect("3/23 #2", [1, 3, 4, 5, 7, 13, 18], Fraction(3, 23))
ok &= expect("5/37", [1, 3, 4, 5, 7, 11, 30], Fraction(5, 37))

print("NEW spectrum point")
ok &= expect("9/65", [2, 7, 9, 11, 13, 20, 54], Fraction(9, 65), want_t=Fraction(11, 65))

print("Theorem A equality, r = 1..25 (independent enumeration)")
for rr in range(1, 26):
    ok &= expect(f"A r={rr}", [1, 2, 3, 4, 5, 7, 6 * rr + 12],
                 Fraction(rr + 2, 6 * rr + 17))

print("Theorem B equality, r = 1..25 (independent enumeration)")
for rr in range(1, 26):
    ok &= expect(f"B r={rr}", [1, 3, 4, 5, 7, 11, 6 * rr + 24],
                 Fraction(rr + 4, 6 * rr + 31))

print(f"\nV3 EXACT-ML AUDIT: {'ALL CHECKS PASSED' if ok else 'FAILURES FOUND'}")
sys.exit(0 if ok else 1)
