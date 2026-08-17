#!/usr/bin/env python3
"""Two candidate infinite families crossing the k=7 spectrum, with exact checks.

Family A: v(r) = (1,2,3,4,5,7,6r+12)
Claim A-lower (PROVED symbolically for all r >= 1, verified here for many r):
    f(t_r) = (r+2)/(6r+17) at t_r = (r+3)/(6r+17), hence ML >= (r+2)/(6r+17).
    Symbolic proof (each runner, exact fractional parts, r >= 1):
      v=1: t_r < 1/2                        -> ||t|| = (r+3)/(6r+17) >= (r+2)/(6r+17)
      v=2: 2t_r < 1/2                       -> (2r+6)/(6r+17) >= ...
      v=3: 1/2 < 3t_r < 1                   -> (3r+8)/(6r+17) >= ...
      v=4: 1/2 < 4t_r < 1                   -> (2r+5)/(6r+17) >= ...
      v=5: 1/2 < 5t_r < 1                   -> (r+2)/(6r+17)  (MIN, equality)
      v=7: 7t_r = 1 + (r+4)/(6r+17)         -> (r+4)/(6r+17) >= ...
      v=6r+12: (6r+12)t_r = (r+2) + (r+2)/(6r+17) -> (r+2)/(6r+17) (MIN, equality)
Claim A-upper (verified exactly for r <= R_MAX only; open for general r):
    ML(v(r)) = (r+2)/(6r+17)   [j-offset = 3 - r: r=1 gives j=2 value 3/23]

Family B: v(r) = (1,3,4,5,7,11,6r+24), conjectured ML = (r+4)/(6r+31) for r>=1
    (r=1 gives the j=2 value 5/37); verified exactly for r <= R_MAX.
"""

from fractions import Fraction
from lrc_verifier import ml_exact

R_MAX = 12


def check_family_A():
    print("Family A: {1,2,3,4,5,7,6r+12}")
    for r in range(1, R_MAX + 1):
        v = [1, 2, 3, 4, 5, 7, 6 * r + 12]
        t = Fraction(r + 3, 6 * r + 17)
        want = Fraction(r + 2, 6 * r + 17)
        got_f = min(min((t * vi) % 1, 1 - (t * vi) % 1) for vi in v)
        assert got_f == want, (r, got_f, want)
        ml, _ = ml_exact(v)
        status = "ML == lower bound (exact)" if ml == want else f"ML={ml} > bound!"
        print(f"  r={r:2d}: f(t_r)={want} exact; {status}")
        assert ml == want, f"upper-bound claim fails at r={r}: ML={ml}"


def check_family_B():
    print("Family B: {1,3,4,5,7,11,6r+24}")
    for r in range(1, R_MAX + 1):
        v = [1, 3, 4, 5, 7, 11, 6 * r + 24]
        want = Fraction(r + 4, 6 * r + 31)
        ml, t = ml_exact(v)
        assert ml == want, f"family B claim fails at r={r}: ML={ml} want {want}"
        print(f"  r={r:2d}: ML={ml} = (r+4)/(6r+31) exact, witness t={t}")


if __name__ == "__main__":
    check_family_A()
    check_family_B()
    print(f"OK: both families verified exactly for 1 <= r <= {R_MAX}")
