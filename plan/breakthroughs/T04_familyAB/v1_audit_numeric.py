#!/usr/bin/env python3
"""V1 adversarial audit — independent NUMERIC verification (exact rationals).

Fully self-contained: does NOT import anything from T04's code/ directory.

Mathematical basis (re-derived independently by V1):
  f(t) = min_i ||v_i t|| is continuous, piecewise linear, 1-periodic,
  every linear piece has slope +-v_i != 0. Hence sup f is attained at a
  breakpoint that is a local max: either a tent apex t=(2a+1)/(2 v_i) or a
  crossing of an ascending piece of ||v_i t|| with a descending piece of
  ||v_j t||: v_i t - a = b - v_j t  =>  t=(a+b)/(v_i+v_j).
  So ML(v) = max{ f(c/d) : d in D(v), 0<=c<=d },  D(v)={2v_i} u {v_i+v_j}.
  Safety net: we also evaluate on the FULL grid d=1..2*max(v) (a superset);
  any grid value is <= ML, and the D(v)-grid already attains ML, so the two
  maxima must agree.
"""
from fractions import Fraction
import sys

def f_at(c, d, v):
    """d * f(c/d) as an integer (exact)."""
    m = d
    for vi in v:
        x = (c * vi) % d
        if d - x < x:
            x = d - x
        if x < m:
            m = x
            if m == 0:
                return 0
    return m

def ml_exact_v1(v):
    """Independent exact ML via critical-time denominators."""
    v = sorted(set(v))
    D = set()
    for i, a in enumerate(v):
        D.add(2 * a)
        for b in v[i + 1:]:
            D.add(a + b)
    bn, bd, bt = 0, 1, Fraction(0)
    for d in sorted(D):
        for c in range(1, d // 2 + 1):
            m = f_at(c, d, v)
            if m * bd > bn * d:
                bn, bd, bt = m, d, Fraction(c, d)
    return Fraction(bn, bd), bt

def ml_fullgrid_v1(v):
    """Superset grid d = 1..2*max(v): cross-check (== ML by the theorem)."""
    v = sorted(set(v))
    bn, bd = 0, 1
    for d in range(1, 2 * max(v) + 1):
        for c in range(1, d // 2 + 1):
            m = f_at(c, d, v)
            if m * bd > bn * d:
                bn, bd = m, d
    return Fraction(bn, bd)

FAILS = []
def check(label, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {label} {detail}")
    if not cond:
        FAILS.append(label)

print("=" * 72)
print("SECTION 1: Families A and B, exact ML for r = 1..16 and spot checks")
print("=" * 72)
for r in list(range(1, 17)) + [20, 50, 100, 200]:
    vA = [1, 2, 3, 4, 5, 7, 6 * r + 12]
    aA = Fraction(r + 2, 6 * r + 17)
    mlA, tA = ml_exact_v1(vA)
    check(f"A r={r}: ML={mlA} == (r+2)/(6r+17)={aA}", mlA == aA, f"witness t={tA}")
for r in list(range(1, 17)) + [20, 50, 100, 200]:
    vB = [1, 3, 4, 5, 7, 11, 6 * r + 24]
    aB = Fraction(r + 4, 6 * r + 31)
    mlB, tB = ml_exact_v1(vB)
    check(f"B r={r}: ML={mlB} == (r+4)/(6r+31)={aB}", mlB == aB, f"witness t={tB}")

print()
print("Full-grid cross-check (superset of D(v)) on small instances:")
for r in (1, 2, 3, 5, 8, 12):
    vA = [1, 2, 3, 4, 5, 7, 6 * r + 12]
    vB = [1, 3, 4, 5, 7, 11, 6 * r + 24]
    check(f"A r={r} fullgrid == D-grid", ml_fullgrid_v1(vA) == ml_exact_v1(vA)[0])
    check(f"B r={r} fullgrid == D-grid", ml_fullgrid_v1(vB) == ml_exact_v1(vB)[0])

print()
print("Witness value check at claimed t*(r) (both families, r = 1..200 sample):")
for r in (1, 2, 3, 7, 12, 13, 37, 200):
    q = 6 * r + 17
    t = Fraction(r + 3, q)
    fA = min(min((t * vi) % 1, 1 - (t * vi) % 1) for vi in [1, 2, 3, 4, 5, 7, 6 * r + 12])
    check(f"A r={r}: f(t*)= {fA} == alpha", fA == Fraction(r + 2, q))
    q = 6 * r + 31
    t = Fraction(r + 5, q)
    fB = min(min((t * vi) % 1, 1 - (t * vi) % 1) for vi in [1, 3, 4, 5, 7, 11, 6 * r + 24])
    check(f"B r={r}: f(t*)= {fB} == alpha", fB == Fraction(r + 4, q))

print()
print("=" * 72)
print("SECTION 2: new spectrum point 9/65 and the other j>=2 instances")
print("=" * 72)
v965 = [2, 7, 9, 11, 13, 20, 54]
ml, t = ml_exact_v1(v965)
check(f"ML(2,7,9,11,13,20,54) = {ml} == 9/65", ml == Fraction(9, 65), f"witness t={t}")
check("fullgrid agrees for 9/65 instance", ml_fullgrid_v1(v965) == Fraction(9, 65))
tw = Fraction(11, 65)
fw = min(min((tw * vi) % 1, 1 - (tw * vi) % 1) for vi in v965)
check(f"claimed witness t=11/65 gives f = {fw} == 9/65", fw == Fraction(9, 65))

J2_INSTANCES = [
    ([1, 7, 8, 15], Fraction(5, 22), 4),
    ([1, 5, 6, 11, 16, 17], Fraction(5, 33), 6),
    ([5, 6, 11, 17, 23, 28], Fraction(8, 51), 6),
    ([1, 2, 3, 4, 5, 7, 18], Fraction(3, 23), 7),
    ([1, 3, 4, 5, 7, 13, 18], Fraction(3, 23), 7),
    ([1, 3, 4, 5, 7, 11, 30], Fraction(5, 37), 7),
    ([2, 7, 9, 11, 13, 20, 54], Fraction(9, 65), 7),
]
print()
print("All 7 claimed j>=2 instances: exact ML + P1 (denominator pair) + P2 (j<=s):")
for v, want, k in J2_INSTANCES:
    ml, _ = ml_exact_v1(v)
    q, s = want.denominator, want.numerator
    j = q - k * s
    pairs = [(a, b) for i, a in enumerate(v) for b in v[i + 1:]
             if (a + b) % q == 0 or (a - b) % q == 0]
    check(f"ML{tuple(v)} = {ml} == {want} (s={s},j={j})", ml == want)
    check(f"  P1 pair exists mod {q}: {pairs}", len(pairs) > 0)
    check(f"  P2: 1 <= j={j} <= s={s}", 1 <= j <= s)

print()
print("=" * 72)
print("SECTION 3: anchor sets (Corollary in findings 7)")
print("=" * 72)
for v in ([1, 2, 3, 4, 5, 7], [1, 3, 4, 5, 7, 11]):
    ml, t = ml_exact_v1(v)
    check(f"ML{tuple(v)} = {ml} == 1/6", ml == Fraction(1, 6), f"witness t={t}")

print()
print("=" * 72)
print("SECTION 4: certificate chains — exact interval-union cover check")
print("=" * 72)
# Chains exactly as in results/certificates_AB.txt (hard-coded here, independent).
def cover_check(V, w, alpha, chain, r):
    """chain entries: (speed_or_'w', m as int or ('r+',c)); returns (covered, gaps)."""
    ivs = []
    for spd, m in chain:
        vv = w if spd == "w" else spd
        mm = m if isinstance(m, int) else r + m[1]
        lo = (mm - alpha) / vv
        hi = (mm + alpha) / vv
        ivs.append((lo, hi))
    # merge in given order per the chain argument; also do full union merge
    ivs_sorted = sorted(ivs)
    merged = []
    for lo, hi in ivs_sorted:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    target_lo, target_hi = Fraction(0), Fraction(1, 2)
    covered = any(lo <= target_lo and hi >= target_hi for lo, hi in merged)
    return covered, merged

CHAIN_A = [(1, 0), (7, 1), ("w", ("r+", 2)), (5, 1), (4, 1), (3, 1), (5, 2), (2, 1)]
CHAIN_B = [(1, 0), (7, 1), ("w", ("r+", 4)), (11, 2), (5, 1), (4, 1), (3, 1),
           (5, 2), (7, 3), (11, 5), (4, 2)]
for r in (3, 4, 7, 9, 12, 30, 101):
    a = Fraction(r + 2, 6 * r + 17)
    ok, merged = cover_check([1, 2, 3, 4, 5, 7], 6 * r + 12, a, CHAIN_A, r)
    check(f"A r={r}: chain bands cover [0,1/2] exactly", ok,
          "" if ok else f"merged={merged}")
for r in (3, 4, 7, 9, 12, 30, 101):
    a = Fraction(r + 4, 6 * r + 31)
    ok, merged = cover_check([1, 3, 4, 5, 7, 11], 6 * r + 24, a, CHAIN_B, r)
    check(f"B r={r}: chain bands cover [0,1/2] exactly", ok,
          "" if ok else f"merged={merged}")

print()
print("Documenting: the symbolic chain genuinely FAILS for r = 1, 2 (expected;")
print("those r are handled by exact enumeration, Lemma 3). Gap shown for r=1,2:")
for r in (1, 2):
    a = Fraction(r + 2, 6 * r + 17)
    ok, merged = cover_check([1, 2, 3, 4, 5, 7], 6 * r + 12, a, CHAIN_A, r)
    print(f"  A r={r}: covered={ok}; union pieces={[(str(l), str(h)) for l, h in merged]}")

print()
print("=" * 72)
if FAILS:
    print(f"RESULT: {len(FAILS)} FAILURES:")
    for x in FAILS:
        print("  -", x)
    sys.exit(1)
print("RESULT: ALL NUMERIC CHECKS PASSED")
