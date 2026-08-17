#!/usr/bin/env python3
"""Machine-verified symbolic upper-bound prover for one-parameter LR families.

Goal: prove  ML(V u {w(r)}) <= alpha(r)  for ALL integers r >= 1, where
  V = fixed small speeds, w(r) = w1*r + w0, alpha(r) = (P*r+Q)/(D*r+E).

Method (bad-band covering certificate):
  ML <= alpha  <=>  every t in [0,1] has some runner with ||v t|| <= alpha
                <=>  [0,1] is covered by the closed "bad bands"
                        B(v,m) = [ (m-alpha)/v , (m+alpha)/v ].
  By the symmetry f(t)=f(1-t) it suffices to cover [0, 1/2].
  A certificate is an ordered list of bands (v_j, m_j) such that
      lo(v_1,m_1) <= 0,   lo(v_{j+1},m_{j+1}) <= hi(v_j,m_j),   hi(last) >= 1/2.
  Each inequality is (A/B <= C/D) with A,B,C,D integer polynomials in r and
  B,D > 0 on r>=1, i.e. equivalent to the polynomial inequality  C*B - A*D >= 0
  on r >= 1.  We verify "p(r) >= 0 for all real r >= 1" rigorously via the
  sufficient criterion: all coefficients of p(s+1) are >= 0 (exact integer
  arithmetic; substitution r = s+1, s >= 0).  If that fails we report failure
  (no unsound fallback).

The certificate itself is discovered numerically (exact Fractions) at two
sample values of r and required to be structurally identical; then every link
is verified symbolically as above, so the final proof is valid for ALL r >= 1
regardless of how the certificate was found.

Bands of the parametric speed w(r) get m = mu1*r + mu0 (a linear polynomial),
discovered from the samples and verified symbolically like everything else.
"""

from fractions import Fraction
from math import gcd
import itertools
import sys

# ---------- exact polynomial arithmetic (integer coefficients, low->high) ----

def pmul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out

def padd(a, b):
    n = max(len(a), len(b))
    return [ (a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n) ]

def pneg(a):
    return [-x for x in a]

def psub(a, b):
    return padd(a, pneg(b))

def pshift(p, r0):
    """coefficients of p(s + r0)"""
    out = [0] * len(p)
    for i, c in enumerate(p):
        row = [1]
        for _ in range(i):
            row = pmul(row, [r0, 1])
        for j, b in enumerate(row):
            out[j] += c * b
    return out

def peval(p, r: Fraction) -> Fraction:
    acc = Fraction(0)
    for c in reversed(p):
        acc = acc * r + c
    return acc

def nonneg_on_r_ge(p, r0):
    """Rigorous sufficient check: p(r) >= 0 for all real r >= r0."""
    q = pshift(p, r0)                   # p(s+r0), s >= 0
    if all(c >= 0 for c in q):
        return True
    return False                        # sound: refuse to certify

# ---------- rational functions A/B with B > 0 on r >= 1 ----------------------

class RF:
    """A/B, both integer polynomials; caller guarantees B(r) > 0 for r >= 1."""
    def __init__(self, num, den):
        self.num, self.den = list(num), list(den)
    def eval(self, r: Fraction) -> Fraction:
        return peval(self.num, r) / peval(self.den, r)
    def __repr__(self):
        return f"({self.num})/({self.den})"

def rf_leq(f: RF, g: RF, r0=1):
    """certify f(r) <= g(r) for all r >= r0;  g.num*f.den - f.num*g.den >= 0"""
    return nonneg_on_r_ge(psub(pmul(g.num, f.den), pmul(f.num, g.den)), r0)

# ---------- family specification ---------------------------------------------

class Family:
    def __init__(self, name, V, w_poly, alpha_num, alpha_den):
        self.name = name
        self.V = V                       # fixed small speeds
        self.w = w_poly                  # [w0, w1]
        self.an, self.ad = alpha_num, alpha_den   # alpha = an/ad as polys

    def alpha(self):
        return RF(self.an, self.ad)

    def w_of(self, r: Fraction) -> Fraction:
        return peval(self.w, r)

    def band_small(self, v: int, m: int):
        """B(v,m) endpoints as RFs:  (m -+ alpha)/v = (m*ad -+ an)/(v*ad)."""
        lo = RF(psub(pmul([m], self.ad), self.an), pmul([v], self.ad))
        hi = RF(padd(pmul([m], self.ad), self.an), pmul([v], self.ad))
        return lo, hi

    def band_w(self, mpoly):
        """B(w, m(r)) endpoints: (m(r) -+ alpha)/w = (m*ad -+ an)/(w*ad)."""
        lo = RF(psub(pmul(mpoly, self.ad), self.an), pmul(self.w, self.ad))
        hi = RF(padd(pmul(mpoly, self.ad), self.an), pmul(self.w, self.ad))
        return lo, hi

# ---------- numeric certificate discovery ------------------------------------

def bands_at(fam: Family, r: Fraction):
    """all bad bands intersecting [0, 1/2+margin], as (lo, hi, tag) Fractions"""
    a = fam.alpha().eval(r)
    out = []
    for v in fam.V:
        for m in range(0, v // 2 + 2):
            lo, hi = Fraction(m - 0, 1), None
            lo = (m - a) / v
            hi = (m + a) / v
            if hi < 0 or lo > Fraction(1, 2) + a:
                continue
            out.append((lo, hi, ("s", v, m)))
    w = fam.w_of(r)
    for m in range(0, int(w // 2) + 2):
        lo = (m - a) / w
        hi = (m + a) / w
        if hi < 0 or lo > Fraction(1, 2) + a:
            continue
        out.append((lo, hi, ("w", m)))
    return out

def greedy_cover(bands, target=Fraction(1, 2)):
    """greedy chain covering [0, target]; returns list of tags or None"""
    chain, cur = [], Fraction(0)
    used = set()
    while cur < target:
        cands = [(hi, tag) for lo, hi, tag in bands if lo <= cur and hi > cur and tag not in used]
        if not cands:
            return None
        hi, tag = max(cands)
        chain.append(tag)
        used.add(tag)
        cur = hi
    return chain

# ---------- symbolic certificate verification --------------------------------

def verify_family(fam: Family, r_min=1, r_samples=None, verbose=True):
    if r_samples is None:
        r_samples = (r_min, r_min + 6, r_min + 43, r_min + 193)
    # 1) discover certificate at samples; require identical structure
    chains = []
    for r in r_samples:
        r = Fraction(r)
        ch = greedy_cover(bands_at(fam, r))
        if ch is None:
            print(f"[{fam.name}] no numeric cover at r={r} — upper bound FALSE?")
            return False
        chains.append(ch)
    # w-band m values must be affine in r: fit from samples and check
    def tag_key(tag):
        return tag[:2] if tag[0] == "s" else ("w",)
    if not all(len(c) == len(chains[0]) for c in chains) or \
       not all(tag_key(c[i]) == tag_key(chains[0][i]) for c in chains for i in range(len(c))):
        print(f"[{fam.name}] certificate structure differs across samples; need split")
        for r, c in zip(r_samples, chains):
            print(f"   r={r}: {c}")
        return False
    # build symbolic chain
    sym = []
    for i, tag in enumerate(chains[0]):
        if tag[0] == "s":
            _, v, m = tag
            sym.append(("s", v, m))
        else:
            # fit m(r) = mu1*r + mu0 through first two samples; verify on third
            ms = [chains[j][i][1] for j in range(len(r_samples))]
            r0, r1 = r_samples[0], r_samples[1]
            mu1, mu0 = None, None
            den = r1 - r0
            mu1 = Fraction(ms[1] - ms[0], den)
            mu0 = Fraction(ms[0]) - mu1 * r0
            if mu1.denominator != 1 or mu0.denominator != 1:
                print(f"[{fam.name}] w-band m(r) not integer-affine: {ms}")
                return False
            mu1, mu0 = int(mu1), int(mu0)
            for rj, mj in zip(r_samples, ms):
                if mu1 * rj + mu0 != mj:
                    print(f"[{fam.name}] w-band m(r) fit fails at r={rj}")
                    return False
            sym.append(("w", [mu0, mu1]))
    # 2) symbolic link verification
    def endpoints(item):
        if item[0] == "s":
            return fam.band_small(item[1], item[2])
        return fam.band_w(item[1])

    ok = True
    lo0, _ = endpoints(sym[0])
    if not rf_leq(lo0, RF([0], [1]), r_min):
        print(f"[{fam.name}] FAIL: first band lo > 0: {sym[0]}")
        ok = False
    for i in range(len(sym) - 1):
        _, hi_i = endpoints(sym[i])
        lo_n, _ = endpoints(sym[i + 1])
        if not rf_leq(lo_n, hi_i, r_min):
            print(f"[{fam.name}] FAIL link {i}: {sym[i]} -> {sym[i+1]}")
            ok = False
    _, hi_last = endpoints(sym[-1])
    if not rf_leq(RF([1], [2]), hi_last, r_min):
        print(f"[{fam.name}] FAIL: last band hi < 1/2: {sym[-1]}")
        ok = False
    if ok and verbose:
        print(f"[{fam.name}] UPPER BOUND PROVED for all r >= {r_min}.")
        print(f"  certificate ({len(sym)} bands covering [0,1/2], symmetric completion):")
        for item in sym:
            if item[0] == "s":
                print(f"    band(v={item[1]}, m={item[2]})")
            else:
                mu0, mu1 = item[1][0], item[1][1]
                print(f"    band(w=6r+{fam.w[0]}, m={mu1}r+{mu0})")
    return ok


FAMILY_A = Family("FamilyA (1,2,3,4,5,7,6r+12); alpha=(r+2)/(6r+17)",
                  V=[1, 2, 3, 4, 5, 7], w_poly=[12, 6],
                  alpha_num=[2, 1], alpha_den=[17, 6])

FAMILY_B = Family("FamilyB (1,3,4,5,7,11,6r+24); alpha=(r+4)/(6r+31)",
                  V=[1, 3, 4, 5, 7, 11], w_poly=[24, 6],
                  alpha_num=[4, 1], alpha_den=[31, 6])

if __name__ == "__main__":
    r_min = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    okA = verify_family(FAMILY_A, r_min)
    print()
    okB = verify_family(FAMILY_B, r_min)
    print()
    print(f"NOTE: combined with the exact equality checks for 1 <= r <= 12")
    print(f"(family_j2.py, rigorous rational arithmetic), r_min={r_min} <= 13 yields")
    print(f"the theorem for ALL r >= 1.")
    sys.exit(0 if (okA and okB) else 1)
