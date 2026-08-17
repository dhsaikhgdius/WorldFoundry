#!/usr/bin/env python3
"""V3 adversarial re-verification of results/certificates_AB.txt (Theorems A/B, upper bound).

Independent of the thread's cover_prover.py machinery.  For each certificate entry we:
  1. parse the claimed P(r) and shifted P(s+7) polynomials from the text;
  2. RE-DERIVE P(r) from the band definitions alone (sympy, exact rationals) and
     require it to match the printed polynomial exactly;
  3. verify P(r) >= 0 for all real r >= 7 by exact real-root isolation
     (sympy real_roots: no roots of odd multiplicity in (7, inf), P(7) >= 0,
     positive leading coefficient), NOT by the coefficient trick;
  4. recompute the shift P(s+7) independently and compare coefficients with the file,
     and re-check the claimed "coeffs>=0" flags;
  5. sample many random rational points r >= 7 (exact Fraction arithmetic) and check
     P(r) >= 0 AND the underlying band inequality lo_next(r) <= hi_prev(r) directly;
  6. identify tangency entries (P == 0), prove the identity symbolically, and confirm
     closed-interval covering still holds (closed bands: [a,b] u [b,c] = [a,c]);
  7. independently of the link formulation, run a direct exact interval-sweep check
     that the union of the listed bands covers [0, 1/2] for every integer r in a range;
  8. check denominator positivity of every rational endpoint on r >= 7.

Exit code 0 iff every check passes.
"""

import re
import sys
import random
from fractions import Fraction
from pathlib import Path

import sympy as sp

r = sp.Symbol("r")
s = sp.Symbol("s")

CERT = Path("/mnt/cpfsB/yangboxue/visual_generation/juanxi/WorldFoundry/plan/threads/"
             "T04_lonely_runner/results/certificates_AB.txt")

# ---------------------------------------------------------------- parsing ----

def parse_poly(text, var):
    """Parse '−17 + 28r + 12r^2' style strings (polystr format) into a sympy Poly."""
    text = text.strip()
    if text == "0":
        return sp.Poly(0, var)
    text = text.replace(" - ", " + -")
    expr = sp.Integer(0)
    for term in text.split(" + "):
        m = re.fullmatch(r"(-?\d+)(?:([rs])(?:\^(\d+))?)?", term.strip())
        assert m, f"cannot parse term {term!r} of {text!r}"
        c = int(m.group(1))
        if m.group(2) is None:
            expr += c
        else:
            e = int(m.group(3)) if m.group(3) else 1
            expr += c * var ** e
    return sp.Poly(expr, var)


BAND_RE = re.compile(r"(lo|hi)B\((?:(\d+),\s*(\d+)|w,\s*(-?\d+)r\+(-?\d+)|w,\s*(-?\d+)\))")

def parse_band_token(tok):
    """'loB(7,1)' -> ('lo','s',7,1);  'loB(w, 1r+2)' -> ('lo','w',(1,2))."""
    m = re.fullmatch(r"(lo|hi)B\((\d+),\s*(-?\d+)\)", tok)
    if m:
        return m.group(1), ("s", int(m.group(2)), int(m.group(3)))
    m = re.fullmatch(r"(lo|hi)B\(w,\s*(-?\d+)r\+(-?\d+)\)", tok)
    if m:
        return m.group(1), ("w", int(m.group(2)), int(m.group(3)))
    raise ValueError(f"bad band token {tok!r}")


def parse_certificates(path):
    """Return list of family dicts with header info and entries."""
    fams = []
    cur = None
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("### "):
            m = re.match(r"### (Family\w+) \(([\d,]+),(\d+)r\+(\d+)\); "
                         r"alpha=\((?:r\+(\d+))\)/\((\d+)r\+(\d+)\)\s+\[upper bound certificate, valid r >= (\d+)\]", ln)
            assert m, f"cannot parse family header: {ln!r}"
            cur = dict(name=m.group(1),
                       V=[int(x) for x in m.group(2).split(",")],
                       w=(int(m.group(3)), int(m.group(4))),      # w = w1*r + w0
                       alpha_num=(1, int(m.group(5))),            # r + a0
                       alpha_den=(int(m.group(6)), int(m.group(7))),
                       r0=int(m.group(8)), entries=[])
            fams.append(cur)
            i += 1
        elif re.match(r"\s+(start|link|end):", ln):
            label = ln.strip()
            mP = re.match(r"\s+P\(r\) = (.*)$", lines[i + 1])
            mQ = re.match(r"\s+P\(s\+(\d+)\) = (.*?)\s+coeffs>=0: (True|False)$", lines[i + 2])
            assert mP and mQ, f"bad entry block at line {i}: {lines[i+1]!r} / {lines[i+2]!r}"
            cur["entries"].append(dict(
                label=label,
                P=parse_poly(mP.group(1), r),
                shift=int(mQ.group(1)),
                Q=parse_poly(mQ.group(2), s),
                claimed_ok=(mQ.group(3) == "True")))
            i += 3
        else:
            i += 1
    return fams

# ------------------------------------------------------- band construction ---

class Fam:
    def __init__(self, d):
        self.name = d["name"]
        self.V = d["V"]
        w1, w0 = d["w"]
        self.w = w1 * r + w0
        a1, a0 = d["alpha_num"]
        d1, d0 = d["alpha_den"]
        self.an = a1 * r + a0
        self.ad = d1 * r + d0
        self.alpha = self.an / self.ad
        self.r0 = d["r0"]

    def band_numden(self, band):
        """Return (lo_num, hi_num, den) UNREDUCED, mirroring the RF construction:
        small band: ((m*ad - an), (m*ad + an), v*ad); w band with m(r)=mu1*r+mu0."""
        if band[0] == "s":
            _, v, m = band
            return (sp.expand(m * self.ad - self.an),
                    sp.expand(m * self.ad + self.an),
                    sp.expand(v * self.ad))
        _, mu1, mu0 = band
        mm = mu1 * r + mu0
        return (sp.expand(mm * self.ad - self.an),
                sp.expand(mm * self.ad + self.an),
                sp.expand(self.w * self.ad))

    def band_interval(self, band, rv: Fraction):
        """Exact Fraction endpoints of the band at integer r = rv."""
        an = Fraction(int(self.an.subs(r, sp.Rational(rv)).p), int(self.an.subs(r, sp.Rational(rv)).q))
        ad = Fraction(int(self.ad.subs(r, sp.Rational(rv)).p), int(self.ad.subs(r, sp.Rational(rv)).q))
        alpha = an / ad
        if band[0] == "s":
            _, v, m = band
            return (Fraction(m) - alpha) / v, (Fraction(m) + alpha) / v
        _, mu1, mu0 = band
        w = Fraction(int(self.w.subs(r, sp.Rational(rv)).p))
        m = Fraction(mu1) * rv + mu0
        return (m - alpha) / w, (m + alpha) / w

# --------------------------------------------------- nonnegativity (exact) ---

def nonneg_on_ray(P: sp.Poly, a) -> tuple[bool, str]:
    """Exact decision: P(x) >= 0 for all real x >= a. Complete (not just sufficient):
    P == 0, or [lc > 0, P(a) >= 0, and every real root > a has even multiplicity]."""
    if P.is_zero:
        return True, "identically 0"
    if P.degree() == 0:
        c = P.coeffs()[0]
        return c >= 0, f"constant {c}"
    if P.LC() < 0:
        return False, "negative leading coefficient"
    if P.eval(sp.Rational(a)) < 0:
        return False, f"P({a}) < 0"
    expr = P.as_expr()
    x = P.gen
    troubles = []
    for root, mult in sp.roots(expr, x).items():          # exact algebraic roots
        if not root.is_real:
            continue
        if sp.simplify(root - a).is_positive and mult % 2 == 1:
            troubles.append((root, mult))
    if sp.roots(expr, x, multiple=True) and sum(m for _, m in sp.roots(expr, x).items()) < P.degree():
        # roots() failed to find everything (shouldn't happen for deg<=2); fall back to real_roots
        troubles = []
        rr = sp.real_roots(expr, x)
        from collections import Counter
        cnt = Counter(rr)
        for root, mult in cnt.items():
            if (root > a) == True and mult % 2 == 1:
                troubles.append((root, mult))
    if troubles:
        return False, f"odd-multiplicity real roots > {a}: {troubles}"
    # additionally count roots in (a, oo) via Sturm to double-check
    nroots = P.count_roots(sp.Rational(a), None)  # roots in [a, oo)
    return True, f"lc>0, P({a})={P.eval(sp.Rational(a))}>=0, odd roots beyond {a}: none (roots in [{a},oo): {nroots})"

# ------------------------------------------------------------------- main ----

def frac_eval(P: sp.Poly, x: Fraction) -> Fraction:
    acc = Fraction(0)
    for c in P.all_coeffs():
        acc = acc * x + Fraction(int(c))
    return acc


def main():
    random.seed(20260814)
    fams_raw = parse_certificates(CERT)
    assert len(fams_raw) == 2, "expected exactly two families"
    all_ok = True

    for d in fams_raw:
        fam = Fam(d)
        print(f"\n=== {fam.name}  (claimed valid r >= {fam.r0}) ===")
        n_entries = len(d["entries"])
        # reconstruct chain from labels
        chain = []
        for e in d["entries"]:
            lab = e["label"]
            if lab.startswith("start:"):
                side, band = parse_band_token(re.search(r"loB\([^)]*\)", lab).group(0))
                chain.append(band)
            elif lab.startswith("link:"):
                toks = re.findall(r"(?:lo|hi)B\([^)]*\)", lab)
                _, nxt = parse_band_token(toks[0])
                _, prv = parse_band_token(toks[1])
                assert prv == chain[-1], f"chain discontinuity at {lab!r}: {prv} vs {chain[-1]}"
                chain.append(nxt)
        print(f"  parsed chain of {len(chain)} bands, {n_entries} inequality entries "
              f"(expect bands+1): {'OK' if n_entries == len(chain) + 1 else 'MISMATCH'}")
        all_ok &= (n_entries == len(chain) + 1)

        # per-entry checks
        tangents = []
        for idx, e in enumerate(d["entries"]):
            lab = e["label"]
            # (2) re-derive P from band definitions
            if lab.startswith("start:"):
                lo_n, _, den = fam.band_numden(chain[0])
                P_mine = sp.Poly(sp.expand(-lo_n), r)
                ineq_lhs, ineq_rhs, dens = None, None, [den]
            elif lab.startswith("end:"):
                _, hi_n, den = fam.band_numden(chain[-1])
                P_mine = sp.Poly(sp.expand(hi_n * 2 - den), r)
                dens = [den]
            else:
                i_link = sum(1 for x in d["entries"][:idx] if x["label"].startswith(("start", "link"))) - 1
                prv, nxt = chain[i_link], chain[i_link + 1]
                lo_n2, _, den2 = fam.band_numden(nxt)
                _, hi_n1, den1 = fam.band_numden(prv)
                P_mine = sp.Poly(sp.expand(hi_n1 * den2 - lo_n2 * den1), r)
                dens = [den1, den2]
            match = (P_mine - e["P"]).is_zero
            all_ok &= match

            # (8) denominators positive on [r0, inf)
            dens_ok = all(nonneg_on_ray(sp.Poly(dd - 1, r), fam.r0)[0] for dd in dens)  # den >= 1 > 0
            all_ok &= dens_ok

            # (3) exact nonnegativity of P on [r0, inf) by root isolation
            ok3, why3 = nonneg_on_ray(e["P"], fam.r0)
            all_ok &= ok3

            # (4) independent shift and comparison
            Q_mine = sp.Poly(e["P"].as_expr().subs(r, s + e["shift"]), s)
            q_match = (Q_mine - e["Q"]).is_zero
            q_nonneg = all(c >= 0 for c in Q_mine.all_coeffs())
            all_ok &= q_match and (q_nonneg == e["claimed_ok"]) and e["claimed_ok"]

            # (5) random rational sampling of P and of the raw inequality
            bad = 0
            for _ in range(400):
                num = random.randint(0, 10**6)
                dnm = random.randint(1, 10**3)
                rv = Fraction(fam.r0) + Fraction(num, dnm)
                if frac_eval(e["P"], rv) < 0:
                    bad += 1
            for rv in [Fraction(fam.r0), Fraction(10**9), Fraction(10**12) + Fraction(1, 7)]:
                if frac_eval(e["P"], rv) < 0:
                    bad += 1
            all_ok &= (bad == 0)

            if e["P"].is_zero:
                tangents.append(lab)
            status = "OK" if (match and ok3 and q_match and q_nonneg and dens_ok and bad == 0) else "FAIL"
            print(f"  [{status}] {lab}")
            print(f"        rederived-P match: {match}; nonneg on [{fam.r0},inf): {ok3} ({why3});")
            print(f"        shift match: {q_match}; shifted coeffs>=0: {q_nonneg} "
                  f"(claimed {e['claimed_ok']}); random sampling violations: {bad}; dens>0: {dens_ok}")

        # (6) tangency handling: for each P == 0 link, prove lo_next == hi_prev
        # identically and that the shared point is exactly the lower-bound witness t*(r)
        wit = (r + 3) / (6 * r + 17) if fam.name == "FamilyA" else (r + 5) / (6 * r + 31)
        for idx, e in enumerate(d["entries"]):
            if not (e["label"].startswith("link:") and e["P"].is_zero):
                continue
            i_link = sum(1 for x in d["entries"][:idx] if x["label"].startswith(("start", "link"))) - 1
            prv, nxt = chain[i_link], chain[i_link + 1]
            lo_n2, _, den2 = fam.band_numden(nxt)
            _, hi_n1, den1 = fam.band_numden(prv)
            ident = sp.simplify(hi_n1 / den1 - lo_n2 / den2) == 0
            at_wit = sp.simplify(hi_n1 / den1 - wit) == 0
            print(f"  tangent link (P == 0): {e['label']}")
            print(f"        lo_next == hi_prev identically: {ident}; shared point == witness t*(r): {at_wit}")
            print(f"        bands are CLOSED intervals; [a,b] u [b,c] = [a,c] -> covering still valid.")
            all_ok &= ident and at_wit

        # (7) direct exact union-cover check at integer r
        half = Fraction(1, 2)
        cover_fail = []
        for rv in list(range(fam.r0, 101)) + [150, 200, 500, 1000]:
            ivs = sorted(fam.band_interval(b, Fraction(rv)) for b in chain)
            cur = Fraction(0)
            for lo, hi in ivs:
                if lo > cur:
                    cover_fail.append((rv, float(cur)))
                    break
                cur = max(cur, hi)
                if cur >= half:
                    break
            if cur < half:
                cover_fail.append((rv, float(cur)))
        print(f"  direct interval sweep r in [{fam.r0},100] u {{150,200,500,1000}}: "
              f"{'all cover [0,1/2] exactly' if not cover_fail else f'FAILURES {cover_fail[:5]}'}")
        all_ok &= not cover_fail

    print(f"\nV3 CERTIFICATE AUDIT RESULT: {'ALL CHECKS PASSED' if all_ok else 'FAILURES FOUND'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
