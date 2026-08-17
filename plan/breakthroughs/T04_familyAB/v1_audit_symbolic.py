#!/usr/bin/env python3
"""V1 adversarial audit — independent SYMBOLIC re-derivation (sympy).

Re-derives from scratch (no import from T04 code):
  (a) every start/link/end inequality of both covering certificates as an
      integer polynomial in r, compares against results/certificates_AB.txt,
      determines the exact minimal real r0 from which each link holds, and
      re-checks the r -> s+7 nonnegative-coefficient criterion;
  (b) every witness inequality of Lemma 1 (nearest integer identity,
      distance >= alpha, distance <= 1/2) for all 7 speeds of each family,
      valid for all r >= 1 via r -> s+1 coefficient nonnegativity.
"""
import sympy as sp

r, s = sp.symbols("r s")
FAILS = []

def check(label, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {label} {detail}")
    if not cond:
        FAILS.append(label)

def coeffs_nonneg_after_shift(P, shift):
    Q = sp.Poly(sp.expand(P.subs(r, s + shift)), s)
    return all(c >= 0 for c in Q.all_coeffs()), Q.all_coeffs()

def min_valid_r(P):
    """largest real root of P (above which P >= 0, checked), or 'all' """
    Pp = sp.Poly(sp.expand(P), r)
    if Pp.degree() == 0:
        return "any" if Pp.all_coeffs()[0] >= 0 else "NEVER"
    roots = [sp.nsimplify(x) for x in sp.real_roots(Pp)]
    if not roots:
        return "any" if Pp.eval(1) >= 0 else "NEVER"
    m = max(roots)
    # sanity: nonneg beyond m
    assert Pp.eval(sp.Rational(m) + 1) >= 0 and Pp.eval(sp.Rational(m)) >= 0
    return m

# ---------------------------------------------------------------- families
def audit_cover(name, w, an, ad, chain, cert_polys):
    """chain: list of (v_poly_or_None_for_w, m_poly). alpha = an/ad."""
    alpha = an / ad
    print(f"\n--- {name}: covering chain, {len(chain)} bands ---")
    exprs = []   # (label, polynomial P with P>=0 <=> inequality)
    # start: lo(first) <= 0  <=>  -( m*ad - an ) >= 0   (v>0, ad>0)
    v0, m0 = chain[0]
    exprs.append(("start lo<=0", sp.expand(-(m0 * ad - an))))
    for i in range(len(chain) - 1):
        v1, m1 = chain[i]
        v2, m2 = chain[i + 1]
        V1 = w if v1 is None else v1
        V2 = w if v2 is None else v2
        # lo_{i+1} <= hi_i  <=>  (m1*ad+an)*V2 - (m2*ad-an)*V1 >= 0
        P = sp.expand((m1 * ad + an) * V2 - (m2 * ad - an) * V1)
        exprs.append((f"link {i}: lo(band{i+1}) <= hi(band{i})", P))
    vl, ml = chain[-1]
    Vl = w if vl is None else vl
    # hi_last >= 1/2  <=>  2*(ml*ad+an) - Vl*ad >= 0
    exprs.append(("end hi>=1/2", sp.expand(2 * (ml * ad + an) - Vl * ad)))

    assert len(exprs) == len(cert_polys), "certificate entry count mismatch"
    worst = -sp.oo
    for (label, P), cert in zip(exprs, cert_polys):
        Pc = sp.expand(cert)
        same = sp.expand(P - Pc) == 0
        # Equivalent up to a positive polynomial factor: my derivation cancels
        # the common denominator polynomial ad(r) > 0, the certificate keeps
        # the raw cross-multiplied product.  Accept cert = P * (poly with
        # nonneg coeffs, not identically 0), which is > 0 for r >= 1.
        if same:
            same_scaled, factor = True, sp.Integer(1)
        elif Pc == 0 or P == 0:
            same_scaled, factor = False, None
        else:
            quo, rem = sp.div(Pc, P, r)
            qp = sp.Poly(sp.expand(quo), r)
            same_scaled = (sp.expand(rem) == 0
                           and all(c >= 0 for c in qp.all_coeffs())
                           and any(c > 0 for c in qp.all_coeffs()))
            factor = sp.expand(quo)
        r0 = min_valid_r(P)
        ok7, _ = coeffs_nonneg_after_shift(P, 7)
        check(f"{label}: P(r) = {sp.expand(P)}",
              same_scaled and ok7,
              f"| cert = P * ({factor if not same else 1}) | holds for real r >= {r0} | s+7 coeffs>=0: {ok7}")
        if r0 not in ("any",) and r0 != "NEVER":
            worst = max(worst, r0)
    print(f"  ==> chain valid for ALL real r >= {worst} (certificate claims r >= 7: "
          f"{'consistent, conservative' if worst <= 7 else 'PROBLEM'})")
    check(f"{name}: chain valid from r0 = {worst} <= 7", worst <= 7)

def audit_witness(name, speeds_w, tn, td, an, ad, nearest):
    """speeds_w: list of speed polys (last = w). t* = tn/td, alpha = an/ad.
    nearest: list of n(r) polys. Verifies |v t* - n| >= alpha, <= 1/2, sign fixed."""
    print(f"\n--- {name}: witness t* = ({tn})/({td}) ---")
    eq_achievers = []
    for vp, np_ in zip(speeds_w, nearest):
        # diff = v t* - n = (v*tn - n*td)/td
        num = sp.expand(vp * tn - np_ * td)
        sgn = 1 if num.subs(r, 100) > 0 else -1
        dist_num = sp.expand(sgn * num)          # dist = dist_num/td >= 0 needed
        ok_sign, _ = coeffs_nonneg_after_shift(dist_num, 1)
        # dist >= alpha  <=>  dist_num*ad - an*td >= 0
        gap = sp.expand(dist_num * ad - an * td)
        ok_alpha, _ = coeffs_nonneg_after_shift(gap, 1)
        # dist <= 1/2  <=>  td - 2*dist_num >= 0
        ok_half, _ = coeffs_nonneg_after_shift(sp.expand(td - 2 * dist_num), 1)
        is_eq = sp.expand(gap) == 0
        if is_eq:
            eq_achievers.append(str(vp))
        check(f"v={vp}: ||v t*|| = ({dist_num})/q  >= alpha, <= 1/2, sign fixed (r>=1)",
              ok_sign and ok_alpha and ok_half,
              "EQUALITY" if is_eq else "strict")
    check(f"{name}: exactly two equality achievers {eq_achievers}", len(eq_achievers) == 2)

# ================= Family A =================
w_A = 6 * r + 12
an_A, ad_A = r + 2, 6 * r + 17
CHAIN_A = [(1, 0), (7, 1), (None, r + 2), (5, 1), (4, 1), (3, 1), (5, 2), (2, 1)]
CERT_A = [  # transcribed from results/certificates_AB.txt
    2 + r,
    -17 + 28 * r + 12 * r**2,
    68 + 58 * r + 12 * r**2,
    sp.Integer(0),
    17 + 57 * r + 18 * r**2,
    -51 - r + 6 * r**2,
    -17 + 28 * r + 12 * r**2,
    -51 - r + 6 * r**2,
    4 + 2 * r,
]
audit_cover("Family A", w_A, an_A, ad_A, CHAIN_A, CERT_A)
audit_witness("Family A", [1, 2, 3, 4, 5, 7, w_A], r + 3, 6 * r + 17, an_A, ad_A,
              nearest=[0, 0, 1, 1, 1, 1, r + 2])

# ================= Family B =================
w_B = 6 * r + 24
an_B, ad_B = r + 4, 6 * r + 31
CHAIN_B = [(1, 0), (7, 1), (None, r + 4), (11, 2), (5, 1), (4, 1), (3, 1),
           (5, 2), (7, 3), (11, 5), (4, 2)]
CERT_B = [
    4 + r,
    31 + 68 * r + 12 * r**2,
    sp.Integer(0),
    496 + 220 * r + 24 * r**2,
    1023 + 508 * r + 60 * r**2,
    155 + 123 * r + 18 * r**2,
    -93 + 13 * r + 6 * r**2,
    31 + 68 * r + 12 * r**2,
    527 + 288 * r + 36 * r**2,
    310 + 246 * r + 36 * r**2,
    -62 + 81 * r + 18 * r**2,
    8 + 2 * r,
]
audit_cover("Family B", w_B, an_B, ad_B, CHAIN_B, CERT_B)
audit_witness("Family B", [1, 3, 4, 5, 7, 11, w_B], r + 5, 6 * r + 31, an_B, ad_B,
              nearest=[0, 0, 1, 1, 1, 2, r + 4])

# ============ tangency identities (the P==0 links), explicit ============
print("\n--- tangency identities ---")
tA = (r + 3) / (6 * r + 17)
check("A: hi B(w,r+2) == lo B(5,1) == t*  (exact tangency at witness)",
      sp.simplify((r + 2 + an_A / ad_A) / w_A - tA) == 0
      and sp.simplify((1 - an_A / ad_A) / 5 - tA) == 0)
tB = (r + 5) / (6 * r + 31)
check("B: hi B(7,1) == lo B(w,r+4) == t*  (exact tangency at witness)",
      sp.simplify((1 + an_B / ad_B) / 7 - tB) == 0
      and sp.simplify((r + 4 - an_B / ad_B) / w_B - tB) == 0)

# w-band arithmetic identity used in Lemma 1 (A):
check("A: (6r+12)(r+3) == (6r+17)(r+2) + (r+2)",
      sp.expand((6 * r + 12) * (r + 3) - (6 * r + 17) * (r + 2) - (r + 2)) == 0)
check("B: (6r+24)(r+5) == (6r+31)(r+4) - (r+4)",
      sp.expand((6 * r + 24) * (r + 5) - (6 * r + 31) * (r + 4) + (r + 4)) == 0)

print()
if FAILS:
    print(f"RESULT: {len(FAILS)} FAILURES:")
    for x in FAILS:
        print("  -", x)
    raise SystemExit(1)
print("RESULT: ALL SYMBOLIC CHECKS PASSED")
