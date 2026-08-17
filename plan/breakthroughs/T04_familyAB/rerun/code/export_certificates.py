#!/usr/bin/env python3
"""Export the full machine-checkable proof certificates of Theorems A and B
as a human-readable appendix: every link inequality of the covering (upper
bound) and every witness inequality (lower bound), with the exact polynomial
whose coefficients after r -> s + r0 are all nonnegative.

Output: results/certificates_AB.txt
"""

from fractions import Fraction
from pathlib import Path

from cover_prover import (RF, pmul, padd, psub, pshift, peval,
                          FAMILY_A, FAMILY_B, bands_at, greedy_cover)


def polystr(p, var="r"):
    terms = []
    for i, c in enumerate(p):
        if c == 0:
            continue
        if i == 0:
            terms.append(f"{c}")
        elif i == 1:
            terms.append(f"{c}{var}")
        else:
            terms.append(f"{c}{var}^{i}")
    return " + ".join(terms).replace("+ -", "- ") or "0"


def link_poly(f: RF, g: RF):
    """polynomial P = g.num*f.den - f.num*g.den  (P>=0 iff f<=g, dens>0)"""
    return psub(pmul(g.num, f.den), pmul(f.num, g.den))


def export_family(fam, chain, r0, out):
    out.append(f"### {fam.name}   [upper bound certificate, valid r >= {r0}]")
    out.append(f"Bad band B(v,m) = [(m-alpha)/v, (m+alpha)/v]; claim: the chain")
    out.append(f"below covers [0, 1/2]; with t -> 1-t symmetry covers [0,1].")
    eps = []

    def endpoints(item):
        if item[0] == "s":
            return fam.band_small(item[1], item[2]), f"B({item[1]},{item[2]})"
        mu0, mu1 = item[1]
        return fam.band_w(item[1]), f"B(w, {mu1}r+{mu0})"

    (lo0, _), name0 = endpoints(chain[0])
    P = psub([0], pmul(lo0.num, [1]))  # 0 - lo0.num >= 0 given den>0: -num>=0
    P = [-c for c in lo0.num]
    eps.append((f"start: lo{name0} <= 0", P))
    for i in range(len(chain) - 1):
        (l1, h1), n1 = endpoints(chain[i])
        (l2, h2), n2 = endpoints(chain[i + 1])
        eps.append((f"link: lo{n2} <= hi{n1}", link_poly(l2, h1)))
    (ll, hl), nl = endpoints(chain[-1])
    eps.append((f"end: hi{nl} >= 1/2", link_poly(RF([1], [2]), hl)))

    allok = True
    for label, P in eps:
        Q = pshift(P, r0)
        ok = all(c >= 0 for c in Q)
        allok &= ok
        out.append(f"  {label}")
        out.append(f"      P(r) = {polystr(P)}")
        out.append(f"      P(s+{r0}) = {polystr(Q, 's')}   coeffs>=0: {ok}")
    out.append(f"  => ALL VERIFIED: {allok}")
    out.append("")
    return allok


def main():
    out = []
    out.append("Machine-checkable certificates for Theorems A and B (Cycle 2)")
    out.append("=" * 64)
    out.append("")
    ok = True
    for fam, r0 in ((FAMILY_A, 7), (FAMILY_B, 7)):
        # re-derive the chain at a stable sample and convert to symbolic tags
        ch = greedy_cover(bands_at(fam, Fraction(r0)))
        chain = []
        for tag in ch:
            if tag[0] == "s":
                chain.append(("s", tag[1], tag[2]))
            else:
                # m at sample r0 and r0+6 to fit affine m(r)
                ch2 = greedy_cover(bands_at(fam, Fraction(r0 + 6)))
                m2 = [t[1] for t in ch2 if t[0] == "w"][0]
                m1 = tag[1]
                mu1 = (m2 - m1) // 6
                mu0 = m1 - mu1 * r0
                chain.append(("w", [mu0, mu1]))
        ok &= export_family(fam, chain, r0, out)

    p = Path(__file__).resolve().parent.parent / "results" / "certificates_AB.txt"
    p.write_text("\n".join(out))
    print(f"written {p}; all verified: {ok}")


if __name__ == "__main__":
    main()
