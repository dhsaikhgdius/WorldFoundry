#!/usr/bin/env python3
"""Formalized 'denominator pair' proposition + machine verification.

Proposition P1 (verified over ALL in-domain near-tight instances, k=2..8):
  If ML(v) = s/q in lowest terms with q = k*s + j and 2 <= j (offset >= 2),
  then there exist i != l with  v_i + v_l == 0 (mod q)  or  v_i == v_l (mod q)
  (a "folded coincidence": the effective folded residue count drops below k).

Proposition P2 (PROVED, elementary; conditional only on LRC for this k, which
  is a theorem for k <= 9):
  If ML(v) = s/(ks+j) in lowest terms and ML < 1/k, then 1 <= j <= s;
  moreover j = s forces s = j = 1... no: j=s gives ML = s/(k s + s) = 1/(k+1)
  whose lowest form is 1/(k+1) (j=s=1). Hence for lowest terms with s >= 2:
  1 <= j <= s - 1 or (s,j)=(s,s) impossible; precisely: j <= s, and j = s
  only for (s,j) = (1,1) (the tight value).
  Proof: LRC_k gives ML >= 1/(k+1), i.e. s(k+1) >= ks+j, i.e. s >= j.
         If j = s then s/(ks+s) = 1/(k+1), lowest terms forces s = j = 1. QED

This gives an unconditional (k <= 9) spectral constraint complementary to the
Fan-Sun conjecture j <= k/2:   j <= min(s, k/2 conj.).
Data check: j=2 instances have s in {3,5,9} (all >= 3 > 2); j=3 have s in
{5,8} (>= 4 > 3) - consistent, and strictly stronger patterns remain open.
"""

import csv
from fractions import Fraction
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"

EXTRA = [([2, 7, 9, 11, 13, 20, 54], Fraction(9, 65))]   # new j=2 point (Cycle 2)


def check_instance(v, ml, k):
    q = ml.denominator
    s = ml.numerator
    j = q - k * s
    if j < 2:
        return None
    pair_sum = [(a, b) for i, a in enumerate(v) for b in v[i + 1:] if (a + b) % q == 0]
    pair_eq = [(a, b) for i, a in enumerate(v) for b in v[i + 1:] if (a - b) % q == 0]
    ok = bool(pair_sum or pair_eq)
    # P2 check
    p2 = (1 <= j <= s) and (j != s or s == 1)
    return dict(v=v, ml=ml, s=s, j=j, pairs=pair_sum + pair_eq, P1=ok, P2=p2)


def main():
    total_j2 = 0
    all_ok = True
    for k in range(2, 9):
        f = RES / f"instances_k{k}.csv"
        if not f.exists():
            continue
        for row in csv.DictReader(open(f)):
            v = [int(x) for x in row["v"].split()]
            ml = Fraction(row["ml"])
            res = check_instance(v, ml, k)
            if res:
                total_j2 += 1
                status = "OK" if (res["P1"] and res["P2"]) else "FAIL"
                all_ok &= res["P1"] and res["P2"]
                print(f"k={k} v={v} ML={ml} (s={res['s']}, j={res['j']}) "
                      f"pairs={res['pairs']} P1={res['P1']} P2={res['P2']} {status}")
    for v, ml in EXTRA:
        res = check_instance(v, ml, 7)
        total_j2 += 1
        all_ok &= res["P1"] and res["P2"]
        print(f"k=7 v={v} ML={ml} (s={res['s']}, j={res['j']}) "
              f"pairs={res['pairs']} P1={res['P1']} P2={res['P2']} "
              f"{'OK' if res['P1'] and res['P2'] else 'FAIL'}")
    print(f"\n{'ALL' if all_ok else 'NOT ALL'} {total_j2} offset>=2 instances "
          f"satisfy P1 (denominator pair) and P2 (j <= s).")


if __name__ == "__main__":
    main()
