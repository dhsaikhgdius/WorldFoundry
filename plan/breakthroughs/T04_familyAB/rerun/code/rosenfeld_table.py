#!/usr/bin/env python3
"""Collect the Rosenfeld criterion table: for which (k, p) does Lemma 6/7 of
Rosenfeld (arXiv:2509.14111) hold?

Criterion (cover formulation, Lemma 7): with N = (k+1)p, say v in
[1, N/2] \\ pZ covers t in [1, N/2] iff fold(t*v mod N) < p (fold = distance
to 0 mod N). The criterion for (k,p) HOLDS iff there is NO k-subset
{v_1..v_k} covering all of [1, N/2] and satisfying the gcd side condition
(every (k-1)-subset S has gcd(S u {N}) = 1). If it holds, every LRC
counterexample for k has product of speeds divisible by p.

We search WITHOUT the gcd side condition first (more covers allowed); if a
cover is found we check the side condition, and keep searching if violated.
Hence "HOLDS" answers are exact; "fails" reports a witness cover.

Backtracking with bitmask covers, branching on the lowest uncovered t,
candidates ordered by coverage size, with a capacity prune.

Output: results/rosenfeld_table.csv with columns k,p,holds,witness,seconds.
"""

import sys
import time
from math import gcd
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"


def criterion(k: int, p: int, budget: float = 60.0):
    N = (k + 1) * p
    half = N // 2
    cand = [v for v in range(1, half + 1) if v % p != 0]
    full = (1 << half) - 1
    cover = {}
    for v in cand:
        m = 0
        for t in range(1, half + 1):
            r = (t * v) % N
            if N - r < r:
                r = N - r
            if r < p:
                m |= 1 << (t - 1)
        cover[v] = m
    covers_of_t = [[] for _ in range(half + 1)]
    for v in cand:
        mv = cover[v]
        for t in range(1, half + 1):
            if (mv >> (t - 1)) & 1:
                covers_of_t[t].append(v)
    order = {v: bin(cover[v]).count("1") for v in cand}
    deadline = time.time() + budget
    sol = []

    def side_condition_ok(vs):
        for skip in range(len(vs)):
            g = N
            for i, v in enumerate(vs):
                if i != skip:
                    g = gcd(g, v)
            if g != 1:
                return False
        return True

    def bt(chosen, acc):
        if time.time() > deadline:
            raise TimeoutError
        if acc == full:
            if side_condition_ok(chosen):
                sol.append(list(chosen))
                return True
            return False
        if len(chosen) == k:
            return False
        rem = full & ~acc
        slots = k - len(chosen)
        # capacity prune
        maxgain = 0
        low_t = (rem & -rem).bit_length()      # lowest uncovered t
        pool = [v for v in covers_of_t[low_t] if not chosen or v > -1]
        # avoid duplicates: require increasing? subsets unordered; but branching
        # on forced t keeps completeness even with repeats filtered:
        pool = [v for v in pool if v not in chosen]
        if not pool:
            return False
        # prune: even taking the globally best candidates can't cover rem
        rb = bin(rem).count("1")
        best_possible = sorted((bin(cover[v] & rem).count("1") for v in pool), reverse=True)
        # rough bound: best single gain from pool times slots
        if best_possible and best_possible[0] * slots < rb:
            # candidates outside pool may cover more of rem; use global bound
            gb = max(bin(cover[v] & rem).count("1") for v in cand if v not in chosen)
            if gb * slots < rb:
                return False
        for v in sorted(pool, key=lambda x: -bin(cover[x] & rem).count("1")):
            if bt(chosen + [v], acc | cover[v]):
                return True
        return False

    t0 = time.time()
    try:
        found = bt([], 0)
        holds = not found
        witness = sol[0] if sol else None
        return holds, witness, time.time() - t0
    except TimeoutError:
        return None, None, time.time() - t0


def main():
    out = RES / "rosenfeld_table.csv"
    rows = ["k,p,holds,witness,seconds"]
    ranges = {3: range(2, 61), 4: range(2, 41), 5: range(2, 25), 6: range(2, 18)}
    if len(sys.argv) > 1:
        ks = [int(sys.argv[1])]
        ranges = {ks[0]: ranges.get(ks[0], range(2, 20))}
    for k, ps in ranges.items():
        for p in ps:
            holds, wit, secs = criterion(k, p, budget=45.0)
            tag = {True: "HOLDS", False: "fails", None: "timeout"}[holds]
            rows.append(f"{k},{p},{tag},{'' if not wit else ' '.join(map(str, wit))},{secs:.1f}")
            print(rows[-1], flush=True)
    if out.exists():
        old = [l for l in out.read_text().splitlines()[1:] if not l.startswith(f'{list(ranges)[0]},')]
        rows = [rows[0]] + old + rows[1:]
    out.write_text("\n".join(rows) + "\n")
    print(f"written {out}")


if __name__ == "__main__":
    main()
