#!/usr/bin/env python3
"""Self-tests for lrc_verifier: known exact values, cross-implementation
agreement, and float-grid sanity lower bounds."""

import random
from fractions import Fraction

from lrc_verifier import ml_exact, ml_sweep, has_witness

KNOWN = [
    # (speeds, exact ML)
    ([1], Fraction(1, 2)),
    ([1, 2], Fraction(1, 3)),
    ([2, 3], Fraction(2, 5)),          # Kravitz spectrum s=2: s/(2s+1)
    ([1, 2, 3], Fraction(1, 4)),       # tight k=3
    ([1, 2, 3, 4], Fraction(1, 5)),    # tight k=4
    ([1, 3, 4, 7], Fraction(1, 5)),    # sporadic tight k=4 (Goddyn-Wong)
    ([1, 2, 3, 4, 5], Fraction(1, 6)),
    ([1, 3, 4, 5, 9], Fraction(1, 6)),         # sporadic tight k=5
    ([1, 2, 3, 4, 5, 6], Fraction(1, 7)),      # tight k=6
    ([1, 2, 3, 4, 5, 6, 7], Fraction(1, 8)),   # tight k=7
    ([1, 4, 5, 6, 7, 11, 13], Fraction(1, 8)), # sporadic tight k=7 (GW)
    ([1, 2, 3, 4, 5, 7, 12], Fraction(1, 8)),  # sporadic tight k=7 (GW)
]


def test_known():
    for v, want in KNOWN:
        got, t = ml_exact(v)
        assert got == want, f"ml_exact({v}) = {got}, want {want}"
        # witness returned must actually achieve the value (exactness check)
        val = min(min((t * vi) % 1, 1 - (t * vi) % 1) for vi in v)
        assert val == got, f"witness t={t} gives {val} != {got} for {v}"
    print(f"[ok] {len(KNOWN)} known exact values reproduced")


def test_cross_implementation(n_trials=60, seed=42):
    rng = random.Random(seed)
    for _ in range(n_trials):
        k = rng.randint(1, 5)
        v = rng.sample(range(1, 30), k)
        a, ta = ml_exact(v)
        b, tb = ml_sweep(v)
        assert a == b, f"mismatch on {sorted(v)}: ml_exact={a} ml_sweep={b}"
    print(f"[ok] ml_exact == ml_sweep on {n_trials} random instances (k<=5, v<30)")


def test_float_grid_sanity(n_trials=20, seed=7):
    """Dense float grid never beats the exact sup (it's a lower-bound sampler)."""
    rng = random.Random(seed)
    for _ in range(n_trials):
        k = rng.randint(2, 5)
        v = rng.sample(range(1, 25), k)
        exact, _ = ml_exact(v)
        N = 20011  # prime grid
        grid = max(
            min(min((c * vi / N) % 1.0, 1 - (c * vi / N) % 1.0) for vi in v)
            for c in range(1, N)
        )
        assert grid <= float(exact) + 1e-12, f"grid {grid} exceeds exact {exact} on {v}"
        assert grid >= float(exact) - max(v) / N - 1e-9, \
            f"grid too far below exact on {v} (grid={grid}, exact={float(exact)})"
    print(f"[ok] float-grid sanity on {n_trials} random instances")


def test_witness_decision():
    assert has_witness([1, 2, 3, 4, 5, 6, 7], 1, 8) is not None
    assert has_witness([1, 2, 3, 4, 5, 6, 7], 1001, 8000) is None  # ML=1/8 exactly
    assert has_witness([2, 3], 2, 5) is not None
    assert has_witness([2, 3], 40001, 100000) is None
    print("[ok] witness decision procedure (exact threshold behaviour)")


if __name__ == "__main__":
    test_known()
    test_witness_decision()
    test_cross_implementation()
    test_float_grid_sanity()
    print("ALL TESTS PASSED")
